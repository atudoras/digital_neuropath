"""Detect primary cilia in the ac3 channel of one fov.
background -> ridge -> floor -> candidates -> skeleton seeds -> grow -> join -> tables.
A missed cilium is unrecoverable, so the premask casts wide and the capped grow keeps precision.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import skimage
from scipy import ndimage, spatial
from skimage.filters import apply_hysteresis_threshold

from digital_neuropath.cilia import filters, measure, params

# by name: the package's own load() function shadows its load module as an attribute
from digital_neuropath.cilia.load import Fov, read_plane, write_labels, write_run_json, write_table

__all__ = ["segment_cilia", "detect", "stages", "CiliaResult"]

# arc length behind an end over which its outward direction is taken
END_DIRECTION_UM = 1.0
# radius of the cylinder around an end's heading line that is searched for the object's tip
TIP_RADIUS_UM = 0.3


@dataclass
class CiliaResult:
    """The label map, its two tables and where they were written."""

    labels: np.ndarray
    morphometry: pd.DataFrame
    skeleton: pd.DataFrame
    out_dir: Path


def segment_cilia(fov: Fov, out_dir: Path, **overrides: Any) -> CiliaResult:
    """Detect cilia in the ac3 channel of one fov and write the label map and its two tables."""
    p, status = params.resolve(params.CILIA, overrides)
    spacing = fov.spacing_um
    raw = read_plane(fov, "ac3")
    labels = detect(raw, spacing, p)
    morphometry = measure.morphometry(labels, raw, spacing, foci=False)
    skeleton = measure.axoneme_table(labels, raw, spacing, bridge_gap_um=p["bridge_gap_um"],
                                     prune_um=p["prune_um"],
                                     curvature_smooth_um=p["curvature_smooth_um"],
                                     collapse_radius_um=p["collapse_radius_um"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_labels(out_dir / "labels.tif", labels)
    write_table(out_dir, "morphometry", morphometry)
    write_table(out_dir, "skeleton", skeleton)
    write_run_json(out_dir, fov=fov, pipeline="cilia", values=p, status=status,
                   inputs={"image": str(fov.image)})
    fov.cilia_dir = out_dir
    return CiliaResult(labels, morphometry, skeleton, out_dir)


def detect(raw: np.ndarray, spacing: tuple[float, float, float], p: dict[str, Any]) -> np.ndarray:
    """Return the cilium label map of one ac3 volume under resolved CILIA values."""
    return stages(raw, spacing, p)["final"]


def stages(raw: np.ndarray, spacing: tuple[float, float, float], p: dict[str, Any]
           ) -> dict[str, np.ndarray]:
    """Return every intermediate mask of the detector, ending in the final label map."""
    # the premask ran in float32 and the grow reads the plane as stored; each is its own bits
    raw32 = raw.astype(np.float32)
    clean = filters.subtract_background(raw32, spacing, sigma_um=p["sigma_bg_um"])
    response = filters.ridge(clean, sigma_vox=ridge_sigma_vox(p["ridge_sigma_um"], spacing),
                             z_chunk=p["z_chunk"], z_halo=p["z_halo"])
    level, scale = background(raw32, spacing, p)
    ridge = response > filters.positive_percentile(response, p["ridge_pct"])
    floor = raw32 > level + float(p["floor_k"]) * scale
    mask = ridge & floor
    premask = candidates(mask, min_voxels=p["min_voxels"], min_aspect=p["min_aspect"])
    # candidates are disjoint, so one skeleton of their union skeletonizes each alone
    seeds = skimage.morphology.skeletonize(premask > 0)
    if p["grow"] == "capped":
        volume = filters.subtract_background(np.asarray(raw, dtype=float), spacing,
                                             p["grow_sigma_bg_um"])
        blurred = filters.smooth(volume, spacing, p["sigma_blur_um"])
        grow_thr = blurred > filters.positive_percentile(blurred, p["threshold_pct"])
        grow_cap = grow_thr & (np.asarray(ndimage.distance_transform_edt(~seeds, sampling=spacing))
                               <= p["max_grow_um"])
    elif p["grow"] == "hysteresis":
        # background measured on the blurred image it thresholds, so k is in that image's noise
        blurred = filters.smooth(raw32, spacing, p["sigma_blur_um"])
        level_b, scale_b = background(blurred, spacing, p)
        contrast = (blurred - level_b) / np.maximum(scale_b, np.finfo(np.float32).tiny)
        grow_thr = np.asarray(apply_hysteresis_threshold(contrast, p["grow_low_k"],
                                                         p["grow_high_k"]))
        grow_cap = grow_thr
    else:
        raise ValueError(f"grow must be 'capped' or 'hysteresis', not {p['grow']!r}")
    labels = seeded(grow_cap, seeds, min_volume_vox=p["min_volume_vox"])
    if p["join_gap_um"] > 0:
        labels = join(labels, spacing, gap_um=p["join_gap_um"], angle_deg=p["join_angle_deg"])
    return {"bg": clean > 0, "ridge": ridge, "floor": floor, "mask": mask, "premask": premask,
            "seeds": seeds, "grow_thr": grow_thr, "grow_cap": grow_cap, "final": labels}


def ridge_sigma_vox(sigma_um: float | tuple[float, float, float],
                    spacing: tuple[float, float, float]) -> float | tuple[float, float, float]:
    """Return the ridge σ in voxels: a float is the shipped y-converted σ on every axis, a (z, y, x)
    tuple is per axis in µm."""
    if isinstance(sigma_um, (tuple, list)):
        sz, sy, sx = (float(s) for s in sigma_um)
        return (sz / spacing[0], sy / spacing[1], sx / spacing[2])
    return filters.um_to_vox(sigma_um, spacing)[1]


def background(raw32: np.ndarray, spacing: tuple[float, float, float], p: dict[str, Any]
               ) -> tuple[Any, Any]:
    """Return the background level and σ: field-wide scalars, or per-voxel maps within a radius."""
    if p["background"] == "global":
        return filters.robust_background(raw32)
    if p["background"] == "local":
        return filters.local_background(raw32, spacing, p["background_radius_um"])
    raise ValueError(f"background must be 'global' or 'local', not {p['background']!r}")


def candidates(mask: np.ndarray, *, min_voxels: int, min_aspect: float) -> np.ndarray:
    """Label the mask's components and keep those large and elongated enough, renumbered from 1."""
    labels = np.asarray(skimage.measure.label(mask, connectivity=3))
    keep = np.zeros(int(labels.max()) + 1, dtype=bool)
    for region in skimage.measure.regionprops(labels):
        if int(region.area) < int(min_voxels):
            continue
        try:
            ratio = float(region.axis_major_length) / (float(region.axis_minor_length) + 1.0)
        except ValueError:
            # skimage raises when a region is so thin an inertia eigenvalue goes negative: the
            # most cilium-like case, so it is kept
            ratio = float("inf")
        keep[int(region.label)] = ratio >= float(min_aspect)
    # the survivors are reconnected, so two kept ids that touch become one candidate
    return np.asarray(skimage.measure.label(keep[labels], connectivity=3)).astype(np.int32)


def seeded(binary: np.ndarray, seeds: np.ndarray, *, min_volume_vox: int) -> np.ndarray:
    """Label the grown mask and keep the components that hold a seed and are large enough."""
    labels = np.asarray(skimage.measure.label(binary, connectivity=3)).astype(np.int32)
    keep = np.bincount(labels.ravel()) >= int(min_volume_vox)
    hit = np.zeros(keep.size, dtype=bool)
    hit[labels[seeds]] = True
    # renumbered, not reconnected: two seeded components that touch stay two objects
    return filters.relabel(labels, keep & hit)


def join(labels: np.ndarray, spacing: tuple[float, float, float], *, gap_um: float,
         angle_deg: float) -> np.ndarray:
    """Join object tips closer than gap_um whose outward directions both point at the other tip
    within angle_deg, drawing the gap in, and renumber from 1."""
    array = np.asarray(labels).copy()
    scale = np.asarray(spacing, dtype=float)
    pad = [int(np.ceil(gap_um / s)) + 2 for s in spacing]
    owner: list[int] = []
    where: list[np.ndarray] = []
    heading: list[np.ndarray] = []
    for label, box in enumerate(ndimage.find_objects(array), start=1):
        if box is None:
            continue
        crop = filters.padded_bbox(box, array.shape, pad)
        window = array[crop]
        # the centreline is the costly step, so only objects with a neighbour in reach get one
        if not np.any((window > 0) & (window != label)):
            continue
        points, _, length, path = measure.centreline(window == label, spacing)
        if length <= 0:
            continue
        trail = (points[path] + np.array([s.start for s in crop])) * scale
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(trail, axis=0), axis=1))])
        voxels = (np.argwhere(window == label) + np.array([s.start for s in crop])) * scale
        for ordered, run in ((trail, arc), (trail[::-1], arc[-1] - arc[::-1])):
            back = ordered[min(int(np.searchsorted(run, END_DIRECTION_UM)), len(ordered) - 1)]
            out = ordered[0] - back
            out = out / np.linalg.norm(out)
            # the skeleton end sits inside the tip: move out along the heading to the last voxel
            # of the object within the tip cylinder, so flank voxels cannot push the tip
            offset = voxels - ordered[0]
            along = offset @ out
            across = np.linalg.norm(offset - np.outer(along, out), axis=1)
            reach = along[(along > 0) & (across <= TIP_RADIUS_UM)]
            owner.append(label)
            where.append(ordered[0] + out * (reach.max() if reach.size else 0.0))
            heading.append(out)
    if len(owner) < 2:
        return array
    ends = np.asarray(where)
    cos_min = float(np.cos(np.deg2rad(angle_deg)))
    accepted: list[tuple[float, int, int]] = []
    for a, b in spatial.KDTree(ends).query_pairs(gap_um):
        if owner[a] == owner[b]:
            continue
        step = ends[b] - ends[a]
        dist = float(np.linalg.norm(step))
        if dist == 0.0:
            continue
        if heading[a] @ step / dist >= cos_min and heading[b] @ -step / dist >= cos_min:
            accepted.append((dist, a, b))
    root = np.arange(int(array.max()) + 1)

    def find(x: int) -> int:
        """Return the merged id a label now belongs to."""
        while root[x] != x:
            x = int(root[x])
        return x

    used: set[int] = set()
    for dist, a, b in sorted(accepted):
        ra, rb = find(owner[a]), find(owner[b])
        if a in used or b in used or ra == rb:
            continue
        used |= {a, b}
        keep, gone = min(ra, rb), max(ra, rb)
        root[gone] = keep
        n = int(np.ceil(dist / scale.min())) * 2 + 2
        line = np.rint(np.linspace(ends[a], ends[b], n) / scale).astype(int)
        line = line[np.all((line >= 0) & (line < array.shape), axis=1)]
        cells = tuple(line.T)
        array[cells] = np.where(array[cells] == 0, keep, array[cells])
    lut = np.array([find(x) for x in range(root.size)], dtype=np.int32)
    merged = lut[array]
    return filters.relabel(merged, np.bincount(merged.ravel(), minlength=root.size) > 0)
