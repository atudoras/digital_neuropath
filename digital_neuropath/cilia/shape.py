"""Per-object cilium shape on a label map, built only from measure.centreline and measure.axoneme_table.
cilium_shape -> arc (= length_signal_um), chord between the same path's ends, tortuosity, curvature,
z-extent and stack-face contact. straight_tube is the phantom for the tortuosity quantisation floor;
component_length is the length of a label painted in pieces (summed, not bridged).
"""
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import morphology

from digital_neuropath.cilia import filters, measure

__all__ = ["cilium_shape", "straight_tube", "component_length", "SHAPE_COLUMNS"]

SHAPE_COLUMNS = ["label", "n_voxels", "arc_um", "chord_um", "tortuosity", "bridged_arc_um",
                 "total_curvature_rad", "max_curvature_inv_um", "mean_curvature_inv_um",
                 "z_min", "z_max", "z_extent_um", "touches_z_face", "touches_xy_face",
                 "branch_count", "blob_r_eff_um", "blob_flag",
                 "end_a_z", "end_a_y", "end_a_x", "end_b_z", "end_b_y", "end_b_x"]


def _shape_one(binary: np.ndarray, raw: np.ndarray, origin: np.ndarray, full_shape: tuple[int, ...],
               spacing: tuple[float, float, float], p: dict[str, float]) -> dict[str, Any]:
    """Shape of one object given as a boolean crop whose margin exceeds the bridging pad."""
    sub = binary.astype(np.int32)
    table = measure.axoneme_table(sub, raw, spacing, bridge_gap_um=p["bridge_gap_um"],
                                  prune_um=p["prune_um"], curvature_smooth_um=p["curvature_smooth_um"],
                                  collapse_radius_um=p["collapse_radius_um"])
    row = table.iloc[0]
    scale = np.asarray(spacing, float)
    points, _, arc, path = measure.centreline(binary, spacing)
    if arc > 0 and len(path) >= 2:
        a, b = points[path[0]] + origin, points[path[-1]] + origin
    elif len(points):
        a = b = points[0] + origin
    else:
        a = b = np.full(3, -1)
    chord = float(np.linalg.norm((a - b) * scale)) if arc > 0 else 0.0
    closing = filters.ellipsoid_footprint(p["bridge_gap_um"] / 2.0, spacing)
    _, _, bridged, _ = measure.centreline(morphology.closing(binary, closing), spacing)
    zs = np.nonzero(binary.any(axis=(1, 2)))[0] + origin[0]
    nz, ny, nx = full_shape
    vox = np.argwhere(binary) + origin
    turning = float(row["total_curvature_rad"])
    return dict(
        n_voxels=int(binary.sum()), arc_um=float(arc), chord_um=chord,
        tortuosity=(arc / chord) if chord > 0 else float("nan"), bridged_arc_um=float(bridged),
        total_curvature_rad=turning, max_curvature_inv_um=float(row["max_curvature_inv_um"]),
        mean_curvature_inv_um=(turning / bridged) if (bridged > 0 and turning == turning) else float("nan"),
        z_min=int(zs.min()), z_max=int(zs.max()), z_extent_um=float((zs.max() - zs.min()) * spacing[0]),
        touches_z_face=bool((vox[:, 0] == 0).any() or (vox[:, 0] == nz - 1).any()),
        touches_xy_face=bool((vox[:, 1] == 0).any() or (vox[:, 1] == ny - 1).any()
                             or (vox[:, 2] == 0).any() or (vox[:, 2] == nx - 1).any()),
        branch_count=int(row["branch_count"]), blob_r_eff_um=float(row["blob_r_eff_um"]),
        blob_flag=bool(row["blob_flag"]),
        end_a_z=int(a[0]), end_a_y=int(a[1]), end_a_x=int(a[2]),
        end_b_z=int(b[0]), end_b_y=int(b[1]), end_b_x=int(b[2]))


def cilium_shape(labels: np.ndarray, raw: np.ndarray, spacing: tuple[float, float, float],
                 ids: list[int] | None = None, **params: float) -> pd.DataFrame:
    """Return one row per requested label: centreline arc and chord, tortuosity, curvature, z-extent
    and face contact. params are the axoneme_table knobs (bridge_gap_um, prune_um,
    curvature_smooth_um, collapse_radius_um); the shipped params.CILIA values are the defaults."""
    from digital_neuropath.cilia import params as P
    p = {k: P.CILIA[k].value for k in ("bridge_gap_um", "prune_um", "curvature_smooth_um", "collapse_radius_um")}
    p.update(params)
    array = np.asarray(labels)
    boxes = ndimage.find_objects(array)
    if ids is None:
        ids = [i + 1 for i, b in enumerate(boxes) if b is not None]
    # two voxels beyond axoneme_table's own pad, so its crop of our crop equals its crop of the frame
    pad = [max(1, int(round(p["bridge_gap_um"] / s))) + 4 for s in spacing]
    rows = []
    for label in ids:
        box = boxes[label - 1] if 0 < label <= len(boxes) else None
        if box is None:
            continue
        crop = filters.padded_bbox(box, array.shape, pad)
        origin = np.array([c.start for c in crop])
        row = _shape_one(array[crop] == label, np.asarray(raw[crop]), origin, array.shape, spacing, p)
        rows.append({"label": int(label), **row})
    return pd.DataFrame(rows, columns=SHAPE_COLUMNS)


def straight_tube(length_um: float, radius_um: float, direction: np.ndarray,
                  spacing: tuple[float, float, float], offset_vox: np.ndarray | None = None,
                  margin_vox: int = 6) -> np.ndarray:
    """Boolean phantom of a straight capsule: voxel centres within radius_um of a segment of length
    length_um along direction (z, y, x), centred in its grid with a sub-voxel offset."""
    s = np.asarray(spacing, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    half = 0.5 * length_um * d
    ext = np.abs(half) + radius_um
    n = (2 * np.ceil(ext / s)).astype(int) + 2 * margin_vox + 1
    centre = (n - 1) / 2.0 + (np.zeros(3) if offset_vox is None else np.asarray(offset_vox, float))
    grid = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), axis=-1).astype(float)
    pos = (grid - centre) * s
    t = np.clip(pos @ d, -0.5 * length_um, 0.5 * length_um)
    nearest = t[..., None] * d
    return np.linalg.norm(pos - nearest, axis=-1) <= radius_um


def component_length(binary: np.ndarray, spacing: tuple[float, float, float]) -> tuple[float, int]:
    """Sum of measure.centreline length over the 26-connected components of a mask, and their count."""
    comps, n = ndimage.label(binary, structure=np.ones((3, 3, 3), bool))
    total = 0.0
    for k, box in enumerate(ndimage.find_objects(comps), start=1):
        if box is None:
            continue
        crop = filters.padded_bbox(box, comps.shape, [2, 2, 2])
        total += measure.centreline(comps[crop] == k, spacing)[2]
    return float(total), int(n)
