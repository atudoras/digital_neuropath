"""Assign cilia and foci to neurons and decide, per neuron, whether it has a cilium.
censor_neurons -> describe -> select_cilia -> pcnt_readouts, mtoc_geometry -> rows, edges, table.
A real cilium starts at the soma surface and points out, which the ac3 channel alone cannot show.
"""
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import morphology

from digital_neuropath.cilia import filters, measure, params
from digital_neuropath.cilia.cilia import CiliaResult

# by name: the package's own load() function shadows its load module as an attribute
from digital_neuropath.cilia.load import (
    GEOMETRY,
    Fov,
    frame_of,
    read_cmap,
    read_labels,
    read_plane,
    read_table,
    write_run_json,
    write_table,
)
from digital_neuropath.cilia.mtocs import MtocResult

__all__ = ["assign_to_neurons", "LinkResult", "Review", "decisions_digest", "kept_of"]

# R1 of stages_refined_78.md: refine counts as included; a close-up decision overrides the status
STATUS_KEPT = {"good": True, "whole": True, "refine": True, "cut": False, "discard": False}
CLOSEUP_KEPT = {"include": True, "mask_wrong": True, "exclude": False}

NOMINATION_REASON = ("fully_intracellular: every voxel inside the neuron instance; the object "
                     "never crosses the membrane")


@dataclass
class Review:
    """One fov's completed step-2 review: status and close-up decision per instance label."""

    status: dict[int, str]
    closeup: dict[int, str]


@dataclass
class LinkResult:
    """The six tables of one fov and where they were written."""

    censoring: pd.DataFrame
    cilia: pd.DataFrame
    neurons: pd.DataFrame
    mtocs: pd.DataFrame
    links: pd.DataFrame
    neuron_table: pd.DataFrame
    out_dir: Path


def assign_to_neurons(fov: Fov, cilia: CiliaResult | None, mtocs: MtocResult | None,
                      out_dir: Path, *, refined: list[dict[str, str]] | None = None,
                      review: Review | None = None, **overrides: Any) -> LinkResult:
    """Link one fov's cilia and foci to its neurons and write the six tables."""
    p, status = params.resolve(params.LINK, overrides)
    spacing = fov.spacing_um
    if fov.neun_instances is None:
        raise ValueError("assign_to_neurons needs the neun instance masks; pass neun_instances to load")
    if cilia is None:
        if fov.cilia_dir is None:
            raise ValueError("segment_cilia has not run on this fov; pass its result")
        cilia = CiliaResult(read_labels(fov.cilia_dir / "labels.tif"),
                            read_table(fov.cilia_dir, "morphometry"),
                            read_table(fov.cilia_dir, "skeleton"), fov.cilia_dir)
    if mtocs is None:
        if fov.mtoc_dir is None:
            raise ValueError("segment_mtocs has not run on this fov; pass its result")
        mtocs = MtocResult(read_labels(fov.mtoc_dir / "labels.tif"),
                           read_table(fov.mtoc_dir, "pcnt_morphometry"), fov.mtoc_dir)
    # the uid namespace is the instance mask's stem
    donor, fov_id = fov.donor, fov.neun_instances.stem
    records = [json.loads((r.out_dir / "run.json").read_text()) for r in (cilia, mtocs)]
    neurons_all, nucleus_all, excluded = read_cmap(fov.neun_instances)
    # censoring sees every delivered instance; everything after it, only those not excluded,
    # unless a review exists: the rater then judged every body and pairing runs over all of them
    drop = [] if review is not None else excluded
    neurons = np.where(np.isin(neurons_all, drop), 0, neurons_all) if drop else neurons_all
    nucleus = np.where(np.isin(nucleus_all, drop), 0, nucleus_all) if drop else nucleus_all
    censoring = censor_neurons(neurons_all, nucleus_all, spacing, excluded=excluded,
                               donor=donor, fov_id=fov_id)
    geometry = describe(cilia.labels, neurons, nucleus, spacing, max_reach_um=p["max_reach_um"],
                        inside_frac=p["inside_frac"])
    objects = select_cilia(geometry, cilia, spacing, donor=donor, fov_id=fov_id,
                           max_base_dist_um=p["max_base_dist_um"],
                           min_tip_out_um=p["min_tip_out_um"],
                           bridge_gap_um=records[0]["values"]["bridge_gap_um"])
    readouts = pcnt_readouts(read_plane(fov, "pcnt"), mtocs.labels, neurons, spacing,
                             max_reach_um=p["max_reach_um"], peak_radius_um=p["peak_radius_um"],
                             peak_k_sigma=p["peak_k_sigma"])
    foci = mtoc_geometry(mtocs.labels, neurons, spacing, objects, donor=donor, fov_id=fov_id,
                         max_reach_um=p["max_reach_um"], inside_frac=p["inside_frac"])
    neuron_rows = neuron_rows_of(readouts, objects, foci, donor=donor, fov_id=fov_id)
    links = link_edges(objects, foci, max_base_dist_um=p["max_base_dist_um"])
    # the weakest status over the whole chain, not over this call's knobs alone
    status = max([status, *(r["status"] for r in records)], key=params.STATUSES.index)
    kept = kept_of([int(v) for v in censoring["neuron_id"]], excluded, refined, review)
    table = neuron_table(censoring=censoring, neurons=neuron_rows, cilia=objects, status=status,
                         kept=kept)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("censoring", censoring), ("cilia", objects), ("neurons", neuron_rows),
                        ("mtocs", foci), ("links", links), ("neuron_table", table)):
        write_table(out_dir, name, frame)
    write_run_json(out_dir, fov=fov, pipeline="link", values=p, status=status,
                   inputs={"image": str(fov.image), "neun_instances": str(fov.neun_instances),
                           "cilia": str(cilia.out_dir), "mtocs": str(mtocs.out_dir),
                           **decisions_digest(refined, review)})
    return LinkResult(censoring, objects, neuron_rows, foci, links, table, out_dir)


def censor_neurons(neurons: np.ndarray, nucleus: np.ndarray, spacing: tuple[float, float, float],
                   *, excluded: list[int], donor: str, fov_id: str) -> pd.DataFrame:
    """Record, per delivered instance, the image faces it reaches, whether the producer's own
    lateral-nucleus rule fires, and the producer's exclusion."""
    nucleus_boxes = ndimage.find_objects(nucleus)
    counts = np.bincount(neurons.ravel())
    voxel_um3 = spacing[0] * spacing[1] * spacing[2]
    shape = tuple(int(v) for v in neurons.shape)
    rows = []
    for neuron_id, box in enumerate(ndimage.find_objects(neurons), start=1):
        if box is None:
            continue
        touches: dict[str, bool] = {}
        for axis, sl, extent in zip("zyx", box, shape, strict=True):
            touches[f"touches_{axis}_min"] = sl.start == 0
            touches[f"touches_{axis}_max"] = sl.stop == extent
        gaps = [(float(sl.start) * step, float(extent - sl.stop) * step)
                for sl, extent, step in zip(box, shape, spacing, strict=True)]
        axial = touches["touches_z_min"] or touches["touches_z_max"]
        lateral = any(touches[f"touches_{axis}_{end}"] for axis in "yx" for end in ("min", "max"))
        word = "+".join(name for name, fired in (("axial", axial), ("lateral", lateral)) if fired)
        nucleus_box = nucleus_boxes[neuron_id - 1] if neuron_id <= len(nucleus_boxes) else None
        n_voxels = int(counts[neuron_id])
        rows.append({"donor": donor, "fov": fov_id,
                     "neuron_uid": f"{donor}:{fov_id}:n{neuron_id:04d}", "neuron_id": neuron_id,
                     "group": "cell_body", "n_voxels": n_voxels,
                     "volume_um3": n_voxels * voxel_um3, **touches,
                     "z_span_planes": int(box[0].stop - box[0].start), "z_margin_um": min(gaps[0]),
                     "lateral_margin_um": min(*gaps[1], *gaps[2]), "censoring": word or "none",
                     # the segmenter's rule is a nucleus on a lateral face, never z; a cell body
                     # with no nucleus has nothing for it to fire on
                     "producer_rule_fires": nucleus_box is not None and any(
                         sl.start == 0 or sl.stop == extent
                         for sl, extent in zip(nucleus_box[1:], shape[1:], strict=True)),
                     "producer_excluded": neuron_id in excluded})
    frame = frame_of(rows, "censoring")
    frame["producer_rule_fires"] = frame["producer_rule_fires"].astype("boolean")
    return frame


def neuron_field(neurons: np.ndarray, instance_id: int, box: Sequence[slice], pad: Sequence[int],
                 spacing: tuple[float, float, float], *, signed: bool) -> tuple[np.ndarray, np.ndarray]:
    """Distance to one instance on its box padded by the reach (negative inside when signed), with its origin."""
    crop = filters.padded_bbox(box, neurons.shape, pad)
    soma = neurons[crop] == instance_id
    field = np.asarray(ndimage.distance_transform_edt(~soma, sampling=spacing))
    if signed:
        field = field - np.asarray(ndimage.distance_transform_edt(soma, sampling=spacing))
    return field, np.array([axis.start for axis in crop])


def crop_readout(neurons: np.ndarray, crop: Sequence[slice], inside: np.ndarray, coords: np.ndarray,
                 owner: int, spacing: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray | None]:
    """Signed distance at the object's voxels and its gradient at the lowest one, from the object's own crop."""
    soma = neurons[crop] == owner
    # negative inside, positive outside, so the gradient exists on both sides of the wall
    sdf = (np.asarray(ndimage.distance_transform_edt(~soma, sampling=spacing))
           - np.asarray(ndimage.distance_transform_edt(soma, sampling=spacing)))
    values = sdf[inside]
    if min(sdf.shape) < 2:
        return values, None
    b = int(np.argmin(values))
    return values, np.array([float(g[tuple(coords[b])]) for g in np.gradient(sdf, *spacing)],
                            dtype=np.float64)


def field_readout(field: np.ndarray, origin: np.ndarray, coords: np.ndarray, spacing: tuple[float, float, float],
                  *, reach: float, shape: Sequence[int], gradient: bool) -> tuple[np.ndarray, np.ndarray | None] | None:
    """Same readout from the neuron's field, or None where only the object's own crop gives the old value.

    Within the reach the field and the crop agree exactly; beyond it the crop's cut-off changes the old value.
    """
    index = coords - origin
    if (index < 0).any() or (index >= field.shape).any():
        return None
    values = field[tuple(index.T)]
    if np.abs(values).max() > reach:
        return None
    if not gradient:
        return values, None
    base = index[int(np.argmin(values))]
    normal = np.empty(3, dtype=np.float64)
    for a in range(3):
        n, i = field.shape[a], int(base[a])
        lo, hi = max(i - 1, 0), min(i + 1, n - 1)
        # an array edge that is not the volume edge has no counterpart in the crop
        if lo == hi or (i == 0 and origin[a] != 0) or (i == n - 1 and origin[a] + n != shape[a]):
            return None
        below, above = base.copy(), base.copy()
        below[a], above[a] = lo, hi
        pair = field[tuple(below)], field[tuple(above)]
        if max(abs(pair[0]), abs(pair[1])) > reach:
            return None
        normal[a] = (pair[1] - pair[0]) / ((hi - lo) * spacing[a])
    return values, normal


def describe(objects: np.ndarray, neurons: np.ndarray, nucleus: np.ndarray | None,
             spacing: tuple[float, float, float], *, max_reach_um: float,
             inside_frac: float) -> pd.DataFrame:
    """Measure each object against the instance that owns it, from one distance field per neuron."""
    scale = np.asarray(spacing, dtype=np.float64)
    voxel_um3 = spacing[0] * spacing[1] * spacing[2]
    pad = [int(np.ceil(max_reach_um / s)) + 1 for s in spacing]
    neuron_boxes = ndimage.find_objects(neurons)
    rows: dict[int, dict[str, Any]] = {}
    crops: dict[int, tuple[slice, ...]] = {}
    near: dict[int, list[int]] = {}
    for label, box in enumerate(ndimage.find_objects(objects), start=1):
        if box is None:
            continue
        crop = filters.padded_bbox(box, objects.shape, pad)
        crops[label] = crop
        origin = np.array([axis.start for axis in crop])
        inside = objects[crop] == label
        territory = neurons[crop]
        coords = np.argwhere(inside)
        n_voxels = len(coords)
        # the owner shares the most voxels; an object touching none goes to the nearest in reach
        owner, n_shared, owner_by = 0, 0, "overlap"
        shared = territory[inside]
        shared = shared[shared > 0]
        if shared.size:
            ids, counts = np.unique(shared, return_counts=True)
            best = int(np.argmax(counts))
            owner, n_shared = int(ids[best]), int(counts[best])
        else:
            present = [int(v) for v in np.unique(territory) if v != 0]
            if present:
                near[label] = present
            owner_by = "none"
        frac_in_soma = n_shared / n_voxels if n_voxels else 0.0
        frac_in_nucleus = 0.0
        if nucleus is not None and n_voxels:
            frac_in_nucleus = float((nucleus[crop][inside] > 0).sum()) / n_voxels
        base_dist = tip_out = sdf_span = outward_cos = float("nan")
        base: tuple[Any, ...] = (None,) * 3
        tip: tuple[Any, ...] = (None,) * 3
        aspect = float("inf")
        if n_voxels >= 2:
            cloud = coords * scale
            eigenvalues = np.clip(np.linalg.eigvalsh(np.cov(cloud - cloud.mean(axis=0),
                                                            rowvar=False)), 0.0, None)
            major, minor = float(np.sqrt(eigenvalues[-1])), float(np.sqrt(eigenvalues[0]))
            aspect = float("inf") if minor <= 0.0 else major / minor
        centroid = (coords.mean(axis=0) + origin) if n_voxels else np.zeros(3)
        rows[label] = ({"label": label, "owner_neuron": owner, "owner_by": owner_by,
                     "n_voxels": n_voxels, "volume_um3": n_voxels * voxel_um3,
                     "frac_in_soma": frac_in_soma, "frac_in_nucleus": frac_in_nucleus,
                     "containment": ("INSIDE" if frac_in_soma >= inside_frac
                                     else "CROSSING" if frac_in_soma > 0.0 else "FREE"),
                     "base_dist_um": base_dist, "tip_out_um": tip_out, "sdf_span_um": sdf_span,
                     "outward_cos": outward_cos, "aspect": aspect,
                     **{f"centroid_{a}": float(v) for a, v in zip("zyx", centroid, strict=True)},
                     **{f"base_{a}": v for a, v in zip("zyx", base, strict=True)},
                     **{f"tip_{a}": v for a, v in zip("zyx", tip, strict=True)}})
    shape = objects.shape

    def voxels(label: int) -> np.ndarray:
        crop = crops[label]
        return np.argwhere(objects[crop] == label) + np.array([axis.start for axis in crop])

    # an object touching no neuron takes the nearest one in reach, from each neuron's distance field
    nearest = {label: (0, float("inf")) for label in near}
    for instance_id in sorted({i for ids in near.values() for i in ids}):
        field, field_origin = neuron_field(neurons, instance_id, neuron_boxes[instance_id - 1], pad,
                                           spacing, signed=False)
        for label, ids in near.items():
            if instance_id not in ids:
                continue
            index = voxels(label) - field_origin
            # a voxel outside the padded box is further than the reach from this neuron
            index = index[((index >= 0) & (index < field.shape)).all(axis=1)]
            gap = float(field[tuple(index.T)].min()) if len(index) else float("inf")
            if gap < nearest[label][1]:
                nearest[label] = (instance_id, gap)
    for label, (instance_id, gap) in nearest.items():
        if gap <= max_reach_um:
            rows[label]["owner_neuron"], rows[label]["owner_by"] = instance_id, "proximity"
    by_owner: dict[int, list[int]] = {}
    for label, row in rows.items():
        if row["owner_neuron"]:
            by_owner.setdefault(int(row["owner_neuron"]), []).append(label)
    for owner, labels in sorted(by_owner.items()):
        field, field_origin = neuron_field(neurons, owner, neuron_boxes[owner - 1], pad, spacing, signed=True)
        for label in labels:
            crop = crops[label]
            origin = np.array([axis.start for axis in crop])
            inside = objects[crop] == label
            coords = np.argwhere(inside)
            gradient = min(inside.shape) >= 2
            # the field matches the crop's own distance only within the reach; elsewhere the crop is the reference
            read = field_readout(field, field_origin, coords + origin, spacing, reach=max_reach_um,
                                 shape=shape, gradient=gradient)
            values, normal = read or crop_readout(neurons, crop, inside, coords, owner, spacing)
            b, t = int(np.argmin(values)), int(np.argmax(values))
            # 0 once the object reaches inside, never a small positive gap
            base_dist = max(0.0, float(values[b]))
            tip_out = float(values[t])
            outward_cos = float("nan")
            if normal is not None:
                axis = (coords[t].astype(np.float64) - coords[b].astype(np.float64)) * scale
                normal_norm, axis_norm = float(np.linalg.norm(normal)), float(np.linalg.norm(axis))
                if normal_norm != 0.0 and axis_norm != 0.0:
                    outward_cos = float(np.dot(axis / axis_norm, normal / normal_norm))
            rows[label].update(
                {"base_dist_um": base_dist, "tip_out_um": tip_out, "sdf_span_um": tip_out - float(values[b]),
                 "outward_cos": outward_cos,
                 **{f"base_{a}": int(v) for a, v in zip("zyx", coords[b] + origin, strict=True)},
                 **{f"tip_{a}": int(v) for a, v in zip("zyx", coords[t] + origin, strict=True)}})
    return pd.DataFrame(list(rows.values()), columns=[c.name for c in GEOMETRY])


def select_cilia(geometry: pd.DataFrame, cilia: CiliaResult, spacing: tuple[float, float, float],
                 *, donor: str, fov_id: str, max_base_dist_um: float, min_tip_out_um: float,
                 bridge_gap_um: float) -> pd.DataFrame:
    """Key each object, attach its measurements, nominate on geometry, apply the eligibility gate
    and choose one object per neuron, ranking by bridged length."""
    out = geometry.copy()
    out.insert(0, "donor", donor)
    out.insert(1, "fov", fov_id)
    out.insert(2, "cilium_uid", [f"{donor}:{fov_id}:c{int(v):04d}" for v in out["label"]])
    out.insert(3, "neuron_uid", [f"{donor}:{fov_id}:n{int(v):04d}" if v else ""
                                 for v in out["owner_neuron"]])
    measured = cilia.morphometry[["label", "ac3_mean", "ac3_integrated"]].merge(
        cilia.skeleton[["label", "length_signal_um", "blob_flag", "blob_r_eff_um", "branch_count"]],
        on="label", how="left")
    out = out.merge(measured, on="label", how="left")
    # nominated on geometry alone; nothing is dropped
    nominated = np.asarray((out["owner_neuron"] > 0) & (out["containment"] == "INSIDE"), dtype=bool)
    nomination = np.array([""] * len(out), dtype=object)
    nomination[nominated] = NOMINATION_REASON
    out["nominated"] = nominated
    out["nomination_reason"] = nomination
    # the first predicate to fire owns the reason; the predicates overlap, so it is not a cause.
    # "fully contained" never fires after the nomination, whatever inside_frac is, so it is gone
    reasons = np.array([""] * len(out), dtype=object)
    keep = np.ones(len(out), dtype=bool)
    for fired, reason in (
            (out["owner_neuron"] <= 0, "no neuron within reach"),
            (out["nominated"], "nominated for discard"),
            (out["base_dist_um"].fillna(np.inf) > max_base_dist_um,
             f"base further than {max_base_dist_um:g} um from the surface"),
            (out["tip_out_um"].fillna(-np.inf) < min_tip_out_um,
             f"clears the surface by less than {min_tip_out_um:g} um")):
        fired = np.asarray(fired, dtype=bool) & keep
        reasons[fired] = reason
        keep[fired] = False
    # the sort key, measured where the gate still needs it; a missing skeleton row reads NaN
    length = pd.Series(np.nan, index=out.index)
    closing = filters.ellipsoid_footprint(bridge_gap_um / 2.0, spacing)
    pad = [max(1, int(round(bridge_gap_um / s))) + 2 for s in spacing]
    boxes = ndimage.find_objects(cilia.labels)
    for i in out.index[keep & out["length_signal_um"].notna().to_numpy()]:
        label = int(out.at[i, "label"])
        box = boxes[label - 1]
        assert box is not None
        binary = cilia.labels[filters.padded_bbox(box, cilia.labels.shape, pad)] == label
        length[i] = max(measure.centreline(morphology.closing(binary, closing), spacing)[2],
                        measure.centreline(binary, spacing)[2])
    # a missing centreline row passes, as v0.1.0 did; only a measured zero is refused
    fired = np.asarray(length.fillna(1.0) <= 0.0, dtype=bool) & keep
    reasons[fired] = "no centerline path"
    keep[fired] = False
    out["eligible"] = keep
    out["exclusion_reason"] = reasons
    out["chosen"] = False
    eligible = out[keep].assign(length_bridged_um=length[keep])
    if len(eligible):
        # label breaks the remaining ties, arbitrarily
        ordered = eligible.sort_values(["length_bridged_um", "tip_out_um", "label"],
                                       ascending=[False, False, True])
        out.loc[ordered.groupby("owner_neuron", sort=False).head(1).index, "chosen"] = True
    return out


def pcnt_readouts(pcnt: np.ndarray, foci: np.ndarray, neurons: np.ndarray,
                  spacing: tuple[float, float, float], *, max_reach_um: float,
                  peak_radius_um: float, peak_k_sigma: float) -> pd.DataFrame:
    """Return per-instance pcnt statistics from the raw plane and the focus map, side by side."""
    image = np.asarray(pcnt, dtype=np.float64)
    median, sigma = filters.robust_background(image)
    voxel_um3 = spacing[0] * spacing[1] * spacing[2]
    local_max = ndimage.maximum_filter(
        image, footprint=filters.ellipsoid_footprint(peak_radius_um, spacing), mode="nearest")
    peaks = (image >= local_max) & (image > median + peak_k_sigma * sigma) & (neurons > 0)
    ids = [int(v) for v in np.unique(neurons) if v != 0]
    raw = []
    for neuron_id in ids:
        soma = neurons == neuron_id
        values = image[soma]
        n = values.size
        volume = n * voxel_um3
        ordered = np.sort(values)
        top = ordered[int(np.floor(n * 0.99)):]
        total = float(ordered.sum())
        # gini: 0 when uniform, toward 1 when concentrated in few voxels
        clipped = np.clip(ordered, 0.0, None)
        weight = clipped.sum()
        gini = (float((2.0 * (np.arange(1, n + 1, dtype=np.float64) * clipped).sum())
                      / (n * weight) - (n + 1.0) / n) if n and weight > 0 else float("nan"))
        raw.append({
            "neuron_id": neuron_id, "soma_n_voxels": int(n), "soma_volume_um3": volume,
            "pcnt_mean": float(values.mean()), "pcnt_median": float(np.median(values)),
            "pcnt_p99": float(np.percentile(values, 99)), "pcnt_max": float(values.max()),
            "pcnt_integrated": total,
            "pcnt_integrated_bgsub": float(np.clip(values - median, 0.0, None).sum()),
            "pcnt_gini": gini,
            "pcnt_top1pct_frac": float(top.sum() / total) if total > 0 else float("nan"),
            "pcnt_p99_over_bg": ((float(np.percentile(values, 99)) - median) / sigma
                                 if sigma > 0 else float("nan")),
            "n_peaks": int(peaks[soma].sum()),
            "peak_density_per_100um3": (float(peaks[soma].sum()) / volume * 100.0
                                        if volume else 0.0)})
    # a focus belongs to the instance it overlaps most; one overlapping none is counted apart
    inside: dict[int, list[tuple[int, float]]] = {}
    nearby: dict[int, list[tuple[int, float]]] = {}
    any_soma = neurons > 0
    if any_soma.any():
        distance, indices = ndimage.distance_transform_edt(  # pyright: ignore[reportGeneralTypeIssues]
            ~any_soma, sampling=spacing, return_indices=True)
        nearest = neurons[tuple(indices)]
    else:
        distance, nearest = np.full(neurons.shape, np.inf), np.zeros_like(neurons)
    for label, box in enumerate(ndimage.find_objects(foci), start=1):
        if box is None:
            continue
        mask = foci[box] == label
        n_voxels = int(mask.sum())
        intensity = float(image[box][mask].sum())
        overlap = neurons[box][mask]
        overlap = overlap[overlap > 0]
        if overlap.size:
            owners, counts = np.unique(overlap, return_counts=True)
            inside.setdefault(int(owners[int(np.argmax(counts))]), []).append(
                (n_voxels, intensity))
            continue
        centre = [int(round(c)) for c in ndimage.center_of_mass(mask)]  # pyright: ignore[reportArgumentType]
        point = tuple(int(box[axis].start + centre[axis]) for axis in range(3))
        if float(distance[point]) <= max_reach_um and int(nearest[point]):
            nearby.setdefault(int(nearest[point]), []).append((n_voxels, intensity))
    counted = []
    for neuron_id in ids:
        within, around = inside.get(neuron_id, []), nearby.get(neuron_id, [])
        volumes = [n * voxel_um3 for n, _ in within]
        counted.append({
            "neuron_id": neuron_id, "n_foci_inside": len(within),
            "n_foci_in_territory": len(around),
            "foci_volume_um3_total": float(sum(volumes)),
            "foci_volume_um3_max": float(max(volumes)) if volumes else 0.0,
            "foci_integrated_total": float(sum(i for _, i in within)),
            "foci_integrated_max": float(max((i for _, i in within), default=0.0))})
    return pd.DataFrame(raw).merge(pd.DataFrame(counted), on="neuron_id", how="left")


def mtoc_geometry(foci: np.ndarray, neurons: np.ndarray, spacing: tuple[float, float, float],
                  objects: pd.DataFrame, *, donor: str, fov_id: str, max_reach_um: float,
                  inside_frac: float) -> pd.DataFrame:
    """Place each focus against its neuron and measure its distance to the chosen cilium's base."""
    geometry = describe(foci, neurons, None, spacing, max_reach_um=max_reach_um,
                        inside_frac=inside_frac).rename(columns={"owner_neuron": "neuron_id"})
    chosen = objects[objects["chosen"]].set_index("owner_neuron")
    scale = np.asarray(spacing, dtype=float)
    rows = []
    for focus in geometry.to_dict("records"):
        owner = focus["neuron_id"]
        row = {"donor": donor, "fov": fov_id,
               "mtoc_uid": f"{donor}:{fov_id}:m{int(focus['label']):04d}",
               "neuron_uid": f"{donor}:{fov_id}:n{int(owner):04d}" if owner else "",
               **{k: focus[k] for k in ("label", "neuron_id", "volume_um3", "frac_in_soma",
                                        "containment", "base_dist_um", "centroid_z",
                                        "centroid_y", "centroid_x")},
               "dist_to_cilium_base_um": float("nan"), "cilium_uid": ""}
        # euclidean between two geometric voxel positions on one grid, not chromatically corrected
        if owner in chosen.index and not np.isnan(float(chosen.at[owner, "base_z"])):
            base = chosen.loc[owner, ["base_z", "base_y", "base_x"]].to_numpy(dtype=float)
            delta = (np.asarray([focus["centroid_z"], focus["centroid_y"], focus["centroid_x"]],
                                dtype=float) - base) * scale
            row["dist_to_cilium_base_um"] = float(np.linalg.norm(delta))
            row["cilium_uid"] = str(chosen.at[owner, "cilium_uid"])
        rows.append(row)
    return frame_of(rows, "mtocs")


def neuron_rows_of(readouts: pd.DataFrame, objects: pd.DataFrame, foci: pd.DataFrame, *,
                   donor: str, fov_id: str) -> pd.DataFrame:
    """Key the instances and attach the chosen cilium and the nearest focus."""
    out = readouts.copy()
    ids = [int(v) for v in out["neuron_id"]]
    out.insert(0, "donor", donor)
    out.insert(1, "fov", fov_id)
    out.insert(2, "neuron_uid", [f"{donor}:{fov_id}:n{i:04d}" for i in ids])
    chosen = objects[objects["chosen"]].set_index("owner_neuron")
    out["cilium_uid"] = [str(chosen.at[i, "cilium_uid"]) if i in chosen.index else "" for i in ids]
    out["has_cilium"] = out["cilium_uid"] != ""
    out["cilium_tip_out_um"] = [chosen.at[i, "tip_out_um"] if i in chosen.index else np.nan
                                for i in ids]
    nearest = (foci.loc[foci["cilium_uid"] != ""].sort_values("dist_to_cilium_base_um")
               .groupby("neuron_uid", sort=False).head(1).set_index("neuron_uid"))
    uids = list(out["neuron_uid"])
    out["nearest_mtoc_uid"] = [str(nearest.at[u, "mtoc_uid"]) if u in nearest.index else ""
                               for u in uids]
    out["nearest_mtoc_dist_um"] = [float(nearest.at[u, "dist_to_cilium_base_um"])
                                   if u in nearest.index else np.nan for u in uids]
    return out


def link_edges(objects: pd.DataFrame, foci: pd.DataFrame, *,
               max_base_dist_um: float) -> pd.DataFrame:
    """Return the edge list: chosen cilium to neuron, focus to neuron, focus to cilium base."""
    edges: list[dict[str, Any]] = [
        {"src_type": "cilium", "src_uid": row["cilium_uid"], "dst_type": "neuron",
         "dst_uid": row["neuron_uid"], "relation": "emanates_from",
         "distance_um": float(row["base_dist_um"]), "threshold_um": float(max_base_dist_um),
         "method": "longest"}
        for row in objects.loc[objects["chosen"]].to_dict("records")]
    for focus in foci.to_dict("records"):
        if focus["neuron_uid"]:
            edges.append({"src_type": "mtoc", "src_uid": focus["mtoc_uid"], "dst_type": "neuron",
                          "dst_uid": focus["neuron_uid"], "relation": "inside",
                          "distance_um": 0.0, "threshold_um": np.nan, "method": "voxel overlap"})
        if focus["cilium_uid"] and not np.isnan(focus["dist_to_cilium_base_um"]):
            edges.append({"src_type": "mtoc", "src_uid": focus["mtoc_uid"], "dst_type": "cilium",
                          "dst_uid": focus["cilium_uid"], "relation": "near_base",
                          "distance_um": float(focus["dist_to_cilium_base_um"]),
                          "threshold_um": np.nan,
                          "method": "euclidean, not chromatically corrected"})
    return frame_of(edges, "links")


def decisions_digest(refined: list[dict[str, str]] | None,
                     review: Review | None) -> dict[str, str]:
    """sha256 of the refinement rows and of the review given to link, '' for none."""
    def sha(obj: Any) -> str:
        return "" if obj is None else hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
    return {"refined_sha256": sha(refined),
            "review_sha256": sha(None if review is None else {k: sorted(v.items()) for k, v in asdict(review).items()})}


def kept_of(ids: list[int], excluded: list[int], refined: list[dict[str, str]] | None,
            review: Review | None) -> dict[int, tuple[bool | None, str]]:
    """Per instance, the rater's inclusion and its reason; None where neither refinement nor review exists."""
    labelled: dict[int, list[str]] | None = None
    if refined is not None:
        labelled = {}
        for r in refined:
            if r.get("mask_label", ""):
                labelled.setdefault(int(r["mask_label"]), []).append(r["kept"])
    out: dict[int, tuple[bool | None, str]] = {}
    for i in ids:
        if labelled is not None and i not in labelled:
            out[i] = (False, "no census neuron")
        elif review is not None and i in review.closeup:
            decision = review.closeup[i]
            out[i] = (CLOSEUP_KEPT[decision], f"close-up {decision}")
        elif review is not None:
            if i not in review.status:
                raise ValueError(f"instance {i} has no step-2 status in a completed review")
            out[i] = (STATUS_KEPT[review.status[i]], f"reviewed {review.status[i]}")
        elif labelled is not None and i in excluded:
            out[i] = (False, "producer excluded")
        elif labelled is not None:
            yes = any(k != "no" for k in labelled[i])
            out[i] = (yes, "refined kept" if yes else "refined kept=no")
        else:
            out[i] = (None, "")
    return out


def neuron_table(*, censoring: pd.DataFrame, neurons: pd.DataFrame, cilia: pd.DataFrame,
                 status: str, kept: dict[int, tuple[bool | None, str]] | None = None) -> pd.DataFrame:
    """Return one row per delivered instance with its eligibility, decision and chosen length."""
    excluded = censoring["producer_excluded"].astype(bool)
    eligible_ids = {int(v) for v in censoring.loc[~excluded, "neuron_id"]}
    chosen = cilia[cilia["chosen"].astype(bool)].set_index("cilium_uid")
    by_id = neurons.set_index("neuron_id")
    ids = [int(v) for v in censoring["neuron_id"]]
    kept = kept or {}
    rater = [kept.get(i, (None, ""))[0] for i in ids]
    # the rater's decision, where one exists, replaces the producer's exclusion
    eligible = [k if k is not None else i in eligible_ids for i, k in zip(ids, rater, strict=True)]
    # an ineligible row is not asked, so it carries no answer
    rows = list(zip(ids, eligible, strict=True))
    uids = [str(by_id.at[i, "cilium_uid"]) if ok else "" for i, ok in rows]
    return pd.DataFrame({
        "donor": [str(v) for v in censoring["donor"]],
        "fov": [str(v) for v in censoring["fov"]],
        "neuron_uid": [str(v) for v in censoring["neuron_uid"]],
        "neuron_id": pd.Series(ids, dtype="int64"),
        "censoring": [str(v) for v in censoring["censoring"]],
        "eligible": pd.Series(eligible, dtype="bool"),
        "has_cilium": pd.Series([bool(by_id.at[i, "has_cilium"]) if ok else pd.NA
                                 for i, ok in rows], dtype="boolean"),
        "cilium_id": pd.Series([int(chosen.at[u, "label"]) if u else pd.NA for u in uids],
                               dtype="Int64"),
        "length_um": [float(chosen.at[u, "length_signal_um"]) if u else np.nan for u in uids],
        "ac3_mean": [float(chosen.at[u, "ac3_mean"]) if u else np.nan for u in uids],
        "nearest_mtoc_dist_um": [float(by_id.at[i, "nearest_mtoc_dist_um"]) if ok else np.nan
                                 for i, ok in rows],
        "kept": pd.Series(rater, dtype="boolean"),
        "kept_reason": [kept.get(i, (None, ""))[1] for i in ids],
        "status": [status] * len(ids),
    })
