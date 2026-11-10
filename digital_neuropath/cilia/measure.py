"""What the package measures on a detected object, and the per-fov summary of the decisions.
morphometry -> shape_columns; centreline -> curvature, pruned_branch_count -> axoneme_table.
The bridged length is returned to the caller and never written: it is a sort key, not a length.
"""
import itertools
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from skimage import measure, morphology

from digital_neuropath.cilia import filters
from digital_neuropath.cilia.load import SCHEMAS, frame_of

__all__ = ["morphometry", "axoneme_table", "summarize_cilia"]

DENOMINATOR = "n_eligible: instances not on the producer's own excluded list"

# the z the v0.1.0 record was computed with; scipy's ppf differs in the ninth decimal
WILSON_Z = 1.959963985

POOLED_WORDS = frozenset({"all", "pooled", "total", "overall", "combined"})


def morphometry(labels: np.ndarray, image: np.ndarray, spacing: tuple[float, float, float], *,
                foci: bool) -> pd.DataFrame:
    """Return one row per object with size, centroid and intensity; for foci, also shape and
    background, an intensity-weighted centroid, and brightest first."""
    marker, name = ("pcnt", "pcnt_morphometry") if foci else ("ac3", "morphometry")
    array = np.asarray(labels)
    counts = np.bincount(array.ravel())
    present = np.nonzero(counts)[0]
    present = present[present != 0]
    if present.size == 0:
        # finding nothing is a result, so the rowless table keeps its kinds
        return frame_of([], name)
    ids = [int(value) for value in present]
    n_voxels = counts[present].astype(np.int64)
    unit = np.ones(array.shape, dtype=np.uint8)
    centroids = np.asarray(ndimage.center_of_mass(unit, array, ids), dtype=float)
    if foci:
        weights = np.clip(np.asarray(image, dtype=np.float64), 0.0, None)
        weighted = np.asarray(ndimage.center_of_mass(weights, array, ids), dtype=float)
        # an object whose weight sums to zero has no intensity centroid; it keeps its geometric one
        blank = np.isnan(weighted).any(axis=1)
        weighted[blank] = centroids[blank]
        centroids = weighted
    sums = np.asarray(ndimage.sum_labels(image, array, ids), dtype=float)
    data: dict[str, Any] = {
        "label": present.astype(int),
        "n_voxels": n_voxels,
        "volume_um3": n_voxels * (spacing[0] * spacing[1] * spacing[2]),
        "centroid_z_um": centroids[:, 0] * float(spacing[0]),
        "centroid_y_um": centroids[:, 1] * float(spacing[1]),
        "centroid_x_um": centroids[:, 2] * float(spacing[2]),
        f"{marker}_integrated": sums,
        f"{marker}_mean": sums / n_voxels,
    }
    if not foci:
        return frame_of(data, name)
    data["surface_area_um2"], data["sphericity"], data["anisotropy"] = shape_columns(
        array, weights, ids, spacing)
    median, sigma = filters.robust_background(np.asarray(image, dtype=np.float64))
    data["pcnt_bg_median"] = np.full(len(ids), median)
    data["pcnt_bg_sigma"] = np.full(len(ids), sigma)
    # brightest first is an order for review; it makes no call about what an object is
    return frame_of(data, name).sort_values("pcnt_integrated", ascending=False,
                                            kind="stable").reset_index(drop=True)


def shape_columns(labels: np.ndarray, weights: np.ndarray, ids: list[int],
                  spacing: tuple[float, float, float]
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return surface area, sphericity and anisotropy per object, aligned to ids."""
    boxes = ndimage.find_objects(labels)
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    area = np.full(len(ids), np.nan)
    sphericity = np.full(len(ids), np.nan)
    anisotropy = np.full(len(ids), np.nan)
    for position, label in enumerate(ids):
        box = boxes[label - 1]
        if box is None:
            continue
        inside = labels[box] == label
        n_voxels = int(inside.sum())
        surface = float("nan")
        if inside.sum() >= 2:
            # padded by one voxel so an object on the crop edge still meshes as a closed surface
            try:
                verts, faces, _, _ = measure.marching_cubes(
                    np.pad(inside.astype(np.float32), 1), level=0.5, spacing=spacing)
                surface = float(measure.mesh_surface_area(verts, faces))
            except (RuntimeError, ValueError):
                pass
        area[position] = surface
        if surface == surface and surface > 0:
            volume = n_voxels * voxel_volume
            sphericity[position] = float(
                np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / surface)
        zs, ys, xs = np.nonzero(inside)
        oz, oy, ox = (axis.start for axis in box)
        coords_um = np.column_stack([(zs + oz) * spacing[0], (ys + oy) * spacing[1],
                                     (xs + ox) * spacing[2]])
        weight = weights[box][inside]
        weight = weight if weight.sum() > 0 else np.ones(n_voxels)
        if coords_um.shape[0] < 2:
            continue
        normalised = weight / weight.sum()
        centred = coords_um - (coords_um * normalised[:, None]).sum(axis=0)
        eigenvalues = np.clip(np.linalg.eigvalsh((centred * normalised[:, None]).T @ centred),
                              0.0, None)
        if eigenvalues.max() > 0:
            anisotropy[position] = float(1.0 - np.sqrt(eigenvalues.min() / eigenvalues.max()))
    return area, sphericity, anisotropy


def centreline(binary: np.ndarray, spacing: tuple[float, float, float]
               ) -> tuple[np.ndarray, sparse.csr_matrix | None, float, list[int]]:
    """Skeletonise one object; return its voxels, its 26-connected graph weighted in µm, and the
    length and voxel path of the graph's longest shortest path."""
    points = np.argwhere(morphology.skeletonize(binary))
    if len(points) < 2:
        return points, None, 0.0, []
    index = {tuple(int(c) for c in point): i for i, point in enumerate(points)}
    scale = np.asarray(spacing, dtype=float)
    offsets = [o for o in itertools.product((-1, 0, 1), repeat=3) if any(o)]
    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    for i, point in enumerate(points):
        for offset in offsets:
            j = index.get(tuple(int(point[k] + offset[k]) for k in range(3)))
            if j is not None and j > i:
                distance = float(np.linalg.norm(np.asarray(offset) * scale))
                rows += [i, j]
                cols += [j, i]
                weights += [distance, distance]
    if not rows:
        return points, None, 0.0, []
    graph = sparse.csr_matrix((weights, (rows, cols)), shape=(len(points), len(points)))
    # double dijkstra: exact on a tree; on a looped skeleton, the furthest pair reachable this way
    from_zero = csgraph.shortest_path(graph, method="D", indices=0, directed=False)
    start = int(np.nanargmax(np.where(np.isfinite(from_zero), from_zero, -1.0)))
    distances, predecessors = csgraph.shortest_path(graph, method="D", indices=start,
                                                    directed=False, return_predecessors=True)
    end = int(np.nanargmax(np.where(np.isfinite(distances), distances, -1.0)))
    path = [end]
    current = end
    while current != start and predecessors[current] >= 0:
        current = int(predecessors[current])
        path.append(current)
    path.reverse()
    return points, graph, float(distances[end]), path


def curvature(path_um: np.ndarray, smooth_um: float) -> tuple[float, float]:
    """Return total turning in radians and the largest local curvature in 1/µm of a path."""
    path = np.asarray(path_um, dtype=float)
    if len(path) < 3:
        return float("nan"), float("nan")
    # resampled at uniform arc length and smoothed, so voxel stepping does not dominate turning
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segments)])
    total = float(arc[-1])
    step = max(smooth_um * 0.5, 1e-3)
    if total < 4 * step:
        return float("nan"), float("nan")
    n_samples = max(5, int(total / step) + 1)
    uniform = np.linspace(0.0, total, n_samples)
    resampled = np.vstack([np.interp(uniform, arc, path[:, k]) for k in range(3)]).T
    sigma = max(1.0, smooth_um / step)
    smoothed = np.vstack([ndimage.gaussian_filter1d(resampled[:, k], sigma, mode="nearest")
                          for k in range(3)]).T
    deltas = np.diff(smoothed, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    usable = lengths > 1e-9
    tangents = deltas[usable] / lengths[usable, None]
    if len(tangents) < 2:
        return 0.0, 0.0
    dots = np.clip((tangents[:-1] * tangents[1:]).sum(axis=1), -1.0, 1.0)
    angles = np.arccos(dots)
    mid_lengths = 0.5 * (lengths[usable][:-1] + lengths[usable][1:])
    local = angles / np.where(mid_lengths > 0, mid_lengths, np.inf)
    return float(angles.sum()), float(local.max()) if local.size else 0.0


def pruned_branch_count(points: np.ndarray, graph: sparse.csr_matrix,
                        spacing: tuple[float, float, float], prune_um: float) -> int:
    """Return the junctions left after repeatedly removing terminal twigs shorter than prune_um."""
    n_points = len(points)
    adjacency = {i: set(graph.indices[graph.indptr[i]:graph.indptr[i + 1]].tolist())
                 for i in range(n_points)}
    scale = np.asarray(spacing, dtype=float)
    alive = set(range(n_points))
    changed = True
    while changed:
        changed = False
        degrees = {i: len(adjacency[i] & alive) for i in alive}
        for endpoint in [i for i in alive if degrees.get(i) == 1]:
            if endpoint not in alive:
                continue
            walk = [endpoint]
            previous, current = -1, endpoint
            while True:
                neighbours = [x for x in adjacency[current] & alive if x != previous]
                if len(neighbours) != 1:
                    break
                previous, current = current, neighbours[0]
                walk.append(current)
                if len(adjacency[current] & alive) != 2:
                    break
            stop = walk[-1]
            length = sum(float(np.linalg.norm((points[walk[k + 1]] - points[walk[k]]) * scale))
                         for k in range(len(walk) - 1))
            if len(adjacency[stop] & alive) >= 3 and length < prune_um:
                for node in walk[:-1]:
                    alive.discard(node)
                changed = True
    degrees = {i: len(adjacency[i] & alive) for i in alive}
    return int(sum(1 for i in alive if degrees[i] >= 3))


def axoneme_table(labels: np.ndarray, raw: np.ndarray, spacing: tuple[float, float, float], *,
                  bridge_gap_um: float, prune_um: float, curvature_smooth_um: float,
                  collapse_radius_um: float) -> pd.DataFrame:
    """Return one skeleton row per object: its detected length, curvature and intensity along its
    bridged centreline, a branch qc flag, the image background and the collapse flag."""
    array = np.asarray(labels)
    background_median, background_sigma = filters.robust_background(raw)
    scale = np.asarray(spacing, float)
    closing = filters.ellipsoid_footprint(bridge_gap_um / 2.0, spacing)
    # the pad exceeds the closing element, so bridging in the crop equals bridging on the frame
    pad = [max(1, int(round(bridge_gap_um / s))) + 2 for s in spacing]
    rows: list[dict[str, Any]] = []
    for label, box in enumerate(ndimage.find_objects(array), start=1):
        if box is None:
            continue
        crop = filters.padded_bbox(box, array.shape, pad)
        binary = array[crop] == label
        signal_points, signal_graph, signal_length, _ = centreline(binary, spacing)
        points, _, length, path = centreline(morphology.closing(binary, closing), spacing)
        row: dict[str, Any] = dict(
            label=label, length_signal_um=signal_length,
            total_curvature_rad=float("nan"), total_curvature_deg=float("nan"),
            max_curvature_inv_um=float("nan"),
            axoneme_mean=float("nan"), axoneme_max=float("nan"), axoneme_cv=float("nan"),
            axoneme_bg_median=background_median, axoneme_bg_sigma=background_sigma,
            branch_count=0)
        if length > 0:
            turning, max_local = curvature(points[path] * scale, curvature_smooth_um)
            row["total_curvature_rad"] = turning
            row["total_curvature_deg"] = (turning * 180.0 / np.pi if turning == turning
                                          else float("nan"))
            row["max_curvature_inv_um"] = max_local
            values = np.asarray(raw[crop])
            sampled = values[tuple(np.clip(points[path][:, k], 0, values.shape[k] - 1)
                                   for k in range(3))].astype(float)
            if sampled.size:
                row["axoneme_mean"] = float(sampled.mean())
                row["axoneme_max"] = float(sampled.max())
                row["axoneme_cv"] = (float(sampled.std() / sampled.mean())
                                     if sampled.mean() != 0 else float("nan"))
        # a qc flag on the detected skeleton's roughness, not on the reconstruction
        if signal_graph is not None:
            row["branch_count"] = pruned_branch_count(signal_points, signal_graph, spacing,
                                                      prune_um)
        volume_um3 = int(binary.sum()) * (spacing[0] * spacing[1] * spacing[2])
        # a zero length is the strongest collapse, so its radius is inf rather than missing
        r_eff = (float(np.sqrt(volume_um3 / (np.pi * signal_length))) if signal_length > 0
                 else float("inf"))
        row["blob_r_eff_um"] = r_eff
        row["blob_flag"] = bool(r_eff > collapse_radius_um)
        rows.append(row)
    return frame_of(rows, "skeleton")


def summarize_cilia(table: pd.DataFrame) -> pd.DataFrame:
    """Return one row per fov with k of a named n, a Wilson bound and a median length, refusing a
    pooled row or a table that mixes statuses."""
    pooled = sorted({str(v) for v in table["fov"]
                     if str(v) == "" or str(v).lower() in POOLED_WORDS})
    if pooled:
        raise ValueError(f"summarize_cilia: row(s) on {pooled}; every count here is over one named fov")
    statuses = sorted({str(v) for v in table["status"]})
    if len(statuses) > 1:
        raise ValueError(f"summarize_cilia: rows carry more than one status {statuses}")
    rows: list[dict[str, Any]] = []
    z = WILSON_Z
    groups: Any = table.groupby(["donor", "fov"], sort=False)
    for (donor, fov), group in groups:
        eligible = group[group["eligible"].astype(bool)]
        ciliated = eligible[eligible["has_cilium"].fillna(False).astype(bool)]
        k, n = len(ciliated), len(eligible)
        lo = hi = float("nan")
        if n:
            p = k / n
            denominator = 1.0 + z * z / n
            centre = (p + z * z / (2 * n)) / denominator
            half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
            lo, hi = max(0.0, centre - half), min(1.0, centre + half)
        lengths = ciliated["length_um"].to_numpy(dtype=float)
        lengths = lengths[np.isfinite(lengths)]
        rows.append({
            "donor": str(donor), "fov": str(fov),
            "n_neurons": len(group), "n_eligible": n, "k_ciliated": k,
            "ciliated_wilson_lo": lo, "ciliated_wilson_hi": hi,
            "n_length": int(lengths.size),
            # a median with its n, never a mean with an interval: about half the chosen objects
            # on the mask arm are the wrong object
            "median_length_um": float(np.median(lengths)) if lengths.size else np.nan,
            "denominator": DENOMINATOR, "status": str(group["status"].iloc[0]),
        })
    return pd.DataFrame(rows, columns=[c.name for c in SCHEMAS["fov_summary"]])
