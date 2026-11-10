"""Synthetic OME-TIFF fovs for the two tutorials; no patient data.
Chain: write -> fov (place neurons and glia, paint DAPI, NeuN, TDP-43, blur, noise) -> OME-TIFF;
write_cilia -> neurons, PCNT foci, curved AC3 tubes -> OME-TIFF, .cmap instance masks, truth.
Which neuron got which TDP-43 phenotype is never written to disk, so the demo can be scored blind.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

__all__ = ["write", "write_cilia"]

SHAPE = (40, 256, 256)
SPACING_UM = (1.0, 0.4, 0.4)
CHANNEL_NAMES = ["DAPI", "NeuN", "TDP-43", "AC3", "PCNT"]
N_FOVS = 2
NEURONS_PER_FOV = 10
GLIA_PER_FOV = 8
SEED = 20260930
SOMA_RADIUS_UM = (5.0, 6.5, 6.5)
NUCLEUS_RADIUS_UM = (3.5, 3.5, 3.5)
GLIA_RADIUS_UM = (2.5, 2.5, 2.5)
PUNCTUM_RADIUS_UM = 0.8
CENTRE_Z_UM = (11.0, 28.0)
PHENOTYPES = ("cytoplasmic inclusion", "nuclear clearance only", "normal nuclear")
# TDP-43 (nucleus, cytoplasm) for each phenotype; inclusions add bright perinuclear puncta.
TDP43_LEVELS = {"cytoplasmic inclusion": (20.0, 25.0), "nuclear clearance only": (20.0, 40.0),
                "normal nuclear": (170.0, 15.0)}


def ellipsoid(centre_um: np.ndarray, radius_um: tuple[float, float, float],
              shape: tuple[int, int, int] = SHAPE,
              spacing_um: tuple[float, float, float] = SPACING_UM) -> tuple[tuple, np.ndarray]:
    """Bounding box and boolean mask of an ellipsoid, in voxels of shape."""
    spacing = np.array(spacing_um)
    lo = np.maximum(np.floor((centre_um - radius_um) / spacing).astype(int), 0)
    hi = np.minimum(np.ceil((centre_um + radius_um) / spacing).astype(int) + 1, shape)
    grids = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    dist = sum(((g * s - c) / r) ** 2 for g, s, c, r in zip(grids, spacing, centre_um, radius_um))
    return tuple(slice(a, b) for a, b in zip(lo, hi)), np.asarray(dist <= 1.0)


def place(rng: np.random.Generator, n: int, min_gap_um: float, taken: list[np.ndarray]) -> list:
    """n centres in um inside the z range, at least min_gap_um apart in y-x from every other."""
    extent = np.array(SHAPE) * np.array(SPACING_UM)
    out = []
    while len(out) < n:
        c = np.array([rng.uniform(*CENTRE_Z_UM), rng.uniform(8.0, extent[1] - 8.0),
                      rng.uniform(8.0, extent[2] - 8.0)])
        if all(np.hypot(*(c[1:] - t[1:])) >= min_gap_um for t in taken + out):
            out.append(c)
    return out


def fov(rng: np.random.Generator) -> np.ndarray:
    """One (C, Z, Y, X) uint16 stack with neurons of all three phenotypes and DAPI-only glia."""
    dapi, neun, tdp = (np.zeros(SHAPE, np.float32) for _ in range(3))
    neurons = place(rng, NEURONS_PER_FOV, 15.0, [])
    glia = place(rng, GLIA_PER_FOV, 9.0, neurons)
    phenotypes = [PHENOTYPES[i % 3] for i in rng.permutation(NEURONS_PER_FOV)]
    for centre, phenotype in zip(neurons, phenotypes):
        soma_box, soma = ellipsoid(centre, SOMA_RADIUS_UM)
        nuc_box, nucleus = ellipsoid(centre, NUCLEUS_RADIUS_UM)
        nuc_level, cyto_level = TDP43_LEVELS[phenotype]
        neun[soma_box][soma] = 150.0
        tdp[soma_box][soma] = cyto_level
        dapi[nuc_box][nucleus] = 190.0
        neun[nuc_box][nucleus] = 110.0
        tdp[nuc_box][nucleus] = nuc_level
        if phenotype == "cytoplasmic inclusion":
            for _ in range(int(rng.integers(3, 6))):
                angle = rng.uniform(0, 2 * np.pi)
                reach = NUCLEUS_RADIUS_UM[1] + rng.uniform(1.0, 2.0)
                offset = np.array([rng.uniform(-1.5, 1.5), reach * np.sin(angle),
                                   reach * np.cos(angle)])
                r = PUNCTUM_RADIUS_UM
                box, punctum = ellipsoid(centre + offset, (r, r, r))
                tdp[box][punctum] = 230.0
    for centre in glia:
        box, nucleus = ellipsoid(centre, GLIA_RADIUS_UM)
        dapi[box][nucleus] = 220.0
    ac3 = np.zeros(SHAPE, np.float32)
    pcnt = np.zeros(SHAPE, np.float32)
    out = []
    for channel in (dapi, neun, tdp, ac3, pcnt):
        blurred = ndimage.gaussian_filter(channel, sigma=(0.8, 1.5, 1.5))
        noisy = rng.poisson(blurred.clip(0))
        out.append(np.clip(noisy, 0, 65535).astype(np.uint16))
    return np.stack(out)


def write(folder: str | Path) -> list[Path]:
    """Write the demo fovs as OME-TIFFs into folder (same pixels every call); return the paths."""
    import tifffile

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    paths = []
    for i in range(1, N_FOVS + 1):
        path = folder / f"demo_fov{i}.ome.tif"
        tifffile.imwrite(path, fov(rng), ome=True, photometric="minisblack", metadata={
            "axes": "CZYX", "PhysicalSizeZ": SPACING_UM[0], "PhysicalSizeY": SPACING_UM[1],
            "PhysicalSizeX": SPACING_UM[2], "Channel": {"Name": CHANNEL_NAMES}})
        paths.append(path)
    return paths


# The cilia fov: one stack on the grid of the real cilia lifs (xy ~2x under Nyquist, z near it).
CILIA_SHAPE = (60, 420, 420)
CILIA_SPACING_UM = (0.3, 0.18, 0.18)
CILIA_CHANNELS = {"dapi": 0, "neun": 1, "ac3": 3, "pcnt": 4}
CILIA_SEED = 20261002
# (centre y, centre x) in um; the last neuron is cut by the x=0 face and excluded by the producer
CILIA_NEURONS_YX = ((14.0, 14.0), (14.0, 40.0), (16.0, 64.0), (38.0, 26.0), (40.0, 54.0),
                    (62.0, 16.0), (62.0, 42.0), (64.0, 64.0), (38.0, 3.0))
CILIATED = (True, True, False, True, True, False, True, False, True)
CILIA_GLIA = 6
CILIUM_LENGTH_UM = (4.0, 8.0)
CILIUM_CURVATURE_PER_UM = (0.03, 0.15)
PSF_SIGMA_UM = (0.28, 0.08, 0.08)  # FWHM 0.66 um axial, 0.19 um lateral
TUBE_SIGMA_UM = 0.1  # a 250 nm axoneme
FOCUS_SIGMA_UM = 0.2
BACKGROUND = 100.0
READ_NOISE = 4.0


def blob(volume: np.ndarray, points_um: np.ndarray, sigma_um: float, peak: float) -> None:
    """Add one object to volume: its points, blurred by sigma_um and the PSF, brightest at peak."""
    spacing = np.array(CILIA_SPACING_UM)
    sigma_vox = np.hypot(sigma_um, PSF_SIGMA_UM) / spacing
    pad = np.ceil(4 * sigma_vox).astype(int) + 1
    vox = np.round(np.atleast_2d(points_um) / spacing).astype(int)
    lo = np.maximum(vox.min(0) - pad, 0)
    hi = np.minimum(vox.max(0) + pad + 1, volume.shape)
    local = np.zeros(hi - lo, np.float32)
    # each voxel once, so a tube along z is not brighter than one along x
    local[tuple(np.unique(vox - lo, axis=0).T)] = 1.0
    local = ndimage.gaussian_filter(local, sigma_vox)
    volume[tuple(slice(a, b) for a, b in zip(lo, hi))] += peak * local / local.max()


def surface_reach(direction: np.ndarray, radius_um: np.ndarray) -> float:
    """Distance from an ellipsoid's centre to its surface along a unit direction."""
    return float(1.0 / np.sqrt(np.sum((direction / radius_um) ** 2)))


def arc(rng: np.random.Generator, base: np.ndarray, outward: np.ndarray,
        length_um: float) -> np.ndarray:
    """Points every 0.03 um along a planar arc of the given length leaving base along outward."""
    normal = np.cross(outward, rng.normal(size=3))
    normal /= np.linalg.norm(normal)
    kappa = rng.uniform(*CILIUM_CURVATURE_PER_UM)
    s = np.arange(0.0, length_um, 0.03)[:, None]
    return base + outward * np.sin(kappa * s) / kappa + normal * (1 - np.cos(kappa * s)) / kappa


def write_cilia(folder: str | Path) -> tuple[Path, Path, pd.DataFrame]:
    """Write one synthetic cilia fov (OME-TIFF, CZYX: DAPI, NeuN, AC3, PCNT) and its .cmap
    instance masks into folder; return both paths and the planted truth, one row per neuron."""
    import h5py
    import tifffile

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(CILIA_SEED)
    spacing = np.array(CILIA_SPACING_UM)
    extent = np.array(CILIA_SHAPE) * spacing
    dapi, neun, ac3, pcnt = (np.zeros(CILIA_SHAPE, np.float32) for _ in range(4))
    cell_body = np.zeros(CILIA_SHAPE, np.int32)
    nucleus = np.zeros(CILIA_SHAPE, np.int32)
    somata = [(np.array([rng.uniform(8.0, 10.0), y, x]),
               np.array([rng.uniform(4.0, 5.0), rng.uniform(6.0, 7.5), rng.uniform(6.0, 7.5)]))
              for y, x in CILIA_NEURONS_YX]
    taken = []
    rows = []
    for neuron_id, ((centre, radius), ciliated) in enumerate(zip(somata, CILIATED), start=1):
        nuc_radius = (3.0, 4.0, 4.0)
        box, soma = ellipsoid(centre, tuple(radius), CILIA_SHAPE, CILIA_SPACING_UM)
        cell_body[box][soma] = neuron_id
        neun[box][soma] = 120.0
        ac3[box][soma] = 8.0
        pcnt[box][soma] = 6.0
        box, nuc = ellipsoid(centre, nuc_radius, CILIA_SHAPE, CILIA_SPACING_UM)
        nucleus[box][nuc] = neuron_id
        dapi[box][nuc] = 150.0
        neun[box][nuc] = 90.0
        others = [s for i, s in enumerate(somata, start=1) if i != neuron_id]
        base, length = np.full(3, np.nan), np.nan
        while True:
            # mostly lateral, so the cilium stays inside the 18 um of z
            outward = np.array([rng.uniform(-0.3, 0.3), *rng.normal(size=2)])
            outward /= np.linalg.norm(outward)
            reach = surface_reach(outward, radius)
            if not ciliated:
                focus = centre + outward * rng.uniform(4.5, reach - 0.8)
                break
            # the basal body sits just under the membrane, the axoneme runs out from it
            focus = centre + outward * (reach - 0.6)
            length = rng.uniform(*CILIUM_LENGTH_UM)
            points = arc(rng, centre + outward * (reach - 0.3), outward, length)
            inside = np.all((points > 0.5) & (points < extent - 0.5))
            clear = all(np.all(np.sum(((points - c) / (r + 1.0)) ** 2, axis=1) > 1.0)
                        for c, r in others)
            if inside and clear:
                blob(ac3, points, TUBE_SIGMA_UM, 450.0)
                taken.append(points)
                base = points[0]
                break
        blob(pcnt, focus, FOCUS_SIGMA_UM, 500.0)
        # a soma cut by the x=0 face is on the producer's excluded list
        rows.append({"neuron_id": neuron_id, "excluded": bool(centre[2] < radius[2]),
                     "planted_cilium": ciliated, "length_um": length,
                     **{f"base_{a}_um": float(v) for a, v in zip("zyx", base)},
                     **{f"pcnt_{a}_um": float(v) for a, v in zip("zyx", focus)}})
    # glia: DAPI-only nuclei with their own centrosome, away from neurons and cilia
    glia = 0
    while glia < CILIA_GLIA:
        c = np.array([rng.uniform(4.0, 14.0), *rng.uniform(4.0, extent[1] - 4.0, size=2)])
        if all(np.sum(((c - s) / (r + 3.0)) ** 2) > 1.0 for s, r in somata) and all(
                np.min(np.linalg.norm(p - c, axis=1)) > 5.0 for p in taken):
            box, nuc = ellipsoid(c, (2.5, 3.0, 3.0), CILIA_SHAPE, CILIA_SPACING_UM)
            dapi[box][nuc] = 180.0
            blob(pcnt, c + np.array([0.0, 3.6, 0.0]), FOCUS_SIGMA_UM, 450.0)
            glia += 1
    tdp = np.zeros(CILIA_SHAPE, np.float32)
    stack = []
    for channel in (dapi, neun, tdp, ac3, pcnt):
        channel = ndimage.gaussian_filter(channel, np.array(PSF_SIGMA_UM) / spacing)
        noisy = rng.poisson(BACKGROUND + channel) + rng.normal(0.0, READ_NOISE, CILIA_SHAPE)
        stack.append(np.clip(noisy, 0, 65535).astype(np.uint16))
    image = folder / "demo_cilia.ome.tif"
    tifffile.imwrite(image, np.stack(stack), ome=True, photometric="minisblack", metadata={
        "axes": "CZYX", "PhysicalSizeZ": spacing[0], "PhysicalSizeY": spacing[1],
        "PhysicalSizeX": spacing[2],
        "Channel": {"Name": ["DAPI", "NeuN", "TDP-43", "AC3", "PCNT"]}})
    excluded = [r["neuron_id"] for r in rows if r["excluded"]]
    cmap = folder / "demo_cilia.cmap"
    with h5py.File(cmap, "w") as handle:
        handle["cell_body/data"] = cell_body
        handle["nucleus/data"] = nucleus
        handle["qc/excluded_ids"] = np.array(excluded, dtype=np.int64)
    return image, cmap, pd.DataFrame(rows)
