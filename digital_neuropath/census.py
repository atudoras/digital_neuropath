"""Pass 0, the neuron census: one click per neuron at its own centre, DAPI and NeuN only, then seal.
Chain: run -> read_fov(dapi, neun) -> start -> open_viewer (record on every click) -> seal.
The denominator is fixed before any stain is loaded, so no neuron enters it because of its stain.
"""
import datetime
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from digital_neuropath import io

__all__ = [
    "CensusResult",
    "add_nuclei",
    "check_seal",
    "display_settings",
    "echo",
    "is_sealed",
    "load_nuclei",
    "load_soma",
    "marks",
    "pair_nuclei",
    "project_display",
    "record",
    "run",
    "seal",
    "start",
    "sweep_stats",
]

CHANNELS = ("dapi", "neun")
FIELDS = ["fov", "neuron_id", "z_um", "y_um", "x_um", "z_vox", "y_vox", "x_vox"]

DAPI_PCT = (1.0, 99.9)
NEUN_GAMMA = 1.0
NEUN_PCT = (1.0, 99.9)
MARK_COLOUR = "#FFFF00"
MARK_BORDER_WIDTH = 0.08
MARK_OPACITY = 0.7
# Echo: a ring on every plane for each neuron counted away from it, as the cilia3d Pass 0 did.
ECHO_COLOUR = "#FF3B30"
ECHO_SIZE_UM = 5.0
NEURON_DIAMETER_UM = 15.0
CONTRASTS = ("fov", "project")
# soma_whole (EXPERIMENTAL, NeuN only, never used in scoring): a soma is whole when, toward each
# stack face it reaches, NeuN at the face has faded below WHOLE_FADE of its peak over the last
# WHOLE_TRACE_UM; each plane is divided by the fov's NeuN level at that depth (p DEPTH_LEVEL_PCT).
WHOLE_FADE = 0.5
WHOLE_TRACE_UM = 5.0
DEPTH_LEVEL_PCT = 99.0


@dataclass(frozen=True)
class CensusResult:
    """Where the census for one fov lives and what it holds."""
    fov_dir: Path
    n_neurons: int
    sealed: bool


def census_dir(fov_dir: Path) -> Path:
    """Folder holding the census files of one fov."""
    return Path(fov_dir) / "census"



def guard_remove(layer) -> None:
    """napari 0.9.2: deleting a hovered point raises TypeError (_value_stored is None); seed it first."""
    original = layer.remove

    def remove(indices):
        if getattr(layer, "_value", None) is not None and getattr(layer, "_value_stored", 0) is None:
            layer._value_stored = layer._value
        return original(indices)

    layer.remove = remove

def is_sealed(fov_dir: Path) -> bool:
    """True once seal() has run for this fov."""
    return (census_dir(fov_dir) / "sealed.json").exists()


def percentile_positive(volume: np.ndarray, pct: float) -> float:
    """Percentile over positive voxels, on a (1, 2, 2) subsample to bound memory."""
    sample = volume[:, ::2, ::2]
    positive = sample[sample > 0]
    return float(np.percentile(positive, pct)) if positive.size else 0.0


def positive_counts(volume: np.ndarray) -> np.ndarray:
    """Histogram of positive voxel values on the (1, 2, 2) subsample percentile_positive uses."""
    if not np.issubdtype(volume.dtype, np.integer):
        raise ValueError(f"project contrast pools integer histograms; this volume is {volume.dtype}")
    sample = volume[:, ::2, ::2]
    counts = np.bincount(sample[sample > 0].ravel())
    counts[:1] = 0
    return counts


def add_counts(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sum two histograms of different lengths."""
    out = np.zeros(max(len(a), len(b)), dtype=np.int64)
    out[:len(a)] += a
    out[:len(b)] += b
    return out


def percentile_counts(counts: np.ndarray, pct: float) -> float:
    """np.percentile (linear) of the values a histogram holds, without expanding it."""
    total = int(counts.sum())
    if total == 0:
        return 0.0
    rank = pct / 100 * (total - 1)
    cum = np.cumsum(counts)
    lo = float(np.searchsorted(cum, np.floor(rank), side="right"))
    hi = float(np.searchsorted(cum, np.ceil(rank), side="right"))
    return lo + (hi - lo) * (rank - np.floor(rank))


def pooled_limits(counts: np.ndarray, pct: tuple[float, float]) -> list[float]:
    """[low, high] contrast limits from a pooled histogram, high at least low + 1."""
    lo = percentile_counts(counts, pct[0])
    return [lo, max(percentile_counts(counts, pct[1]), lo + 1.0)]


def project_display(sources: list[tuple[str, int]], channels: dict[str, int]) -> dict[str, dict]:
    """display_settings with each channel's limits pooled over every (source, series) fov."""
    counts = {name: np.zeros(0, dtype=np.int64) for name in CHANNELS}
    for source, series in sources:
        fov = io.read_fov(source, series, channels)
        for name in CHANNELS:
            counts[name] = add_counts(counts[name], positive_counts(fov.data[name]))
    rule = f"pooled over {len(sources)} fov(s)"
    return {"dapi": {"limits": pooled_limits(counts["dapi"], DAPI_PCT), "gamma": NEUN_GAMMA,
                     "rule": f"p{DAPI_PCT[0]:g} to p{DAPI_PCT[1]:g} {rule}"},
            "neun": {"limits": pooled_limits(counts["neun"], NEUN_PCT), "gamma": NEUN_GAMMA,
                     "rule": f"p{NEUN_PCT[0]:g} to p{NEUN_PCT[1]:g} {rule}"}}


def display_settings(data: dict[str, np.ndarray]) -> dict[str, dict]:
    """Contrast limits and gamma per channel, computed once for the whole fov."""
    dapi, neun = data["dapi"], data["neun"]
    dapi_lo = percentile_positive(dapi, DAPI_PCT[0])
    dapi_hi = max(percentile_positive(dapi, DAPI_PCT[1]), dapi_lo + 1.0)
    neun_lo = percentile_positive(neun, NEUN_PCT[0])
    neun_hi = max(percentile_positive(neun, NEUN_PCT[1]), neun_lo + 1.0)
    return {"dapi": {"limits": [dapi_lo, dapi_hi], "gamma": NEUN_GAMMA,
                     "rule": f"p{DAPI_PCT[0]:g} to p{DAPI_PCT[1]:g}"},
            "neun": {"limits": [neun_lo, neun_hi], "gamma": NEUN_GAMMA,
                     "rule": f"p{NEUN_PCT[0]:g} to p{NEUN_PCT[1]:g}"}}


def start(fov_dir: Path, fov: io.Fov, neuron_diameter_um: float = NEURON_DIAMETER_UM,
          contrast: str = "fov", display: dict[str, dict] | None = None) -> dict:
    """Fix display once for this fov (its own, or the project's when contrast="project", passed
    as display); refuse changed channels or contrast on resume."""
    if contrast not in CONTRASTS:
        raise ValueError(f"contrast must be one of {CONTRASTS}, got {contrast!r}")
    if set(fov.data) != set(CHANNELS):
        raise ValueError(f"the census may hold only {CHANNELS}; this fov holds {sorted(fov.data)}")
    path = census_dir(fov_dir) / "census.json"
    if path.exists():
        meta = json.loads(path.read_text())
        loaded = {name: fov.channel_index[name] for name in sorted(fov.data)}
        if meta["channels_loaded"] != loaded:
            raise ValueError(f"this census was started with channels {meta['channels_loaded']}, "
                             f"not {loaded}; its display and marks belong to those channels. "
                             f"Start a new project instead.")
        if meta.get("contrast", "fov") != contrast:
            raise ValueError(f"this census was started with contrast {meta.get('contrast', 'fov')!r}"
                             f", not {contrast!r}; its marks were made on that display.")
        return meta
    if (contrast == "project") != (display is not None):
        raise ValueError("contrast='project' needs the project display (project_display), and "
                         "only it does")
    meta = {"fov": fov.fov_id, "source": fov.source, "series": fov.series, "name": fov.name,
            "channels_loaded": {name: fov.channel_index[name] for name in sorted(fov.data)},
            "neuron_diameter_um": neuron_diameter_um,
            "spacing_um": list(fov.spacing_um), "shape_zyx": list(fov.shape),
            "contrast": contrast, "display": display or display_settings(fov.data),
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))
    return meta


def read_meta(fov_dir: Path) -> dict:
    """The census.json written by start()."""
    return json.loads((census_dir(fov_dir) / "census.json").read_text())


def sweep_stats(planes: list[int], z0: int, z1: int) -> dict[str, float]:
    """Marks per worked plane and the largest run of planes, stack faces included, with no mark."""
    if not planes:
        return {"marks_per_worked_plane": 0.0, "largest_z_gap_planes": z1 - z0}
    worked = sorted(set(planes))
    edges = [z0] + worked + [z1 - 1]
    return {"marks_per_worked_plane": len(planes) / len(worked),
            "largest_z_gap_planes": int(max(np.diff(edges)))}


def record(fov_dir: Path, marks_um: np.ndarray) -> int:
    """Rewrite neurons.csv from the current marks; refused after sealing."""
    if is_sealed(fov_dir):
        raise PermissionError(f"{fov_dir} is sealed. neuron_id is positional, so any edit would "
                              f"renumber neurons a scoring pass already points at.")
    meta = read_meta(fov_dir)
    dz, dy, dx = meta["spacing_um"]
    n_z = meta["shape_zyx"][0]
    points = np.asarray(marks_um, float).reshape(-1, 3)
    voxels = np.rint(points / np.array([dz, dy, dx])).astype(int)
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    rows = [{"fov": meta["fov"], "neuron_id": f"N{rank:03d}",
             "z_um": f"{points[i, 0]:.4f}", "y_um": f"{points[i, 1]:.4f}",
             "x_um": f"{points[i, 2]:.4f}",
             "z_vox": int(voxels[i, 0]), "y_vox": int(voxels[i, 1]), "x_vox": int(voxels[i, 2])}
            for rank, i in enumerate(order, 1)]
    stats = sweep_stats([int(v[0]) for v in voxels], 0, n_z)
    display = meta["display"]
    io.write_rows(census_dir(fov_dir) / "neurons.csv", FIELDS, rows, header=[
        (f"CENSUS (Pass 0) of {meta['fov']}: {meta['source']} series {meta['series']} "
         f"({meta['name']})"),
        f"channels loaded: {meta['channels_loaded']} -- no other channel was read in this process",
        (f"spacing_um (z, y, x): {tuple(meta['spacing_um'])}   shape (z, y, x): "
         f"{tuple(meta['shape_zyx'])}"),
        "display: " + "; ".join(f"{name} limits {s['limits'][0]:.1f}-{s['limits'][1]:.1f} "
                                f"gamma {s['gamma']} ({s['rule']})"
                                for name, s in sorted(display.items())),
        (f"marks per worked plane: {stats['marks_per_worked_plane']:.2f} (1.0 = one neuron "
         f"per plane; well above 1 means several neurons were marked from one plane)"),
        (f"largest z gap: {stats['largest_z_gap_planes']} planes = "
         f"{stats['largest_z_gap_planes'] * dz:.1f} um without a mark, stack faces included"),
    ])
    return len(rows)


def echo(marks_um: np.ndarray, z_um: float, radius_um: float) -> np.ndarray:
    """(y, x) of every mark whose centre is more than one radius from plane z_um."""
    points = np.asarray(marks_um, float).reshape(-1, 3)
    return points[np.abs(points[:, 0] - z_um) > radius_um][:, 1:]


def marks(fov_dir: Path) -> np.ndarray:
    """Marks already on disk in um (z, y, x); this is what resume reloads."""
    rows = io.read_rows(census_dir(fov_dir) / "neurons.csv")
    return np.array([[float(r["z_um"]), float(r["y_um"]), float(r["x_um"])] for r in rows],
                    float).reshape(-1, 3)


def digest(path: Path) -> str:
    """sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def soma_whole(neun: np.ndarray, spacing_um: list[float], marks_um: np.ndarray,
               diameter_um: float) -> list[str]:
    """Per mark, 'yes' when the soma (a sphere of diameter_um about the click) fades before every
    stack face it reaches, judged on NeuN alone; the worst face decides."""
    dz, dy, dx = spacing_um
    n_z = neun.shape[0]
    level = np.array([max(percentile_positive(neun[z:z + 1], DEPTH_LEVEL_PCT), 1e-9)
                      for z in range(n_z)])
    span = int(np.floor(WHOLE_TRACE_UM / dz + 1e-9))
    radius = diameter_um / 2
    out = []
    for z_um, y_um, x_um in np.asarray(marks_um, float).reshape(-1, 3):
        worst = 0.0
        for face, planes in ((0, range(min(span, n_z - 1) + 1)),
                             (n_z - 1, range(max(n_z - 1 - span, 0), n_z))):
            d = abs(z_um - face * dz)
            if d >= radius:
                continue
            # the sphere's cross-section on the face plane, at least one pixel wide
            r = max(float(np.sqrt(radius ** 2 - d ** 2)), dy, dx)
            yy, xx = np.ogrid[:neun.shape[1], :neun.shape[2]]
            disc = (yy * dy - y_um) ** 2 + (xx * dx - x_um) ** 2 <= r ** 2
            trace = {z: float(neun[z][disc].mean()) / level[z] for z in planes}
            peak = max(trace.values())
            worst = max(worst, trace[face] / peak if peak > 0 else 0.0)
        out.append("yes" if worst < WHOLE_FADE else "no")
    return out


def add_soma_whole(fov_dir: Path, neurons: Path, rows: list[dict[str, str]]) -> None:
    """Rewrite neurons.csv with the soma_whole column, read from the fov's NeuN channel; left
    empty when the source cannot be read now."""
    meta = read_meta(fov_dir)
    marks_um = np.array([[float(r[f"{a}_um"]) for a in "zyx"] for r in rows])
    try:
        neun = io.read_fov(meta["source"], meta["series"],
                           {"neun": meta["channels_loaded"]["neun"]}).data["neun"]
        values = soma_whole(neun, meta["spacing_um"], marks_um, meta["neuron_diameter_um"])
        how = (f"yes = the soma (sphere of {meta['neuron_diameter_um']:g} um about the click) has "
               f"faded below {WHOLE_FADE:g} of its peak NeuN over the last {WHOLE_TRACE_UM:g} um "
               f"before every stack face it reaches")
    except OSError as error:
        values = [""] * len(rows)
        how = f"not computed: {error}"
    header = [line[2:] for line in neurons.read_text().splitlines() if line.startswith("# ")]
    io.write_rows(neurons, FIELDS + ["soma_whole"],
                  [{**r, "soma_whole": v} for r, v in zip(rows, values)],
                  header=[*header, ("soma_whole: EXPERIMENTAL, NeuN only, never used in "
                                    f"scoring; {how}")])


def _cmap_in_masks(source: Path, name: str, masks_dir: Path | None) -> Path | None:
    """Find a .cmap in masks_dir matching source stem or fov name."""
    if masks_dir is None:
        return None
    masks_dir = Path(masks_dir)
    if not masks_dir.is_dir():
        return None
    stem = source.stem.removesuffix(".ome")
    for candidate_name in (stem, name):
        candidate = masks_dir / f"{candidate_name}.cmap"
        if candidate.exists():
            return candidate
    return None


def load_nuclei(fov_dir: Path, nuclei_dir: Path | None,
                masks_dir: Path | None = None) -> np.ndarray | None:
    """Instance-labelled nuclei for this FOV: from a .cmap beside the source (nucleus/data),
    from nuclei_dir/<stem>.tif, or from a .cmap in masks_dir. Returns int32 labels or None."""
    meta = read_meta(fov_dir)
    source = Path(meta["source"])
    cmap = source.with_suffix(".cmap")
    if cmap.exists():
        import h5py
        with h5py.File(cmap, "r") as f:
            if "nucleus/data" in f:
                return np.asarray(f["nucleus/data"], dtype=np.int32)
    if nuclei_dir is not None:
        nuclei_dir = Path(nuclei_dir)
        stem = source.stem.removesuffix(".ome")
        for candidate in (nuclei_dir / f"{stem}.tif", nuclei_dir / f"{meta['name']}.tif"):
            if candidate.exists():
                import tifffile
                return np.asarray(tifffile.imread(str(candidate)), dtype=np.int32)
    fallback = _cmap_in_masks(source, meta["name"], masks_dir)
    if fallback is not None:
        import h5py
        with h5py.File(fallback, "r") as f:
            if "nucleus/data" in f:
                return np.asarray(f["nucleus/data"], dtype=np.int32)
    return None


def load_soma(fov_dir: Path, masks_dir: Path | None = None) -> np.ndarray | None:
    """Instance-labelled soma masks from a .cmap beside the source (cell_body/data),
    or from a .cmap in masks_dir."""
    meta = read_meta(fov_dir)
    source = Path(meta["source"])
    cmap = source.with_suffix(".cmap")
    if cmap.exists():
        import h5py
        with h5py.File(cmap, "r") as f:
            if "cell_body/data" in f:
                return np.asarray(f["cell_body/data"], dtype=np.int32)
    fallback = _cmap_in_masks(source, meta["name"], masks_dir)
    if fallback is not None:
        import h5py
        with h5py.File(fallback, "r") as f:
            if "cell_body/data" in f:
                return np.asarray(f["cell_body/data"], dtype=np.int32)
    return None


def pair_nuclei(nuclei: np.ndarray, spacing_um, marks_um: np.ndarray,
                diameter_um: float) -> list[tuple[int, float]]:
    """For each mark, find the nucleus instance with the largest overlap inside a sphere of
    diameter_um. Returns (nucleus_id, volume_um3) per mark; (0, nan) when none."""
    dz, dy, dx = spacing_um
    radius = diameter_um / 2
    voxel_vol = dz * dy * dx
    # precompute total volume per instance
    instance_ids, instance_counts = np.unique(nuclei, return_counts=True)
    total_vol = dict(zip(instance_ids.tolist(), (instance_counts * voxel_vol).tolist()))
    paired = set()
    results = []
    for z_um, y_um, x_um in np.asarray(marks_um, float).reshape(-1, 3):
        zc, yc, xc = z_um / dz, y_um / dy, x_um / dx
        rz, ry, rx = radius / dz, radius / dy, radius / dx
        z0 = max(0, int(np.floor(zc - rz)))
        z1 = min(nuclei.shape[0], int(np.ceil(zc + rz)) + 1)
        y0 = max(0, int(np.floor(yc - ry)))
        y1 = min(nuclei.shape[1], int(np.ceil(yc + ry)) + 1)
        x0 = max(0, int(np.floor(xc - rx)))
        x1 = min(nuclei.shape[2], int(np.ceil(xc + rx)) + 1)
        sub = nuclei[z0:z1, y0:y1, x0:x1]
        zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
        dist2 = ((zz * dz - z_um) / radius) ** 2 + ((yy * dy - y_um) / radius) ** 2 + \
                 ((xx * dx - x_um) / radius) ** 2
        sphere = dist2 <= 1.0
        inside = sub[sphere]
        inside = inside[inside > 0]
        if inside.size == 0:
            results.append((0, float("nan")))
            continue
        ids, counts = np.unique(inside, return_counts=True)
        best_idx = int(np.argmax(counts))
        nuc_id = int(ids[best_idx])
        paired.add(nuc_id)
        results.append((nuc_id, total_vol.get(nuc_id, float("nan"))))
    # log unpaired nuclei
    all_ids = set(instance_ids.tolist()) - {0}
    unpaired = sorted(all_ids - paired)
    if unpaired:
        print(f"  found {len(unpaired)} nuclei without a neuron soma "
              f"(ids: {unpaired[:20]}{'...' if len(unpaired) > 20 else ''}); "
              f"labelled as non-neuronal")
    return results


def add_nuclei(fov_dir: Path, neurons: Path, rows: list[dict[str, str]],
               nuclei: np.ndarray | None, spacing_um) -> None:
    """Rewrite neurons.csv with nucleus_id and nucleus_volume_um3."""
    meta = read_meta(fov_dir)
    marks_um = np.array([[float(r[f"{a}_um"]) for a in "zyx"] for r in rows])
    if nuclei is not None:
        pairs = pair_nuclei(nuclei, spacing_um, marks_um, meta["neuron_diameter_um"])
        values = [{"nucleus_id": str(nid), "nucleus_volume_um3": f"{vol:.4f}" if nid else ""}
                  for nid, vol in pairs]
        how = (f"paired by largest overlap in a sphere of {meta['neuron_diameter_um']:g} um "
               f"about the click")
    else:
        values = [{"nucleus_id": "", "nucleus_volume_um3": ""}] * len(rows)
        how = "not computed: no nuclei masks available"
    # read current header from the file
    header = [line[2:] for line in neurons.read_text().splitlines() if line.startswith("# ")]
    # determine current fields from the file
    all_fields = list(rows[0].keys()) if rows else FIELDS + ["soma_whole"]
    if "nucleus_id" not in all_fields:
        all_fields += ["nucleus_id", "nucleus_volume_um3"]
    io.write_rows(neurons, all_fields,
                  [{**r, **v} for r, v in zip(rows, values)],
                  header=[*header, f"nucleus: {how}"])


def seal(fov_dir: Path, nuclei_dir: Path | None = None,
         masks_dir: Path | None = None) -> dict:
    """Make the census final; any later edit is refused and a hand edit is detected. Sealing adds
    the experimental soma_whole column (add_soma_whole) before the hash is taken."""
    if is_sealed(fov_dir):
        raise PermissionError(f"{fov_dir} is already sealed")
    neurons = census_dir(fov_dir) / "neurons.csv"
    rows = io.read_rows(neurons)
    n = len(rows)
    if n == 0:
        raise ValueError(f"{fov_dir} has no marked neurons to seal")
    add_soma_whole(fov_dir, neurons, rows)
    rows = io.read_rows(neurons)  # re-read after soma_whole added its column
    nuclei = load_nuclei(fov_dir, nuclei_dir, masks_dir)
    meta = read_meta(fov_dir)
    add_nuclei(fov_dir, neurons, rows, nuclei, meta["spacing_um"])
    stamp = {"n_neurons": n, "sha256": digest(neurons),
             "sealed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (census_dir(fov_dir) / "sealed.json").write_text(json.dumps(stamp, indent=2))
    return stamp


def check_seal(fov_dir: Path) -> list[dict[str, str]]:
    """Sealed neuron rows, refused if unsealed or if neurons.csv changed after sealing."""
    if not is_sealed(fov_dir):
        raise PermissionError(f"{fov_dir} is not sealed; seal the census before scoring")
    stamp = json.loads((census_dir(fov_dir) / "sealed.json").read_text())
    neurons = census_dir(fov_dir) / "neurons.csv"
    if digest(neurons) != stamp["sha256"]:
        raise PermissionError(f"{neurons} changed after it was sealed; its neuron_ids can no longer "
                              f"be trusted")
    return io.read_rows(neurons)


def run(fov_dir: Path, path: str | Path, series: int, channels: dict[str, int],
        neuron_diameter_um: float = NEURON_DIAMETER_UM, screenshot: Path | None = None,
        contrast: str = "fov", display: dict[str, dict] | None = None) -> CensusResult:
    """Open (or resume) the census of one fov in napari; every click is saved as it is made.
    contrast="project" takes the display from project_display over every fov of the project."""
    if set(channels) != set(CHANNELS):
        raise ValueError(f"channels must map exactly {CHANNELS} to indices, got {channels}")
    if is_sealed(fov_dir):
        return CensusResult(Path(fov_dir), len(marks(fov_dir)), True)
    fov = io.read_fov(path, series, channels)
    meta = start(fov_dir, fov, neuron_diameter_um, contrast, display)
    open_viewer(fov, meta, marks(fov_dir), lambda points: record(fov_dir, points), screenshot)
    return CensusResult(Path(fov_dir), len(marks(fov_dir)), False)


def show_until_closed(viewer, screenshot: Path | None) -> None:
    """Show the window and return once it closes, saving it to screenshot just before."""
    from qtpy.QtCore import QEvent, QEventLoop, QObject

    class BeforeClose(QObject):
        """Saves the whole window when the close event arrives, while it is still drawn."""

        def eventFilter(self, watched, event) -> bool:
            """Screenshot on close; never consume the event."""
            if event.type() == QEvent.Type.Close and screenshot is not None:
                viewer.screenshot(str(screenshot), canvas_only=False)
            return False

    window = viewer.window._qt_window
    window.installEventFilter(BeforeClose(window))
    # Not napari.run(), which returns at once under IPython because the kernel already drives Qt,
    # nor viewer.show(block=True), which touches the window again after Qt has deleted it.
    loop = QEventLoop()
    window.destroyed.connect(loop.quit)
    loop.exec()


def points_size(um: float) -> Any:
    """A float size for napari Points; its size argument is unannotated, so pyright reads the
    default 10 as int although napari takes any float."""
    return um


def open_viewer(fov: io.Fov, meta: dict, existing: np.ndarray, on_change,
                screenshot: Path | None = None) -> None:
    """napari window with DAPI and NeuN; calls on_change(marks_um) after every edit."""
    import napari
    import napari.layers

    viewer = napari.Viewer(title=f"CENSUS {meta['fov']}  DAPI + NeuN only", ndisplay=2)
    for name, colormap in (("dapi", "blue"), ("neun", "gray")):
        setting = meta["display"][name]
        viewer.add_image(fov.data[name], name=name.upper(), scale=fov.spacing_um,
                         colormap=colormap, blending="additive",
                         contrast_limits=setting["limits"], gamma=setting["gamma"],
                         projection_mode="none")
    radius = meta["neuron_diameter_um"] / 2
    z0_um = (fov.shape[0] // 2) * fov.spacing_um[0]
    # Added before the marks so the marks draw on top; 2D, so it shows on every plane.
    rings = napari.layers.Points(echo(existing, z0_um, radius), name="counted on other planes",
                                 ndim=2,
                                 size=points_size(ECHO_SIZE_UM), face_color="transparent",
                                 border_color=ECHO_COLOUR, border_width=0.18, opacity=MARK_OPACITY)
    viewer.add_layer(rings)
    rings.editable = False
    # No scale on the points layer, so its data are world micrometres.
    layer = napari.layers.Points(existing, name="neurons", ndim=3, size=meta["neuron_diameter_um"],
                                 face_color="transparent", border_color=MARK_COLOUR,
                                 border_width=MARK_BORDER_WIDTH, opacity=MARK_OPACITY,
                                 projection_mode="rescale_spherical")
    viewer.add_layer(layer)
    guard_remove(layer)
    # A mark is drawn within one radius of its centre plane. napari's Image default projection_mode
    # is "mean", which would average every plane inside this margin; "none" above keeps one plane.
    viewer.dims.margin_left = (radius, 0, 0)
    viewer.dims.margin_right = (radius, 0, 0)
    viewer.dims.set_point(0, z0_um)
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = "click each neuron once, on its centre plane"
    state = {"last": np.asarray(existing, float).reshape(-1, 3)}

    def saved(event=None) -> None:
        """Persist the layer, or undo the click that was refused."""
        points = np.asarray(layer.data, float).reshape(-1, 3)
        try:
            on_change(points)
            state["last"] = points
        except ValueError as error:
            layer.events.data.disconnect(saved)
            layer.data = state["last"]
            layer.events.data.connect(saved)
            viewer.status = str(error)
            print(f"  refused: {error}")

    def redraw_echo(event=None) -> None:
        """Ring every counted neuron whose own disc is not drawn on this plane."""
        rings.data = echo(layer.data, float(viewer.dims.point[0]), radius)

    layer.events.data.connect(saved)
    layer.events.data.connect(redraw_echo)
    viewer.dims.events.point.connect(redraw_echo)
    viewer.layers.selection.active = layer
    layer.mode = "add"
    header = f"\u2500\u2500 NEURONS  {meta['fov']} "
    rule = "\u2500" * max(2, 50 - len(header))
    print(f"\n{header}{rule}")
    print("  Click once on each neuron, on the plane where it is widest.")
    print("  Change plane: slider at the bottom, or \u2190 \u2192.")
    print("  Yellow ring = counted here \u00b7 red ring = counted on another plane.")
    print("  Done: close the window, then type y here to seal.\n")
    show_until_closed(viewer, screenshot)
