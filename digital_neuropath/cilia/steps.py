"""Step objects (mtocs, cilia, link) and the per-case summary."""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from digital_neuropath import io, masks
from digital_neuropath.cilia import measure, params
from digital_neuropath.cilia.cilia import segment_cilia
from digital_neuropath.cilia.link import Review, assign_to_neurons, decisions_digest
from digital_neuropath.cilia.load import Fov, read_cmap, read_labels, read_plane, read_table
from digital_neuropath.cilia.load import load as load_fov
from digital_neuropath.cilia.mtocs import segment_mtocs
from digital_neuropath.settings import changed, from_params, values

__all__ = ["STEPS", "cilia", "link", "mtocs", "summary"]


def _entries(project_dir: Path, fovs: list[str] | None) -> list[dict]:
    path = project_dir / "project.json"
    if not path.exists():
        raise FileNotFoundError(f"no project.json in {project_dir}")
    entries = json.loads(path.read_text())["fovs"]
    if fovs is not None:
        wanted = set(fovs)
        entries = [e for e in entries if e["fov"] in wanted]
    return entries


def _masks_dir(project_dir: Path) -> str:
    from digital_neuropath.app import read_toml
    return read_toml(project_dir).get("masks", {}).get("soma", "")

def _review_dir(project_dir: Path) -> str:
    from digital_neuropath.app import read_toml
    return read_toml(project_dir).get("masks", {}).get("review", "")


def read_review(review_dir: str, fov_id: str) -> Review | None:
    """The fov's step-2 review when its CSV exists and every row is done, with its close-up decisions."""
    path = Path(review_dir) / f"{fov_id}.csv" if review_dir else None
    if path is None or not masks.is_done(path):
        return None
    closeups = Path(review_dir) / "closeup_decisions.csv"
    closeup = ({int(r["cell_body_id"]): r["decision"] for r in io.read_rows(closeups) if r["fov"] == fov_id}
               if closeups.exists() else {})
    return Review({int(r["cell_body_id"]): r["status"] for r in io.read_rows(path)}, closeup)


def _load_fov(entry: dict, project_dir: Path, masks_dir: str = "") -> Fov:
    cmap = None
    if masks_dir:
        from digital_neuropath.neurons import match_mask
        stem = Path(entry["source"]).stem.removesuffix(".ome")
        mask_path = match_mask([entry["fov"], stem], masks_dir)
        if mask_path is not None and mask_path.suffix == ".cmap":
            cmap = mask_path
    return load_fov(
        Path(entry["source"]), channels=entry.get("channels", {}),
        neun_instances=cmap,
        series=int(entry.get("series", 0)), fov_id=entry["fov"],
        donor=entry.get("case") or entry.get("name") or entry["fov"])


def _resume_ok(out_dir: Path, resolved: dict[str, Any], inputs: dict[str, str] | None = None) -> bool:
    run_json = out_dir / "run.json"
    if not run_json.exists():
        return False
    record = json.loads(run_json.read_text())
    recorded = record.get("inputs", {})
    return record.get("values") == resolved and all(recorded.get(k) == v for k, v in (inputs or {}).items())


COLOURS = {"dapi": (0.0, 0.25, 1.0), "neun": (0.75, 0.75, 0.75), "ac3": (0.1, 1.0, 0.2), "pcnt": (1.0, 0.1, 0.1)}
NAMES = {"dapi": "DAPI", "neun": "NeuN", "ac3": "AC3", "pcnt": "PCNT"}


def _show_channels(v, fov: Fov, names: tuple[str, ...]) -> None:
    """Add each channel the fov has, additive, in the manual's colours."""
    from napari.utils import Colormap
    for name in names:
        try:
            data = read_plane(fov, name)
        except (KeyError, ValueError):
            continue
        lo, hi = np.percentile(data[::2, ::4, ::4], [75 if name in ("ac3", "pcnt") else 50, 99.9])
        v.add_image(data, name=NAMES[name], scale=fov.spacing_um, blending="additive",
                    contrast_limits=(float(lo), float(max(hi, lo + 1))),
                    colormap=Colormap([[0, 0, 0, 1], [*COLOURS[name], 1]], name=NAMES[name]))


def _show_mask(v, labels: np.ndarray, name: str, colour: tuple[float, float, float], contour: int, sp) -> None:
    """One-colour outline layer."""
    from napari.utils.colormaps import DirectLabelColormap
    layer = v.add_labels(labels, name=name, scale=sp, opacity=0.9)  # type: ignore[attr-defined]
    layer.contour = contour
    layer.colormap = DirectLabelColormap(color_dict={None: (*colour, 1.0), 0: (0, 0, 0, 0)})


# -- mtocs ------------------------------------------------------------------

class _Mtocs:
    STEP = "mtocs"
    TITLE = "PCNT / MTOC segmentation"
    SETTINGS = from_params(params.MTOC)

    def run(self, project_dir: Path, vals: dict[str, Any],
            fovs: list[str] | None = None) -> None:
        resolved = values(self.SETTINGS, vals)
        kw = changed(self.SETTINGS, resolved)
        md = _masks_dir(project_dir)
        for entry in _entries(project_dir, fovs):
            out = project_dir / "cilia" / entry["fov"] / "mtocs"
            if _resume_ok(out, resolved):
                continue
            fov = _load_fov(entry, project_dir, md)
            if "pcnt" not in fov.channels:
                print(f"  {entry['fov']}: no pcnt channel, skipping mtocs")
                continue
            try:
                segment_mtocs(fov, out, **kw)
            except (KeyError, ValueError) as exc:
                print(f"  {entry['fov']}: skipping mtocs ({exc})")

    def view(self, project_dir: Path, fov: str) -> None:
        import napari
        f = _load_fov(_entries(project_dir, [fov])[0], project_dir, _masks_dir(project_dir))
        labels = read_labels(project_dir / "cilia" / fov / "mtocs" / "labels.tif")
        v = napari.Viewer(title=f"MTOC — {fov}")
        _show_channels(v, f, ("dapi", "neun", "pcnt"))
        _show_mask(v, labels, "PCNT foci", (1.0, 1.0, 1.0), 1, f.spacing_um)
        napari.run()


mtocs = _Mtocs()


# -- cilia ------------------------------------------------------------------

class _Cilia:
    STEP = "cilia"
    TITLE = "AC3 cilium segmentation"
    SETTINGS = from_params(params.CILIA)

    def run(self, project_dir: Path, vals: dict[str, Any],
            fovs: list[str] | None = None) -> None:
        resolved = values(self.SETTINGS, vals)
        kw = changed(self.SETTINGS, resolved)
        md = _masks_dir(project_dir)
        for entry in _entries(project_dir, fovs):
            out = project_dir / "cilia" / entry["fov"] / "cilia"
            if _resume_ok(out, resolved):
                continue
            fov = _load_fov(entry, project_dir, md)
            if "ac3" not in fov.channels:
                print(f"  {entry['fov']}: no ac3 channel, skipping cilia")
                continue
            try:
                segment_cilia(fov, out, **kw)
            except (KeyError, ValueError) as exc:
                print(f"  {entry['fov']}: skipping cilia ({exc})")

    def view(self, project_dir: Path, fov: str) -> None:
        import napari
        f = _load_fov(_entries(project_dir, [fov])[0], project_dir, _masks_dir(project_dir))
        labels = read_labels(project_dir / "cilia" / fov / "cilia" / "labels.tif")
        v = napari.Viewer(title=f"Cilia — {fov}")
        _show_channels(v, f, ("dapi", "neun", "ac3", "pcnt"))
        _show_mask(v, labels, "cilium objects", (1.0, 0.85, 0.0), 2, f.spacing_um)
        napari.run()


cilia = _Cilia()


# -- link -------------------------------------------------------------------

class _Link:
    STEP = "link"
    TITLE = "Cilium -> neuron assignment"
    SETTINGS = from_params(params.LINK)

    def run(self, project_dir: Path, vals: dict[str, Any],
            fovs: list[str] | None = None) -> None:
        resolved = values(self.SETTINGS, vals)
        kw = changed(self.SETTINGS, resolved)
        md = _masks_dir(project_dir)
        rd = _review_dir(project_dir)
        for entry in _entries(project_dir, fovs):
            out = project_dir / "cilia" / entry["fov"] / "link"
            refined_csv = project_dir / "fovs" / entry["fov"] / "refined" / "refined.csv"
            refined = io.read_rows(refined_csv) if refined_csv.exists() else None
            review = read_review(rd, entry["fov"])
            # a changed refinement or review re-runs link even with the same knobs
            if _resume_ok(out, resolved, decisions_digest(refined, review)):
                continue
            fov = _load_fov(entry, project_dir, md)
            missing = [ch for ch in ("pcnt", "ac3") if ch not in fov.channels]
            if missing:
                print(f"  {entry['fov']}: no {'/'.join(missing)} channel(s), skipping link")
                continue
            if fov.neun_instances is None:
                print(f"  {entry['fov']}: no NeuN masks (.cmap), skipping link")
                continue
            fov.cilia_dir = project_dir / "cilia" / entry["fov"] / "cilia"
            fov.mtoc_dir = project_dir / "cilia" / entry["fov"] / "mtocs"
            try:
                assign_to_neurons(fov, None, None, out, refined=refined, review=review, **kw)
            except (KeyError, ValueError) as exc:
                print(f"  {entry['fov']}: skipping link ({exc})")

    def view(self, project_dir: Path, fov_id: str) -> None:
        import napari
        entry = _entries(project_dir, [fov_id])[0]
        fov = _load_fov(entry, project_dir, _masks_dir(project_dir))
        base = project_dir / "cilia" / fov_id
        sp = fov.spacing_um

        cilia_lab = read_labels(base / "cilia" / "labels.tif")
        mtoc_lab = read_labels(base / "mtocs" / "labels.tif")
        cilia_tbl = read_table(base / "link", "cilia")
        nt = read_table(base / "link", "neuron_table")
        neurons_vol, _, excluded = read_cmap(fov.neun_instances)  # type: ignore[arg-type]
        if excluded:
            neurons_vol = np.where(np.isin(neurons_vol, excluded), 0, neurons_vol)

        chosen = {int(v) for v in cilia_tbl.loc[cilia_tbl["chosen"], "label"]}
        chosen_mask = np.where(np.isin(cilia_lab, list(chosen)), cilia_lab, 0)

        v = napari.Viewer(title=f"Link — {fov_id}")
        _show_channels(v, fov, ("dapi", "neun", "ac3", "pcnt"))
        _show_mask(v, neurons_vol, "neurons", (0.55, 0.05, 0.12), 2, sp)
        _show_mask(v, mtoc_lab, "PCNT foci", (1.0, 1.0, 1.0), 1, sp)
        _show_mask(v, chosen_mask, "chosen cilia", (1.0, 0.85, 0.0), 2, sp)

        rows = nt[nt["eligible"]].to_dict("records")  # type: ignore[call-overload]
        positions, texts = [], []
        for row in rows:
            nid = int(row["neuron_id"])
            mask = neurons_vol == nid
            if not mask.any():
                continue
            c = np.argwhere(mask).mean(axis=0) * np.array(sp)
            has = row["has_cilium"]
            txt = f"n{nid}: {'cil' if has else 'no'}"
            if has and pd.notna(row["length_um"]):
                txt += f" {float(row['length_um']):.1f}µm"
            positions.append(c)
            texts.append(txt)
        if positions:
            v.add_points(np.array(positions), name="labels",  # type: ignore[attr-defined]
                         text={"string": texts, "size": 9, "color": "yellow"},
                         size=0.5, face_color="transparent", border_color="transparent")
        napari.run()


link = _Link()

STEPS = [mtocs, cilia, link]


# -- summary ----------------------------------------------------------------

def summary(project_dir: Path) -> pd.DataFrame:
    """One table per case: k ciliated of n eligible, Wilson 95%, median length."""
    project_dir = Path(project_dir)
    entries = json.loads((project_dir / "project.json").read_text())["fovs"]
    tables = []
    for entry in entries:
        nt = project_dir / "cilia" / entry["fov"] / "link" / "neuron_table.csv"
        if nt.exists():
            tables.append(read_table(project_dir / "cilia" / entry["fov"] / "link",
                                     "neuron_table"))
    if not tables:
        return pd.DataFrame()
    return measure.summarize_cilia(pd.concat(tables, ignore_index=True))
