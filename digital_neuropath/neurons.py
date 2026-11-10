"""Pass 1, neuron refinement: match masks or fall back to census clicks, refine, seal.
Chain: run -> (match masks | census clicks) -> view (refinement window) -> seal.
Neurons the fade rule marks whole=no start as suggested exclusions; the rater clicks to toggle.
"""
import datetime
import json
from pathlib import Path

import numpy as np

from digital_neuropath import census, io
from digital_neuropath.settings import Field

__all__ = ["SETTINGS", "STEP", "TITLE", "run", "seal_refined", "view"]

STEP = "neurons"
TITLE = "Neurons"
FADE_THRESHOLD = 0.5

REFINED_FIELDS = ["fov", "neuron_id", "mask_label", "z_um", "y_um", "x_um", "kept", "whole",
                  "reason", "decided_by", "source"]

SETTINGS: dict[str, Field] = {
    "neuron_diameter_um": Field(15.0, unit="µm", help="diameter of a neuron soma"),
}


def refined_dir(fov_dir: Path) -> Path:
    return Path(fov_dir) / "refined"


def refined_path(fov_dir: Path) -> Path:
    return refined_dir(fov_dir) / "refined.csv"


def is_refined(fov_dir: Path) -> bool:
    return refined_path(fov_dir).exists()


def read_refined(fov_dir: Path) -> list[dict[str, str]]:
    return io.read_rows(refined_path(fov_dir))


def kept_neurons(fov_dir: Path) -> list[dict[str, str]]:
    """Neurons not marked x."""
    return [r for r in read_refined(fov_dir) if r["kept"] != "no"]


def match_mask(names: list[str], masks_dir: str) -> Path | None:
    """Find a .cmap or label .tif in masks_dir matching one of the names."""
    if not masks_dir:
        return None
    d = Path(masks_dir)
    if not d.is_dir():
        return None
    for name in names:
        for ext in (".cmap", ".tif", ".tiff"):
            candidate = d / f"{name}{ext}"
            if candidate.exists():
                return candidate
    return None


def load_mask_instances(mask_path: Path) -> tuple[np.ndarray, list[int]]:
    """Load a mask and return (label_volume, instance_ids) with excluded dropped."""
    if mask_path.suffix == ".cmap":
        from digital_neuropath.cilia.load import read_cmap
        body, _nucleus, excluded = read_cmap(mask_path)
    else:
        import tifffile
        body = np.asarray(tifffile.imread(str(mask_path))).astype(np.int32)
        excluded = []
    all_ids = sorted({int(v) for v in np.unique(body)} - {0} - set(excluded))
    body[np.isin(body, excluded)] = 0
    return body, all_ids


def delivered_body(mask_path: Path) -> np.ndarray:
    """Cell-body labels as delivered, producer exclusions included."""
    if mask_path.suffix == ".cmap":
        from digital_neuropath.cilia.load import read_cmap
        return read_cmap(mask_path)[0]
    import tifffile
    return np.asarray(tifffile.imread(str(mask_path))).astype(np.int32)


def centroid_um(body: np.ndarray, label: int, spacing_um: tuple[float, ...]) -> tuple[float, ...]:
    """Centroid of one label in physical coordinates."""
    mean_vox = np.argwhere(body == label).mean(axis=0)
    return tuple(float(mean_vox[i]) * spacing_um[i] for i in range(3))


def mask_neurons(body: np.ndarray, instance_ids: list[int],
                 spacing_um: tuple[float, ...]) -> list[dict]:
    """One row per mask instance."""
    rows = []
    for label in instance_ids:
        z, y, x = centroid_um(body, label, spacing_um)
        rows.append({"z_um": f"{z:.4f}", "y_um": f"{y:.4f}", "x_um": f"{x:.4f}",
                      "mask_label": label})
    return rows


def label_at_clicks(body: np.ndarray, rows: list[dict[str, str]]) -> list[str]:
    """Mask label under each census click's voxel, '' when off the grid or on background."""
    out = []
    for r in rows:
        p = tuple(int(r[k]) for k in ("z_vox", "y_vox", "x_vox"))
        inside = all(0 <= v < s for v, s in zip(p, body.shape, strict=True))
        label = int(body[p]) if inside else 0
        out.append(str(label) if label else "")
    return out


def fade_whole_sphere(neun: np.ndarray, spacing_um: tuple[float, ...],
                      z_um: float, y_um: float, x_um: float,
                      diameter_um: float) -> str:
    """'' when a sphere fades below FADE_THRESHOLD before both z faces, else the nearer z face."""
    whole = census.soma_whole(neun, list(spacing_um),
                              np.array([[z_um, y_um, x_um]]), diameter_um)[0]
    if whole == "yes":
        return ""
    return "cut: z top" if z_um <= (neun.shape[0] - 1) * spacing_um[0] - z_um else "cut: z bottom"


def sphere_edge(shape: tuple[int, ...], spacing_um: tuple[float, ...],
                y_um: float, x_um: float, diameter_um: float) -> str:
    """'cut: y edge' / 'cut: x edge' when the click is within a soma radius of that face."""
    for axis, name, at in ((1, "cut: y edge", y_um), (2, "cut: x edge", x_um)):
        if min(at, (shape[axis] - 1) * spacing_um[axis] - at) < diameter_um / 2:
            return name
    return ""


def mask_edge(body: np.ndarray, label: int) -> str:
    """'cut: y edge' / 'cut: x edge' when the label has a voxel on that face of the stack."""
    mask = body == label
    if mask[:, 0, :].any() or mask[:, -1, :].any():
        return "cut: y edge"
    if mask[:, :, 0].any() or mask[:, :, -1].any():
        return "cut: x edge"
    return ""


def fade_whole_mask(body: np.ndarray, neun: np.ndarray, label: int,
                    spacing_um: tuple[float, ...]) -> str:
    """'' when the mask's NeuN fades below FADE_THRESHOLD before both z faces, else why it is cut."""
    dz = spacing_um[0]
    n_z = neun.shape[0]
    mask = body == label
    zs = np.where(mask.any(axis=(1, 2)))[0]
    if len(zs) < 2:
        return "one z plane"
    z_min, z_max = int(zs[0]), int(zs[-1])
    level = np.array([max(float(np.percentile(neun[z][neun[z] > 0], 99.0))
                          if (neun[z] > 0).any() else 1e-9, 1e-9)
                      for z in range(n_z)])
    trace = {}
    for z in range(n_z):
        plane_mask = mask[z]
        if plane_mask.any():
            trace[z] = float(neun[z][plane_mask].mean()) / level[z]
    if not trace:
        return "one z plane"
    peak = max(trace.values())
    if peak <= 0:
        return "no neun in mask"
    for face, name in ((0, "cut: z top"), (n_z - 1, "cut: z bottom")):
        dist_um = abs(face * dz - (z_min if face == 0 else z_max) * dz)
        if dist_um < 2 * dz and trace.get(face, 0.0) / peak >= FADE_THRESHOLD:
            return name
    return ""


def suggest_whole(body: np.ndarray | None, neun: np.ndarray | None,
                  spacing_um: tuple[float, ...], row: dict,
                  diameter_um: float, mask_label: int | None) -> tuple[str, str]:
    """(whole, reason): 'no' with the cut face, 'yes' / 'inside'; mask-based when a mask exists."""
    if body is not None and mask_label is not None:
        cut = mask_edge(body, mask_label)
        if not cut and neun is not None:
            cut = fade_whole_mask(body, neun, mask_label, spacing_um)
    elif neun is not None:
        y_um, x_um = float(row["y_um"]), float(row["x_um"])
        cut = (sphere_edge(neun.shape, spacing_um, y_um, x_um, diameter_um)
               or fade_whole_sphere(neun, spacing_um, float(row["z_um"]), y_um, x_um, diameter_um))
    else:
        return "", "no neun"
    if cut:
        return "no", cut
    return ("yes", "inside") if neun is not None else ("", "no neun")


def build_refined(fov_dir: Path, neurons: list[dict], source: str,
                  body: np.ndarray | None, neun: np.ndarray | None,
                  spacing_um: tuple[float, ...], diameter_um: float) -> list[dict[str, str]]:
    """Initial refined.csv rows before the rater opens the refinement window."""
    rows = []
    for n in neurons:
        mask_label = n.get("mask_label")
        whole, reason = suggest_whole(body, neun, spacing_um, n, diameter_um,
                                      int(mask_label) if mask_label else None)
        kept = "no" if whole == "no" else "yes"
        rows.append({"fov": n["fov"], "neuron_id": n["neuron_id"],
                      "mask_label": str(mask_label or ""),
                      "z_um": n["z_um"], "y_um": n["y_um"], "x_um": n["x_um"],
                      "kept": kept, "whole": whole, "reason": reason,
                      "decided_by": "tool", "source": source})
    return rows


def write_refined(fov_dir: Path, rows: list[dict[str, str]], fov_id: str) -> None:
    """Write refined.csv."""
    io.write_rows(refined_path(fov_dir), REFINED_FIELDS, rows, header=[
        f"REFINEMENT of {fov_id}",
        f"{len(rows)} neurons, {sum(1 for r in rows if r['kept'] == 'no')} excluded",
        (f"whole: {sum(1 for r in rows if r['whole'] == 'yes')} yes, "
         f"{sum(1 for r in rows if r['whole'] == 'no')} no, "
         f"{sum(1 for r in rows if r['whole'] == '')} unknown"),
    ])


def seal_refined(fov_dir: Path) -> dict:
    """Seal the refinement; refuse if empty or already sealed."""
    seal_path = refined_dir(fov_dir) / "sealed.json"
    if seal_path.exists():
        raise PermissionError(f"{fov_dir} refinement is already sealed")
    rows = read_refined(fov_dir)
    if not rows:
        raise ValueError(f"{fov_dir} has no refined neurons to seal")
    stamp = {"n_neurons": len(rows),
             "n_kept": sum(1 for r in rows if r["kept"] != "no"),
             "n_whole": sum(1 for r in rows if r["whole"] == "yes"),
             "sha256": census.digest(refined_path(fov_dir)),
             "sealed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    seal_path.write_text(json.dumps(stamp, indent=2))
    return stamp


def run(project_dir: Path, values: dict, fovs: list[str] | None = None) -> None:
    """Per fov: census from masks or clicks, then refinement window, seal."""
    from digital_neuropath import app, database, project as proj

    project_dir = Path(project_dir)
    config = app.read_toml(project_dir)
    masks_dir = config.get("masks", {}).get("soma", "")
    nuclei_dir_str = config.get("masks", {}).get("nuclei")
    nuclei_dir = Path(nuclei_dir_str) if nuclei_dir_str else None
    diameter_um = values["neuron_diameter_um"]

    for entry in database.fovs(project_dir):
        fov_id = entry["fov"]
        if fovs is not None and fov_id not in fovs:
            continue
        fov_dir = project_dir / "fovs" / fov_id

        if is_refined(fov_dir):
            kept = len(kept_neurons(fov_dir))
            print(f"  {fov_id}: already refined, {kept}/{len(read_refined(fov_dir))} kept; skipped")
            continue

        source_stem = Path(entry["source"]).stem.removesuffix(".ome")
        mask_path = match_mask([fov_id, source_stem], masks_dir)
        body, instance_ids = (load_mask_instances(mask_path) if mask_path else (None, []))

        # Census: from mask centroids or interactive clicks
        from_masks = False
        if not census.is_sealed(fov_dir):
            if body is not None:
                own = entry.get("channels") or config.get("channels", {})
                fov_channels = {name: own[name] for name in ("dapi", "neun")}
                fov = io.read_fov(entry["source"], entry["series"], fov_channels)
                census.start(fov_dir, fov)
                marks_um = np.array([centroid_um(body, label, tuple(fov.spacing_um))
                                     for label in instance_ids])
                census.record(fov_dir, marks_um)
                census.seal(fov_dir, nuclei_dir=nuclei_dir, masks_dir=masks_dir)
                from_masks = True
                print(f"  {fov_id}: {len(instance_ids)} neurons from masks, census sealed")
            else:
                own = entry.get("channels") or config.get("channels")
                proj.label_neurons(project_dir, own, diameter_um, only=[fov_id])
                if not census.is_sealed(fov_dir):
                    print(f"  {fov_id}: census not sealed, skipping refinement")
                    continue

        # Refinement
        census_rows = census.check_seal(fov_dir)
        meta = census.read_meta(fov_dir)
        spacing_um = tuple(meta["spacing_um"])
        neun = None
        source = "clicks"

        if from_masks and body is not None:
            neuron_list = [{"fov": fov_id, "neuron_id": f"N{i:03d}",
                            "z_um": mr["z_um"], "y_um": mr["y_um"], "x_um": mr["x_um"],
                            "mask_label": mr["mask_label"]}
                           for i, mr in enumerate(mask_neurons(body, instance_ids, spacing_um), 1)]
            source = "masks"
        else:
            # a clicked census keeps its ids; a later mask adds the delivered label under each click
            labels = (label_at_clicks(delivered_body(mask_path), census_rows) if mask_path
                      else [""] * len(census_rows))
            neuron_list = [{"fov": r["fov"], "neuron_id": r["neuron_id"], "mask_label": label,
                            "z_um": r["z_um"], "y_um": r["y_um"], "x_um": r["x_um"]}
                           for r, label in zip(census_rows, labels, strict=True)]

        try:
            neun = io.read_fov(meta["source"], meta["series"],
                               {"neun": meta["channels_loaded"]["neun"]}).data["neun"]
        except OSError:
            pass

        rows = build_refined(fov_dir, neuron_list, source, body, neun, spacing_um, diameter_um)
        write_refined(fov_dir, rows, fov_id)
        view(project_dir, fov_id, _body=body)
        seal_refined(fov_dir)
        n_kept = sum(1 for r in read_refined(fov_dir) if r["kept"] != "no")
        print(f"  {fov_id}: sealed, {n_kept}/{len(read_refined(fov_dir))} kept")


def view(project_dir: Path, fov: str, *,
         _body: np.ndarray | None = None) -> None:
    """Refinement window: napari with DAPI + NeuN, click to toggle kept, close to accept."""
    import napari

    fov_dir = Path(project_dir) / "fovs" / fov
    meta = census.read_meta(fov_dir)
    spacing_um = tuple(meta["spacing_um"])
    rows = read_refined(fov_dir)
    if not rows:
        raise ValueError(f"{fov} has no refined neurons")

    channels = {name: meta["channels_loaded"][name] for name in ("dapi", "neun")}
    fov_data = io.read_fov(meta["source"], meta["series"], channels)
    display = meta["display"]
    has_masks = any(r.get("source") == "masks" for r in rows)

    n_suggested = sum(1 for r in rows if r["kept"] == "no" and r["decided_by"] == "tool")
    extra = (f"\u2500\u2500 {len(rows)} neurons, {n_suggested} suggested "
             f"exclusion{'s' if n_suggested != 1 else ''} ") if n_suggested else ""
    header = f"\u2500\u2500 REFINE  {fov} {extra}"
    rule = "\u2500" * max(2, 50 - len(header))
    print(f"\n{header}{rule}")
    print("  Click a neuron to include / exclude.")
    print("  Green = included \u00b7 orange = suggested exclusion \u00b7 red = excluded by you.")
    print("  Done: close the window.\n")

    viewer = napari.Viewer(title=f"REFINE {fov}  DAPI + NeuN only", ndisplay=2)
    for name, colormap in (("dapi", "blue"), ("neun", "gray")):
        s = display[name]
        viewer.add_image(fov_data.data[name], name=name.upper(), scale=spacing_um,
                         colormap=colormap, blending="additive",
                         contrast_limits=s["limits"], gamma=s["gamma"],
                         projection_mode="none")

    if has_masks and _body is not None:
        viewer.add_labels(_body, name="masks", scale=spacing_um, opacity=0.3)  # type: ignore[attr-defined]

    marks_um = np.array([[float(r["z_um"]), float(r["y_um"]), float(r["x_um"])] for r in rows])
    colours = [_neuron_colour(r) for r in rows]
    diameter = float(meta.get("neuron_diameter_um", 15.0))
    layer = napari.layers.Points(marks_um, name="neurons", ndim=3,  # type: ignore[attr-defined]
                                 size=diameter,
                                 face_color="transparent", border_color=colours,
                                 border_width=0.08, opacity=0.7,
                                 projection_mode="rescale_spherical")
    viewer.add_layer(layer)
    viewer.dims.margin_left = (diameter / 2, 0, 0)
    viewer.dims.margin_right = (diameter / 2, 0, 0)
    layer.editable = False

    state = {"rows": list(rows)}

    def _update_colours():
        layer.border_color = [_neuron_colour(r) for r in state["rows"]]

    viewer.text_overlay.visible = True
    viewer.text_overlay.text = "click: include / exclude   close when done"

    @viewer.mouse_drag_callbacks.append
    def _on_click(viewer_ref, event):
        dragged = False
        yield
        while event.type == "mouse_move":
            dragged = True
            yield
        if dragged:
            return
        pos = np.array(viewer_ref.cursor.position[:3])
        dists_yx = np.linalg.norm(marks_um[:, 1:] - pos[1:3], axis=1)
        idx = int(np.argmin(dists_yx))
        if dists_yx[idx] > diameter / 2:
            return
        r = state["rows"][idx]
        r["kept"] = "yes" if r["kept"] == "no" else "no"
        r["decided_by"] = "rater"
        write_refined(fov_dir, state["rows"], fov)
        _update_colours()

    census.show_until_closed(viewer, None)


def _neuron_colour(row: dict[str, str]) -> str:
    if row["kept"] == "no":
        return "#FFA500" if row["decided_by"] == "tool" else "#FF0000"
    return "#00FF00"
