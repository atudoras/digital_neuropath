"""Project setup, configuration and workflow entry points for digital-neuropath."""
import argparse
import json
from pathlib import Path

from digital_neuropath import io, settings

__all__ = [
    "STEPS", "apply", "load_settings", "locks", "main", "read_toml", "run",
    "run_from_settings", "scan_images", "status", "write_settings", "write_toml",
]

HEADER = "# Written by digital-neuropath.\n\n"


# --- Scoring step adapter ---

class _Scoring:
    """Wraps score.py's per-stain workflow as a pipeline step."""
    STEP = "scoring"
    TITLE = "Scoring"
    SETTINGS: dict[str, settings.Field] = {
        "blind": settings.Field(True, help="shuffled, no image names; False = in order, named"),
        "score": settings.Field("all", help="stains to score this session; others keep their calls"),
    }

    def run(self, project_dir: Path, vals: dict, fovs: list[str] | None = None) -> None:
        from digital_neuropath import score
        config = read_toml(project_dir)
        nuclei_dir_str = config.get("masks", {}).get("nuclei")
        nuclei_dir = Path(nuclei_dir_str) if nuclei_dir_str else None
        masks_dir_str = config.get("masks", {}).get("soma")
        masks_dir = Path(masks_dir_str) if masks_dir_str else None
        if not config.get("stains"):
            return
        chosen = vals.get("score", "all")
        for sc in config.get("stains", []):
            name, ch = sc["name"], sc["channel"]
            if chosen != "all" and name not in chosen:
                print(f"  {name}: not in SCORE; its calls are kept")
                continue
            waiting = [e for e in sc.get("skip_from", [])
                       if not score.scoring_dir(project_dir, e).exists()
                       or score.unanswered(project_dir, e)]
            if waiting:
                print(f"  {name}: waits until {', '.join(waiting)} is fully answered")
                break
            if not score.scoring_dir(project_dir, name).exists():
                n_existing = len(list(Path(project_dir).glob("scoring_*")))
                score.build(project_dir, name, ch, sc.get("categories"),
                            seed=score.SEED + n_existing,
                            skip_from=sc.get("skip_from"), nuclei_dir=nuclei_dir,
                            masks_dir=masks_dir, context=sc.get("also_show", []))
            else:
                score.update_context(project_dir, name, sc.get("also_show", []))
                score.extend(project_dir, name, nuclei_dir=nuclei_dir,
                             masks_dir=masks_dir)
            result = score.run(project_dir, name, blind=vals.get("blind", True))
            print(f"  {name}: {result.n_answered}/{result.n_crops}")
            if result.n_answered < result.n_crops:
                print(f"  stopped in {name}; run again to resume")
                break
            score.unseal(project_dir, name)

    def view(self, project_dir: Path, fov: str) -> None:
        pass


scoring = _Scoring()


def _load_steps() -> list:
    """neurons -> scoring -> mtocs -> cilia -> link."""
    from digital_neuropath import neurons
    from digital_neuropath.cilia.steps import STEPS as cilia_steps
    return [neurons, scoring, *cilia_steps]


STEPS = _load_steps()


# --- TOML ---

def read_toml(root: Path) -> dict:
    """Parse project.toml; empty dict if missing."""
    path = Path(root) / "project.toml"
    if not path.exists():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
    return tomllib.loads(path.read_text())


def _quote(v: str) -> str:
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _format_toml(config: dict) -> str:
    lines = [HEADER.rstrip()]
    if "channels" in config:
        lines.append("[channels]")
        lines.extend(f"{k} = {v}" for k, v in config["channels"].items())
        lines.append("")
    for s in config.get("sources", []):
        lines.append("[[sources]]")
        lines.append(f"path = {_quote(s['path'])}")
        if "series" in s:
            lines.append(f"series = {json.dumps(s['series'])}")
        if "case" in s:
            lines.append(f"case = {_quote(s['case'])}")
        lines.append("")
    for s in config.get("stains", []):
        lines.append("[[stains]]")
        lines.append(f"name = {_quote(s['name'])}")
        lines.append(f"channel = {s['channel']}")
        if "categories" in s:
            lines.append(f"categories = {json.dumps(s['categories'])}")
        if "skip_from" in s:
            lines.append(f"skip_from = {json.dumps(s['skip_from'])}")
        if "also_show" in s:
            lines.append(f"also_show = {json.dumps(s['also_show'])}")
        lines.append("")
    if "masks" in config:
        lines.append("[masks]")
        lines.extend(f"{k} = {_quote(v)}" for k, v in config["masks"].items())
        lines.append("")
    for sid, sv in config.get("steps", {}).items():
        lines.append(f"[steps.{sid}]")
        for k, v in sv.items():
            if isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                lines.append(f"{k} = {v}")
            elif isinstance(v, str):
                lines.append(f"{k} = {_quote(v)}")
            else:
                lines.append(f"{k} = {json.dumps(v)}")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_toml(root: Path, config: dict) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tmp = (root / "project.toml").with_suffix(".toml.tmp")
    tmp.write_text(_format_toml(config))
    tmp.replace(root / "project.toml")


# --- Locking ---

def locks(root: Path) -> dict[str, str]:
    root = Path(root)
    out: dict[str, str] = {}
    any_answer = False
    for d in sorted(root.glob("scoring_*/answers.csv")):
        stain = d.parent.name.removeprefix("scoring_")
        if io.read_rows(d):
            out[f"stain:{stain}:categories"] = f"{stain} has answers"
            any_answer = True
    if any_answer:
        out["channels"] = "a stain has answers scored with these channels"
    return out


# --- Image scanning ---

def scan_images(folder: str | Path) -> list[dict]:
    """Every .lif and .tif in a folder, as sources; files that cannot be read are left out."""
    return image_sources([p for p in sorted(Path(folder).iterdir())
                          if p.suffix.lower() in (".lif", ".tif", ".tiff")], strict=False)


def image_sources(paths: list[Path], strict: bool = True) -> list[dict]:
    """Sources from image files, each file one case named by its stem: every series of a .lif,
    the one image of a .tif. strict: a file that cannot be read stops with its reason."""
    results: list[dict] = []
    for p in map(Path, paths):
        case = p.stem.removesuffix(".ome")
        try:
            infos = io.list_series(p)
        except (ValueError, OSError) as error:
            if strict:
                raise ValueError(f"{p.name}: {error}") from None
            continue
        if p.suffix.lower() != ".lif":
            infos = infos[:1]
        results += [{"path": str(p), "name": info.name, "series": info.series, "case": case,
                     "n_channels": info.n_channels} for info in infos]
    return results


# --- Settings file ---

SETTINGS_TEMPLATE = '''\
# digital_neuropath: per-neuron scoring of 3D confocal stacks

# =============================== INPUT ===============================

# Your files, where they already are, one per line. When they share a folder, write it once
# here and each file as f"{{FOLDER}}/<file name>"; otherwise give each file's full path.
# Masks pair with images by position (the 1st mask belongs to the 1st image), so each mask
# list has as many files as IMAGES.
FOLDER = "{folder}"

IMAGES = {images}   # .tif (OME or ImageJ) or .lif

NEURON_MASKS = {neuron_masks}   # labelled neuron masks (.tif or .cmap); [] = click each neuron

NUCLEI_MASKS = {nuclei_masks}   # labelled nucleus masks (.tif), optional; [] = none

OUTPUT = "{project}"   # folder where the tool writes everything; tables in {project}/results/

# Channel of each stain in your images; the first channel is 0
DAPI = 0
NEUN = 1

# Stains, scored one after the other in this order: "name": (channel, [categories])
# "unsure" and "not scoreable" are added to every list; [] = loaded, not scored.
# A neuron called "not scoreable" on one stain is not shown on the next.
STAINS = {stains}

# Other stains to load in a stain's window, e.g. {{"tdp43": ["acvr1c"]}}. Seeing another stain
# can bias the call; {{}} keeps each window to DAPI, NeuN and its own stain.
ALSO_SHOW = {{}}

# True : every session, neurons of all images in a new random order, no image names
# False: image by image, in order, image name and neuron shown
BLIND = True
SCORE = list(STAINS)   # stains to score this session; the others keep their calls and are not shown

# ======================================================================

RUN = {run}   # add "mtocs", "cilia", "link" for 3D cilia (needs "ac3" and "pcnt")
PARAMS = {{}}

if __name__ == "__main__":
    from digital_neuropath import app
    app.run_from_settings(__file__)
'''

DEFAULT_STAINS = {
    "tdp43": (2, ["Normal", "Nuclear clearance", "NCI"]),
    "ac3": (3, ["yes, normal", "abnormally long", "abnormally short", "not found"]),
}


def _stains_text(stains: dict) -> str:
    rows = [f"    {name!r}: ({ch}, {list(cats)!r})," for name, (ch, cats) in stains.items()]
    return "{\n" + "\n".join(rows).replace("'", '"') + "\n}"


def _list_text(items: list[str], folder: str = "") -> str:
    """A python list, one file a line; inside FOLDER when one is given."""
    if not items:
        return "[]"
    lines = [f'    f"{{FOLDER}}/{i}",\n' if folder else f'    "{i}",\n' for i in items]
    return "[\n" + "".join(lines) + "]"


def write_settings(path: Path, images: list[str], project: str,
                   neuron_masks: list[str] = (), nuclei_masks: list[str] = (),
                   stains: dict | None = None, run: list[str] | None = None,
                   folder: str = "") -> None:
    """Write the settings script: choices on top, runnable with python3. With a folder, the
    files are names inside it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SETTINGS_TEMPLATE.format(
        name=path.name, folder=folder, images=_list_text(list(images), folder), project=project,
        neuron_masks=_list_text(list(neuron_masks), folder),
        nuclei_masks=_list_text(list(nuclei_masks), folder),
        stains=_stains_text(stains or DEFAULT_STAINS),
        run=json.dumps(run or ["neurons", "scoring"])))


def load_settings(path: str | Path) -> dict:
    """Execute a settings.py and return its uppercase variables."""
    path = Path(path).resolve()
    ns: dict = {"__name__": "digital_neuropath_settings", "__file__": str(path)}
    exec(compile(path.read_text(), path, "exec"), ns)
    out = {}
    for k, v in ns.items():
        if k.isupper() and not k.startswith("_"):
            if isinstance(v, str) and ("~" in v or "$" in v):
                v = str(Path(v).expanduser())
            out[k] = v
    return out


def _expand(v: str, base: Path) -> str:
    """Expand ~ and resolve a relative path against base, the settings script's folder."""
    if not v:
        return v
    p = Path(v).expanduser()
    return str(p if p.is_absolute() else base / p)


def _files(s: dict, key: str, base: Path, n: int | None = None) -> list[Path]:
    """A list of files from the settings, each checked; n = the length it must have when not []."""
    value = s.get(key, [])
    if isinstance(value, (str, Path)):
        raise ValueError(f'{key} must be a list of files, e.g. {key} = ["data/images/a.tif"]')
    files = [Path(_expand(str(v), base)) for v in value]
    if n is not None and files and len(files) != n:
        raise ValueError(f"{key} has {len(files)} file(s) and IMAGES has {n}: give one per image, "
                         f"in the same order")
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"{key}: not found: {', '.join(missing)}")
    return files


def link_inputs(project_dir: Path, images: list[Path], files: list[Path], kind: str) -> Path:
    """project/inputs/<kind>/<image name><mask suffix> -> each mask, so mask i goes with image i."""
    folder = Path(project_dir) / "inputs" / kind
    folder.mkdir(parents=True, exist_ok=True)
    for image, f in zip(images, files, strict=True):
        if image.suffix.lower() == ".lif":
            raise ValueError(f"{image.name}: a .lif holds several images; give masks for "
                             f"single-image .tif files only")
        link = folder / (image.stem.removesuffix(".ome") + f.suffix.lower())
        if link.is_symlink() or link.exists():
            if link.resolve() == f.resolve():
                continue
            link.unlink()
        link.symlink_to(f.resolve())
    return folder


def run_from_settings(path: str | Path) -> None:
    """Load a settings.py file and run the pipeline."""
    s = load_settings(path)
    base = Path(path).resolve().parent
    if not s.get("OUTPUT"):
        raise ValueError('OUTPUT is empty: name the folder the tool writes to, e.g. OUTPUT = "output"')
    project_dir = Path(_expand(s["OUTPUT"], base))
    images = _files(s, "IMAGES", base)
    if not images:
        raise ValueError("IMAGES is empty: list at least one image file")
    stems = [i.stem.removesuffix(".ome") for i in images]
    if len(set(stems)) < len(stems):
        raise ValueError("two IMAGES have the same file name; rename one")
    neuron_masks = _files(s, "NEURON_MASKS", base, len(images))
    nuclei_masks = _files(s, "NUCLEI_MASKS", base, len(images))

    # channels
    channels = {"dapi": s.get("DAPI", 0), "neun": s.get("NEUN", 1)}
    stains_def = s.get("STAINS", {})
    for name, spec in stains_def.items():
        ch = spec[0] if isinstance(spec, (list, tuple)) else spec
        channels[name] = ch

    # scored stains in order; each skips neurons an earlier one called not scoreable
    from digital_neuropath import score
    also_show = score.check_also_show(s.get("ALSO_SHOW", {}), stains_def)
    stains: list[dict] = []
    for name, spec in stains_def.items():
        if isinstance(spec, (list, tuple)) and len(spec) >= 2 and spec[1]:
            earlier = [st["name"] for st in stains]
            stains.append({"name": name, "channel": spec[0], "categories": list(spec[1]),
                           **({"skip_from": earlier} if earlier else {}),
                           **({"also_show": also_show[name]} if also_show.get(name) else {})})
    chosen = list(s.get("SCORE", stains_def))
    unknown = [name for name in chosen if name not in stains_def]
    if unknown:
        raise ValueError(f"SCORE names {unknown}, not in STAINS {list(stains_def)}")

    # masks: mask i belongs to image i
    masks: dict[str, str] = {}
    if neuron_masks:
        masks["soma"] = str(link_inputs(project_dir, images, neuron_masks, "masks"))
    if nuclei_masks:
        masks["nuclei"] = str(link_inputs(project_dir, images, nuclei_masks, "nuclei"))

    sources = image_sources(images)

    # steps to run
    run_list = list(s.get("RUN", [st.STEP for st in STEPS]))
    stain_names = set(stains_def.keys()) | {"dapi", "neun"}
    cilia_steps = {"mtocs", "cilia", "link"}
    for step_name in run_list:
        if step_name in cilia_steps:
            for needed in ("ac3", "pcnt"):
                if needed not in stain_names:
                    raise ValueError(
                        f"step '{step_name}' needs channel '{needed}' but it is not in STAINS")

    # build config
    config: dict = {"channels": channels, "sources": sources}
    if stains:
        config["stains"] = stains
    if masks:
        config["masks"] = masks
    params = {k: dict(v) for k, v in s.get("PARAMS", {}).items()}
    if not s.get("BLIND", True):
        params.setdefault("scoring", {})["blind"] = False
    if {st["name"] for st in stains} - set(chosen):
        params.setdefault("scoring", {})["score"] = chosen
    if params:
        config["steps"] = params

    # write toml + run
    write_toml(project_dir, config)
    apply(project_dir, config)
    from digital_neuropath import project
    print(f"  registered {len(project.fovs(project_dir))} fov(s)")
    stored = config.get("steps", {})
    for step_mod in STEPS:
        if step_mod.STEP not in run_list:
            continue
        vals = settings.values(step_mod.SETTINGS, stored.get(step_mod.STEP) or {})
        print(f"  {step_mod.TITLE}...")
        step_mod.run(project_dir, vals)
    from digital_neuropath import summary
    print(f"  wrote {summary.write_results(project_dir)}")


# --- Apply ---

def apply(root: Path, config: dict) -> None:
    """Register sources via project.add(); idempotent."""
    from digital_neuropath import project
    root = Path(root)
    channel_map = dict(config.get("channels", {}))
    for s in config.get("stains", []):
        channel_map.setdefault(s["name"], s["channel"])
    for source in config.get("sources", []):
        path, case = source["path"], source.get("case")
        if path.lower().endswith(".lif"):
            project.add_lif(root, path, series=source.get("series"),
                            channels=channel_map or None, case=case)
        else:
            project.add_files(root, [path], channels=channel_map or None, case=case)


# --- Workflow ---

def run(root: str | Path, ticked: set[str] | None = None) -> None:
    """Register sources, then run each pipeline step."""
    root = Path(root)
    config = read_toml(root)
    if not config:
        raise FileNotFoundError(f"no project.toml in {root}")
    from digital_neuropath import project
    apply(root, config)
    print(f"  registered {len(project.fovs(root))} fov(s)")
    stored = config.get("steps", {})
    for step_mod in STEPS:
        if ticked is not None and step_mod.STEP not in ticked:
            continue
        vals = settings.values(step_mod.SETTINGS, stored.get(step_mod.STEP) or {})
        print(f"  {step_mod.TITLE}...")
        step_mod.run(root, vals)


def status(root: str | Path) -> str:
    from digital_neuropath import census, project, score
    root = Path(root)
    config = read_toml(root)
    entries = project.fovs(root) if (root / "project.json").exists() else []
    sealed = sum(1 for e in entries if census.is_sealed(Path(root) / "fovs" / e["fov"]))
    lines = [f"project: {root}", f"fovs: {len(entries)} ({sealed} sealed)"]
    for sc in config.get("stains", []):
        name = sc["name"]
        if score.scoring_dir(root, name).exists():
            q, a = len(score.queue(root, name)), len(score.answered(root, name))
            lines.append(f"  {name}: {a}/{q} answered")
            if (root / "results" / f"{name}_unsealed.csv").exists():
                lines.append("    unsealed")
        else:
            lines.append(f"  {name}: not started")
    return "\n".join(lines)


# --- CLI ---

def main() -> None:
    parser = argparse.ArgumentParser(prog="digital-neuropath",
                                     description="Blind per-neuron scoring of 3D confocal stacks")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("new", help="Write a settings.py template").add_argument("folder", type=Path)

    demo_p = sub.add_parser("demo", help="Write synthetic demo images + settings.py")
    demo_p.add_argument("folder", type=Path)

    run_p = sub.add_parser("run", help="Run pipeline from settings.py")
    run_p.add_argument("path", type=Path)

    sub.add_parser("params", help="Print all step parameters")

    st_p = sub.add_parser("status", help="Print project status")
    st_p.add_argument("path", type=Path)

    view_p = sub.add_parser("view", help="View a pipeline step result in napari")
    view_p.add_argument("step", choices=[s.STEP for s in STEPS])
    view_p.add_argument("fov")
    view_p.add_argument("path", type=Path, nargs="?", default=Path("."))

    args = parser.parse_args()

    if args.command == "new":
        dest = args.folder / "settings.py"
        write_settings(dest, folder="/path/to/your/files", images=["image1.tif"],
                       project="output", neuron_masks=["image1_neuron_mask.tif"],
                       nuclei_masks=["image1_nuclei.tif"])
        print(f"Wrote {dest}")
        print(f"Edit INPUT, then:  python3 {dest}")

    elif args.command == "demo":
        import shutil
        from digital_neuropath import demo
        tiff, cmap, _ = demo.write_cilia(args.folder / "images")
        masks = args.folder / "masks"
        masks.mkdir(parents=True, exist_ok=True)
        shutil.move(str(cmap), str(masks / cmap.name))
        dest = args.folder / "settings.py"
        write_settings(dest,
                       images=[str(tiff)],
                       project=str(args.folder / "output"),
                       neuron_masks=[str(masks / cmap.name)],
                       stains={"tdp43": DEFAULT_STAINS["tdp43"], "ac3": (3, []), "pcnt": (4, [])},
                       run=[st.STEP for st in STEPS])
        print(f"Wrote 1 demo image to {args.folder / 'images'}")
        print(f"Wrote masks to {masks}")
        print(f"Wrote {dest}")
        print(f"Next:  digital-neuropath run {dest}")

    elif args.command == "run":
        p = args.path
        if p.is_dir():
            p = p / "settings.py"
        run_from_settings(p)

    elif args.command == "params":
        for step_mod in STEPS:
            print(f"\n{step_mod.STEP}")
            print("-" * len(step_mod.STEP))
            if not step_mod.SETTINGS:
                print("  (no parameters)")
                continue
            for name, f in step_mod.SETTINGS.items():
                unit = f" ({f.unit})" if f.unit else ""
                print(f"  {name}{unit} = {f.default}")

    elif args.command == "status":
        print(status(args.path if args.path.is_dir() else args.path.parent))

    elif args.command == "view":
        root = args.path if args.path.is_dir() else args.path.parent
        step = next(s for s in STEPS if s.STEP == args.step)
        step.view(root, args.fov)

    else:
        parser.print_help()
