"""A project folder with one subfolder per fov, run as census for every fov, then scoring per stain.
Chain: add_lif / add_files -> label_neurons (open, save per click, seal on request) -> score_stain -> summary.
Each step reads what is on disk and skips what is done, so any step can be stopped and rerun, and a
project can take new files at any time: their fovs are counted, then join each stain in a new round.
"""
import json
from pathlib import Path

from digital_neuropath import app, census, database, io, score, summary

__all__ = ["add_files", "add_lif", "fovs", "label_neurons", "relocate", "score_stain"]


def fovs(root: Path) -> list[dict]:
    """Every fov of the project (database.fovs): fov id, source file, series and the rest."""
    return database.fovs(root)


def add(root: Path, entries: list[tuple[str, int, str]],
        channels: dict[str, int] | None = None, case: str | None = None) -> list[str]:
    """Register (source, series, name) fovs; a fov already registered is skipped, with a warning
    when it was registered from another file or under another case. channels (name -> index,
    e.g. dapi, neun and each stain) and case are stored on each new fov, so files with different
    channel orders and cases share one project."""
    root = Path(root)
    known = fovs(root)
    by_id = {f["fov"]: f for f in known}
    added = []
    for source, series, name in entries:
        fid = io.fov_id(source, series)
        source = str(Path(source).resolve())
        if fid in by_id:
            old = by_id[fid]
            if old["source"] != source:
                print(f"  WARNING {fid}: already registered from {old['source']}; skipped")
            elif case is not None and old.get("case") not in (None, case):
                raise ValueError(f"{fid} is registered as case {old['case']}, not {case}")
            elif channels and old.get("channels") and old["channels"] != channels:
                raise ValueError(f"{fid} is registered with channels {old['channels']}, "
                                 f"not {channels}")
            continue
        entry = {"fov": fid, "source": source, "series": series, "name": name,
                 **({"channels": dict(channels)} if channels else {}),
                 **({"case": case} if case is not None else {})}
        known.append(entry)
        by_id[fid] = entry
        (root / "fovs" / fid).mkdir(parents=True, exist_ok=True)
        added.append(fid)
    database.write_fovs(root, known)
    return added


def add_lif(root: Path, lif_path: str | Path, series: list[int] | None = None,
            channels: dict[str, int] | None = None, case: str | None = None) -> list[str]:
    """Register the series (one fov each) of a .lif: all of them, or only those listed in `series`.
    Refused when a channel index is past the channels a series holds."""
    chosen = io.list_series(lif_path, series)
    for info in chosen:
        if channels and max(channels.values()) >= info.n_channels:
            raise ValueError(f"{Path(lif_path).name} s{info.series:02d} has {info.n_channels} "
                             f"channels; {channels} names index {max(channels.values())}")
    return add(root, [(info.source, info.series, info.name) for info in chosen], channels, case)


def add_files(root: Path, paths: list[str | Path], channels: dict[str, int] | None = None,
              case: str | None = None) -> list[str]:
    """Register a list of single-fov files (OME-TIFF), each as series 0."""
    return add(root, [(str(p), 0, Path(p).stem) for p in paths], channels, case)


def relocate(root: Path, path: str | Path) -> list[str]:
    """Point every fov registered from a file of this name at `path`, its new place on disk (the
    same file moved, or downloaded again from cloud storage). Only the source path changes: the
    neuron list, seal and scores stay as they are. Returns the fovs moved ([] when none)."""
    root, path = Path(root), Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    entries = fovs(root)
    moved = []
    for entry in entries:
        if Path(entry["source"]).name == path.name and entry["source"] != str(path):
            entry["source"] = str(path)
            meta_path = census.census_dir(root / "fovs" / entry["fov"]) / "census.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                meta["source"] = str(path)
                meta_path.write_text(json.dumps(meta, indent=2))
            moved.append(entry["fov"])
    if moved:
        database.write_fovs(root, entries)
    return moved


def label_neurons(root: Path, channels: dict[str, int] | None,
                  neuron_diameter_um: float = census.NEURON_DIAMETER_UM, confirm=input,
                  screenshot: Path | None = None, contrast: str = "fov",
                  only: list[str] | None = None) -> list[census.CensusResult]:
    """Open the census of every unsealed fov in turn (or of the fov ids in `only`), and seal each
    one the rater confirms; a sealed fov is reported with its neuron count and not opened.
    screenshot names one PNG per fov: <stem>_<fov><suffix> beside the path given.
    contrast="project" pools DAPI and NeuN limits over every fov, computed once, before the first
    fov whose census has not started. A fov registered with its own channels uses their dapi and
    neun; channels is the default for the others."""
    results = []
    display = None
    toml = app.read_toml(root)
    nuclei_dir_str = toml.get("masks", {}).get("nuclei")
    nuclei_dir = Path(nuclei_dir_str) if nuclei_dir_str else None
    masks_dir_str = toml.get("masks", {}).get("soma")
    masks_dir = Path(masks_dir_str) if masks_dir_str else None
    for entry in fovs(root):
        if only is not None and entry["fov"] not in only:
            continue
        fov_dir = Path(root) / "fovs" / entry["fov"]
        if census.is_sealed(fov_dir):
            n = len(census.check_seal(fov_dir))
            print(f"  {entry['fov']}: already counted, {n} neuron(s); skipped")
            results.append(census.CensusResult(fov_dir, n, True))
            continue
        started = (census.census_dir(fov_dir) / "census.json").exists()
        own = entry.get("channels") or channels
        if own is None:
            raise ValueError(f"{entry['fov']} has no channel map; pass channels or register it "
                             f"with one")
        fov_channels = {name: own[name] for name in census.CHANNELS}
        if contrast == "project" and display is None and not started:
            display = census.project_display([(f["source"], f["series"]) for f in fovs(root)],
                                             fov_channels)
        result = census.run(fov_dir, entry["source"], entry["series"], fov_channels,
                            neuron_diameter_um, contrast=contrast,
                            screenshot=None if screenshot is None else Path(screenshot).with_name(
                                f"{Path(screenshot).stem}_{entry['fov']}{Path(screenshot).suffix}"),
                            display=None if started else display)
        if result.n_neurons:
            reply = confirm(f"seal {entry['fov']} with {result.n_neurons} neuron(s)? After "
                            f"sealing no edit is possible. [y/N] ")
            if reply.strip().lower() == "y":
                census.seal(fov_dir, nuclei_dir=nuclei_dir, masks_dir=masks_dir)
                result = census.CensusResult(fov_dir, result.n_neurons, True)
        results.append(result)
        print(f"  {entry['fov']}: {result.n_neurons} neuron(s), "
              f"{'sealed' if result.sealed else 'NOT sealed'}")
    return results


def score_stain(root: Path, stain: str, stain_channel: int | None, categories: list[str] | None,
                seed: int = score.SEED, crop_half_um: float | None = None,
                screenshot: Path | None = None, target_size_um: float = score.TARGET_SIZE_UM,
                skip_from: list[str] | None = None, contrast: str = "fov",
                only: list[str] | None = None, blind: bool = True) -> score.ScoreResult:
    """Score this stain (blind: shuffled, no names; not blind: in order, named): the first call builds a crop per neuron of every sealed fov; a
    later call adds a round for neurons counted since. Unanswered crops open in napari; once every
    crop is answered, unseal and summarise. stain_channel is the default; a fov registered with its
    own channels uses its index for stain. Neurons answered in imported/<stain>.csv get no crop.
    `only` (fov ids) limits the crops cut now to those fovs; the rest join in a later round.
    crop_half_um: the build's half-size (default score.CROP_HALF_UM); for a later round, None
    keeps the half-size the build fixed and a number sets this round's (recorded per round)."""
    root = Path(root)
    toml = app.read_toml(root)
    nuclei_dir_str = toml.get("masks", {}).get("nuclei")
    nuclei_dir = Path(nuclei_dir_str) if nuclei_dir_str else None
    masks_dir_str = toml.get("masks", {}).get("soma")
    masks_dir = Path(masks_dir_str) if masks_dir_str else None
    if not score.scoring_dir(root, stain).exists():
        half = crop_half_um if crop_half_um is not None else score.CROP_HALF_UM
        new = score.build(root, stain, stain_channel, categories, seed, half, skip_from,
                          contrast, only, nuclei_dir=nuclei_dir, masks_dir=masks_dir)
        print(f"  {stain}: built {new.n_crops} blind crop(s)")
    else:
        new = score.extend(root, stain, only, crop_half_um, nuclei_dir=nuclei_dir,
                           masks_dir=masks_dir)
        if new.n_crops:
            cases = sorted({str(f.get("case")) for f in fovs(root)
                            if f["fov"] in score.manifest_rounds(root, stain)[-1]["fovs"]})
            print(f"  {stain}: new round of {new.n_crops} blind crop(s), case(s) {cases}")
    left = len(score.unanswered(root, stain))
    unsealed = root / "results" / f"{stain}_unsealed.csv"
    joined = {r["blind_id"] for r in io.read_rows(unsealed)}
    if not left and unsealed.exists() and set(score.queue(root, stain)) <= joined:
        print(f"  {stain}: nothing left to score{' in these fovs' if only else ''}; skipped")
        n = len(score.queue(root, stain))
        return score.ScoreResult(stain, n, n)
    filled, left_out = score.add_context(root, stain)
    if filled or left_out:
        print(f"  {stain}: context {score.SCORING_CONTEXT.get(stain)} added to {filled} crop(s)"
              + (f"; {left_out} crop(s) shown without it (file not on disk)" if left_out else ""))
    result = score.run(root, stain, screenshot=screenshot, target_size_um=target_size_um,
                       blind=blind)
    print(f"  {stain}: {result.n_answered} of {result.n_crops} crops answered")
    if result.n_answered == result.n_crops:
        print(f"  unsealed -> {score.unseal(root, stain)}")
        for path in summary.write(root):
            print(f"  wrote {path}")
    return result
