"""Pass 2, blind scoring of one stain: one crop per sealed neuron, shuffled across every fov of a project.
Chain: build (crops + manifest outside the scoring folder) -> run (answer, flush, resume) -> unseal.
The rater sees a random id and three channels; which fov and which neuron come back only at unseal.
A neuron already answered in imported/<stain>.csv gets no crop and unseals with that answer.
"""
import csv
import datetime
import hashlib
import json
import os
import re
import secrets
import shutil
import string
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from digital_neuropath import census, io

__all__ = ["ScoreResult", "all_categories", "answer", "build", "extend", "run", "tune_display",
           "unanswered", "unseal"]

DEFAULT_CATEGORIES = ["positive", "negative"]
FIXED_CATEGORIES = ["unsure", "not scoreable"]
ANSWER_FIELDS = ["blind_id", "category", "note", "answered_utc", "blind"]
MANIFEST_FIELDS = ["blind_id", "fov", "neuron_id", "z_um", "y_um", "x_um",
                   "crop_z0", "crop_z1", "crop_y0", "crop_y1", "crop_x0", "crop_x1",
                   "target_z_um", "target_y_um", "target_x_um", "spacing_um", "display", "pad_vox"]
SKIPPED_FIELDS = ["fov", "neuron_id", "z_um", "y_um", "x_um", "category"]
IMPORTED_REQUIRED = ["fov", "neuron_id", "category", "note"]
BLIND_ID_ALPHABET = string.ascii_uppercase + string.digits
BLIND_ID_LENGTH = 6
STAIN_PCT = (1.0, 99.9)
# "raw" = linear 0 to RAW_TOP_FRACTION of the detector range; "stretch" = 0 to crop p99.99.
VIEW_STAIN_LIMITS: dict[str, list[float] | str] = {"ac3": "raw"}
RAW_TOP_FRACTION = 0.5
SCORING_CONTEXT: dict[str, list[str]] = {"ac3": ["pcnt"]}
CONTEXT_COLOUR = "red"
CONTEXT_PCT = (90.0, 99.99)
TARGET_COLOUR = "#FF9F1C"
# Scoring look: DAPI blue (hidden by default), NeuN grey, stain green.
SCORING_DISPLAY: dict[str, tuple[str, bool]] = {"dapi": ("blue", False),
                                                "neun": ("gray", True),
                                                "stain": ("green", True)}
# Opacity per channel in the scoring window: additive NeuN at full weight whitened the soma under
# the stain (labmate figs 2-3). Channels not listed are drawn at 1.
SCORING_WEIGHTS: dict[str, float] = {"neun": 0.45}
# Other sealed census marks inside a crop; the neuron scored stays the orange cross.
OTHER_MARK_COLOUR = "#9E9E9E"
OTHER_MARK_SIZE_UM = 2.0
VIEWS = ("planes", "mip")
TUNE_SAMPLE = 12
TUNE_TRIES = 1000  # random draws per fov for a point clear of every census mark
PENDING_FIELDS = ["blind_id", "note", "noted_utc"]
# Unseal QC, never shown to the rater: the fraction of the stain's top COBRIGHT_STAIN voxels that
# also sit in another channel's top COBRIGHT_OTHER (chance = COBRIGHT_OTHER).
COBRIGHT_STAIN = 0.002
COBRIGHT_OTHER = 0.01
TARGET_SIZE_UM = 3
CROP_HALF_UM = 30.0
# Base shuffle seed; the i-th stain uses SEED + i, so no two stains share an order.
SEED = 20261001


@dataclass(frozen=True)
class ScoreResult:
    """Progress of one stain's scoring pass."""
    stain: str
    n_crops: int
    n_answered: int


def _drop_excluded_neurons(sealed: dict[Path, list[dict[str, str]]]
                           ) -> dict[Path, list[dict[str, str]]]:
    """Drop neurons marked kept=no in refined.csv, so they get no crop."""
    from digital_neuropath import neurons
    out = {}
    for fov_dir, rows in sealed.items():
        if neurons.is_refined(fov_dir):
            excluded = {r["neuron_id"] for r in neurons.read_refined(fov_dir) if r["kept"] == "no"}
            out[fov_dir] = [n for n in rows if n["neuron_id"] not in excluded]
        else:
            out[fov_dir] = rows
    return out


def scoring_dir(project_dir: Path, stain: str) -> Path:
    """Folder the rater works in; holds crops and answers and nothing that identifies a neuron."""
    return Path(project_dir) / f"scoring_{stain}"


def manifest_path(project_dir: Path, stain: str) -> Path:
    """Blind manifest, deliberately outside the scoring folder."""
    return Path(project_dir) / "manifests" / f"{stain}.csv"


def all_categories(categories: list[str]) -> list[str]:
    """The caller's categories followed by 'unsure' and 'not scoreable', which are always present."""
    own = [c.strip() for c in categories if c.strip() and c.strip() not in FIXED_CATEGORIES]
    if not own:
        raise ValueError("give at least one category besides 'unsure' and 'not scoreable'")
    if len(set(own)) != len(own):
        raise ValueError(f"categories repeat: {own}")
    out = own + FIXED_CATEGORIES
    if len(out) > 9:
        raise ValueError(f"{len(out)} categories do not fit on keys 1-9")
    return out


def fov_dirs(project_dir: Path) -> list[Path]:
    """Every fov folder of a project, in name order."""
    root = Path(project_dir) / "fovs"
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []


def blind_ids(n: int, rng: np.random.Generator) -> list[str]:
    """n distinct random ids that carry no fov, neuron or position, in the order drawn, so the
    seed alone fixes which neuron gets which id."""
    out: dict[str, None] = {}
    while len(out) < n:
        out["".join(rng.choice(list(BLIND_ID_ALPHABET), BLIND_ID_LENGTH))] = None
    return list(out)


def stain_limits(volume: np.ndarray) -> list[float]:
    """Stain contrast for a whole fov, computed once: p1 to p99.9 of positive voxels."""
    lo = census.percentile_positive(volume, STAIN_PCT[0])
    return [lo, max(census.percentile_positive(volume, STAIN_PCT[1]), lo + 1.0)]


def crop_bounds(centre_vox: tuple[int, int, int], half_vox: tuple[int, int, int],
                shape: tuple[int, int, int]) -> list[int]:
    """[z0, z1, y0, y1, x0, x1] of a box about the neuron, clipped to the fov."""
    out = []
    for c, h, n in zip(centre_vox, half_vox, shape):
        out += [max(0, c - h), min(n, c + h + 1)]
    return out


def skipped_path(project_dir: Path, stain: str) -> Path:
    """Neurons given no crop because an earlier stain called them not scoreable."""
    return Path(project_dir) / "manifests" / f"{stain}_skipped.csv"


def imported_path(project_dir: Path, stain: str) -> Path:
    """Answers made before this project (e.g. another pipeline), one row per (fov, neuron_id)."""
    return Path(project_dir) / "imported" / f"{stain}.csv"


def imported(project_dir: Path, stain: str) -> dict[tuple[str, str], dict[str, str]]:
    """Imported answers by (fov, neuron_id); refused if a column is missing, a category is empty
    or a neuron is answered twice. Columns beyond IMPORTED_REQUIRED are carried to unseal."""
    path = imported_path(project_dir, stain)
    rows = io.read_rows(path)
    if rows and set(IMPORTED_REQUIRED) - set(rows[0]):
        raise ValueError(f"{path} lacks columns {sorted(set(IMPORTED_REQUIRED) - set(rows[0]))}")
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        unit = (row["fov"], row["neuron_id"])
        if not row["category"].strip():
            raise ValueError(f"{path}: {unit} has no category")
        if unit in out:
            raise ValueError(f"{path}: {unit} is answered twice")
        out[unit] = row
    return out


def imported_digest(project_dir: Path, stain: str) -> str:
    """sha256 of imported/<stain>.csv, '' when there is none; unseal refuses a changed file."""
    path = imported_path(project_dir, stain)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def fov_channels(project_dir: Path) -> dict[str, dict[str, int]]:
    """Channel map per fov, from project.json entries that carry one (project.add channels=)."""
    path = Path(project_dir) / "project.json"
    entries = json.loads(path.read_text())["fovs"] if path.exists() else []
    return {e["fov"]: e["channels"] for e in entries if e.get("channels")}


def context_channels(project_dir: Path, fovs: list[str], stain: str) -> dict[str, dict[str, int]]:
    """Per fov, the SCORING_CONTEXT channels of this stain its channel map declares."""
    maps = fov_channels(project_dir)
    names = SCORING_CONTEXT.get(stain, [])
    return {f: {n: int(maps[f][n]) for n in names if n in maps.get(f, {})} for f in fovs}


def context_limits(volume: np.ndarray) -> np.ndarray:
    """Display limits of a context channel for one fov: CONTEXT_PCT of its positive voxels."""
    lo = census.percentile_positive(volume, CONTEXT_PCT[0])
    return np.array([lo, max(census.percentile_positive(volume, CONTEXT_PCT[1]), lo + 1.0)])


def add_context(project_dir: Path, stain: str) -> tuple[int, int]:
    """Add the SCORING_CONTEXT channels to unanswered crops that lack them, cut with the crop's own
    bounds and padding from its fov's source. Returns (crops filled, crops left: file not on disk)."""
    names = SCORING_CONTEXT.get(stain, [])
    if not names:
        return 0, 0
    crops = scoring_dir(project_dir, stain) / "crops"
    todo = set(unanswered(project_dir, stain))
    by_fov: dict[str, list[dict[str, str]]] = {}
    for row in io.read_rows(manifest_path(project_dir, stain)):
        if row["blind_id"] in todo:
            with np.load(crops / f"{row['blind_id']}.npz") as crop:
                if all(n in crop.files for n in names):
                    continue
            by_fov.setdefault(row["fov"], []).append(row)
    filled = left = 0
    for fov, rows in by_fov.items():
        ctx = context_channels(project_dir, [fov], stain)[fov]
        meta = census.read_meta(Path(project_dir) / "fovs" / fov)
        if not ctx or not Path(meta["source"]).exists():
            left += len(rows)
            continue
        data = io.read_fov(meta["source"], meta["series"], ctx).data
        limits = {n: context_limits(data[n]) for n in ctx}
        for row in rows:
            box = tuple(slice(int(row[f"crop_{a}0"]), int(row[f"crop_{a}1"])) for a in "zyx")
            pad = json.loads(row["pad_vox"])
            path = crops / f"{row['blind_id']}.npz"
            with np.load(path) as crop:
                arrays = {k: crop[k] for k in crop.files}
            for n in ctx:
                arrays[n] = np.pad(data[n][box], pad)
                arrays[f"{n}_limits"] = limits[n]
            tmp = path.with_name(path.stem + ".tmp.npz")
            np.savez_compressed(tmp, **arrays)
            tmp.replace(path)
            filled += 1
        del data
    return filled, left


def stain_channel_of(project_dir: Path, fov: str, stain: str, default: int | None) -> int:
    """This fov's index for the stain: its own channel map first, else the default given."""
    index = fov_channels(project_dir).get(fov, {}).get(stain, default)
    if index is None:
        raise ValueError(f"{fov} has no channel for {stain!r} in project.json and no default was "
                         f"given; a channel map is declared, never guessed")
    return int(index)


def not_scoreable(project_dir: Path, stain: str) -> set[tuple[str, str]]:
    """(fov, neuron_id) of every neuron a fully answered stain called not scoreable."""
    if not scoring_dir(project_dir, stain).exists():
        raise ValueError(f"stain {stain!r} is not built, so it cannot decide which neurons to skip")
    left = unanswered(project_dir, stain)
    if left:
        raise PermissionError(f"{len(left)} crop(s) of {stain!r} are unanswered; skip only "
                              f"from a stain that is fully answered")
    done = answered(project_dir, stain)
    out = {(u["fov"], u["neuron_id"]) for u in io.read_rows(manifest_path(project_dir, stain))
           if done[u["blind_id"]]["category"] == "not scoreable"}
    skipped = skipped_path(project_dir, stain)
    if skipped.exists():
        out |= {(r["fov"], r["neuron_id"]) for r in io.read_rows(skipped)}
    out |= {unit for unit, r in imported(project_dir, stain).items()
            if r["category"] == "not scoreable"}
    return out


def project_stain_limits(project_dir: Path, dirs: list[Path], stain: str,
                         stain_channel: int | None) -> list[float]:
    """Stain contrast pooled over every fov: p1 to p99.9 of their positive voxels together."""
    counts = np.zeros(0, dtype=np.int64)
    for fov_dir in dirs:
        meta = census.read_meta(fov_dir)
        index = stain_channel_of(project_dir, fov_dir.name, stain, stain_channel)
        fov = io.read_fov(meta["source"], meta["series"], {"stain": index})
        counts = census.add_counts(counts, census.positive_counts(fov.data["stain"]))
    return census.pooled_limits(counts, STAIN_PCT)


def build(project_dir: Path, stain: str, stain_channel: int | None, categories: list[str] | None,
          seed: int = SEED, crop_half_um: float = CROP_HALF_UM,
          skip_from: list[str] | None = None, contrast: str = "fov",
          only: list[str] | None = None, nuclei_dir: Path | None = None,
          masks_dir: Path | None = None) -> ScoreResult:
    """Cut one crop per neuron of every sealed fov (or of the fov ids in `only`), under fresh blind ids: the first round of
    this stain; fovs sealed later join with extend().
    No categories (None or []) means DEFAULT_CATEGORIES. A neuron answered in imported/<stain>.csv
    gets no crop and unseals with that answer; then a neuron any stain in skip_from called not
    scoreable gets no crop and unseals as 'not scoreable (from <first such stain>)'.
    stain_channel is the default index; a fov whose project.json entry carries channels uses its
    own. contrast="fov" sets stain limits per fov; "project" pools them over every fov."""
    project_dir = Path(project_dir)
    categories = categories or DEFAULT_CATEGORIES
    skip_from = skip_from or []
    if contrast not in census.CONTRASTS:
        raise ValueError(f"contrast must be one of {census.CONTRASTS}, got {contrast!r}")
    if stain in skip_from:
        raise ValueError(f"{stain!r} cannot skip from itself")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", stain):
        raise ValueError(f"stain name {stain!r} must be letters, digits, '_' or '-'")
    keys = all_categories(categories)
    final, manifest = scoring_dir(project_dir, stain), manifest_path(project_dir, stain)
    out_dir = final.with_name(final.name + ".partial")
    if out_dir.exists():  # an interrupted build: nothing in it was ever shown or answered
        shutil.rmtree(out_dir)
        for leftover in (manifest, manifest.with_suffix(".json"), skipped_path(project_dir, stain)):
            leftover.unlink(missing_ok=True)
    if final.exists() and not manifest.exists():
        raise FileExistsError(f"{final} has no manifest, so its crops cannot be joined to neurons; "
                              f"move it aside and build again")
    if final.exists() or manifest.exists():
        raise FileExistsError(f"scoring for {stain!r} is already built; call run() to resume")
    for other in manifest.parent.glob("*.json"):
        if json.loads(other.read_text())["seed"] == seed:
            raise ValueError(f"seed {seed} is already used by {other.stem}; each stain needs its "
                             f"own seed so its order owes nothing to another pass")
    dirs = fov_dirs(project_dir)
    if not any(census.is_sealed(d) for d in dirs):
        raise ValueError(f"{project_dir} has no sealed fov")
    if contrast == "project" and not all(census.is_sealed(d) for d in dirs):
        raise PermissionError("contrast='project' pools every fov; seal every census first")
    # Fovs counted later join through extend(), in a round of their own.
    sealed = {d: census.check_seal(d) for d in dirs if census.is_sealed(d)}
    sealed = _drop_excluded_neurons(sealed)
    known = {(n["fov"], n["neuron_id"]) for neurons in sealed.values() for n in neurons}
    if only is not None:
        sealed = {d: neurons for d, neurons in sealed.items() if d.name in only}
        if not sealed:
            raise ValueError(f"none of {only} is a sealed fov of {project_dir}")
    already = imported(project_dir, stain)
    stray = sorted(set(already) - known)
    if stray:
        raise ValueError(f"{imported_path(project_dir, stain)} answers {len(stray)} neuron(s) no "
                         f"sealed census holds, e.g. {stray[:3]}")
    skip: dict[tuple[str, str], str] = {}
    for earlier in skip_from:
        for unit in not_scoreable(project_dir, earlier):
            skip.setdefault(unit, earlier)
    skipped = [{**{k: n[k] for k in ("fov", "neuron_id", "z_um", "y_um", "x_um")},
                "category": f"not scoreable (from {skip[(n['fov'], n['neuron_id'])]})"}
               for neurons in sealed.values() for n in neurons
               if (n["fov"], n["neuron_id"]) in skip and (n["fov"], n["neuron_id"]) not in already]
    sealed = {d: [n for n in neurons if (n["fov"], n["neuron_id"]) not in skip
                  and (n["fov"], n["neuron_id"]) not in already]
              for d, neurons in sealed.items()}
    to_read = dirs if contrast == "project" else [d for d, neurons in sealed.items() if neurons]
    missing = sorted({census.read_meta(d)["source"] for d in to_read
                      if not Path(census.read_meta(d)["source"]).exists()})
    if missing:
        raise FileNotFoundError(f"{len(missing)} source file(s) needed for {stain!r} crops are not "
                                f"on disk: {missing}")
    index = {d.name: stain_channel_of(project_dir, d.name, stain, stain_channel) for d in to_read}
    pooled = (project_stain_limits(project_dir, dirs, stain, stain_channel)
              if contrast == "project" else None)
    rng = np.random.default_rng(seed)
    ids = iter(blind_ids(sum(len(rows) for rows in sealed.values()), rng))

    (out_dir / "crops").mkdir(parents=True)
    rows = cut_crops(out_dir, sealed, index, pooled, crop_half_um, ids,
                     context_channels(project_dir, [d.name for d in sealed], stain),
                     nuclei_dir, masks_dir)
    write_manifest(manifest, stain, rows, index, seed, crop_half_um, contrast)
    manifest.with_suffix(".json").write_text(json.dumps(
        {"stain": stain, "stain_channel": stain_channel, "stain_channels": index, "seed": seed,
         "categories": keys, "crop_half_um": crop_half_um, "fovs": [d.name for d in dirs],
         "skip_from": skip_from, "contrast": contrast,
         "imported": {"n": len(already), "sha256": imported_digest(project_dir, stain)},
         "rounds": [round_record(1, rows)]},
        indent=2))
    if skip_from:
        io.write_rows(skipped_path(project_dir, stain), SKIPPED_FIELDS, skipped, header=[
            f"stain {stain!r}: neurons given no crop; skip_from {skip_from}, first stain named."])
    (out_dir / "scoring.json").write_text(json.dumps(
        {"stain": stain, "categories": keys, "n_crops": len(rows),
         "channels": ["dapi", "neun", stain]}, indent=2))
    out_dir.rename(final)  # the build exists only from here: crops, manifest and settings at once
    return ScoreResult(stain, len(rows), 0)


def write_manifest(manifest: Path, stain: str, rows: list[dict], index: dict[str, int], seed: int,
                   crop_half_um: float, contrast: str) -> None:
    """The blind manifest, every round, sorted by blind id."""
    io.write_rows(manifest, MANIFEST_FIELDS, sorted(rows, key=lambda r: r["blind_id"]), header=[
        (f"BLIND MANIFEST for stain {stain!r}. Do not open until unseal(); it joins blind ids "
         f"to fov and neuron."),
        (f"channels per crop: dapi, neun, {stain} (index per fov {index}); seed {seed}; "
         f"crop +/-{crop_half_um} um in z, y and x, zero-padded past the fov edge (pad_vox); "
         f"stain contrast per {contrast}"),
        ("display is per fov (computed once over the whole fov) and lives here, not beside the "
         "answers, because identical limits would group crops by fov."),
    ])


def manifest_rounds(project_dir: Path, stain: str) -> list[dict]:
    """The rounds of crops this stain has had (round_record), oldest first."""
    meta = json.loads(manifest_path(project_dir, stain).with_suffix(".json").read_text())
    return meta.get("rounds", [])


def round_record(number: int, rows: list[dict], crop_half_um: float | None = None) -> dict:
    """What one round of crops added: its size, its fovs, its crop half-size and when."""
    return {"round": number, "n_crops": len(rows), "fovs": sorted({r["fov"] for r in rows}),
            **({"crop_half_um": crop_half_um} if crop_half_um is not None else {}),
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def extend(project_dir: Path, stain: str, only: list[str] | None = None,
           crop_half_um: float | None = None, nuclei_dir: Path | None = None,
           masks_dir: Path | None = None) -> ScoreResult:
    """Add a round of crops for sealed neurons this stain has not met yet (fovs counted after
    build), under fresh blind ids interleaved with any crops still open. Imported answers and
    skip_from apply as in build; a skip_from stain must be fully answered first. Settings
    (categories, contrast) are the ones build fixed; the crop half-size too unless `crop_half_um`
    is given, and then it is recorded in this round's record (rounds may differ: a later round
    can show more context). `only` limits the round to those fov ids. Returns the new crops."""
    project_dir = Path(project_dir)
    manifest = manifest_path(project_dir, stain)
    if not manifest.exists():
        raise FileNotFoundError(f"scoring for {stain!r} is not built; call build() first")
    meta = json.loads(manifest.with_suffix(".json").read_text())
    rows = io.read_rows(manifest)
    finish_round(project_dir, stain, {r["blind_id"] for r in rows})
    met = {(r["fov"], r["neuron_id"]) for r in rows}
    skipped_rows = io.read_rows(skipped_path(project_dir, stain))
    met |= {(r["fov"], r["neuron_id"]) for r in skipped_rows}
    already = imported(project_dir, stain)
    dirs = [d for d in fov_dirs(project_dir) if census.is_sealed(d)
            and (only is None or d.name in only)]
    sealed = {d: [n for n in census.check_seal(d) if (n["fov"], n["neuron_id"]) not in met
                  and (n["fov"], n["neuron_id"]) not in already] for d in dirs}
    sealed = _drop_excluded_neurons(sealed)
    sealed = {d: neurons for d, neurons in sealed.items() if neurons}
    if not sealed:
        return ScoreResult(stain, 0, 0)
    if meta.get("contrast", "fov") == "project":
        raise ValueError(f"{stain!r} pools stain contrast over the fovs it was built on; a new "
                         f"round would be shown on other limits. Start a new project instead.")
    skip: dict[tuple[str, str], str] = {}
    for earlier in meta.get("skip_from", []):
        for unit in not_scoreable(project_dir, earlier):
            skip.setdefault(unit, earlier)
    new_skipped = [{**{k: n[k] for k in ("fov", "neuron_id", "z_um", "y_um", "x_um")},
                    "category": f"not scoreable (from {skip[(n['fov'], n['neuron_id'])]})"}
                   for neurons in sealed.values() for n in neurons
                   if (n["fov"], n["neuron_id"]) in skip]
    sealed = {d: [n for n in neurons if (n["fov"], n["neuron_id"]) not in skip]
              for d, neurons in sealed.items()}
    sealed = {d: neurons for d, neurons in sealed.items() if neurons}
    missing = sorted({census.read_meta(d)["source"] for d in sealed
                      if not Path(census.read_meta(d)["source"]).exists()})
    if missing:
        raise FileNotFoundError(f"{len(missing)} source file(s) needed for {stain!r} crops are not "
                                f"on disk: {missing}")
    index = {d.name: stain_channel_of(project_dir, d.name, stain, meta["stain_channel"])
             for d in sealed}
    number = len(meta.get("rounds", [{}])) + 1
    rng = np.random.default_rng([meta["seed"], number])
    taken = {r["blind_id"] for r in rows}
    fresh = [b for b in blind_ids(sum(map(len, sealed.values())) + len(taken), rng)
             if b not in taken][:sum(map(len, sealed.values()))]
    incoming = scoring_dir(project_dir, stain) / "incoming"
    (incoming / "crops").mkdir(parents=True)
    half_um = float(crop_half_um) if crop_half_um is not None else float(meta["crop_half_um"])
    new = cut_crops(incoming, sealed, index, None, half_um, iter(fresh),
                    context_channels(project_dir, [d.name for d in sealed], stain),
                    nuclei_dir, masks_dir)
    meta["stain_channels"] = {**meta.get("stain_channels", {}), **index}
    meta["fovs"] = [d.name for d in fov_dirs(project_dir)]
    meta["rounds"] = meta.get("rounds", [{"round": 1, "n_crops": len(rows)}]) + [
        round_record(number, new, half_um)]
    write_manifest(manifest, stain, rows + new, meta["stain_channels"], meta["seed"],
                   meta["crop_half_um"], meta.get("contrast", "fov"))
    manifest.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    if new_skipped:
        io.write_rows(skipped_path(project_dir, stain), SKIPPED_FIELDS, skipped_rows + new_skipped,
                      header=[(f"stain {stain!r}: neurons given no crop; skip_from "
                               f"{meta['skip_from']}, first stain named.")])
    finish_round(project_dir, stain, {r["blind_id"] for r in rows + new})
    scoring = settings(project_dir, stain)
    scoring["n_crops"] = len(queue(project_dir, stain))
    (scoring_dir(project_dir, stain) / "scoring.json").write_text(json.dumps(scoring, indent=2))
    return ScoreResult(stain, len(new), 0)


def finish_round(project_dir: Path, stain: str, in_manifest: set[str]) -> None:
    """Move a round's crops from incoming/ into the queue once the manifest lists them; a round
    cut but never written to the manifest (interrupted) is discarded, as nobody has seen it."""
    incoming = scoring_dir(project_dir, stain) / "incoming"
    if not incoming.exists():
        return
    crops = sorted((incoming / "crops").glob("*.npz"))
    if crops and {c.stem for c in crops} <= in_manifest:
        for crop in crops:
            crop.replace(scoring_dir(project_dir, stain) / "crops" / crop.name)
    shutil.rmtree(incoming)
    now = datetime.datetime.now(datetime.timezone.utc).timestamp()
    for crop in (scoring_dir(project_dir, stain) / "crops").iterdir():
        os.utime(crop, (now, now))


def cut_crops(out_dir: Path, sealed: dict[Path, list[dict[str, str]]], index: dict[str, int],
              pooled: list[float] | None, crop_half_um: float, ids,
              context: dict[str, dict[str, int]] | None = None,
              nuclei_dir: Path | None = None,
              masks_dir: Path | None = None) -> list[dict]:
    """One zero-padded crop per neuron under the next blind id; manifest rows in return."""
    rows = []
    for fov_dir, neurons in sealed.items():
        if not neurons:
            continue
        meta = census.read_meta(fov_dir)
        ctx = (context or {}).get(fov_dir.name, {})
        channels = {**meta["channels_loaded"], "stain": index[fov_dir.name], **ctx}
        fov = io.read_fov(meta["source"], meta["series"], channels)
        extra = {n: context_limits(fov.data[n]) for n in ctx}
        spacing = np.array(fov.spacing_um)
        half = (round(crop_half_um / spacing[0]), round(crop_half_um / spacing[1]),
                round(crop_half_um / spacing[2]))
        display = {name: {"limits": s["limits"], "gamma": s["gamma"]}
                   for name, s in meta["display"].items()}
        display["stain"] = {"limits": pooled or stain_limits(fov.data["stain"]), "gamma": 1.0}
        nuclei_vol = census.load_nuclei(fov_dir, nuclei_dir, masks_dir)
        soma_vol = census.load_soma(fov_dir, masks_dir)
        for neuron in neurons:
            blind_id = next(ids)
            centre = (int(neuron["z_vox"]), int(neuron["y_vox"]), int(neuron["x_vox"]))
            z0, z1, y0, y1, x0, x1 = crop_bounds(centre, half, fov.shape)
            box = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
            # Zero-padded to the full 2 * half + 1 box, so a crop's size never shows a fov edge.
            pad = [(h - (c - lo), (c + h + 1) - hi)
                   for c, h, lo, hi in zip(centre, half, (z0, y0, x0), (z1, y1, x1))]
            mask_arrays = {}
            nuc_id = int(neuron.get("nucleus_id") or 0)
            if nuclei_vol is not None and nuc_id > 0:
                mask_arrays["nucleus"] = np.pad(
                    (nuclei_vol[box] == nuc_id).astype(np.uint8), pad)
            if soma_vol is not None:
                # find the soma instance at the click point
                zv, yv, xv = centre
                soma_id = int(soma_vol[min(zv, soma_vol.shape[0] - 1),
                                       min(yv, soma_vol.shape[1] - 1),
                                       min(xv, soma_vol.shape[2] - 1)])
                if soma_id > 0:
                    mask_arrays["soma"] = np.pad(
                        (soma_vol[box] == soma_id).astype(np.uint8), pad)
            np.savez_compressed(out_dir / "crops" / f"{blind_id}.npz",
                                dapi=np.pad(fov.data["dapi"][box], pad),
                                neun=np.pad(fov.data["neun"][box], pad),
                                stain=np.pad(fov.data["stain"][box], pad),
                                **{n: np.pad(fov.data[n][box], pad) for n in ctx},
                                **{f"{n}_limits": lim for n, lim in extra.items()},
                                **mask_arrays)
            target = np.array(half) * spacing
            rows.append({"blind_id": blind_id, "fov": neuron["fov"],
                         "neuron_id": neuron["neuron_id"], "z_um": neuron["z_um"],
                         "y_um": neuron["y_um"], "x_um": neuron["x_um"],
                         "crop_z0": z0, "crop_z1": z1, "crop_y0": y0, "crop_y1": y1,
                         "crop_x0": x0, "crop_x1": x1,
                         "target_z_um": f"{target[0]:.4f}", "target_y_um": f"{target[1]:.4f}",
                         "target_x_um": f"{target[2]:.4f}",
                         "spacing_um": json.dumps(list(fov.spacing_um)),
                         "display": json.dumps(display), "pad_vox": json.dumps(pad)})
        del fov
    # Crops were written fov by fov; one shared mtime keeps `ls -t` from regrouping them.
    now = datetime.datetime.now(datetime.timezone.utc).timestamp()
    for crop in (out_dir / "crops").iterdir():
        os.utime(crop, (now, now))

    return rows


def settings(project_dir: Path, stain: str) -> dict:
    """The scoring.json written by build()."""
    return json.loads((scoring_dir(project_dir, stain) / "scoring.json").read_text())


def answered(project_dir: Path, stain: str) -> dict[str, dict[str, str]]:
    """Answers on disk, by blind id; a crop answered again keeps its latest row."""
    rows = io.read_rows(scoring_dir(project_dir, stain) / "answers.csv")
    return {r["blind_id"]: r for r in rows}


def queue(project_dir: Path, stain: str) -> list[str]:
    """Every blind id in presentation order; ids are random, so sorting them is a shuffle."""
    return sorted(p.stem for p in (scoring_dir(project_dir, stain) / "crops").glob("*.npz"))


def unanswered(project_dir: Path, stain: str) -> list[str]:
    """Blind ids still waiting for an answer, in presentation order."""
    done = answered(project_dir, stain)
    return [b for b in queue(project_dir, stain) if b not in done]


def answer_fields(path: Path) -> list[str]:
    """Columns of an answers.csv: its own header when it exists (files before 0.4 have no blind)."""
    if path.exists():
        with path.open(newline="") as handle:
            for line in handle:
                if not line.startswith("#"):
                    return next(csv.reader([line]))
    return ANSWER_FIELDS


def answer(project_dir: Path, stain: str, blind_id: str, category: str, note: str = "",
           again: bool = False, blind: bool = True) -> None:
    """Record one answer and flush it to disk before anything else happens. again=True answers a
    crop once more (the window's back key); the latest row wins and earlier rows stay on disk.
    blind=False records that the rater saw the image name; a pre-0.4 file, blind only, refuses it."""
    keys = settings(project_dir, stain)["categories"]
    if category not in keys:
        raise ValueError(f"{category!r} is not one of {keys}")
    if blind_id not in queue(project_dir, stain):
        raise KeyError(f"{blind_id} is not a crop of stain {stain!r}")
    if blind_id in answered(project_dir, stain) and not again:
        raise PermissionError(f"{blind_id} is already answered; answers are not overwritten")
    path = scoring_dir(project_dir, stain) / "answers.csv"
    fields = answer_fields(path)
    if "blind" not in fields and not blind:
        raise PermissionError(f"{path} was started blind and has no blind column; score this "
                              f"stain blind, or start a new project for open scoring")
    row = {"blind_id": blind_id, "category": category, "note": " ".join(note.split()),
           "answered_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "blind": "yes" if blind else "no"}
    io.append_row(path, fields, {k: row[k] for k in fields},
                  header=[f"ANSWERS, stain {stain!r}. channels shown: dapi, neun, {stain}.",
                          f"categories: {keys}",
                          "blind_id only; fov, neuron and display limits are joined at unseal().",
                          "blind: yes = shuffled, no image name; no = in order, image name shown."])


def unseal(project_dir: Path, stain: str) -> Path:
    """Join answers back to fov and neuron; refused while any crop is unanswered."""
    from digital_neuropath import __version__

    left = unanswered(project_dir, stain)
    if left:
        raise PermissionError(f"{len(left)} crop(s) of {stain!r} are unanswered; unsealing now "
                              f"would reveal fovs while calls are still to be made")
    project_dir = Path(project_dir)
    manifest = manifest_path(project_dir, stain)
    meta = json.loads(manifest.with_suffix(".json").read_text())
    neurons = {(n["fov"], n["neuron_id"]): n for fov_dir in fov_dirs(project_dir)
               if census.is_sealed(fov_dir) for n in census.check_seal(fov_dir)}
    if imported_digest(project_dir, stain) != meta.get("imported", {}).get("sha256", ""):
        raise PermissionError(f"{imported_path(project_dir, stain)} changed after build(); the "
                              f"crops were cut around the answers it held then")
    done = answered(project_dir, stain)
    fields = ["fov", "neuron_id", "z_um", "y_um", "x_um", "blind_id", "category", "note",
              "answered_utc", "display", "source", "blind"]
    rows = []
    for unit in io.read_rows(manifest):
        reply = done[unit["blind_id"]]
        with np.load(scoring_dir(project_dir, stain) / "crops" / f"{unit['blind_id']}.npz") as crop:
            qc = cobright(crop, json.loads(unit["pad_vox"]))
        rows.append({**{k: unit[k] for k in ("fov", "neuron_id", "z_um", "y_um", "x_um",
                                             "blind_id", "display")},
                     "category": reply["category"], "note": reply["note"],
                     "answered_utc": reply["answered_utc"], "source": "this project",
                     "blind": reply.get("blind") or "yes", **{f"cobright_{n}": f"{v:.4f}" for n, v in qc.items()}})
    fields += sorted({k for r in rows for k in r if k.startswith("cobright_")})
    skipped = skipped_path(project_dir, stain)
    if skipped.exists():
        rows += [{**r, "blind_id": "", "note": "", "answered_utc": "", "display": "",
                  "source": "skip_from", "blind": ""}
                 for r in io.read_rows(skipped)]
    for unit, r in imported(project_dir, stain).items():
        n = neurons[unit]
        source = " ".join(r.get(k, "") for k in ("source", "reading") if r.get(k, ""))
        rows.append({**{k: n[k] for k in ("fov", "neuron_id", "z_um", "y_um", "x_um")},
                     "blind_id": "", "category": r["category"], "note": r["note"],
                     "answered_utc": "", "display": "",
                     "source": f"imported: {source}" if source else "imported",
                     "blind": r.get("blind", "")})
    rows.sort(key=lambda r: (r["fov"], r["neuron_id"]))
    out = project_dir / "results" / f"{stain}_unsealed.csv"
    io.write_rows(out, fields, rows, header=[
        (f"digital_neuropath {__version__}; written "
         f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}; "
         f"contrast {meta.get('contrast', 'fov')}"),
        f"UNSEALED stain {stain!r}: one row per sealed neuron, every crop answered.",
        (f"categories: {meta['categories']}; channels shown: dapi, neun, {stain} "
         f"(index per fov {meta.get('stain_channels', meta['stain_channel'])}); "
         f"seed {meta['seed']}; crop +/-{meta['crop_half_um']} um; "
         f"skip_from {meta.get('skip_from', [])}; imported {meta.get('imported', {}).get('n', 0)} "
         f"(source column)"),
        (f"cobright_<channel> (QC, never shown to the rater): fraction of the stain's top "
         f"{COBRIGHT_STAIN:.1%} voxels of the crop (padding excluded) that are in the top "
         f"{COBRIGHT_OTHER:.0%} of that channel; chance = {COBRIGHT_OTHER:g}"),
    ])
    return out


def top_mask(values: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """The top `fraction` of values (at least one) as a mask, ties at the cut drawn at random."""
    k = max(1, int(np.ceil(fraction * values.size)))
    cut = np.partition(values, values.size - k)[values.size - k]
    mask = values > cut
    ties = np.flatnonzero(values == cut)
    mask[rng.choice(ties, k - int(mask.sum()), replace=False)] = True
    return mask


def cobright(crop, pad: list[list[int]]) -> dict[str, float]:
    """Per channel besides the stain, the fraction of the stain's top COBRIGHT_STAIN voxels that
    are in that channel's top COBRIGHT_OTHER, over the crop without its zero padding."""
    rng = np.random.default_rng(SEED)
    inner = tuple(slice(lo, n - hi) for (lo, hi), n in zip(pad, crop["stain"].shape))
    bright = top_mask(crop["stain"][inner].ravel(), COBRIGHT_STAIN, rng)
    return {name: float(top_mask(crop[name][inner].ravel(), COBRIGHT_OTHER, rng)[bright].mean())
            for name in sorted(crop) if name != "stain" and not name.endswith("_limits")}


def run(project_dir: Path, stain: str, screenshot: Path | None = None,
        target_size_um: float = TARGET_SIZE_UM,
        session_seed: int | None = None, blind: bool = True) -> ScoreResult:
    """Score (or resume) every unanswered crop of one stain in napari. blind: a fresh shuffle
    every session, blind ids only; not blind: image by image, neuron by neuron, names shown."""
    todo = unanswered(project_dir, stain)
    if todo:
        manifest = {r["blind_id"]: r for r in io.read_rows(manifest_path(project_dir, stain))}
        if blind:
            seed = secrets.randbits(32) if session_seed is None else session_seed
            todo = session_shuffle(todo, seed)
        else:
            todo.sort(key=lambda b: (manifest[b]["fov"], manifest[b]["neuron_id"]))
        open_viewer(project_dir, stain, todo, manifest, screenshot, target_size_um, blind)
    n = len(queue(project_dir, stain))
    return ScoreResult(stain, n, n - len(unanswered(project_dir, stain)))


def session_shuffle(todo: list[str], seed: int) -> list[str]:
    """Shuffle the presentation order for one scoring session, leaving the on-disk queue unchanged."""
    rng = np.random.default_rng(seed)
    order = list(todo)
    rng.shuffle(order)
    return order


def view_arrays(crop, mip: bool) -> dict[str, np.ndarray]:
    """The crop's arrays for one view: as cut (planes), or each channel's max over z (MIP)."""
    return {k: crop[k].max(axis=0) if mip and not k.endswith("_limits") else crop[k] for k in crop}


def add_channels(viewer, crop, stain: str, spacing: list[float], display: dict) -> None:
    """Add DAPI, NeuN and the stain to `viewer`: visibility from display or SCORING_DISPLAY
    defaults, limits and gamma from display, additive blending at SCORING_WEIGHTS opacity."""
    for name, (colormap, default_visible) in SCORING_DISPLAY.items():
        viewer.add_image(crop[name], name=stain.upper() if name == "stain" else name.upper(),
                         scale=spacing, colormap=colormap, blending="additive",
                         visible=display[name].get("visible", default_visible),
                         contrast_limits=display[name]["limits"],
                         gamma=display[name]["gamma"], opacity=SCORING_WEIGHTS.get(name, 1.0))
    for name in SCORING_CONTEXT.get(stain, []):
        if name in crop:
            look = display.get(name, {"limits": list(crop[f"{name}_limits"]), "gamma": 1.0})
            viewer.add_image(crop[name], name=name.upper(), scale=spacing, colormap=CONTEXT_COLOUR,
                             blending="additive", visible=look.get("visible", True),
                             contrast_limits=look["limits"],
                             gamma=look["gamma"], opacity=SCORING_WEIGHTS.get(name, 1.0))


def add_marks(viewer, unit: dict[str, str], others: np.ndarray, mip: bool,
              target_size_um: float) -> None:
    """The orange cross on this neuron, and small hollow grey points on the other sealed census
    marks inside the crop; in MIP both drop z."""
    import napari.layers

    keep = slice(1, None) if mip else slice(None)
    if len(others):
        grey = napari.layers.Points(others[:, keep], name="other neurons", ndim=2 if mip else 3,
                                    size=census.points_size(OTHER_MARK_SIZE_UM),
                                    face_color="transparent", border_color=OTHER_MARK_COLOUR,
                                    border_width=0.15, projection_mode="all")
        viewer.add_layer(grey)
        grey.editable = False
    target = np.array([[float(unit[f"target_{a}_um"]) for a in "zyx"]])
    marker = napari.layers.Points(target[:, keep], name="this neuron", symbol="cross",
                                  ndim=2 if mip else 3, size=census.points_size(target_size_um),
                                  face_color=TARGET_COLOUR, border_color=TARGET_COLOUR,
                                  projection_mode="all")
    viewer.add_layer(marker)
    marker.editable = False


def add_outlines(viewer, arrays: dict, spacing: list[float], mip: bool,
                 visible: bool) -> None:
    """Add nucleus and soma outline layers from the crop's mask arrays."""
    from skimage.segmentation import find_boundaries

    for name, colour in (("nucleus", "cyan"), ("soma", "yellow")):
        if name not in arrays or not np.asarray(arrays[name]).any():
            continue
        mask = np.asarray(arrays[name])
        if mip:
            mask = mask.max(axis=0) if mask.ndim == 3 else mask
        outline = find_boundaries(mask, mode="inner").astype(int)
        viewer.add_labels(outline, name=name, scale=spacing, opacity=0.6, visible=visible)


def other_marks(project_dir: Path, unit: dict[str, str]) -> np.ndarray:
    """(z, y, x) in the crop's um of every other sealed census mark of its fov inside the crop."""
    spacing = np.array(json.loads(unit["spacing_um"]))
    lo = np.array([int(unit[f"crop_{a}0"]) for a in "zyx"])
    hi = np.array([int(unit[f"crop_{a}1"]) for a in "zyx"])
    pad_lo = np.array([p[0] for p in json.loads(unit["pad_vox"])])
    vox = np.array([[int(r[f"{a}_vox"]) for a in "zyx"]
                    for r in census.check_seal(Path(project_dir) / "fovs" / unit["fov"])
                    if r["neuron_id"] != unit["neuron_id"]], int).reshape(-1, 3)
    inside = np.all((vox >= lo) & (vox < hi), axis=1)
    return (vox[inside] - lo + pad_lo) * spacing


def widen(viewer) -> None:
    """Let each integer image's slider span its full detector range, keeping its limits."""
    for layer in viewer.layers:
        dtype = np.asarray(layer.data).dtype
        if hasattr(layer, "contrast_limits") and dtype.kind in "ui":
            limits = list(layer.contrast_limits)
            layer.contrast_limits_range = [0.0, float(np.iinfo(dtype).max)]
            layer.contrast_limits = limits


def display_path(project_dir: Path, stain: str) -> Path:
    """The look set with tune_display: limits and gamma per channel, per view."""
    return scoring_dir(project_dir, stain) / "display.json"


def saved_display(project_dir: Path, stain: str) -> dict | None:
    """display.json, or None when the look was never tuned."""
    path = display_path(project_dir, stain)
    return json.loads(path.read_text()) if path.exists() else None


def view_display(display: dict, saved: dict | None, view: str) -> dict:
    """A crop's display (per fov, from the manifest) with display.json's look for this view laid
    over it, so every crop of every fov and case is drawn the same."""
    return {**display, **saved["views"][view]} if saved else display


def save_display(project_dir: Path, stain: str, looks: dict, sample: dict) -> Path:
    """Write display.json: the look per view and channel, when it was set and how its boxes were
    drawn (n, seed, rule; no fov or case)."""
    path = display_path(project_dir, stain)
    path.write_text(json.dumps(
        {"stain": stain, "saved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"), "sample": sample, "views": looks}, indent=2))
    return path


def readable(source: Path) -> bool:
    """True when the source file is on disk and not a cloud placeholder."""
    try:
        io.check_on_disk(source)
    except OSError:
        return False
    return True


def tune_points(project_dir: Path, stain: str, n: int,
                seed: int) -> list[tuple[str, tuple[int, int, int]]]:
    """n (fov, centre voxel) pairs drawn round robin over cases (the fov when it has none) and then
    their fovs, each centre at least census.NEURON_DIAMETER_UM from every census mark of its fov;
    only sealed fovs this stain has crops of, whose source is on disk."""
    project_dir = Path(project_dir)
    path = project_dir / "project.json"
    entries = {e["fov"]: e for e in json.loads(path.read_text())["fovs"]} if path.exists() else {}
    built = {r["fov"] for r in io.read_rows(manifest_path(project_dir, stain))}
    by_case: dict[str, list[str]] = {}
    for fov_dir in fov_dirs(project_dir):
        if (fov_dir.name in built and census.is_sealed(fov_dir)
                and readable(Path(census.read_meta(fov_dir)["source"]))):
            case = str(entries.get(fov_dir.name, {}).get("case") or fov_dir.name)
            by_case.setdefault(case, []).append(fov_dir.name)
    rng = np.random.default_rng(seed)
    order = [sorted(by_case[c]) for c in sorted(by_case)]
    cases = [[str(f) for f in rng.permutation(order[i])] for i in rng.permutation(len(order))]
    out: list[tuple[str, tuple[int, int, int]]] = []
    for k in range(n * sum(map(len, cases))):
        if len(out) == n:
            break
        case = cases[k % len(cases)]
        fov = case[(k // len(cases)) % len(case)]
        meta = census.read_meta(project_dir / "fovs" / fov)
        spacing = np.array(meta["spacing_um"])
        marks_um = census.marks(project_dir / "fovs" / fov)
        for _ in range(TUNE_TRIES):
            centre = rng.integers(0, meta["shape_zyx"])
            gap = np.linalg.norm(marks_um - centre * spacing, axis=1)
            if not gap.size or gap.min() >= census.NEURON_DIAMETER_UM:
                out.append((fov, (int(centre[0]), int(centre[1]), int(centre[2]))))
                break
    return out


def tune_crops(project_dir: Path, stain: str,
               points: list[tuple[str, tuple[int, int, int]]]) -> list[dict]:
    """One box per point, cut like a scoring crop (the build's half-size, zero padding), all
    channels shown as the normal image.
    Each item: arrays (channel -> array, context limits too), spacing_um, display of its fov."""
    project_dir = Path(project_dir)
    manifest = manifest_path(project_dir, stain)
    built = json.loads(manifest.with_suffix(".json").read_text())
    display_of = {r["fov"]: json.loads(r["display"]) for r in io.read_rows(manifest)}
    out: list[dict] = [{} for _ in points]
    for fov in dict.fromkeys(f for f, _ in points):
        fov_dir = project_dir / "fovs" / fov
        meta = census.read_meta(fov_dir)
        ctx = context_channels(project_dir, [fov], stain)[fov]
        channels = {**meta["channels_loaded"], "stain": int(built["stain_channels"][fov]), **ctx}
        data = io.read_fov(meta["source"], meta["series"], channels)
        spacing = np.array(data.spacing_um)
        half_um = float(built["crop_half_um"])
        half = (round(half_um / spacing[0]), round(half_um / spacing[1]),
                round(half_um / spacing[2]))
        limits = {f"{name}_limits": context_limits(data.data[name]) for name in ctx}
        for i, (where, centre) in enumerate(points):
            if where != fov:
                continue
            z0, z1, y0, y1, x0, x1 = crop_bounds(centre, half, data.shape)
            box = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
            pad = [(h - (c - lo), (c + h + 1) - hi)
                   for c, h, lo, hi in zip(centre, half, (z0, y0, x0), (z1, y1, x1))]
            arrays = {name: np.pad(data.data[name][box], pad) for name in channels}
            out[i] = {"arrays": {**arrays, **limits}, "spacing_um": list(data.spacing_um),
                      "display": display_of[fov]}
        del data
    return out


def tune_display(project_dir: Path, stain: str, n: int = TUNE_SAMPLE, seed: int = SEED,
                 screenshot: Path | None = None) -> Path:
    """Set the scoring look on n boxes: per channel, limits, gamma and visibility for planes and
    MIP views. Box 1 opens first; more (f/b) optional. Saved = mean of limits/gamma over tuned
    boxes, visibility from the last tuned box. View only: crops, manifest and answers untouched."""
    import napari
    import napari.layers

    project_dir = Path(project_dir)
    points = tune_points(project_dir, stain, n, seed)
    if not points:
        raise ValueError(f"{stain!r} has no built fov with its source on disk to tune on")
    sample = tune_crops(project_dir, stain, points)
    rule = {"n": len(sample), "seed": seed,
            "rule": (f"random points >= {census.NEURON_DIAMETER_UM:g} um from every census mark, "
                     f"round robin over cases and fovs")}
    names = ("dapi", "neun", *SCORING_CONTEXT.get(stain, []))
    channel_of = {stain.upper(): "stain", **{c.upper(): c for c in names}}
    viewer = napari.Viewer(title=f"DISPLAY {stain}", ndisplay=2)
    state = {"i": 0, "view": VIEWS[0]}
    box_state: list[dict[str, dict[str, dict]]] = [{} for _ in sample]

    def _default_visible(ch: str) -> bool:
        if ch in dict(SCORING_DISPLAY):
            return SCORING_DISPLAY[ch][1]
        return True

    def _is_box_tuned(i: int) -> bool:
        fov_display = sample[i]["display"]
        for v in VIEWS:
            for ch, s in box_state[i].get(v, {}).items():
                d = fov_display.get(ch, {})
                if (s["limits"] != d.get("limits")
                        or abs(s["gamma"] - d.get("gamma", 1.0)) > 1e-6
                        or s["visible"] != _default_visible(ch)):
                    return True
        return False

    def harvest() -> None:
        """Capture the current box+view's layer state."""
        v = state["view"]
        box_state[state["i"]][v] = {}
        for layer in viewer.layers:
            if isinstance(layer, napari.layers.Image) and layer.name in channel_of:
                box_state[state["i"]][v][channel_of[layer.name]] = {
                    "limits": [float(x or 0.0) for x in layer.contrast_limits],
                    "gamma": float(layer.gamma or 1.0),
                    "visible": bool(layer.visible)}

    def _compute_and_save() -> bool:
        """Mean limits/gamma over tuned boxes, visibility from last; False if nothing tuned."""
        tuned = [i for i in range(len(sample)) if _is_box_tuned(i)]
        if not tuned:
            print(f"  {stain}: no boxes tuned, nothing saved")
            return False
        result: dict[str, dict] = {}
        for v in VIEWS:
            result[v] = {}
            in_view = [i for i in tuned if box_state[i].get(v)]
            if not in_view:
                continue
            all_ch: set[str] = set()
            for i in in_view:
                all_ch |= set(box_state[i][v])
            for ch in sorted(all_ch):
                entries = [box_state[i][v][ch] for i in in_view if ch in box_state[i][v]]
                result[v][ch] = {
                    "limits": [round(float(np.mean([e["limits"][0] for e in entries])), 2),
                               round(float(np.mean([e["limits"][1] for e in entries])), 2)],
                    "gamma": round(float(np.mean([e["gamma"] for e in entries])), 4),
                    "visible": entries[-1]["visible"]}
        save_display(project_dir, stain, result, rule)
        label = f"{len(tuned)} box{'es' if len(tuned) != 1 else ''}"
        print(f"  {stain}: display saved ({label})")
        return True

    def show() -> None:
        """Draw the current box in the current view."""
        item = sample[state["i"]]
        mip = state["view"] == "mip"
        arrays = view_arrays(item["arrays"], mip)
        spacing = item["spacing_um"]
        captured = box_state[state["i"]].get(state["view"], {})
        viewer.layers.clear()
        add_channels(viewer, arrays, stain, spacing[1:] if mip else spacing,
                     {**item["display"], **captured})
        widen(viewer)
        if not mip:
            viewer.dims.set_point(0, item["arrays"]["stain"].shape[0] // 2 * spacing[0])
        for layer in viewer.layers:
            bind(layer)
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = f"box {state['i'] + 1}/{len(sample)}   {state['view']}"

    def step(k: int) -> None:
        harvest()
        state["i"] = (state["i"] + k) % len(sample)
        show()

    def switch() -> None:
        harvest()
        state["view"] = VIEWS[1 - VIEWS.index(state["view"])]
        show()

    def save() -> None:
        harvest()
        if _compute_and_save():
            viewer.status = "saved display.json"
        else:
            viewer.status = "no boxes tuned"

    actions: dict[str, Callable[[], None]] = {"f": lambda: step(1), "b": lambda: step(-1),
                                              "m": switch, "s": save}

    def bind(target) -> None:
        for key, action in actions.items():
            target.bind_key(key, lambda _obj, action=action: action(), overwrite=True)

    header = f"\u2500\u2500 CONTRAST  {stain} \u2500\u2500 {len(sample)} boxes "
    bar = "\u2500" * max(2, 50 - len(header))
    print(f"\n{header}{bar}")
    print("  Show/hide channels with the eye icons, set contrast and gamma per channel.")
    print("  f = next box \u00b7 b = back \u00b7 m = MIP/planes \u00b7 s = save and continue.")
    print("  Tune box 1; more boxes optional (f = next).")
    print("  Saved = mean over the boxes you tuned; none tuned = defaults.\n")

    bind(viewer)
    show()
    census.show_until_closed(viewer, screenshot)
    harvest()
    _compute_and_save()
    return display_path(project_dir, stain)


def pending_path(project_dir: Path, stain: str) -> Path:
    """Notes typed on crops before their answer, by blind id only."""
    return scoring_dir(project_dir, stain) / "pending_notes.csv"


def pending_notes(project_dir: Path, stain: str) -> dict[str, str]:
    """The pending note of each crop that has one; the latest row wins."""
    return {r["blind_id"]: r["note"] for r in io.read_rows(pending_path(project_dir, stain))}


class Walk:
    """The scoring window's place in the queue, without napari: answer, skip, back and notes. A
    note stays with its crop, across skips and runs, until the crop is answered."""

    def __init__(self, project_dir: Path, stain: str, todo: list[str], blind: bool = True) -> None:
        self.project_dir, self.stain, self.todo = Path(project_dir), stain, list(todo)
        self.blind = blind
        self.i = 0
        self.notes = pending_notes(project_dir, stain)

    @property
    def current(self) -> str | None:
        """The blind id on screen, None once the queue is done."""
        return self.todo[self.i] if self.i < len(self.todo) else None

    @property
    def note(self) -> str:
        """The note that goes with the next answer for the crop on screen."""
        return self.notes.get(self.current or "", "")

    def previous(self) -> str:
        """The category already given to the crop on screen ('' when unanswered)."""
        reply = answered(self.project_dir, self.stain).get(self.current or "", {})
        return reply.get("category", "")

    def set_note(self, text: str) -> None:
        """Attach a one-line note to the crop on screen and keep it on disk until answered."""
        blind_id = self.current
        if blind_id is None:
            return
        self.notes[blind_id] = " ".join(text.split())
        io.append_row(pending_path(self.project_dir, self.stain), PENDING_FIELDS,
                      {"blind_id": blind_id, "note": self.notes[blind_id],
                       "noted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()},
                      header=[(f"PENDING NOTES, stain {self.stain!r}: typed before the crop's "
                               f"answer and saved with it; latest row wins. Not answers.")])

    def pick(self, category: str) -> None:
        """Answer the crop on screen (again, when it was answered in this run), then move on."""
        blind_id = self.current
        if blind_id is None:
            return
        answer(self.project_dir, self.stain, blind_id, category, self.note,
               again=bool(self.previous()), blind=self.blind)
        self.i += 1

    def skip(self) -> None:
        """Move on, leaving the crop on screen as it is."""
        if self.current is not None:
            self.i += 1

    def back(self) -> None:
        """Return to the crop before, to answer it again (the latest answer wins)."""
        self.i = max(self.i - 1, 0)

    def left(self) -> int:
        """Crops of this run still unanswered."""
        done = answered(self.project_dir, self.stain)
        return sum(b not in done for b in self.todo)


def open_viewer(project_dir: Path, stain: str, todo: list[str], manifest: dict,
                screenshot: Path | None = None, target_size_um: float = TARGET_SIZE_UM,
                blind: bool = True) -> None:
    """One napari window stepping through crops: 1..k answer, n adds a note, s skips, b goes back
    one crop, m switches planes <-> MIP. Contrast, gamma or visibility changed on a crop carries to
    every later crop (per view) and is saved to display.json, which the next session starts from."""
    import napari
    import napari.layers
    from qtpy.QtWidgets import QInputDialog

    keys = settings(project_dir, stain)["categories"]
    total = len(queue(project_dir, stain))
    saved = saved_display(project_dir, stain)

    header = f"\u2500\u2500 SCORING  {stain} \u2500\u2500 {len(todo)} crops left "
    bar = "\u2500" * max(2, 50 - len(header))
    print(f"\n{header}{bar}")
    print("  " + " \u00b7 ".join(f"{i} = {c}" for i, c in enumerate(keys, 1)))
    print("  n = note \u00b7 s = skip \u00b7 b = back \u00b7 m = MIP/planes \u00b7 o = outlines")
    print("  blind: shuffled, image names hidden" if blind else "  not blind: in order, image names shown")
    print("  Every answer is saved. Close the window to stop; it resumes here.\n")

    viewer = napari.Viewer(title=f"SCORING {stain}", ndisplay=2)
    walk = Walk(project_dir, stain, todo, blind)
    state = {"view": "raw", "mip": False, "show_outlines": True}
    raw = saved is None and VIEW_STAIN_LIMITS.get(stain) == "raw"
    if raw:
        print("  h = stain raw/stretch")
    print("  Contrast you set on one neuron is kept for the next ones.")
    view_limits: dict[str, list[float]] = {}
    looks = {v: dict((saved or {}).get("views", {}).get(v, {})) for v in VIEWS}
    channel_of = {stain.upper(): "stain", "DAPI": "dapi", "NEUN": "neun",
                  **{c.upper(): c for c in SCORING_CONTEXT.get(stain, [])}}
    shown: dict[str, dict] = {}

    def current() -> dict[str, dict]:
        """Limits, gamma and visibility of each channel layer on screen."""
        return {channel_of[layer.name]: {"limits": [float(x) for x in layer.contrast_limits],
                                         "gamma": float(layer.gamma),
                                         "visible": bool(layer.visible)}
                for layer in viewer.layers
                if isinstance(layer, napari.layers.Image) and layer.name in channel_of}

    def carry() -> None:
        """Keep every channel the rater changed on the crop on screen for the crops after it."""
        if not shown:
            return
        changed = {ch: look for ch, look in current().items() if look != shown["looks"].get(ch)}
        if changed:
            looks[shown["view"]].update(changed)
            save_display(project_dir, stain, looks,
                         {"rule": "set in the scoring window; carried to every later crop"})

    def overlay() -> None:
        """One line: who (blind id, or image and neuron), place, MIP; then an earlier answer, note."""
        blind_id = walk.current or ""
        unit = manifest.get(blind_id, {})
        who = blind_id if blind else f"{unit.get('fov', '')}   {unit.get('neuron_id', '')}"
        place = f"{total - len(todo) + walk.i + 1}/{total}"
        lines = [f"{who}   {place}" + ("   MIP" if state["mip"] else "")]
        if walk.previous():
            lines.append(f"was: {walk.previous()}")
        if walk.note:
            lines.append(f"note: {walk.note}")
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = "\n".join(lines)

    def show() -> None:
        """Load the current crop, or close when the queue is done."""
        carry()
        blind_id = walk.current
        if blind_id is None:
            print(f"  {stain}: queue done, {walk.left()} skipped (asked again next run)")
            # The Qt window, not viewer.close(), which tears the canvas down before the screenshot.
            viewer.window._qt_window.close()
            return
        unit = manifest[blind_id]
        spacing = json.loads(unit["spacing_um"])
        mip = state["mip"]
        carried = looks["mip" if mip else "planes"]
        display = {**json.loads(unit["display"]), **carried}
        with np.load(scoring_dir(project_dir, stain) / "crops" / f"{blind_id}.npz") as crop:
            arrays = view_arrays(crop, mip)
            stain_vox = crop["stain"]
        view = VIEW_STAIN_LIMITS.get(stain)
        if raw and "stain" not in carried:
            top = (float(np.iinfo(stain_vox.dtype).max) if stain_vox.dtype.kind in "ui"
                   else float(stain_vox.max()))
            positive = stain_vox[stain_vox > 0]
            view_limits["raw"] = [0.0, max(top * RAW_TOP_FRACTION, 1.0)]
            view_limits["range"] = [0.0, max(top, 1.0)]
            view_limits["stretch"] = [0.0, max(float(np.percentile(positive, 99.99))
                                               if positive.size else 1.0, 1.0)]
            display["stain"] = {**display["stain"], "limits": view_limits[state["view"]]}
        elif "stain" not in carried and saved is None and isinstance(view, list):
            display["stain"] = {**display["stain"], "limits": view}
        viewer.layers.clear()
        add_channels(viewer, arrays, stain, spacing[1:] if mip else spacing, display)
        add_marks(viewer, unit, other_marks(project_dir, unit), mip, target_size_um)
        add_outlines(viewer, arrays, spacing[1:] if mip else spacing, mip,
                     state["show_outlines"])
        widen(viewer)
        if "range" in view_limits and "stain" not in carried:  # slider spans the full detector range; handle where set
            for layer in viewer.layers:
                if isinstance(layer, napari.layers.Image) and layer.name == stain.upper():
                    layer.contrast_limits_range = view_limits["range"]
                    lo, hi = view_limits[state["view"]]
                    layer.contrast_limits = (lo, hi)
        if not mip:
            viewer.dims.set_point(0, float(unit["target_z_um"]))
        viewer.layers.selection.clear()
        for layer in viewer.layers:
            bind(layer)
        shown.update(view="mip" if mip else "planes", looks=current())
        overlay()

    def pick(category: str) -> None:
        """Write the answer for the crop on screen, then move on."""
        walk.pick(category)
        show()

    def note() -> None:
        """Ask for a one-line note that goes with this crop's answer."""
        text, ok = QInputDialog.getText(None, "note", "one-line note for this crop:",
                                        text=walk.note)
        if ok:
            walk.set_note(text)
            overlay()

    def skip() -> None:
        """Leave the crop on screen as it is and move on; its note stays pending."""
        walk.skip()
        show()

    def back() -> None:
        """Show the crop before again."""
        if walk.i:
            walk.back()
            show()

    def switch_mip() -> None:
        """Planes <-> max projection over the crop's z."""
        state["mip"] = not state["mip"]
        show()

    def switch_view() -> None:
        """Stain display raw (0 to half the detector range) <-> stretch (0 to the crop's p99.99)."""
        if not view_limits:
            return
        state["view"] = "stretch" if state["view"] == "raw" else "raw"
        for layer in viewer.layers:
            if isinstance(layer, napari.layers.Image) and layer.name == stain.upper():
                lo, hi = view_limits[state["view"]]
                layer.contrast_limits = (lo, hi)
        print(f"  {stain} display: {state['view']}")

    def toggle_outlines() -> None:
        """Show or hide nucleus and soma outline layers."""
        state["show_outlines"] = not state["show_outlines"]
        for name in ("nucleus", "soma"):
            layer = next((l for l in viewer.layers if l.name == name), None)
            if layer:
                layer.visible = state["show_outlines"]

    actions: dict[str, Callable[[], None]] = {"n": note, "s": skip, "b": back, "m": switch_mip,
                                               "o": toggle_outlines}
    if raw:
        actions["h"] = switch_view
    for i, category in enumerate(keys, 1):
        actions[str(i)] = lambda category=category: pick(category)

    def bind(target) -> None:
        """Bind every scoring key on a viewer or layer, over napari's own shortcuts."""
        for key, action in actions.items():
            target.bind_key(key, lambda _obj, action=action: action(), overwrite=True)

    bind(viewer)
    show()
    census.show_until_closed(viewer, screenshot)
    carry()  # what was changed on the last crop shown
