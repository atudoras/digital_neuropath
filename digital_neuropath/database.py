"""The project database: one folder, a fixed set of tables, read and checked here.

    project.json                         fovs     one entry per fov (FOV_REQUIRED + FOV_OPTIONAL)
    fovs/<fov>/census/neurons.csv        neurons  one row per counted neuron; final once sealed.json exists
    imported/<stain>.csv                 scores   answers made before this project
    results/<stain>_unsealed.csv         scores   one row per neuron once every crop is answered
    scoring_<stain>/, manifests/<stain>  blind    crops and answers, kept apart from fov identity

A project folder that does not exist yet is an empty database; the first registered fov creates it.
"""
import json
from pathlib import Path

import pandas as pd

from digital_neuropath import census, io, score

__all__ = ["FOV_OPTIONAL", "FOV_REQUIRED", "fovs", "print_status", "status", "write_fovs"]

FOV_REQUIRED = ("fov", "source", "series", "name")
FOV_OPTIONAL = ("channels", "case", "seen_before")


def project_file(project_dir: Path) -> Path:
    """project.json, the fovs table."""
    return Path(project_dir) / "project.json"


def check_fov(entry: dict, where: Path) -> dict:
    """One fovs row; refused if a required field is missing or a field is not in the schema."""
    missing = [k for k in FOV_REQUIRED if k not in entry]
    unknown = sorted(set(entry) - set(FOV_REQUIRED) - set(FOV_OPTIONAL))
    if missing or unknown:
        raise ValueError(f"{where}: fov {entry.get('fov', '?')} has missing {missing} / "
                         f"unknown {unknown} fields; schema is {FOV_REQUIRED} + {FOV_OPTIONAL}")
    return entry


def fovs(project_dir: Path) -> list[dict]:
    """Every fov of the project, checked against the schema; [] for a new project."""
    path = project_file(project_dir)
    if not path.exists():
        return []
    return [check_fov(e, path) for e in json.loads(path.read_text())["fovs"]]


def write_fovs(project_dir: Path, entries: list[dict]) -> None:
    """Rewrite the fovs table atomically, after checking every row."""
    path = project_file(project_dir)
    for entry in entries:
        check_fov(entry, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"fovs": entries}, indent=2))
    tmp.replace(path)


def stain_state(project_dir: Path, stain: str) -> dict[tuple[str, str], str]:
    """(fov, neuron_id) -> 'done' (unsealed, imported or skipped) or 'blind' (in a crop round).
    Blind answers are never read here, so nothing tells which crops are still open."""
    project_dir = Path(project_dir)
    out: dict[tuple[str, str], str] = {}
    if score.manifest_path(project_dir, stain).exists():
        out |= {(r["fov"], r["neuron_id"]): "blind"
                for r in io.read_rows(score.manifest_path(project_dir, stain))}
    skipped = score.skipped_path(project_dir, stain)
    out |= {(r["fov"], r["neuron_id"]): "done" for r in io.read_rows(skipped)}
    out |= {unit: "done" for unit in score.imported(project_dir, stain)}
    unsealed = project_dir / "results" / f"{stain}_unsealed.csv"
    out |= {(r["fov"], r["neuron_id"]): "done" for r in io.read_rows(unsealed)}
    return out


def status(project_dir: Path, stains: list[str] = ()) -> pd.DataFrame:
    """One row per fov: case, census state and neuron count, then per stain how many of its
    sealed neurons are done, in a blind round, or not yet given a crop."""
    rows = []
    states = {stain: stain_state(project_dir, stain) for stain in stains}
    for entry in fovs(project_dir):
        fov_dir = Path(project_dir) / "fovs" / entry["fov"]
        sealed = census.is_sealed(fov_dir)
        started = (census.census_dir(fov_dir) / "census.json").exists()
        n = len(census.marks(fov_dir)) if started else 0
        row = {"fov": entry["fov"], "case": entry.get("case"),
               "census": "sealed" if sealed else ("open" if started else "new"), "neurons": n}
        units = [(entry["fov"], r["neuron_id"]) for r in census.check_seal(fov_dir)] if sealed else []
        for stain in stains:
            got = [states[stain].get(u, "new") for u in units]
            row[stain] = (f"done {got.count('done')}/{len(units)}" if sealed else "-") + (
                f", blind {got.count('blind')}" if got.count("blind") else "") + (
                f", new {got.count('new')}" if got.count("new") else "")
        rows.append(row)
    return pd.DataFrame(rows)


def print_status(project_dir: Path, stains: list[str] = (), only: list[str] | None = None) -> None:
    """status() as one line per fov, limited to the fov ids in `only` when given."""
    table = status(project_dir, list(stains))
    if only is not None:
        table = table[table["fov"].isin(only)]
    for row in table.to_dict("records"):
        parts = [f"{row['census']}, {row['neurons']} neurons"] + [f"{s} {row[s]}" for s in stains]
        case = f" ({row['case']})" if row["case"] is not None else ""
        print(f"  {row['fov']}{case}: " + " | ".join(parts))
