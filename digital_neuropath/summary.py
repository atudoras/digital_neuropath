"""Per-neuron database across stains, and a per-fov count table with Wilson 95% bounds.
Chain: database (sealed census + every unsealed stain) -> table (per fov, then a labelled pooled sum).
The fov is the unit that carries an n here; a pooled row is a sum shown beside them, never a rate alone.
"""
import datetime
import json
import math
from pathlib import Path

import pandas as pd

from digital_neuropath import census, io, score

__all__ = ["database", "provenance", "read", "table", "wilson", "write"]

POOLED_LABEL = "pooled (sum of the per-fov rows above; not a rate)"
Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def unsealed_stains(project_dir: Path) -> list[str]:
    """Stains whose scoring has been unsealed."""
    results = Path(project_dir) / "results"
    return sorted(p.name.removesuffix("_unsealed.csv") for p in results.glob("*_unsealed.csv"))


def database(project_dir: Path) -> pd.DataFrame:
    """One row per neuron of a sealed fov: position, then category and note per unsealed stain."""
    rows = [r for d in score.fov_dirs(project_dir) if census.is_sealed(d) for r in census.check_seal(d)]
    db = pd.DataFrame(rows, columns=["fov", "neuron_id", "z_um", "y_um", "x_um", "soma_whole"])
    for stain in unsealed_stains(project_dir):
        scored = pd.DataFrame([{"fov": r["fov"], "neuron_id": r["neuron_id"], stain: r["category"],
                                f"{stain}_note": r["note"]}
                               for r in io.read_rows(Path(project_dir) / "results" /
                                                     f"{stain}_unsealed.csv")])
        db = db.merge(scored, on=["fov", "neuron_id"], how="left")
    return db


def is_not_scoreable(category: str) -> bool:
    """'not scoreable', or 'not scoreable (from <stain>)' for a neuron an earlier stain skipped."""
    return category == "not scoreable" or category.startswith("not scoreable (from ")


def table(db: pd.DataFrame, stain: str, categories: list[str], unit: str = "fov") -> pd.DataFrame:
    """Per unit (a db column) and category: count, n (all neurons of that unit), proportion and
    Wilson 95% bounds; then n_scored (real categories only), n_unsure, n_not_scoreable and the
    proportion of n_scored with its Wilson bounds, given for real categories only; then
    count_whole and n_whole, the same counts over neurons whose soma_whole is 'yes' (EXPERIMENTAL;
    empty for a unit whose fovs were sealed before soma_whole existed)."""
    real = [c for c in categories if c != "unsure" and not is_not_scoreable(c)]
    rows = []
    for value in sorted(db[unit].unique()):
        here = db.loc[db[unit] == value]
        values = [v for v in here[stain].tolist() if isinstance(v, str)]
        whole = here["soma_whole"].tolist() if "soma_whole" in here else []
        known = any(w in ("yes", "no") for w in whole)
        values_whole = [v for v, w in zip(here[stain].tolist(), whole)
                        if isinstance(v, str) and w == "yes"]
        n = len(values)
        n_scored = sum(v in real for v in values)
        for category in categories:
            k = values.count(category)
            lo, hi = wilson(k, n)
            scored_lo, scored_hi = wilson(k, n_scored) if category in real else (math.nan, math.nan)
            rows.append({unit: value, "category": category, "count": k, "n": n,
                         "proportion": k / n if n else math.nan,
                         "wilson95_lo": lo, "wilson95_hi": hi,
                         "n_scored": n_scored, "n_unsure": values.count("unsure"),
                         "n_not_scoreable": sum(is_not_scoreable(v) for v in values),
                         "proportion_of_scored": (k / n_scored if category in real and n_scored
                                                  else math.nan),
                         "wilson95_lo_scored": scored_lo, "wilson95_hi_scored": scored_hi,
                         "count_whole": values_whole.count(category) if known else math.nan,
                         "n_whole": len(values_whole) if known else math.nan})
    pooled = POOLED_LABEL if unit == "fov" else f"pooled (sum of the per-{unit} rows above; not a rate)"
    for category in categories:
        mine = [r for r in rows if r["category"] == category]
        rows.append({unit: pooled, "category": category,
                     "count": sum(r["count"] for r in mine), "n": sum(r["n"] for r in mine),
                     "proportion": math.nan, "wilson95_lo": math.nan, "wilson95_hi": math.nan,
                     **{c: sum(r[c] for r in mine) for c in ("n_scored", "n_unsure",
                                                             "n_not_scoreable")},
                     "proportion_of_scored": math.nan, "wilson95_lo_scored": math.nan,
                     "wilson95_hi_scored": math.nan,
                     **{c: (sum(r[c] for r in mine if not math.isnan(r[c]))
                            if any(not math.isnan(r[c]) for r in mine) else math.nan)
                        for c in ("count_whole", "n_whole")}})
    return pd.DataFrame(rows)


def provenance(project_dir: Path, stains: list[str]) -> list[str]:
    """`#` header lines: package version, date, and each stain's seed and contrast."""
    from digital_neuropath import __version__

    written = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    lines = [f"digital_neuropath {__version__}; written {written}"]
    for stain in stains:
        meta = json.loads(score.manifest_path(project_dir, stain).with_suffix(".json").read_text())
        lines.append(f"stain {stain}: seed {meta['seed']}, contrast {meta.get('contrast', 'fov')}")
    return lines


def write_frame(path: Path, frame: pd.DataFrame, header: list[str], **to_csv) -> None:
    """A table with its provenance as `#` lines above it."""
    with path.open("w", newline="") as handle:
        handle.writelines(f"# {line}\n" for line in header)
        frame.to_csv(handle, index=False, **to_csv)


def read(path: Path) -> pd.DataFrame:
    """A results table, skipping its `#` provenance header (a note may itself hold a '#')."""
    with Path(path).open() as handle:
        skip = 0
        for line in handle:
            if not line.startswith("#"):
                break
            skip += 1
    return pd.read_csv(path, skiprows=skip)


def write_results(project_dir: Path) -> Path:
    """The one output table, <project>/results.csv: a row per neuron with its image, case,
    position and every stain's call and note."""
    from digital_neuropath import database as fov_table

    project_dir = Path(project_dir)
    db = database(project_dir)
    cases = {e["fov"]: e.get("case", "") for e in fov_table.fovs(project_dir)}
    db.insert(1, "case", db["fov"].map(cases))
    out = project_dir / "results.csv"
    write_frame(out, db, provenance(project_dir, unsealed_stains(project_dir))
                + ["one row per neuron"])
    return out


def write(project_dir: Path) -> list[Path]:
    """Write results/neurons.csv and one results/summary_<stain>.csv per unsealed stain."""
    project_dir = Path(project_dir)
    db = database(project_dir)
    out = [project_dir / "results" / "neurons.csv"]
    write_frame(out[0], db, provenance(project_dir, unsealed_stains(project_dir)))
    for stain in unsealed_stains(project_dir):
        categories = score.settings(project_dir, stain)["categories"]
        # 'not scoreable (from <stain>)' neurons got no crop, so build() never listed that category
        categories += sorted({v for v in db[stain] if isinstance(v, str)} - set(categories))
        path = project_dir / "results" / f"summary_{stain}.csv"
        write_frame(path, table(db, stain, categories), provenance(project_dir, [stain]),
                    float_format="%.4f")
        out.append(path)
    return out
