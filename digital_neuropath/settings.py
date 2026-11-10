"""Settings shared by every step. A step module has STEP, TITLE, SETTINGS, run() and view()."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Field:
    default: Any
    unit: str = ""
    help: str = ""
    choices: tuple = ()


def from_params(group: dict) -> dict[str, Field]:
    """cilia.params group -> Fields."""
    return {name: Field(p.value, p.unit) for name, p in group.items()}


def values(fields: dict[str, Field], stored: dict | None = None) -> dict[str, Any]:
    """Defaults with the saved changes applied."""
    stored = stored or {}
    unknown = set(stored) - set(fields)
    if unknown:
        raise KeyError(f"unknown setting(s) {sorted(unknown)}")
    out = {}
    for k, f in fields.items():
        v = stored.get(k, f.default)
        # a list from toml is a per-axis value, e.g. ridge_sigma_um (z, y, x)
        out[k] = tuple(v) if isinstance(v, (list, tuple)) else type(f.default)(v)
    return out


def changed(fields: dict[str, Field], current: dict[str, Any]) -> dict[str, Any]:
    """Only what differs from the default; this is what project.toml stores."""
    return {k: v for k, v in current.items() if v != fields[k].default}


