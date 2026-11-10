"""Step-2 mask review: per-FOV good / refine / discard / cut / whole calls on NeuroVIT cell bodies."""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["COLUMNS", "Field", "Review", "census_bodies", "face_fraction", "face_ids", "is_done",
           "load_bodies", "load_images"]

COLUMNS = ["cell_body_id", "status", "note", "done"]
# Next status on a click: interior bodies good -> refine -> discard -> good, face-touching bodies cut -> whole -> refine -> cut.
# Three decisions: ok (good, whole), refine 3D (refine), discarded (cut, discard).
INTERIOR_NEXT = {"good": "refine", "refine": "discard", "discard": "good"}
FACE_NEXT = {"cut": "whole", "whole": "refine", "refine": "cut"}


def load_bodies(npz: Path, spacing: tuple[float, float, float], min_um3: float = 10) -> np.ndarray:
    """Cell body labels with bodies under min_um3 removed."""
    body = np.load(npz)["cell_body"]
    n = np.bincount(body.ravel())
    specks = np.flatnonzero((n > 0) & (n * np.prod(spacing) < min_um3))
    return np.where(np.isin(body, specks[specks > 0]), 0, body)


def face_ids(body: np.ndarray) -> set[int]:
    """IDs touching any of the six stack faces."""
    faces = [body[0], body[-1], body[:, 0], body[:, -1], body[:, :, 0], body[:, :, -1]]
    return {int(i) for f in faces for i in np.unique(f) if i}


def face_fraction(body: np.ndarray) -> dict[int, float]:
    """Fraction of each face-touching body's voxels that lie on a stack face."""
    on_face = np.ones(body.shape, bool)
    on_face[1:-1, 1:-1, 1:-1] = False
    total = np.bincount(body.ravel())
    face = np.bincount(body[on_face], minlength=len(total))
    return {i: float(face[i] / total[i]) for i in face_ids(body)}


def census_bodies(body: np.ndarray, neurons_csv: Path) -> set[int]:
    """IDs of bodies that contain at least one census dot (neurons.csv voxel coordinates)."""
    with neurons_csv.open(newline="") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    zyx = np.array([[round(float(r[k])) for k in ("z_vox", "y_vox", "x_vox")] for r in rows], int).reshape(-1, 3)
    zyx = np.clip(zyx, 0, np.array(body.shape) - 1)
    return {int(i) for i in body[tuple(zyx.T)] if i}


@dataclass(frozen=True)
class Field:
    """One pilot fov: patient, mask, spacing, DAPI/NeuN source (None when absent) and census dots."""
    fov: str
    patient: str
    mask: Path
    spacing: tuple[float, float, float]
    images: Path | None
    census: Path


def load_images(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(dapi, neun) from a review npz or a .cmap; no other channel is read."""
    if path.suffix == ".cmap":
        import h5py
        with h5py.File(path, "r") as fh:
            return fh["DAPI"]["data"][()], fh["NeuN"]["data"][()]
    with np.load(path) as z:
        return z["dapi"], z["neun"]


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Rows of a review CSV, refused when its columns are not COLUMNS."""
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"{path} has columns {reader.fieldnames}, not {COLUMNS}")
        return list(reader)


def is_done(path: Path) -> bool:
    """True when the CSV exists, is readable and every row is marked done."""
    try:
        rows = _read_rows(path) if path.exists() else []
    except ValueError:
        return False
    return bool(rows) and all(r["done"] == "1" for r in rows)


class Review:
    """One field's review; the CSV is read on open and written only after a change."""

    def __init__(self, path: Path, body: np.ndarray) -> None:
        self.path, self.body = path, body
        self.ids = [int(i) for i in np.unique(body) if i]
        self.face = face_ids(body)
        self.done = False
        self.current = 0
        self.calls = {i: {"status": "cut" if i in self.face else "good", "note": ""} for i in self.ids}
        self.extra: list[dict[str, str]] = []
        if path.exists():
            rows = _read_rows(path)
            for r in rows:
                cid = int(r["cell_body_id"])
                if cid in self.calls:
                    self.calls[cid].update(status=r["status"], note=r["note"])
                else:
                    self.extra.append(r)
            self.done = all(r["done"] == "1" for r in rows) and set(self.ids) <= {int(r["cell_body_id"]) for r in rows}

    def toggle(self, i: int) -> None:
        """Advance body i one state and save."""
        status = self.calls[i]["status"]
        if i in self.face:
            self.calls[i]["status"] = FACE_NEXT.get(status, "cut")
        else:
            self.calls[i]["status"] = INTERIOR_NEXT.get(status, "refine")
        self.current = i
        self.save()

    def mark_done(self) -> None:
        """Toggle the field-done flag and save."""
        self.done = not self.done
        self.save()

    def note(self, i: int, text: str) -> None:
        """Set body i's note; saves only if the text changed."""
        text = text.strip()
        if text == self.calls[i]["note"]:
            return
        self.calls[i]["note"] = text
        self.save()

    def save(self) -> None:
        """Write every body's state, then the CSV rows absent from the mask unchanged."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(COLUMNS)
            w.writerows([i, c["status"], c["note"], int(self.done)] for i, c in self.calls.items())
            w.writerows([r["cell_body_id"], r["status"], r["note"], int(self.done)] for r in self.extra)

    def progress(self) -> str:
        """One-line status."""
        groups = {"ok": ("good", "whole"), "refine": ("refine",), "discarded": ("cut", "discard")}
        count = {k: sum(c["status"] in v for c in self.calls.values()) for k, v in groups.items()}
        return (f"{len(self.ids)} bodies | {count['ok']} ok | {count['refine']} refine | "
                f"{count['discarded']} discarded | {'DONE' if self.done else 'not done'}")
