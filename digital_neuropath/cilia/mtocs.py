"""Detect centrosomal foci in the pcnt channel of one fov.
tophat -> smooth -> hysteresis -> size window -> table.
No pcnt ground truth exists, so the operating point is eyeballed and the count is steep in high_pct.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.filters import apply_hysteresis_threshold

from digital_neuropath.cilia import filters, measure, params

# by name: the package's own load() function shadows its load module as an attribute
from digital_neuropath.cilia.load import Fov, read_plane, write_labels, write_run_json, write_table

__all__ = ["segment_mtocs", "detect", "MtocResult"]


@dataclass
class MtocResult:
    """The focus map, its table and where they were written."""

    labels: np.ndarray
    table: pd.DataFrame
    out_dir: Path


def segment_mtocs(fov: Fov, out_dir: Path, **overrides: Any) -> MtocResult:
    """Detect foci in the pcnt channel of one fov and write the label map and its table."""
    p, status = params.resolve(params.MTOC, overrides)
    spacing = fov.spacing_um
    pcnt = read_plane(fov, "pcnt")
    labels = detect(pcnt, spacing, p)
    table = measure.morphometry(labels, pcnt.astype(np.float64), spacing, foci=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_labels(out_dir / "labels.tif", labels)
    write_table(out_dir, "pcnt_morphometry", table)
    write_run_json(out_dir, fov=fov, pipeline="mtoc", values=p, status=status,
                   inputs={"image": str(fov.image)})
    fov.mtoc_dir = out_dir
    return MtocResult(labels, table, out_dir)


def detect(pcnt: np.ndarray, spacing: tuple[float, float, float], p: dict[str, Any]) -> np.ndarray:
    """Return the focus label map of one pcnt volume under resolved MTOC values."""
    volume = pcnt.astype(np.float32)
    footprint = filters.ellipsoid_footprint(p["tophat_radius_um"], spacing)
    flat = np.asarray(ndimage.white_tophat(volume, footprint=footprint))
    response = filters.smooth(flat, spacing, sigma_um=p["smooth_um"]).astype(np.float32)
    high = filters.positive_percentile(response, p["high_pct"])
    low = filters.positive_percentile(response, p["low_pct"])
    if high <= low:
        # percentiles coincide when the response has few distinct positive values
        high = low + 1e-6
    mask = apply_hysteresis_threshold(response, low, high)
    # scipy's labeller, as shipped: skimage's may number the same components in another order
    found = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.int8))
    labels = np.asarray(found[0])  # pyright: ignore[reportIndexIssue]
    counts = np.bincount(labels.ravel())
    # renumbered, not reconnected: a pair the hysteresis split stays two foci
    return filters.relabel(labels, (counts >= int(p["min_vox"])) & (counts <= int(p["max_vox"])))
