"""Count a reference region's voxels in the foreground of every step of segment_cilia.
bg -> ridge -> floor -> mask -> min_voxels -> min_aspect -> seeds -> grow_thr -> grow_cap -> final.
Reads cilia.stages, the detector's own intermediates, so the step that drops a cilium is counted, not guessed.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import skimage

from digital_neuropath.cilia import cilia, params

__all__ = ["stage_counts", "STAGES"]

STAGES = ("bg", "ridge", "floor", "mask", "min_voxels", "min_aspect", "seeds", "grow_thr",
          "grow_cap", "final")


def stage_counts(raw: np.ndarray, spacing: tuple[float, float, float], reference: np.ndarray, *,
                 cache: Path | None = None, **overrides: Any
                 ) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    """Return the reference voxels left foreground at each step, and each step's mask (final as labels)."""
    p, _ = params.resolve(params.CILIA, overrides)
    key = json.dumps({"params": p, "shape": list(raw.shape), "spacing": list(spacing)}, sort_keys=True)
    stages: dict[str, np.ndarray] | None = None
    if cache is not None and Path(cache).is_file():
        with np.load(cache) as stored:
            if str(stored["key"]) == key:
                stages = {s: stored[s] for s in STAGES}
    if stages is None:
        steps = cilia.stages(raw, spacing, p)
        components = np.asarray(skimage.measure.label(steps["mask"], connectivity=3))
        big = np.bincount(components.ravel()) >= int(p["min_voxels"])
        big[0] = False
        stages = {s: steps[s] for s in ("bg", "ridge", "floor", "mask", "seeds", "grow_thr",
                                        "grow_cap", "final")}
        stages["min_voxels"] = big[components]
        stages["min_aspect"] = steps["premask"] > 0
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, key=key, **stages)  # pyright: ignore[reportArgumentType]
    counts = {s: int(np.count_nonzero(stages[s][reference])) for s in STAGES}
    return counts, stages
