"""Every knob of the three pipelines and of the stage-2 rule, one Param each, grouped by the code that reads it.
resolve merges a call's overrides onto a group and names the weakest status the call ran under.
The shipped value is the default, so a call with no overrides runs the shipped chain.
"""
from dataclasses import dataclass
from typing import Any

__all__ = ["Param", "CILIA", "MTOC", "LINK", "STAGE2", "INCLUSION", "resolve"]

# strongest first, so the weakest has the highest index
STATUSES = ("calibrated", "LOPO-fitted", "controls-only", "eyeballed", "inherited", "unverified")


@dataclass(frozen=True)
class Param:
    """One knob: its value, unit, how far the value is trusted, and one evidence pointer."""

    value: Any
    unit: str
    status: str
    evidence: str


CILIA = {
    "sigma_bg_um": Param(2.0, "µm", "unverified", "A037"),
    # a float is converted with the y spacing and applied in voxels on every axis (shipped);
    # a (z, y, x) tuple is per axis in µm
    "ridge_sigma_um": Param(0.27, "µm, or (z, y, x) µm", "controls-only", "A012"),
    "ridge_pct": Param(99.5, "percentile of ridge>0", "controls-only", "A037"),
    "floor_k": Param(6.0, "background σ", "controls-only", "A037"),
    "background": Param("global", "global | local", "controls-only", "A037"),
    "background_radius_um": Param(5.0, "µm (local only)", "unverified", "untested"),
    "min_voxels": Param(15, "voxels", "unverified", "A037"),
    "min_aspect": Param(1.25, "ratio major/(minor+1)", "controls-only", "A037"),
    "z_chunk": Param(24, "z-planes", "eyeballed", "A037"),
    "z_halo": Param(8, "z-planes", "unverified", "A037"),
    "grow_sigma_bg_um": Param(1.0, "µm", "unverified", "A037"),
    "sigma_blur_um": Param(0.4, "µm", "unverified", "A037"),
    "grow": Param("capped", "capped | hysteresis", "controls-only", "A037"),
    "threshold_pct": Param(97.0, "percentile of F>0 (capped)", "controls-only", "A037"),
    "max_grow_um": Param(0.5, "µm (capped)", "controls-only", "A037"),
    "grow_low_k": Param(2.0, "background σ (hysteresis)", "unverified", "untested"),
    "grow_high_k": Param(6.0, "background σ (hysteresis)", "unverified", "untested"),
    "min_volume_vox": Param(3, "voxels", "unverified", "A037"),
    "join_gap_um": Param(0.0, "µm between object tips, 0 = off", "controls-only", "A037"),
    "join_angle_deg": Param(30.0, "° between each end's direction and the gap", "unverified",
                            "untested"),
    "bridge_gap_um": Param(0.6, "µm (closing radius = half)", "controls-only", "A036"),
    "prune_um": Param(0.3, "µm", "unverified", "A036"),
    "curvature_smooth_um": Param(1.0, "µm of arc length", "unverified", "A036"),
    "collapse_radius_um": Param(0.9, "µm", "controls-only", "A036"),
}

# no pcnt ground truth exists, so eyeballed is the ceiling here
MTOC = {
    "tophat_radius_um": Param(0.8, "µm", "eyeballed", "A038"),
    "smooth_um": Param(0.12, "µm", "eyeballed", "A038"),
    "high_pct": Param(99.97, "percentile of response>0", "eyeballed", "A038"),
    "low_pct": Param(99.9, "percentile of response>0", "eyeballed", "A038"),
    "min_vox": Param(20, "voxels", "eyeballed", "A038"),
    "max_vox": Param(1500, "voxels", "eyeballed", "A038"),
}

LINK = {
    "max_reach_um": Param(18.0, "µm", "unverified", "A039"),
    "inside_frac": Param(0.99, "fraction of the object's voxels", "unverified", "A039"),
    "max_base_dist_um": Param(3.0, "µm", "unverified", "A039"),
    "min_tip_out_um": Param(0.4, "µm", "unverified", "A039"),
    "peak_radius_um": Param(0.4, "µm", "unverified", "A039"),
    "peak_k_sigma": Param(6.0, "robust sigma above the field median", "unverified", "A039"),
}

# the validated stage-2 rule (validate.py, variants.py), not cilia/link.py, which selects by soma-surface
# geometry; the defaults are R0, harness_r1_4row_2026-11-05/R0_per_neuron.csv
STAGE2 = {
    "object_filter": Param("none", "none | min_length_um | max_norm", "inherited", "R0"),
    "min_length_um": Param(0.0, "µm (min_length_um filter; shorter is dropped)", "unverified", "untested"),
    "max_norm": Param(2.6417, "object max AC3 / field AC3 p99.9 (max_norm filter; at or below is dropped)",
                      "LOPO-fitted", "C2, 182 blind object labels; the pick in 11/12 LOPO folds"),
    "assignment": Param("greedy", "greedy | mtoc_first", "inherited", "R0"),
    "d_um": Param(5.0, "µm base to focus (greedy), near end to focus (mtoc_first)", "unverified",
                  "validation/gates.md"),
    "balloon_margin_um": Param(3.0, "µm cell-body dilation that owns a focus", "unverified", "validation/gates.md"),
}

# the three-way inclusion gate R2 (inclusion.py): observability omega of the cilium search volume and the nucleus face margin
GATE_EVIDENCE = "gate_optimize 2026-11-06 (band 274/2093, cut auto-in 1.9 %, whole auto-out 4.8 %)"
INCLUSION = {
    "core_opening_um": Param(1.5, "µm, ball radius of the soma-core opening", "LOPO-fitted", GATE_EVIDENCE),
    "search_um": Param(5.0, "µm, soma core dilation that defines the search volume", "LOPO-fitted", GATE_EVIDENCE),
    "nucleus_margin_um": Param(0.5, "µm, nucleus to nearest stack face (below: exclude)", "LOPO-fitted", GATE_EVIDENCE),
    "omega_in": Param(0.92, "omega at or above which (with the nucleus margin) a neuron is included", "LOPO-fitted", GATE_EVIDENCE),
    "omega_out": Param(0.78, "omega below which a neuron is excluded", "LOPO-fitted", GATE_EVIDENCE),
}


def resolve(group: dict[str, Param], overrides: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Return every value of the group with the overrides applied, and the weakest status used,
    counting an overridden value as unverified."""
    unknown = sorted(set(overrides) - set(group))
    if unknown:
        raise TypeError(f"unknown parameter(s) {unknown}; known: {sorted(group)}")
    values = {name: overrides.get(name, param.value) for name, param in group.items()}
    used = {"unverified" if values[name] != param.value else param.status
            for name, param in group.items()}
    return values, max(used, key=STATUSES.index)
