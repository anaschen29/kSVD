"""Data-only accuracy and modal-rate diagnostics, independent of stopping/plotting.

A stopping reason is not an accuracy claim. A mode's linearized multiplier is
not necessarily the asymptotic rate of its nonlinearly forced amplitude.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from .metrics import estimate_rate

DIAGNOSTICS_VERSION = "collection-v2"
DEFAULT_TARGETS = {"product_error": 1e-8}
METRIC_DEFINITIONS = {
    "product_error": "||XX^T-M_k||_F / ||M_k||_F; only for a unique optimal product",
    "trace_over_gap": "e(X) / (lambda_k-lambda_(k+1)); dimensionless, strict gap only",
    "trace_deficit": "sum_{i<=k} lambda_i - tr(M Pi_X); absolute spectral mass",
    "objective_gap": "g(X)-g_star; absolute objective gap",
    "factor_error": "absolute error to the known limit; Y for controlled rates, X for rank-one comparison",
    "x_factor_error": "||X-X_infinity||_F; absolute, known target only",
    "orbit_error_y": "min_{O orthogonal} ||Y-E_star O||_F; set distance, NOT distance to a particular limit",
    "orbit_error_x": "min_{O orthogonal} ||X-X_star O||_F; absolute fixed-orbit distance",
    "radial_signed": "Y[k-1,k-1]-1 in the one-column controlled experiment",
    "angular_signed": "Y[k,k-1] in the one-column controlled experiment",
    "radial_normal_norm": "||sym((Y O^T)[:k]-I)||_F after diagnostic Procrustes alignment",
    "angular_normal_norm": "||(Y O^T)[k:]||_F after diagnostic Procrustes alignment",
    "slow_angular_signed": "(Y O^T)[k,k-1] after diagnostic Procrustes alignment",
    "radial_error": "||Y^T Y-I||_F; not a signed Hessian-mode amplitude",
    "manifold_residual": "combined Gram and optimal-family subspace residual; not an exact distance",
}


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_targets(targets: Mapping[str, float]) -> None:
    if not isinstance(targets, Mapping):
        raise ValueError("accuracy_targets must map metric names to nonnegative tolerances")
    for key, tolerance in targets.items():
        if not isinstance(key, str) or not key or not _number(tolerance) or tolerance < 0:
            raise ValueError("accuracy_targets require names and finite nonnegative tolerances")


class AccuracyTracker:
    """Observe every evaluated iterate, even if trajectories are saved sparsely.

    Missing metrics are explicitly unavailable. First hits do not imply the
    tolerance remains met. Final accuracy is unknown if the terminal state
    could not be evaluated (for example after numerical rank loss).
    """

    def __init__(self, targets: Mapping[str, float]) -> None:
        validate_targets(targets)
        self.targets = dict(targets)
        self.observations: dict[str, dict] = {
            key: {"observations": 0, "first_hit_step": None, "last_observed_step": None,
                  "last_observed_value": None, "minimum_observed_value": None}
            for key in targets
        }
        self.last_step: int | None = None

    def observe(self, row: dict) -> bool:
        step = row["t"]
        if not isinstance(step, int) or step < 0 or (self.last_step is not None and step <= self.last_step):
            raise ValueError("accuracy observations require strictly increasing nonnegative integer steps")
        self.last_step = step
        new_hit = False
        for key, threshold in self.targets.items():
            value = row.get(key)
            if not _number(value):
                continue
            if value < 0:
                raise ValueError(f"accuracy metric {key} must be a nonnegative error")
            observation = self.observations[key]
            observation["observations"] += 1
            observation["last_observed_step"] = step
            observation["last_observed_value"] = value
            old = observation["minimum_observed_value"]
            observation["minimum_observed_value"] = value if old is None else min(old, value)
            if value <= threshold and observation["first_hit_step"] is None:
                observation["first_hit_step"] = step
                new_hit = True
        return new_hit

    def summary(self, terminal_step: int | None, *, scope: str = "all_evaluated_iterates") -> dict:
        output = {}
        for key, threshold in self.targets.items():
            observation = self.observations[key]
            is_final = (terminal_step is not None and observation["last_observed_step"] == terminal_step)
            final = observation["last_observed_value"] if is_final else None
            output[key] = {
                "threshold": threshold, "definition": METRIC_DEFINITIONS.get(key, "study-defined error"),
                **observation, "ever_met": observation["first_hit_step"] is not None,
                "final_value": final, "final_met": final <= threshold if final is not None else None,
                "available": observation["observations"] > 0, "observation_scope": scope,
            }
        return output


def accuracy_from_records(records: list[dict], targets: Mapping[str, float],
                          terminal_step: int | None) -> dict:
    """Legacy reanalysis: a first saved hit is not necessarily a first true hit."""
    tracker = AccuracyTracker(targets)
    for row in records:
        tracker.observe(row)
    return tracker.summary(terminal_step, scope="saved_records_only")


def component_rate_fits(records: list[dict], theory: dict, *, fit: dict | None = None) -> dict:
    """Fit magnitudes, keeping alternating signed modes instead of dropping them.

    The same predeclared window rule is used for every series; reference
    factors are attached only AFTER fitting and never select the window.
    """
    config = dict(fit or {})
    specs = {
        "factor_error": (False, theory.get("rho"), "normal_spectral_upper_bound"),
        "x_factor_error": (False, theory.get("rho"), "normal_spectral_upper_bound"),
        "orbit_error_y": (False, theory.get("rho"), "normal_spectral_upper_bound_for_set_distance"),
        "radial_signed": (True, theory.get("radial_multiplier"), "linearized_mode_not_forced_asymptotic_rate"),
        "angular_signed": (True, theory.get("angular_multiplier"), "linearized_selected_mixing_mode"),
        "radial_normal_norm": (False, theory.get("radial_multiplier"), "linearized_radial_block_nonlinearly_forced"),
        "angular_normal_norm": (False, theory.get("angular_multiplier"), "slowest_linearized_mixing_block"),
        "slow_angular_signed": (True, theory.get("angular_multiplier"), "linearized_selected_mixing_mode"),
    }
    output = {}
    for key, (signed, multiplier, meaning) in specs.items():
        observed = [row for row in records if _number(row.get(key))]
        if not observed:
            continue
        series = [{"t": row["t"], key: abs(row[key]) if signed else row[key]} for row in observed]
        result = estimate_rate(series, key, **config)
        magnitudes = [abs(row[key]) for row in observed]
        reference = abs(multiplier) if multiplier is not None else None
        result.update(
            transform="absolute_value" if signed else "identity",
            definition=METRIC_DEFINITIONS[key],
            reference_factor=reference, reference_signed_multiplier=multiplier if signed else None,
            reference_kind=meaning, zero_samples=sum(v == 0 for v in magnitudes),
            note=("Zero linear term; a log-linear fit does not measure quadratic order."
                  if reference == 0 else
                  "Nonlinear forcing and absent slow modes can change a component's measured rate."),
        )
        if signed:
            eligible = [row for row in observed
                        if result["floor"] < abs(row[key]) < result["ceiling"]]
            if result["status"] == "ok":
                eligible = [row for row in eligible if result["t_start"] <= row["t"] <= result["t_end"]]
            result["sign_changes_in_fit_samples"] = sum(
                left[key]*right[key] < 0 for left, right in zip(eligible, eligible[1:]))
        output[key] = result
    return output


def first_radial_dominance(records: list[dict], *, floor: float = 1e-11) -> int | None:
    """First resolved |radial|>|angular| sample; not a persistence guarantee."""
    for row in records:
        radial, angular = row.get("radial_signed"), row.get("angular_signed")
        if _number(radial) and _number(angular) and abs(radial) > abs(angular) > floor:
            return row["t"]
    return None
