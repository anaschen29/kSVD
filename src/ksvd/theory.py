"""Certified constants from the convergence manuscript."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .problem import SpectralProblem


_LOG10 = math.log(10.0)
_LOG_MAX = math.log(float.fromhex("0x1.fffffffffffffp+1023"))
_LOG_MIN_SUBNORMAL = math.log(float.fromhex("0x0.0000000000001p-1022"))


def _exp_or_boundary(log_value: float) -> tuple[float, bool, bool]:
    if log_value < _LOG_MIN_SUBNORMAL:
        return 0.0, True, False
    if log_value > _LOG_MAX:
        return math.inf, False, True
    return math.exp(log_value), False, False


def _logsumexp(values: tuple[float, ...]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


@dataclass(frozen=True)
class CertifiedStep:
    """Sufficient (not sharp) threshold and manuscript-v1 sublevel bounds."""

    C: float
    S_C: float
    a_C: float
    Gamma_C: float
    hessian_bound_C: float
    d_C: float
    eta_C: float
    log10_S_C: float
    log10_a_C: float
    log10_Gamma_C: float
    log10_hessian_bound_C: float
    log10_d_C: float
    log10_eta_C: float
    a_C_underflow: bool
    Gamma_C_overflow: bool
    hessian_bound_C_overflow: bool
    d_C_underflow: bool
    eta_C_underflow: bool
    definition_version: str = "manuscript_v1"

    def to_dict(self) -> dict[str, float | bool | str]:
        """Return flat scalar values, logarithms, flags, and definition version."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def certified_step_quantities(problem: SpectralProblem, k: int, C: float) -> CertifiedStep:
    """Compute the manuscript's sufficient threshold for nonempty ``K_C``.

    ``S_C`` is the larger root of ``phi(s)-C=0`` on ``[k,infinity)``, where
    ``phi(s)=s/2-(k/2)log(Ls/k)`` and ``L=lambda_1``.  The theorem requires the
    strict inequality ``0 < eta < eta_C``; ``eta_C`` is not a sharp cutoff.
    Logarithmic forms are computed before exponentiation.
    """
    if not 1 <= k <= problem.r or not math.isfinite(C):
        raise ValueError("require 1 <= k <= r and finite C")
    L = float(problem.eigenvalues[0])

    def residual(s: float) -> float:
        return 0.5 * s - 0.5 * k * math.log(L * s / k) - C

    at_k = residual(float(k))
    if at_k > 0:
        raise ValueError("C < phi(k), so the potential sublevel certificate is empty")
    if at_k == 0:
        S = float(k)
    else:
        lo, hi = float(k), max(float(k + 1), 2 * (abs(C) + k + abs(k * math.log(L)) + 1))
        while residual(hi) <= 0:
            hi *= 2
        for _ in range(1075):
            mid = (lo + hi) / 2
            if mid == lo or mid == hi:
                break
            if residual(mid) <= 0:
                lo = mid
            else:
                hi = mid
        S = (lo + hi) / 2

    log_S = math.log(S)
    log_L = math.log(L)
    log_a = -2 * C - (k - 1) * (log_L + log_S)
    a, a_underflow, _ = _exp_or_boundary(log_a)
    # Gamma_C = sqrt(S_C) * (1 + L/a_C).
    log_gamma = 0.5 * log_S + _logsumexp((0.0, log_L - log_a))
    gamma, _, gamma_overflow = _exp_or_boundary(log_gamma)
    log_hessian = _logsumexp(
        (0.0, math.log(2.0) + log_L - log_a,
         math.log(8.0) + 2 * log_L + 2 * math.log(math.sqrt(S) + 1) - 2 * log_a)
    )
    hessian, _, hessian_overflow = _exp_or_boundary(log_hessian)
    log_d = min(0.0, log_a - math.log(4.0) - log_L - math.log(math.sqrt(S) + 1))
    d, d_underflow, _ = _exp_or_boundary(log_d)
    log_eta = min(0.0, log_d - log_gamma, -log_hessian)
    eta, eta_underflow, _ = _exp_or_boundary(log_eta)
    return CertifiedStep(
        C, S, a, gamma, hessian, d, eta,
        log_S / _LOG10, log_a / _LOG10, log_gamma / _LOG10,
        log_hessian / _LOG10, log_d / _LOG10, log_eta / _LOG10,
        a_underflow, gamma_overflow, hessian_overflow, d_underflow, eta_underflow,
    )
