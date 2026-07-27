"""Certified sublevel quantities derived from the potential bounds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .problem import SpectralProblem


@dataclass(frozen=True)
class CertifiedStep:
    """Conservative constants for the potential sublevel ``{F <= C}``.

    ``S_C`` bounds ``||Y||_F^2``; ``a_C`` bounds
    ``lambda_min(Y.T Lambda Y)``; ``Gamma_C`` bounds ``||grad F||_F``;
    ``L_C`` bounds the Hessian operator norm; ``d_C`` is a Gram-boundary
    displacement; and ``eta_C=min(1/L_C,d_C/Gamma_C)``.
    """

    C: float
    S_C: float
    a_C: float
    Gamma_C: float
    L_C: float
    d_C: float
    eta_C: float

    def to_dict(self) -> dict[str, float]:
        """Serialize the constants as plain floats."""
        return asdict(self)


def certified_step_quantities(problem: SpectralProblem, k: int, C: float) -> CertifiedStep:
    """Return conservative certified quantities for ``F(Y) <= C``.

    The AM-GM determinant bound gives
    ``F >= h(s)=s/2-k/2 log(lambda_1 s/k)`` for ``s=||Y||_F^2``.
    ``S_C`` is the upper solution of ``h(s)=C``. The remaining bounds follow
    from ``det(A)>=exp(-2C)`` and ``lambda_max(A)<=lambda_1 S_C``.
    These independently derived certificates are intentionally isolated from
    the unavailable paper-specific definitions named in the source document.
    """
    if not 1 <= k <= problem.r or not math.isfinite(C):
        raise ValueError("require 1 <= k <= r and finite C")
    lmax = float(problem.eigenvalues[0])

    def h(s: float) -> float:
        return 0.5 * s - 0.5 * k * math.log(lmax * s / k)

    if C + 1e-14 < h(float(k)):
        raise ValueError("C lies below the determinant-bound minimum")
    lo, hi = float(k), max(float(k + 1), 2 * (C + k + abs(k * math.log(lmax)) + 1))
    while h(hi) <= C:
        hi *= 2
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if h(mid) <= C:
            lo = mid
        else:
            hi = mid
    s_bound = hi
    a = math.exp(-2 * C) / (lmax * s_bound) ** (k - 1)
    gamma = math.sqrt(s_bound) * (1 + lmax * math.sqrt(k) / a)
    lipschitz = 1 + lmax / a + 2 * lmax**2 * s_bound / a**2
    distance = min(1.0, a / (4 * lmax * (math.sqrt(s_bound) + 1)))
    eta = min(1 / lipschitz, distance / gamma)
    return CertifiedStep(C, s_bound, a, gamma, lipschitz, distance, eta)
