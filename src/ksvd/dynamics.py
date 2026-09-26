"""The unmodified paper iteration. No QR normalization or Gram regularization."""
from __future__ import annotations
import math
import torch
from torch import Tensor
from .problem import SpectralProblem, check_factor


def right_solve(numerator: Tensor, gram: Tensor) -> Tensor:
    """numerator @ gram^{-1}, using an SPD solve, never an explicit inverse.

    A failed Cholesky is reported, not repaired by adding a ridge. This
    preserves the distinction between exact rank preservation and numerical
    failure of a finite-precision computation.
    """
    chol, info = torch.linalg.cholesky_ex(gram)
    if bool((info != 0).any()):
        raise ArithmeticError("Gram Cholesky failed (numerical rank loss)")
    return torch.cholesky_solve(numerator.T, chol).T


def _eta(eta: float) -> None:
    # Endpoints / out-of-theorem steps are allowed for explicitly labeled stress tests.
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be finite and positive")


def ambient_step(p: SpectralProblem, x: Tensor, eta: float) -> Tensor:
    _eta(eta)
    check_factor(x, p.n)
    p.check_k(x.shape[1])
    return (1-eta) * x + eta * right_solve(p.apply(x), x.T @ x)


def reduced_step(p: SpectralProblem, y: Tensor, eta: float) -> Tensor:
    _eta(eta)
    check_factor(y, p.r)
    ly = p.lam[:, None] * y
    return (1-eta) * y + eta * right_solve(ly, y.T @ ly)


def gradient_f(p: SpectralProblem, y: Tensor) -> Tensor:
    ly = p.lam[:, None] * y
    return y - right_solve(ly, y.T @ ly)


def hessian_f(p: SpectralProblem, y: Tensor, h: Tensor) -> Tensor:
    """Analytic Hessian action from Appendix B; useful for mode tests."""
    if h.shape != y.shape:
        raise ValueError("H and Y must have the same shape")
    ly, lh = p.lam[:, None] * y, p.lam[:, None] * h
    a = y.T @ ly
    return h - right_solve(lh, a) + right_solve(right_solve(ly, a) @ (h.T @ ly + y.T @ lh), a)


def potential_f(p: SpectralProblem, y: Tensor) -> Tensor:
    sign, logdet = torch.linalg.slogdet(y.T @ (p.lam[:, None] * y))
    if float(sign) <= 0:
        raise ArithmeticError("potential outside positive-definite domain")
    return .5 * (y.square().sum() - logdet)


def quotient(p: SpectralProblem, y: Tensor) -> Tensor:
    signs, logs = zip(*(torch.linalg.slogdet(a) for a in
                       (y.T @ (p.lam[:, None] * y), y.T @ y)))
    if any(float(s) <= 0 for s in signs):
        raise ArithmeticError("quotient outside full-rank domain")
    return logs[0] - logs[1]
