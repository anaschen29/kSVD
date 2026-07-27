"""Objectives, potential, derivatives, and local predictions."""

from __future__ import annotations

import torch
from torch import Tensor

from .problem import SpectralProblem, _check_factor


def potential(problem: SpectralProblem, y: Tensor) -> Tensor:
    """Evaluate ``F(Y)=||Y||_F^2/2-logdet(Y.T Lambda Y)/2``."""
    _check_factor(y, problem.r, "Y")
    sign, logabsdet = torch.linalg.slogdet(y.T @ (problem.eigenvalues[:, None] * y))
    if float(sign) <= 0:
        raise ValueError("Y.T Lambda Y must be positive definite")
    return 0.5 * torch.sum(y * y) - 0.5 * logabsdet


def potential_gradient(problem: SpectralProblem, y: Tensor) -> Tensor:
    """Return Euclidean ``grad F(Y)`` with shape ``(r,k)``."""
    from .dynamics import _right_solve

    _check_factor(y, problem.r, "Y")
    weighted = problem.eigenvalues[:, None] * y
    return y - _right_solve(weighted, y.T @ weighted)


def spectral_objective(problem: SpectralProblem, xbar: Tensor) -> Tensor:
    """Evaluate the spectral formula for ``g(Xbar)`` for ``Xbar (r,k)``."""
    _check_factor(xbar, problem.r, "Xbar")
    gram = xbar.T @ xbar
    return 0.25 * (
        torch.sum(problem.eigenvalues.square())
        - 2 * torch.sum(xbar * (problem.eigenvalues[:, None] * xbar))
        + torch.sum(gram.square())
    )


def dense_objective(matrix: Tensor, x: Tensor) -> Tensor:
    """Evaluate ``g(X)=||M-XX.T||_F^2/4`` in ambient coordinates."""
    return 0.25 * torch.sum((matrix - x @ x.T).square())


def objective_gradient(matrix: Tensor, x: Tensor) -> Tensor:
    """Return ``grad g(X)=(XX.T-M)X`` with shape matching ``X (n,k)``."""
    return (x @ x.T - matrix) @ x


def objective_hessian_action(matrix: Tensor, x: Tensor, h: Tensor) -> Tensor:
    """Apply ``nabla^2 g(X)`` to perturbation ``H (n,k)``."""
    if h.shape != x.shape:
        raise ValueError("H must have the same shape as X")
    return h @ (x.T @ x) + x @ (h.T @ x) + (x @ x.T - matrix) @ h


def optimal_objective(problem: SpectralProblem, k: int) -> Tensor:
    """Return ``g_star=sum_{i=k+1}^r lambda_i^2/4`` (one-based formula)."""
    if not 1 <= k <= problem.r:
        raise ValueError("k must satisfy 1 <= k <= r")
    return 0.25 * torch.sum(problem.eigenvalues[k:].square())


def predicted_local_rate(problem: SpectralProblem, k: int, eta: float) -> tuple[float, float]:
    """Return ``(mu_k,rho_eta)`` for a strict eigengap at indices ``k,k+1``."""
    if not 1 <= k < problem.r:
        raise ValueError("local eigengap rate requires 1 <= k < r")
    lk, next_l = float(problem.eigenvalues[k - 1]), float(problem.eigenvalues[k])
    if lk <= next_l:
        raise ValueError("a strict eigengap lambda_k > lambda_(k+1) is required")
    mu = (lk - next_l) / lk
    return mu, max(abs(1 - 2 * eta), 1 - eta * mu)


def locally_optimal_step(problem: SpectralProblem, k: int) -> tuple[float, float]:
    """Return ``(eta_local_star,rho_local_star)`` at a strict cutoff gap."""
    mu, _ = predicted_local_rate(problem, k, 0.0)
    return 2 / (2 + mu), (2 - mu) / (2 + mu)
