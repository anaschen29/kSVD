"""Metrics for factors, subspaces, and tied optimal families."""

from __future__ import annotations

import torch
from torch import Tensor

from .problem import SpectralProblem, _check_factor


def procrustes_distance(x: Tensor, target: Tensor) -> Tensor:
    """Return ``min_Q ||X-target Q||_F`` for orthogonal ``Q (k,k)``.

    Both factors have shape ``(n,k)``. If ``target.T @ X=U S V.T``, the
    right-oriented optimizer is ``Q=U V.T``.
    """
    if x.ndim != 2 or target.shape != x.shape:
        raise ValueError("X and target must have identical matrix shapes")
    u, _, vh = torch.linalg.svd(target.T @ x)
    q = u @ vh
    return torch.linalg.norm(x - target @ q)


def _orthonormal_basis(x: Tensor) -> Tensor:
    _check_factor(x, x.shape[0], "factor")
    singular = torch.linalg.svdvals(x)
    threshold = torch.finfo(x.dtype).eps * max(x.shape) * singular[0]
    if singular[-1] <= threshold:
        raise ValueError("factor is numerically rank deficient")
    return torch.linalg.qr(x, mode="reduced").Q


def subspace_distance(x: Tensor, target: Tensor) -> Tensor:
    """Return chordal projector distance ``||P_X-P_target||_F/sqrt(2)``."""
    if x.shape != target.shape:
        raise ValueError("factors must have identical shape")
    qx, qt = _orthonormal_basis(x), _orthonormal_basis(target)
    return torch.linalg.norm(qx @ qx.T - qt @ qt.T) / (2.0**0.5)


def tied_optimal_family_distance(problem: SpectralProblem, x: Tensor, k: int) -> Tensor:
    """Measure distance of ``span(X)`` to the optimal tied-cutoff family.

    Let ``lambda_k`` be the cutoff, ``U_>`` contain directions strictly above
    it, and ``U_>=`` contain the complete tied block as well. The squared metric
    is missing mandatory projector mass plus projector mass outside ``U_>=``.
    It is zero exactly for a ``k``-plane containing ``U_>`` and contained in
    ``U_>=``.
    """
    if not 1 <= k <= problem.r:
        raise ValueError("k must satisfy 1 <= k <= r")
    _check_factor(x, problem.n, "X")
    if x.shape[1] != k:
        raise ValueError("X must have k columns")
    q = _orthonormal_basis(x)
    cutoff = problem.eigenvalues[k - 1]
    above = problem.eigenvectors[:, problem.eigenvalues > cutoff]
    eligible = problem.eigenvectors[:, problem.eigenvalues >= cutoff]
    # Compute residuals directly instead of subtracting nearly equal projector
    # traces.  The trace formulas are mathematically equivalent, but at an
    # optimal family member they can leave an O(eps) positive remainder whose
    # square root is only O(sqrt(eps)).
    missing = torch.zeros((), dtype=x.dtype, device=x.device)
    if above.shape[1]:
        mandatory_residual = above - q @ (q.T @ above)
        missing = torch.sum(mandatory_residual.square())
    eligible_residual = q - eligible @ (eligible.T @ q)
    outside = torch.sum(eligible_residual.square())
    return torch.sqrt(missing + outside)
