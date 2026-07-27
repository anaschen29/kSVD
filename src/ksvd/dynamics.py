"""Equivalent PGD dynamics in the three coordinate systems."""

from __future__ import annotations

import torch
from torch import Tensor

from .problem import SpectralProblem, _check_factor


def _right_solve(numerator: Tensor, gram: Tensor) -> Tensor:
    """Evaluate ``numerator @ gram^{-1}`` without forming an inverse."""
    singular = torch.linalg.svdvals(gram)
    threshold = torch.finfo(gram.dtype).eps * gram.shape[0] * singular[0]
    if singular[-1] <= threshold:
        raise ValueError("factor Gram matrix is numerically rank deficient")
    return torch.linalg.solve(gram, numerator.T).T


def reduced_y_update(problem: SpectralProblem, y: Tensor, eta: float) -> Tensor:
    """Apply ``Y+ = (1-eta)Y + eta Lambda Y(Y.T Lambda Y)^-1``.

    ``Y`` has shape ``(r,k)`` and must have numerical column rank ``k``.
    """
    _check_factor(y, problem.r, "Y")
    gram = y.T @ (problem.eigenvalues[:, None] * y)
    return (1.0 - eta) * y + eta * _right_solve(problem.eigenvalues[:, None] * y, gram)


def reduced_xbar_update(problem: SpectralProblem, xbar: Tensor, eta: float) -> Tensor:
    """Apply support PGD to ``Xbar (r,k)``: ``Xbar-eta grad g Gram^-1``."""
    _check_factor(xbar, problem.r, "Xbar")
    gram = xbar.T @ xbar
    gradient = xbar @ gram - problem.eigenvalues[:, None] * xbar
    return xbar - eta * _right_solve(gradient, gram)


def ambient_update(matrix: Tensor, x: Tensor, eta: float) -> Tensor:
    """Apply ambient PGD to symmetric ``M (n,n)`` and ``X (n,k)``."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("M must be a square matrix")
    _check_factor(x, matrix.shape[0], "X")
    tol = 100 * torch.finfo(matrix.dtype).eps * matrix.shape[0]
    if not torch.allclose(matrix, matrix.T, atol=tol, rtol=tol):
        raise ValueError("M must be symmetric")
    gram = x.T @ x
    gradient = (x @ x.T - matrix) @ x
    return x - eta * _right_solve(gradient, gram)
