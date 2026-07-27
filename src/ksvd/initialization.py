"""Deterministic initialization families in reduced coordinates."""

from __future__ import annotations

import torch
from torch import Tensor

from .problem import SpectralProblem


def random_y(problem: SpectralProblem, k: int, seed: int, *, scale: float = 1.0) -> Tensor:
    """Draw reproducible Gaussian full-rank ``Y (r,k)`` on the problem device."""
    if not 1 <= k <= problem.r or scale <= 0:
        raise ValueError("require 1 <= k <= r and positive scale")
    generator = torch.Generator(device=problem.eigenvalues.device).manual_seed(seed)
    y = scale * torch.randn(problem.r, k, generator=generator, dtype=problem.eigenvalues.dtype, device=problem.eigenvalues.device)
    if int(torch.linalg.matrix_rank(y)) != k:
        raise RuntimeError("random initialization unexpectedly lost rank")
    return y


def aligned_y(problem: SpectralProblem, indices: list[int] | Tensor) -> Tensor:
    """Construct ``Y (r,k)`` whose columns are selected zero-based eigenvectors.

    Scaling by ``lambda_i^-1/2`` makes the corresponding ``Xbar`` columns unit
    vectors scaled by one; use ``minimizer_y`` for objective minimizers.
    """
    idx = torch.as_tensor(indices, dtype=torch.long, device=problem.eigenvalues.device)
    if idx.ndim != 1 or idx.numel() == 0 or int(idx.min()) < 0 or int(idx.max()) >= problem.r or idx.unique().numel() != idx.numel():
        raise ValueError("indices must be distinct valid support indices")
    y = torch.zeros(problem.r, idx.numel(), dtype=problem.eigenvalues.dtype, device=problem.eigenvalues.device)
    y[idx, torch.arange(idx.numel(), device=idx.device)] = 1 / problem.eigenvalues[idx].sqrt()
    return y


def minimizer_y(problem: SpectralProblem, k: int) -> Tensor:
    """Construct the canonical top-``k`` minimizer ``Y (r,k)``."""
    if not 1 <= k <= problem.r:
        raise ValueError("k must satisfy 1 <= k <= r")
    y = torch.zeros(problem.r, k, dtype=problem.eigenvalues.dtype, device=problem.eigenvalues.device)
    y[:k, :] = torch.eye(k, dtype=y.dtype, device=y.device)
    return y


def perturbed_minimizer_y(problem: SpectralProblem, k: int, seed: int, noise: float) -> Tensor:
    """Return the canonical minimizer plus seeded Gaussian noise in ``Y``."""
    if noise < 0:
        raise ValueError("noise must be nonnegative")
    return minimizer_y(problem, k) + noise * random_y(problem, k, seed)
