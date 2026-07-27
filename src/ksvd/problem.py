"""Spectral problem representation and coordinate transformations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _floating_matrix(value: Tensor, name: str) -> None:
    if value.ndim != 2 or not value.is_floating_point():
        raise ValueError(f"{name} must be a rank-2 floating-point tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class SpectralProblem:
    """PSD matrix represented as ``M=U diag(lambda) U.T``.

    ``U`` has shape ``(n, r)`` with orthonormal columns and ``eigenvalues`` has
    shape ``(r,)``, is strictly positive, and is nonincreasing. Mathematical
    experiments use float64 unless the caller explicitly supplies another
    floating dtype.
    """

    eigenvalues: Tensor
    eigenvectors: Tensor

    def __post_init__(self) -> None:
        lam, u = self.eigenvalues, self.eigenvectors
        if lam.ndim != 1 or not lam.is_floating_point() or lam.numel() == 0:
            raise ValueError("eigenvalues must be a nonempty floating vector")
        _floating_matrix(u, "eigenvectors")
        if u.shape[1] != lam.numel() or u.dtype != lam.dtype or u.device != lam.device:
            raise ValueError("eigenvectors and eigenvalues must have compatible shape, dtype, and device")
        if not bool(torch.isfinite(lam).all()) or not bool((lam > 0).all()):
            raise ValueError("support eigenvalues must be finite and strictly positive")
        if lam.numel() > 1 and not bool((lam[:-1] >= lam[1:]).all()):
            raise ValueError("eigenvalues must be nonincreasing")
        eye = torch.eye(lam.numel(), dtype=lam.dtype, device=lam.device)
        tol = 100 * torch.finfo(lam.dtype).eps * max(u.shape)
        if not torch.allclose(u.T @ u, eye, atol=tol, rtol=tol):
            raise ValueError("eigenvector columns must be orthonormal")

    @property
    def n(self) -> int:
        """Ambient dimension ``n``."""
        return self.eigenvectors.shape[0]

    @property
    def r(self) -> int:
        """Support rank ``r``."""
        return self.eigenvalues.numel()

    @property
    def matrix(self) -> Tensor:
        """Return the dense ambient matrix with shape ``(n, n)``."""
        return (self.eigenvectors * self.eigenvalues) @ self.eigenvectors.T

    @classmethod
    def diagonal(cls, eigenvalues: Tensor) -> "SpectralProblem":
        """Build a diagonal problem from descending positive ``(r,)`` values."""
        if eigenvalues.ndim != 1:
            raise ValueError("eigenvalues must be a vector")
        return cls(eigenvalues, torch.eye(eigenvalues.numel(), dtype=eigenvalues.dtype, device=eigenvalues.device))


def spectral_problem(
    eigenvalues: Tensor | list[float],
    *,
    eigenvectors: Tensor | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> SpectralProblem:
    """Construct a validated spectral problem, defaulting to float64."""
    lam = torch.as_tensor(eigenvalues, dtype=dtype, device=device)
    u = torch.eye(lam.numel(), dtype=dtype, device=device) if eigenvectors is None else eigenvectors.to(dtype=dtype, device=device)
    return SpectralProblem(lam, u)


def _check_factor(z: Tensor, rows: int, name: str) -> None:
    _floating_matrix(z, name)
    if z.shape[0] != rows or z.shape[1] == 0 or z.shape[1] > rows:
        raise ValueError(f"{name} must have shape ({rows}, k), 1 <= k <= {rows}")


def y_to_xbar(problem: SpectralProblem, y: Tensor) -> Tensor:
    """Map ``Y (r,k)`` to ``Xbar=Lambda^(1/2)Y (r,k)``."""
    _check_factor(y, problem.r, "Y")
    return problem.eigenvalues.sqrt()[:, None] * y


def xbar_to_y(problem: SpectralProblem, xbar: Tensor) -> Tensor:
    """Map support ``Xbar (r,k)`` to ``Y=Lambda^(-1/2)Xbar (r,k)``."""
    _check_factor(xbar, problem.r, "Xbar")
    return xbar / problem.eigenvalues.sqrt()[:, None]


def xbar_to_x(problem: SpectralProblem, xbar: Tensor) -> Tensor:
    """Embed support ``Xbar (r,k)`` as ambient ``X=U Xbar (n,k)``."""
    _check_factor(xbar, problem.r, "Xbar")
    return problem.eigenvectors @ xbar


def x_to_xbar(problem: SpectralProblem, x: Tensor, *, check_support: bool = True) -> Tensor:
    """Project ambient ``X (n,k)`` to support coordinates ``U.T X (r,k)``.

    If ``check_support`` is true, reject a non-negligible null-space component.
    """
    _check_factor(x, problem.n, "X")
    xbar = problem.eigenvectors.T @ x
    if check_support:
        residual = x - problem.eigenvectors @ xbar
        tol = 100 * torch.finfo(x.dtype).eps * max(x.shape) * max(1.0, float(torch.linalg.norm(x)))
        if float(torch.linalg.norm(residual)) > tol:
            raise ValueError("X is not in range(M)")
    return xbar
