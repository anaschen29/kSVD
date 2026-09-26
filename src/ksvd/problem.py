"""PSD test problems and supported coordinates. All generators use float64."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor


def generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def check_factor(x: Tensor, rows: int) -> None:
    if x.ndim != 2 or x.shape[0] != rows or not 1 <= x.shape[1] <= rows:
        raise ValueError(f"expected a ({rows}, k) factor, 1 <= k <= {rows}")
    if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
        raise ValueError("factor must be finite and floating point")


@dataclass(frozen=True)
class SpectralProblem:
    """M = U diag(lam) U.T; lam contains only positive eigenvalues (descending).

    U has shape (n,r). The omitted ambient eigenvalues are exactly zero.
    Spectral multiplicities are specified by the caller, not inferred by a
    numerical eigensolver or a tolerance that could collapse small gaps.
    """
    lam: Tensor
    U: Tensor

    def __post_init__(self) -> None:
        if self.lam.ndim != 1 or self.lam.numel() == 0:
            raise ValueError("lam must be a nonempty vector")
        if not self.lam.is_floating_point() or not bool(torch.isfinite(self.lam).all()):
            raise ValueError("lam must be finite and floating point")
        if not bool((self.lam > 0).all()) or not bool((self.lam[:-1] >= self.lam[1:]).all()):
            raise ValueError("positive eigenvalues must be descending")
        if self.U.ndim != 2 or self.U.shape[1] != self.r or self.n < self.r:
            raise ValueError("U must have shape (n,r), n >= r")
        if self.U.dtype != self.lam.dtype or self.U.device != self.lam.device:
            raise ValueError("U and lam must share dtype and device")
        tol = 100 * torch.finfo(self.lam.dtype).eps * self.n
        eye = torch.eye(self.r, dtype=self.U.dtype, device=self.U.device)
        if not torch.allclose(self.U.T @ self.U, eye, atol=tol, rtol=tol):
            raise ValueError("U must have orthonormal columns")

    @property
    def n(self) -> int:
        return self.U.shape[0]

    @property
    def r(self) -> int:
        return self.lam.numel()

    @classmethod
    def make(cls, eigenvalues, *, n: int | None = None, seed: int | None = None,
             device: str = "cpu") -> SpectralProblem:
        lam = torch.as_tensor(eigenvalues, dtype=torch.float64, device=device)
        n = lam.numel() if n is None else n
        if n < lam.numel():
            raise ValueError("n must be >= support rank")
        if seed is None:
            u = torch.eye(n, dtype=lam.dtype, device=device)[:, :lam.numel()]
        else:
            z = torch.randn(n, lam.numel(), generator=generator(seed, device),
                            dtype=lam.dtype, device=device)
            u = torch.linalg.qr(z, mode="reduced").Q
        return cls(lam, u)

    def check_k(self, k: int) -> None:
        if not 1 <= k <= self.r:
            raise ValueError("require 1 <= k <= r")

    def apply(self, x: Tensor) -> Tensor:
        check_factor(x, self.n)
        return self.U @ (self.lam[:, None] * (self.U.T @ x))

    def dense(self) -> Tensor:
        return (self.U * self.lam) @ self.U.T

    def to_x(self, y: Tensor) -> Tensor:
        check_factor(y, self.r)
        return self.U @ (self.lam.sqrt()[:, None] * y)

    def to_y(self, x: Tensor, *, check_support: bool = True) -> Tensor:
        check_factor(x, self.n)
        z = self.U.T @ x
        if check_support:
            tol = 100 * torch.finfo(x.dtype).eps * self.n * max(1., float(x.norm()))
            if float((x - self.U @ z).norm()) > tol:
                raise ValueError("ambient factor is not supported; cannot use Y dynamics")
        return z / self.lam.sqrt()[:, None]

    def gap(self, k: int) -> float:
        self.check_k(k)
        return float(self.lam[k-1] - (self.lam[k] if k < self.r else 0.))

    def frame(self, k: int) -> Tensor:
        self.check_k(k)
        return torch.eye(self.r, dtype=self.lam.dtype, device=self.lam.device)[:, :k]

    def target(self, k: int) -> Tensor:
        return self.to_x(self.frame(k))

    def optimal_value(self, k: int) -> float:
        self.check_k(k)
        return float(self.lam[k:].square().sum()) / 4
