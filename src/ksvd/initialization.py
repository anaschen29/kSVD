"""Raw power sketches, graph starts, and controlled normal perturbations."""
from __future__ import annotations
import math
import torch
from torch import Tensor
from .problem import SpectralProblem, generator
from .metrics import basis


def gaussian(p: SpectralProblem, k: int, seed: int) -> Tensor:
    p.check_k(k)
    return torch.randn(p.n, k, generator=generator(seed, p.U.device),
                       dtype=p.lam.dtype, device=p.U.device)


def power_sketch(p: SpectralProblem, omega: Tensor, q: int, *, basis_only: bool = False) -> Tensor:
    """Raw M^q Omega by default. QR mode represents ONLY its column space.

    basis_only=True is appropriate for entry measurements. Starting PGD from
    its return value changes the radial initialization and must be labeled.
    """
    if not isinstance(q, int) or q < 1:
        raise ValueError("q must be a positive integer")
    x = omega.clone()
    for _ in range(q):
        x = p.apply(x)
        if basis_only:
            x = basis(x)
    return x


def controlled_trace_start(p: SpectralProblem, k: int, theta: float, *,
                           null_mix: float = 0., condition: float = 1.,
                           scale: float = 1., seed: int = 0) -> Tensor:
    """Construct e(X0)=theta*Delta with an optional genuinely ambient tilt.

    null_mix is the squared nullspace fraction in the omitted direction.
    condition is the intended condition number of G0, not of X0.
    A random right SPD factor generally does not commute with the compression.
    """
    p.check_k(k)
    if p.gap(k) <= 0 or theta < 0 or not 0 <= null_mix <= 1:
        raise ValueError("need a strict gap, theta >= 0, null_mix in [0,1]")
    if not all(math.isfinite(v) for v in (theta, null_mix, condition, scale)) or condition < 1 or scale <= 0:
        raise ValueError("invalid scale or conditioning")
    if k == 1 and condition != 1:
        raise ValueError("a one-column Gram matrix has condition number one")
    if k == p.r and null_mix != 1:
        raise ValueError("k=r requires a purely null omitted direction")
    v = torch.zeros(p.n, dtype=p.U.dtype, device=p.U.device)
    omitted_rayleigh = 0.
    if null_mix < 1:
        v += math.sqrt(1-null_mix) * p.U[:, k]
        omitted_rayleigh = (1-null_mix) * float(p.lam[k])
    if null_mix > 0:
        if p.n == p.r:
            raise ValueError("a nullspace is required")
        # Choose a stable projected coordinate vector, not an approximate eigenvector.
        eye = torch.eye(p.n, dtype=p.U.dtype, device=p.U.device)
        null = eye-p.U @ p.U.T
        j = int(null.square().sum(0).argmax())
        v += math.sqrt(null_mix) * null[:, j]/null[:, j].norm()
    sine2 = theta*p.gap(k)/(float(p.lam[k-1])-omitted_rayleigh)
    if not 0 <= sine2 <= 1:
        raise ValueError("requested deficit is not attainable by this one-direction tilt")
    q = p.U[:, :k].clone()
    q[:, -1] = math.sqrt(1-sine2)*q[:, -1] + math.sqrt(sine2)*v
    gen = generator(seed, p.U.device)
    rot = torch.linalg.qr(torch.randn(k, k, generator=gen, dtype=p.U.dtype, device=p.U.device)).Q
    # Geometrically centered Gram eigenvalues: determinant unaffected by condition.
    s = torch.logspace(-.25*math.log10(condition), .25*math.log10(condition), k,
                      dtype=p.U.dtype, device=p.U.device)
    if k == 1:
        s.fill_(1.)
    return scale * q @ ((rot*s) @ rot.T)


def normal_start(p: SpectralProblem, k: int, mode: str, amplitude: float = .01) -> Tensor:
    """Canonical frame plus a known radial and/or slowest mixing direction."""
    if amplitude <= 0 or not math.isfinite(amplitude):
        raise ValueError("amplitude must be positive")
    y = p.frame(k).clone()
    if mode not in {"radial", "angular", "mixed"}:
        raise ValueError("mode must be radial, angular, or mixed")
    if mode in {"radial", "mixed"}:
        y[k-1, k-1] += amplitude
    if mode in {"angular", "mixed"}:
        if k == p.r or p.gap(k) <= 0:
            raise ValueError("angular mode needs a strict positive-support cutoff")
        y[k, k-1] += amplitude
    return y


def coupled_normal_start(p: SpectralProblem, k: int, amplitude: float = .01,
                         seed: int = 0) -> Tensor:
    """Dense multi-column normal perturbation in Y; not a rank-one embedding.

    The symmetric top block and dense omitted block each have Frobenius norm
    amplitude/sqrt(2). Thus ||Y0-E_star||_F=amplitude in exact arithmetic.
    The small-radius restriction guarantees an invertible selected block.
    No QR or normalization is subsequently applied to the evolving factor.
    """
    p.check_k(k)
    if not 2 <= k < p.r or p.gap(k) <= 0:
        raise ValueError("coupled normal starts require 2<=k<r and a strict cutoff gap")
    if not math.isfinite(amplitude) or not 0 < amplitude < .25:
        raise ValueError("coupled amplitude must be in (0, .25)")
    gen = generator(seed, p.lam.device)
    radial = torch.randn(k, k, generator=gen, dtype=p.lam.dtype, device=p.lam.device)
    radial = (radial + radial.T)/2
    angular = torch.randn(p.r-k, k, generator=gen, dtype=p.lam.dtype, device=p.lam.device)
    y = p.frame(k).clone()
    y[:k] += (amplitude/math.sqrt(2)) * radial/radial.norm()
    y[k:] += (amplitude/math.sqrt(2)) * angular/angular.norm()
    return y
