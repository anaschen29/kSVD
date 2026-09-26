"""Stable diagnostics. Measurement QR never replaces the evolving factor."""
from __future__ import annotations
import math
import torch
from torch import Tensor
from .problem import SpectralProblem, check_factor


def basis(x: Tensor) -> Tensor:
    check_factor(x, x.shape[0])
    s = torch.linalg.svdvals(x)
    if float(s[-1]) <= torch.finfo(x.dtype).eps * max(x.shape) * float(s[0]):
        raise ArithmeticError("factor is numerically rank deficient")
    return torch.linalg.qr(x, mode="reduced").Q


def polar_frame(x: Tensor) -> Tensor:
    """Q=X(X.T X)^(-1/2) via SVD, required when paired with G=X.T X.

    A QR basis has the same projector but cannot be substituted into formulas
    involving this unrotated G (notably the exact ambient trace identity).
    """
    check_factor(x, x.shape[0])
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    if float(s[-1]) <= torch.finfo(x.dtype).eps * max(x.shape) * float(s[0]):
        raise ArithmeticError("factor is numerically rank deficient")
    return u @ vh



def _deficit(p: SpectralProblem, q: Tensor, k: int, power: int = 1) -> Tensor:
    """Nonnegative weighted projector residuals, avoiding tau - captured trace."""
    lam = p.lam ** power
    cut = lam[k-1]
    missing = p.U[:, :k] - q @ (q.T @ p.U[:, :k])
    tail = p.U[:, k:].T @ q
    null = q - p.U @ (p.U.T @ q)
    return ((lam[:k]-cut) * missing.square().sum(0)).sum() + \
        ((cut-lam[k:]) * tail.square().sum(1)).sum() + cut * null.square().sum()


def trace_deficit(p: SpectralProblem, x: Tensor) -> float:
    p.check_k(x.shape[1])
    return float(_deficit(p, basis(x), x.shape[1]))


def residual(p: SpectralProblem, x: Tensor) -> Tensor:
    q = basis(x)
    mq = p.apply(q)
    return mq - q @ (q.T @ mq)


def objective_gap(p: SpectralProblem, x: Tensor) -> float:
    """Exact identity: 4(g-g*) = e_{M^2}(Q) + ||R||_F^2 + ||S-C||_F^2.

    S = (Q.T X)(Q.T X).T. This is not merely product error squared,
    and does not subtract two nearly equal objective values.
    """
    q = basis(x)
    mq = p.apply(q)
    c = q.T @ mq
    r = mq - q @ c
    v = q.T @ x
    return float(_deficit(p, q, x.shape[1], 2) + r.square().sum()
                 + (v @ v.T - c).square().sum()) / 4


def procrustes_distance(x: Tensor, target: Tensor) -> float:
    """Distance to a fixed right-orthogonal orbit, NOT the full tied family."""
    if x.shape != target.shape:
        raise ValueError("factor shapes must agree")
    u, _, vh = torch.linalg.svd(target.T @ x)
    return float((x - target @ (u @ vh)).norm())


def manifold_residual(p: SpectralProblem, y: Tensor) -> float:
    """Zero exactly on M_*; a residual, not an exact nearest-point distance."""
    k = y.shape[1]
    q = basis(y)
    cut = p.lam[k-1]
    eye = torch.eye(p.r, dtype=y.dtype, device=y.device)
    mandatory = eye[:, p.lam > cut]
    missing = mandatory - q @ (q.T @ mandatory)
    outside = q[p.lam < cut]
    gram_error = y.T @ y - torch.eye(k, dtype=y.dtype, device=y.device)
    return float((gram_error.square().sum() + missing.square().sum()
                  + outside.square().sum()).sqrt())


def snapshot(p: SpectralProblem, x: Tensor, *, y: Tensor | None = None) -> dict:
    q = basis(x)
    s = torch.linalg.svdvals(x)
    k = x.shape[1]
    r = residual(p, x)
    e = float(_deficit(p, q, k))
    out = {"trace_deficit": e, "objective_gap": objective_gap(p, x),
           "gram_min": float(s[-1]**2), "gram_max": float(s[0]**2),
           "gram_condition": float((s[0]/s[-1])**2), "residual_norm": float(r.norm()),
           "null_norm": float((x - p.U @ (p.U.T @ x)).norm())}
    if p.gap(k) > 0:
        target = p.target(k)
        product = target @ target.T
        out["product_error"] = float((x @ x.T-product).norm()/product.norm())
        out["subspace_error"] = float((q - p.U[:, :k] @ (p.U[:, :k].T @ q)).norm())
        out["trace_over_gap"] = e/p.gap(k)
    if y is not None:
        out["radial_error"] = float((y.T @ y - torch.eye(k, dtype=y.dtype, device=y.device)).norm())
        out["manifold_residual"] = manifold_residual(p, y)
    return out


def estimate_rate(records: list[dict], key: str, *, floor: float = 1e-11,
                  ceiling: float = 1e-2, min_points: int = 8, window: int = 30) -> dict:
    """Fit a predeclared late error window, never fit theoretical constants.

    Returns 'insufficient_window' rather than inventing a rate. This estimate
    is a log-linear fit, not a numerical proof of a root-asymptotic limit.
    """
    if not 0 < floor < ceiling or min_points < 3 or window < min_points:
        raise ValueError("invalid fitting window")
    eligible = [r for r in records if key in r and math.isfinite(r[key]) and floor < r[key] < ceiling]
    if len(eligible) < min_points:
        return {"status": "insufficient_window", "points": len(eligible), "metric": key,
                "floor": floor, "ceiling": ceiling}
    rows = eligible[-window:]
    t = torch.tensor([r["t"] for r in rows], dtype=torch.float64)
    z = torch.tensor([math.log(r[key]) for r in rows], dtype=torch.float64)
    tc, zc = t-t.mean(), z-z.mean()
    slope = float((tc*zc).sum()/tc.square().sum())
    pred = z.mean() + slope*tc
    sst = float(zc.square().sum())
    return {"status": "ok", "metric": key, "rho_fit": math.exp(slope),
            "slope": slope, "r_squared": 1-float((z-pred).square().sum())/sst if sst else None,
            "t_start": int(t[0]), "t_end": int(t[-1]), "points": len(rows),
            "floor": floor, "ceiling": ceiling}
