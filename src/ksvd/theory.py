"""Manuscript-v15 predictions, separate from empirical fitting.

Trace constants deliberately use beta=Delta^2/lambda_1*(1-e0/Delta),
exactly as the supplied manuscript, not the proposed sharper replacement.
Local rates are for SUPPORTED Y dynamics; omitted null eigenvalues must
not be silently inserted, especially when k=r.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
from torch import Tensor
from .problem import SpectralProblem
from .metrics import trace_deficit
from .dynamics import ambient_step

FORMULA_VERSION = "manuscript-v15-original-beta"


def positive_mixing(p: SpectralProblem, k: int) -> list[float]:
    p.check_k(k)
    return [1-float(b/a) for a in p.lam[:k] for b in p.lam[k:] if bool(a > b)]


def normal_rate(p: SpectralProblem, k: int, eta: float) -> float:
    if not 0 < eta < 1:
        raise ValueError("normal contraction theorem requires 0 < eta < 1")
    return max([abs(1-2*eta)] + [1-eta*g for g in positive_mixing(p, k)])


def optimal_step(p: SpectralProblem, k: int) -> tuple[float, float]:
    mixing = positive_mixing(p, k)
    if not mixing:
        return .5, 0.
    mu = min(mixing)
    return 2/(2+mu), (2-mu)/(2+mu)


@dataclass(frozen=True)
class TraceCertificate:
    e0: float
    delta: float
    c: float
    d: float
    u: float
    k_minus: float
    k_plus: float
    alpha: float
    beta: float
    decrease: float
    log_zeta: float
    zeta: float
    formula_version: str = FORMULA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    def envelope(self, t: int) -> float:
        if t < 1:
            raise ValueError("certificate envelope starts at t=1")
        return self.e0 * math.exp((t-1)*self.log_zeta)


def trace_certificate(p: SpectralProblem, x0: Tensor, eta: float) -> TraceCertificate:
    if not 0 < eta < 1:
        raise ValueError("certificate requires 0 < eta < 1")
    k = x0.shape[1]
    e0, delta = trace_deficit(p, x0), p.gap(k)
    if not 0 <= e0 < delta:
        raise ValueError("initialization is outside the strict trace band")
    c = float(p.lam[k-1])-e0
    a, L = 1-eta, float(p.lam[0])
    d = 4*a*eta*c
    x1 = ambient_step(p, x0, eta)
    u = max(float(torch.linalg.matrix_norm(x1, ord=2)**2), L**2/d)
    km, kp = eta/(a*u+eta*L), eta/(a*d+eta*c)
    alpha = (c+2*a*d/eta)*km**2/(1+kp**2*L**2)
    beta = delta**2/L*(1-e0/delta)
    dec = min(alpha*beta, .5)
    if dec <= 0 or not math.isfinite(dec):
        raise ArithmeticError("certificate coefficient is not representable; use higher precision")
    logz = math.log1p(-dec)  # stable even when 1-decrease rounds to 1
    return TraceCertificate(e0, delta, c, d, u, km, kp, alpha, beta, dec, logz, 1-dec)


def power_log_bound(p: SpectralProblem, k: int, q: int, failure_probability: float) -> float:
    p.check_k(k)
    if k == p.r or p.gap(k) <= 0 or q < 1 or not 0 < failure_probability < 1:
        raise ValueError("power-entry bound requires k<r, strict gap, q>=1, 0<delta<1")
    return (math.log(16/math.pi) + math.log(float(p.lam[0])) + 5*math.log(k)
            + math.log(p.r-k) - 3*math.log(failure_probability)
            + 2*q*math.log(float(p.lam[k]/p.lam[k-1])))


def sufficient_power(p: SpectralProblem, k: int, theta: float, failure_probability: float) -> int:
    if not 0 < theta < 1:
        raise ValueError("theta must be in (0,1)")
    # q=1 validation also checks all spectral assumptions.
    log_ratio = math.log(float(p.lam[k-1]/p.lam[k])) if k < p.r else 0.
    prefactor = power_log_bound(p, k, 1, failure_probability) + 2*log_ratio
    return max(1, math.ceil(max(0., prefactor-math.log(theta*p.gap(k)))/(2*log_ratio)))


def sample_power_log_bound(p: SpectralProblem, omega: Tensor, k: int, q: int) -> float:
    z = p.U.T @ omega
    if k >= p.r or z.shape[1] != k or q < 1:
        raise ValueError("need k<r and an (n,k) sketch")
    smallest = float(torch.linalg.svdvals(z[:k])[-1])
    tail = float(torch.linalg.matrix_norm(z[k:], ord=2))
    if smallest <= 0 or tail <= 0:
        raise ArithmeticError("sample conditioning factor not representable")
    return (math.log(float(p.lam[0])*k) + 2*q*math.log(float(p.lam[k]/p.lam[k-1]))
            + 2*math.log(tail)-2*math.log(smallest))


def rank_one_annulus(p: SpectralProblem, x0: Tensor, eta: float = .5) -> dict:
    """Jedra--Shah v2 Lemma 1, Eqs. 12--14, expressed with max/min explicitly.

    tau is a norm-annulus entry estimate, NOT entry into the local-rate or
    trace-band region. No claimed prefactor from their Lemma 3 is used.
    """
    if x0.shape[1] != 1 or not 0 < eta < 1:
        raise ValueError("annulus requires k=1 and 0<eta<1")
    norm, L = float(x0.norm()), float(p.lam[0])
    if norm <= 0:
        raise ValueError("nonzero initialization required")
    overlap = abs(float(p.U[:, 0] @ x0[:, 0]))/norm
    ell = float(p.lam[-1])
    a = 2*math.sqrt(eta*(1-eta))*max(overlap, math.sqrt(ell/L))
    b = 2*(1-eta) + math.sqrt(eta/(1-eta))*min(1/overlap if overlap else math.inf, math.sqrt(L/ell))
    z = .5*((1-eta)*norm/math.sqrt(L)+eta*math.sqrt(L)/norm)
    tau = max(1, math.ceil(4/(3*eta)*math.log(z)))
    return {"a_js": a, "b_js": b, "tau_js": tau, "initial_overlap": overlap,
            "source": "Jedra--Shah arXiv:2502.00320v2 Lemma 1, equations 12-14"}
