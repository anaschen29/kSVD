from __future__ import annotations
from functools import partial
from itertools import product
import torch
from .common import Job, problem
from ..problem import SpectralProblem, generator
from ..runner import RunConfig, run_trajectory


def _run(cfg, eta, seed):
    p = problem(cfg, 2, .2)
    z = torch.randn(p.r, 2, generator=generator(seed, p.U.device), dtype=p.U.dtype, device=p.U.device)
    y = torch.linalg.qr(z).Q @ torch.diag(torch.tensor([1., 1e-4], dtype=z.dtype, device=z.device))
    result = run_trajectory(p, y, RunConfig(eta=eta, max_steps=cfg["max_steps"], coordinates="y"))
    result["theory"] = {"inside_damping_theorem": 0 < eta < 1,
                        "d_eta": 4*eta*(1-eta) if 0 < eta < 1 else None}
    return result


def _cycle():
    p = SpectralProblem.make([1.])
    return run_trajectory(p, torch.tensor([[2.]], dtype=torch.float64),
                          RunConfig(eta=1., max_steps=10, coordinates="y"))


def jobs(cfg):
    for i, (eta, seed) in enumerate(product(cfg["etas"], cfg["seeds"])):
        yield Job(f"damping-{i:05d}", {"eta": eta, "seed": seed}, partial(_run, cfg, eta, seed))
    yield Job("scalar-endpoint-cycle", {"eta": 1., "y0": 2.}, _cycle)
