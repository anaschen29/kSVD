from __future__ import annotations
import math
from functools import partial
from itertools import product
import torch
from .common import Job
from ..problem import SpectralProblem
from ..initialization import gaussian, power_sketch
from ..metrics import basis, procrustes_distance, manifold_residual
from ..theory import normal_rate
from ..runner import RunConfig, run_trajectory


def _run(cfg, splitting, seed):
    if not 0 <= splitting < 1:
        raise ValueError("tie splitting must be in [0,1)")
    p = SpectralProblem.make([3., 2.+splitting, 2., 1.], device=cfg.get("device", "cpu"))
    tied = SpectralProblem.make([3., 2., 2., 1.], device=cfg.get("device", "cpu"))
    y = p.to_y(power_sketch(p, gaussian(p, 2, seed), 1))
    def extra(s, x, t):
        q = basis(s)
        block = q[1:3] @ q[1:3].T
        vals, vecs = torch.linalg.eigh(block)
        angle = float(torch.atan2(vecs[1, -1], vecs[0, -1])) % math.pi
        return {"reference_orbit_distance": procrustes_distance(s, p.frame(2)),
                "near_tied_family_residual": manifold_residual(tied, s),
                "selected_angle": angle, "angle_identifiable": bool(vals[-1]-vals[0] > 1e-10)}
    result = run_trajectory(p, y, RunConfig(eta=cfg["eta"], max_steps=cfg["max_steps"], coordinates="y"), extra=extra)
    result["theory"] = {"rho": normal_rate(p, 2, cfg["eta"]),
                        "tie_split_curvature": 1-2/(2+splitting),
                        "metric_note": "manifold_residual is not an exact distance; reference orbit is one choice"}
    return result


def jobs(cfg):
    for i, (splitting, seed) in enumerate(product(cfg["splittings"], cfg["seeds"])):
        yield Job(f"manifold-{i:05d}", {"splitting": splitting, "seed": seed},
                  partial(_run, cfg, splitting, seed))
