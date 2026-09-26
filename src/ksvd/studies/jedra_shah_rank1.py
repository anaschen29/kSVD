from __future__ import annotations
import math
from functools import partial
from itertools import product
from .common import Job, problem
from ..initialization import gaussian, power_sketch
from ..metrics import estimate_rate
from ..theory import rank_one_annulus
from ..runner import RunConfig, run_trajectory


def _run(cfg, mu, seed, overlap):
    p = problem(cfg, 1, mu)
    x = power_sketch(p, gaussian(p, 1, seed), 1)
    if overlap is not None:
        if not 0 < overlap < 1 or p.r < 3:
            raise ValueError("controlled comparison needs 0<overlap<1 and r>=3")
        # Both u2 (slow mode) and a lower mode (possible outside-band start).
        tail = (p.U[:, 1]+p.U[:, -1])/math.sqrt(2)
        x = (overlap*p.U[:, 0]+math.sqrt(1-overlap**2)*tail)[:, None]
    annulus = rank_one_annulus(p, x)
    target = p.target(1)
    def extra(s, z, t):
        return {"factor_error": min(float((z-target).norm()), float((z+target).norm())),
                "normalized_norm": float(z.norm())/math.sqrt(float(p.lam[0])),
                "leading_alignment": abs(float(p.U[:, 0] @ z[:, 0]))/float(z.norm())}
    result = run_trajectory(p, x, RunConfig(eta=.5, max_steps=cfg["max_steps"]), extra=extra)
    result["theory"] = {**annulus, "rho": 1-mu/2,
                        "comparison": "ONE optimizer; same rank-one half-step as Jedra--Shah"}
    result["rate_fit"] = estimate_rate(result["records"], "factor_error", **cfg.get("fit", {}))
    result["observed_first_trace_band_step"] = next((r["t"] for r in result["records"]
                                                        if r["trace_over_gap"] < 1), None)
    return result


def jobs(cfg):
    for i, (mu, seed, overlap) in enumerate(product(cfg["mus"], cfg["seeds"], [None]+cfg["overlaps"])):
        yield Job(f"jedra-{i:05d}", {"mu": mu, "seed": seed, "overlap": overlap,
                                   "start": "gaussian" if overlap is None else "controlled"},
                  partial(_run, cfg, mu, seed, overlap))
