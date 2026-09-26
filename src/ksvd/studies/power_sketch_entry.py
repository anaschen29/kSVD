from __future__ import annotations
import math
from itertools import product
from functools import partial
import numpy as np
from .common import Job, problem
from ..initialization import gaussian
from ..metrics import basis, trace_deficit
from ..theory import sufficient_power, power_log_bound, sample_power_log_bound


def wilson(successes: int, trials: int) -> tuple[float, float]:
    """Pointwise 95% Wilson interval, not a simultaneous confidence band."""
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    phat, den = successes/trials, 1+z*z/trials
    center = (phat+z*z/(2*trials))/den
    radius = z*math.sqrt(phat*(1-phat)/trials+z*z/(4*trials**2))/den
    return (0. if successes == 0 else max(0., center-radius),
            1. if successes == trials else min(1., center+radius))


def _run(cfg, k, ratio):
    p = problem(cfg, k, 1-ratio)
    delta = cfg["failure_probability"]
    q_cert = sufficient_power(p, k, cfg["theta"], delta)
    qmax = cfg.get("q_max") or q_cert+5
    trials = []
    for j in range(cfg["trials"]):
        seed = cfg.get("seed", 0)+j
        omega = gaussian(p, k, seed)
        base_log = sample_power_log_bound(p, omega, k, 1)
        x, errors = omega, []
        for q in range(1, qmax+1):
            x = basis(p.apply(x))  # only measuring col(M^q Omega), not running PGD
            errors.append(trace_deficit(p, x))
        trials.append({"seed": seed, "trace_deficits": errors,
                       "sample_log_bound_q1": base_log})
    array = np.array([v["trace_deficits"] for v in trials])
    summary = []
    for j in range(qmax):
        values = array[:, j]
        count = int(np.sum(values <= cfg["theta"]*p.gap(k)))
        lo, hi = wilson(count, len(trials))
        sample_logs = [v["sample_log_bound_q1"]+2*j*math.log(ratio) for v in trials]
        summary.append({"q": j+1, "entry_probability": count/len(trials),
                        "successes": count, "trials": len(trials), "ci_lower": lo, "ci_upper": hi,
                        "median": float(np.quantile(values, .5)), "p90": float(np.quantile(values, .9)),
                        "p95": float(np.quantile(values, .95)),
                        "theorem_log_bound": power_log_bound(p, k, j+1, delta),
                        "median_sample_log_bound": float(np.median(sample_logs))})
    return {"status": "complete", "summary": summary, "trials": trials,
            "theory": {"sufficient_q": q_cert, "theta": cfg["theta"], "delta": p.gap(k),
                       "failure_probability": delta, "q_cert_in_plotted_range": q_cert <= qmax},
            "measurement": "QR-stabilized subspace-only powers; NOT a replacement PGD initialization"}


def jobs(cfg):
    for i, (k, ratio) in enumerate(product(cfg["ks"], cfg["ratios"])):
        yield Job(f"power-{i:04d}", {"k": k, "ratio": ratio, "trials": cfg["trials"]},
                  partial(_run, cfg, k, ratio))
