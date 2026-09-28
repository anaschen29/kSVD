from __future__ import annotations
from itertools import product
from functools import partial
from .common import Job, problem
from ..initialization import controlled_trace_start
from ..theory import trace_certificate
from ..runner import RunConfig, run_trajectory


def _run(cfg, k, mu, theta, eta, cond, null_mix, seed):
    p = problem(cfg, k, mu, seed=seed if cfg.get("rotate", True) else None)
    x = controlled_trace_start(p, k, theta, condition=cond, null_mix=null_mix, seed=seed)
    cert = trace_certificate(p, x, eta) if theta < 1 else None
    def extra(state, x, t):
        return {"certified_envelope": cert.envelope(t) if cert and t >= 1 else None}
    targets = {"product_error": 1e-8, "trace_over_gap": 1e-8, **cfg.get("accuracy_targets", {})}
    result = run_trajectory(p, x, RunConfig(eta=eta, max_steps=cfg["max_steps"],
                                          accuracy_targets=targets, record_every=cfg.get("record_every", 1)), extra=extra)
    result["theory"] = cert.to_dict() if cert else {"status": "outside_trace_band"}
    # Only consecutive, well-resolved samples are one-step contraction ratios.
    rows = result["records"]
    floor = cfg.get("ratio_floor", 1e-12)
    ratios = []
    for before, after in zip(rows, rows[1:]):
        if before["t"] >= 1 and after["t"] == before["t"]+1 and before["trace_deficit"] > floor:
            ratios.append({"t": before["t"], "ratio": after["trace_deficit"]/before["trace_deficit"]})
    result["resolved_trace_ratios"] = ratios
    result["certificate_diagnostics"] = {
        "maximum_resolved_ratio": max((v["ratio"] for v in ratios), default=None),
        "minimum_observed_gram": min((v["gram_min"] for v in rows if v["t"] >= 1), default=None),
        "ratio_floor": floor,
        "maximum_envelope_excess": max((v["trace_deficit"]-v["certified_envelope"]
                                        for v in rows if v["certified_envelope"] is not None), default=None),
        "observation_scope": "saved_records_only; ratios require consecutive steps"}
    return result


def jobs(cfg):
    values = product(cfg["ks"], cfg["mus"], cfg["thetas"], cfg["etas"],
                     cfg["gram_conditions"], cfg["null_mixes"], cfg["seeds"])
    for i, (k, mu, theta, eta, cond, null_mix, seed) in enumerate(values):
        params = dict(k=k, mu=mu, theta=theta, eta=eta, gram_condition=cond,
                      null_mix=null_mix, seed=seed)
        yield Job(f"trace-{i:05d}", params, partial(_run, cfg, k, mu, theta, eta, cond, null_mix, seed))
