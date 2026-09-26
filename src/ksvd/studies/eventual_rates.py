from __future__ import annotations
from functools import partial
from itertools import product
import torch
from .common import Job, problem
from ..problem import SpectralProblem
from ..initialization import normal_start, gaussian, power_sketch
from ..metrics import estimate_rate
from ..theory import normal_rate, optimal_step
from ..runner import RunConfig, run_trajectory


def _run(cfg, k, mu, eta, mode, amplitude):
    p = problem(cfg, k, mu)
    y = normal_start(p, k, mode, amplitude)
    target_y, target_x = p.frame(k), p.target(k)
    def extra(s, x, t):
        return {"factor_error": float((s-target_y).norm()), "x_factor_error": float((x-target_x).norm()),
                "radial_signed": float(s[k-1, k-1]-1), "angular_signed": float(s[k, k-1])}
    result = run_trajectory(p, y, RunConfig(eta=eta, max_steps=cfg["max_steps"], coordinates="y"), extra=extra)
    result["theory"] = {"rho": normal_rate(p, k, eta), "radial_multiplier": 1-2*eta,
                        "angular_multiplier": 1-eta*mu, "optimal_eta": optimal_step(p, k)[0],
                        "interpretation": "rho is a spectral upper bound; missing modes can converge faster"}
    result["rate_fit"] = estimate_rate(result["records"], "factor_error", **cfg.get("fit", {}))
    return result


def _gaussian_run(cfg, k, mu, eta, seed):
    p = problem(cfg, k, mu)
    x = power_sketch(p, gaussian(p, k, seed), 1)
    result = run_trajectory(p, x, RunConfig(eta=eta, max_steps=cfg["max_steps"],
                                          stop_metric="product_error", stop_tolerance=1e-8))
    result["theory"] = {"rho": normal_rate(p, k, eta), "optimal_eta": optimal_step(p, k)[0]}
    return result


def _quadratic(cfg):
    p = SpectralProblem.make([3., 2., 1.], device=cfg.get("device", "cpu"))
    y = torch.diag(torch.tensor([.5, 1.5, 2.], dtype=p.lam.dtype, device=p.lam.device))
    def extra(s, x, t):
        return {"factor_error": float((s-p.frame(3)).norm())}
    result = run_trajectory(p, y, RunConfig(eta=.5, max_steps=20, coordinates="y"), extra=extra)
    result["theory"] = {"rho": 0., "order": 2, "known_limit": "I_3 (positive diagonal initial Y)"}
    return result


def high_precision_quadratic(dps: int = 100, steps: int = 12) -> dict:
    """Optional independent scalar Heron reference for the k=r diagonal case."""
    try:
        import mpmath as mp
    except ImportError as exc:
        raise RuntimeError("install the optional precision extra: pip install '.[precision]'") from exc
    if dps < 30 or steps < 1:
        raise ValueError("need dps>=30 and steps>=1")
    records = []
    with mp.workdps(dps):
        s = [mp.mpf('.5'), mp.mpf('1.5'), mp.mpf('2')]
        for t in range(steps+1):
            err = mp.sqrt(sum((v-1)**2 for v in s))
            records.append({"t": t, "factor_error_decimal": mp.nstr(err, dps),
                            "log_error": float(mp.log(err)) if err else None})
            if not err or err < mp.mpf(10)**(-dps+10):
                break
            s = [(v+1/v)/2 for v in s]
    return {"status": "complete", "records": records, "dps": dps,
            "theory": {"rho": 0., "order": 2}, "backend": "mpmath scalar Heron, diagonal k=r"}


def jobs(cfg):
    idx = 0
    for k, mu in product(cfg["ks"], cfg["mus"]):
        p = problem(cfg, k, mu)
        etas = sorted(set(cfg["etas"]+[optimal_step(p, k)[0]]))
        for eta, mode, amplitude in product(etas, cfg["modes"], cfg["amplitudes"]):
            params = dict(k=k, mu=mu, eta=eta, mode=mode, amplitude=amplitude, start="controlled")
            yield Job(f"rates-{idx:05d}", params, partial(_run, cfg, k, mu, eta, mode, amplitude))
            idx += 1
        for eta, seed in product(etas, cfg.get("gaussian_seeds", [])):
            params = dict(k=k, mu=mu, eta=eta, seed=seed, start="gaussian")
            yield Job(f"rates-{idx:05d}", params, partial(_gaussian_run, cfg, k, mu, eta, seed))
            idx += 1
    yield Job("quadratic-float64", {"k": 3, "r": 3, "eta": .5}, partial(_quadratic, cfg))
    if cfg.get("high_precision", False):
        yield Job("quadratic-high-precision", {"dps": cfg.get("dps", 100)},
                  partial(high_precision_quadratic, cfg.get("dps", 100)))
