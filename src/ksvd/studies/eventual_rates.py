"""Collect isolated and coupled normal modes, keeping point and orbit errors distinct."""
from __future__ import annotations
from functools import partial
from itertools import product
import torch
from .common import Job, problem
from ..problem import SpectralProblem
from ..initialization import normal_start, coupled_normal_start, gaussian, power_sketch
from ..metrics import polar_frame, procrustes_distance
from ..diagnostics import component_rate_fits, first_radial_dominance, METRIC_DEFINITIONS
from ..theory import normal_rate, optimal_step
from ..runner import RunConfig, run_trajectory


def _theory(p, k, eta):
    return {"rho": normal_rate(p, k, eta), "radial_multiplier": 1-2*eta,
            "angular_multiplier": 1-eta*p.gap(k)/float(p.lam[k-1]) if k < p.r else None,
            "optimal_eta": optimal_step(p, k)[0],
            "interpretation": "rho is an upper bound; component multipliers are linearizations, not forced rates"}


def _finish(result, cfg, theory):
    result["theory"] = theory
    result["rate_fit_policy"] = {"floor": 1e-11, "ceiling": 1e-2, "min_points": 8, "window": 30, **cfg.get("fit", {})}
    result["rate_fits"] = component_rate_fits(result["records"], theory, fit=result["rate_fit_policy"])
    # Backward-compatible alias for the existing plotting code.
    result["rate_fit"] = result["rate_fits"].get("factor_error", {"status": "not_applicable"})
    result["first_resolved_radial_dominance_step"] = first_radial_dominance(
        result["records"], floor=cfg.get("fit", {}).get("floor", 1e-11))
    result["metric_definitions"] = METRIC_DEFINITIONS
    return result


def _run(cfg, k, mu, eta, mode, amplitude):
    p = problem(cfg, k, mu)
    y = normal_start(p, k, mode, amplitude)
    target_y, target_x = p.frame(k), p.target(k)
    def extra(s, x, t):
        return {"factor_error": float((s-target_y).norm()), "x_factor_error": float((x-target_x).norm()),
                "radial_signed": float(s[k-1, k-1]-1), "angular_signed": float(s[k, k-1])}
    targets = {"product_error": 1e-8, "factor_error": 1e-8, **cfg.get("accuracy_targets", {})}
    result = run_trajectory(p, y, RunConfig(eta=eta, max_steps=cfg["max_steps"], coordinates="y",
                                          accuracy_targets=targets, record_every=cfg.get("record_every", 1)), extra=extra)
    return _finish(result, cfg, _theory(p, k, eta))


def aligned_normal_metrics(p: SpectralProblem, y: torch.Tensor, x: torch.Tensor) -> dict:
    """Diagnostic gauge only: never replace y by its aligned copy in the solver.

    For a strict cutoff, E_* O(k) is the complete reduced minimizer set.
    The particular trajectory limit is unknown for coupled/Gaussian starts.
    """
    k = y.shape[1]
    if p.gap(k) <= 0:
        raise ValueError("the fixed-orbit metric needs a strict cutoff")
    left, _, right = torch.linalg.svd(y[:k], full_matrices=False)
    aligned = y @ (left @ right).T
    top = aligned[:k] - torch.eye(k, dtype=y.dtype, device=y.device)
    radial = (top+top.T)/2
    return {"orbit_error_y": float((aligned-p.frame(k)).norm()),
            "orbit_error_x": procrustes_distance(x, p.target(k)),
            "radial_normal_norm": float(radial.norm()),
            "angular_normal_norm": float(aligned[k:].norm()),
            "slow_angular_signed": float(aligned[k, k-1]) if k < p.r else None}


def _coupled_run(cfg, k, mu, eta, amplitude, seed):
    p = problem(cfg, k, mu)
    y = coupled_normal_start(p, k, amplitude, seed)
    q = polar_frame(y)
    b, c = y.T @ y, q.T @ (p.lam[:, None]*q)
    noncommutation = float((b@c-c@b).norm()/(b.norm()*c.norm()))
    targets = {"product_error": 1e-8, "orbit_error_y": 1e-8, **cfg.get("accuracy_targets", {})}
    result = run_trajectory(p, y, RunConfig(eta=eta, max_steps=cfg["max_steps"], coordinates="y",
                                          accuracy_targets=targets, record_every=cfg.get("record_every", 1)),
                            extra=lambda s, x, t: aligned_normal_metrics(p, s, x))
    result["initial_gram_compression_commutator_relative"] = noncommutation
    result["error_target"] = "full strict-cutoff minimizer orbit, not an assumed point limit"
    return _finish(result, cfg, _theory(p, k, eta))


def _gaussian_run(cfg, k, mu, eta, seed):
    p = problem(cfg, k, mu)
    x = power_sketch(p, gaussian(p, k, seed), 1)
    tolerance = cfg.get("product_tolerance", 1e-8)
    targets = {"product_error": tolerance, "orbit_error_y": 1e-8, **cfg.get("accuracy_targets", {})}
    result = run_trajectory(p, x, RunConfig(eta=eta, max_steps=cfg["max_steps"],
                                          stop_metric="product_error", stop_tolerance=tolerance,
                                          accuracy_targets=targets, record_every=cfg.get("record_every", 1)),
                            extra=lambda s, x, t: aligned_normal_metrics(p, p.to_y(x), x))
    result["error_target"] = "full strict-cutoff minimizer orbit; no point-limit surrogate"
    return _finish(result, cfg, _theory(p, k, eta))


def _quadratic(cfg):
    p = SpectralProblem.make([3., 2., 1.], device=cfg.get("device", "cpu"))
    y = torch.diag(torch.tensor([.5, 1.5, 2.], dtype=p.lam.dtype, device=p.lam.device))
    def extra(s, x, t):
        return {"factor_error": float((s-p.frame(3)).norm())}
    result = run_trajectory(p, y, RunConfig(eta=.5, max_steps=20, coordinates="y",
                                          accuracy_targets={"factor_error": 1e-12, "product_error": 1e-8}), extra=extra)
    result["theory"] = {"rho": 0., "order": 2, "known_limit": "I_3 (positive diagonal initial Y)"}
    return result


def high_precision_quadratic(dps: int = 100, steps: int = 12) -> dict:
    """Independent scalar Heron reference; zero is precision saturation, not exact termination."""
    try:
        import mpmath as mp
    except ImportError as exc:
        raise RuntimeError("install the optional precision extra: pip install '.[precision]'") from exc
    if dps < 30 or steps < 1:
        raise ValueError("need dps>=30 and steps>=1")
    records = []
    termination = "max_steps"
    with mp.workdps(dps):
        s = [mp.mpf('.5'), mp.mpf('1.5'), mp.mpf('2')]
        for t in range(steps+1):
            err = mp.sqrt(sum((v-1)**2 for v in s))
            records.append({"t": t, "factor_error_decimal": mp.nstr(err, dps),
                            "log_error": float(mp.log(err)) if err else None,
                            "precision_saturated": not bool(err)})
            if not err or err < mp.mpf(10)**(-dps+10):
                termination = "precision_saturation" if not err else "precision_floor"
                break
            s = [(v+1/v)/2 for v in s]
    logs = [row for row in records if row["log_error"] is not None]
    orders = []
    for a, b, c in zip(logs, logs[1:], logs[2:]):
        denominator = b["log_error"]-a["log_error"]
        if denominator:
            orders.append({"t": c["t"], "order": (c["log_error"]-b["log_error"])/denominator})
    return {"status": "complete", "termination_reason": termination, "records": records, "dps": dps,
            "empirical_orders": orders, "theory": {"rho": 0., "order": 2},
            "numeric_metadata": {"dtype": "mpmath.mpf", "device": "cpu", "decimal_digits": dps,
                                 "mpmath": mp.__version__},
            "backend": "mpmath scalar Heron, diagonal k=r"}


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
        if k > 1:
            for eta, amplitude, seed in product(etas, cfg["amplitudes"], cfg.get("coupled_seeds", [])):
                params = dict(k=k, mu=mu, eta=eta, mode="mixed", amplitude=amplitude, seed=seed, start="coupled")
                # Separate IDs do not relabel the existing controlled/Gaussian cases.
                name = f"coupled-k{k}-mu{mu:.8g}-eta{eta:.12g}-amp{amplitude:.8g}-seed{seed}"
                yield Job(name, params, partial(_coupled_run, cfg, k, mu, eta, amplitude, seed))
    yield Job("quadratic-float64", {"k": 3, "r": 3, "eta": .5}, partial(_quadratic, cfg))
    if cfg.get("high_precision", False):
        yield Job("quadratic-high-precision", {"dps": cfg.get("dps", 100)},
                  partial(high_precision_quadratic, cfg.get("dps", 100)))
