"""Independent PNG/SVG figure generation from immutable saved measurements."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(path.parent/(path.name+"."+ext), dpi=180)
    plt.close(fig)


def _series(ax, records, key, label=None, *, log=True, divisor=1.):
    pairs = [(r["t"], r[key]/divisor) for r in records if isinstance(r.get(key), (int, float))
             and math.isfinite(r[key]) and (not log or r[key] > 0)]
    if pairs:
        t, v = zip(*pairs)
        (ax.semilogy if log else ax.plot)(t, v, label=label or key)


def _reference(ax, result, records):
    fit = result.get("rate_fit", {})
    rho = result.get("theory", {}).get("rho")
    if fit.get("status") != "ok" or rho is None or not 0 < rho < 1:
        return
    start = fit["t_start"]
    row = next(r for r in records if r["t"] == start)
    ts = np.array([r["t"] for r in records if start <= r["t"] <= fit["t_end"]])
    ax.semilogy(ts, row["factor_error"] * np.exp((ts-start)*math.log(rho)), "--",
                label="spectral slope, anchored (not a certificate)")


def plot_case(case: dict, out: Path):
    name, result, study = case["id"], case["result"], case["study"]
    rows = result.get("records", [])
    params = case["parameters"]
    if result["status"] == "failed":
        return
    if study == "power_sketch_entry":
        rows = result["summary"]
        q = [r["q"] for r in rows]
        fig, ax = plt.subplots()
        ax.plot(q, [r["entry_probability"] for r in rows], label="empirical entry probability")
        ax.fill_between(q, [r["ci_lower"] for r in rows], [r["ci_upper"] for r in rows], alpha=.2,
                        label="pointwise 95% Wilson interval")
        qcert = result["theory"]["sufficient_q"]
        if qcert <= max(q):
            ax.axvline(qcert, linestyle="--", label=f"theorem sufficient q={qcert}")
        else:
            ax.text(.03, .1, f"theorem sufficient q={qcert} (outside range)", transform=ax.transAxes)
        ax.set(xlabel="Power q", ylabel="Entry probability", ylim=(-.02, 1.02), title=f"k={params['k']}, ratio={params['ratio']}")
        ax.legend(fontsize=8)
        _save(fig, out/(name+"-probability"))
        fig, ax = plt.subplots()
        delta = result["theory"]["delta"]
        for key in ("median", "p90", "p95"):
            pairs = [(r["q"], math.log10(r[key]/delta)) for r in rows if r[key] > 0]
            ax.plot(*zip(*pairs), label=key)
        for key, label in (("theorem_log_bound", "high-probability upper bound"),
                           ("median_sample_log_bound", "median sample-specific upper bound")):
            ax.plot(q, [(r[key]-math.log(delta))/math.log(10) for r in rows], "--", label=label)
        ax.set(xlabel="Power q", ylabel="log10(trace deficit / cutoff gap)")
        ax.legend(fontsize=8)
        _save(fig, out/(name+"-bounds"))
        return
    if not rows:
        return
    if result.get("backend"):
        pairs = [(r["t"], r["log_error"]/math.log(10)) for r in rows if r["log_error"] is not None]
        fig, ax = plt.subplots()
        ax.plot(*zip(*pairs), marker="o")
        ax.set(xlabel="Iteration", ylabel="log10 factor error", title="High-precision Heron reference")
        _save(fig, out/(name+"-error"))
        valid = [(a["log_error"], b["log_error"]) for a, b in zip(rows, rows[1:])
                 if a["log_error"] is not None and b["log_error"] is not None]
        fig, ax = plt.subplots()
        xs, ys = np.array(valid).T/math.log(10)
        ax.plot(xs, ys, "o", label="observed error pairs")
        ax.plot(xs, 2*xs+ys[-1]-2*xs[-1], "--", label="slope 2 reference (anchored)")
        ax.set(xlabel="log10 E_t", ylabel="log10 E_(t+1)")
        ax.legend()
        _save(fig, out/(name+"-order"))
        return
    fig, ax = plt.subplots()
    if study == "trace_contraction":
        e0 = rows[0]["trace_deficit"]
        _series(ax, rows, "trace_deficit", "e_t / e_0", divisor=e0 if e0 else 1.)
        _series(ax, rows, "certified_envelope", "certified envelope / e_0", divisor=e0 if e0 else 1.)
    elif study == "minimizer_manifold":
        for key in ("manifold_residual", "near_tied_family_residual", "reference_orbit_distance", "objective_gap"):
            _series(ax, rows, key)
    elif study in {"eventual_rates", "jedra_shah_rank1"}:
        for key in ("factor_error", "product_error", "objective_gap"):
            _series(ax, rows, key)
        _reference(ax, result, rows)
    else:
        for key in ("radial_error", "manifold_residual", "gram_min"):
            _series(ax, rows, key)
    ax.set(xlabel="Iteration t", ylabel="Error / diagnostic", title=f"{name}: {result['status']}")
    ax.legend(fontsize=8)
    _save(fig, out/(name+"-errors"))
    if study == "eventual_rates" and params.get("start") == "controlled":
        fig, ax = plt.subplots()
        for key in ("radial_signed", "angular_signed"):
            _series(ax, rows[:40], key, log=False)
        ax.set(xlabel="Iteration t", ylabel="Signed mode amplitude", title="Initial normal-mode dynamics")
        ax.legend()
        _save(fig, out/(name+"-signed-modes"))
    if study == "trace_contraction":
        fig, ax = plt.subplots()
        rr = result["resolved_trace_ratios"]
        ax.plot([v["t"] for v in rr], [v["ratio"] for v in rr], label="resolved e_(t+1)/e_t")
        if "zeta" in result["theory"]:
            ax.axhline(result["theory"]["zeta"], linestyle="--", label="certificate zeta")
        ax.set(xlabel="Iteration t", ylabel="Trace ratio (above recorded floor)")
        ax.legend()
        _save(fig, out/(name+"-ratios"))
        fig, ax = plt.subplots()
        _series(ax, rows, "product_error")
        _series(ax, rows, "gram_min")
        ax.set(xlabel="Iteration t", ylabel="Product error / minimum Gram eigenvalue")
        ax.legend()
        _save(fig, out/(name+"-factor"))
    if study == "minimizer_manifold":
        fig, ax = plt.subplots()
        valid = [r for r in rows if r["angle_identifiable"]]
        angles = np.unwrap(2*np.array([r["selected_angle"] for r in valid]))/2
        ax.plot([r["t"] for r in valid], angles)
        ax.set(xlabel="Iteration t", ylabel="Selected line angle (unwrapped modulo pi)", title=f"splitting={params['splitting']}")
        _save(fig, out/(name+"-angle"))
    if study == "jedra_shah_rank1":
        fig, ax = plt.subplots()
        _series(ax, rows, "normalized_norm", log=False)
        th = result["theory"]
        ax.axhline(th["a_js"], linestyle="--", label="JS annulus lower bound")
        ax.axhline(th["b_js"], linestyle="--", label="JS annulus upper bound")
        ax.axvline(th["tau_js"], linestyle=":", label="JS norm-entry estimate")
        ax.set(xlabel="Iteration t", ylabel="Norm / sqrt(lambda_1)")
        ax.legend(fontsize=8)
        _save(fig, out/(name+"-annulus"))
        fig, ax = plt.subplots()
        _series(ax, rows, "trace_over_gap")
        ax.axhline(1., linestyle="--", label="trace-band boundary")
        ax.set(xlabel="Iteration t", ylabel="Trace deficit / cutoff gap")
        ax.legend()
        _save(fig, out/(name+"-alignment"))


def plot_aggregate(cases, out):
    groups = {}
    for case in cases:
        p, r = case["parameters"], case["result"]
        if case["study"] == "eventual_rates" and p.get("start") == "controlled" and p.get("mode") == "mixed":
            groups.setdefault((p["k"], p["mu"], p["amplitude"]), []).append(case)
    for (k, mu, amp), group in groups.items():
        fig, ax = plt.subplots()
        grid = np.linspace(.01, .995, 300)
        ax.plot(grid, np.maximum(np.abs(1-2*grid), 1-grid*mu), label="normal spectral factor")
        good = [c for c in group if c["result"].get("rate_fit", {}).get("status") == "ok"]
        ax.scatter([c["parameters"]["eta"] for c in good],
                   [c["result"]["rate_fit"]["rho_fit"] for c in good], label="late-window empirical fit")
        ax.axvline(2/(2+mu), linestyle="--", label="locally optimal eta")
        ax.set(xlabel="Damping eta", ylabel="Convergence factor", title=f"k={k}, mu={mu}, amplitude={amp}; {len(good)}/{len(group)} fits")
        ax.legend(fontsize=8)
        _save(fig, out/f"eta-k{k}-mu{mu}-amp{amp}")
    global_groups = {}
    for c in cases:
        par = c["parameters"]
        if c["study"] == "eventual_rates" and par.get("start") == "gaussian":
            global_groups.setdefault((par["k"], par["mu"]), []).append(c)
    for (k, mu), group in global_groups.items():
        fig, ax = plt.subplots()
        stats = []
        for eta in sorted({c["parameters"]["eta"] for c in group}):
            subset = [c for c in group if c["parameters"]["eta"] == eta]
            times = [c["result"]["last_evaluated_step"] for c in subset
                     if c["result"].get("status") == "target_reached"]
            item = {"eta": eta, "reached": len(times), "trials": len(subset),
                    "median_steps_among_reached": float(np.median(times)) if times else None}
            stats.append(item)
            if times:
                ax.scatter([eta], [np.median(times)])
                ax.annotate(f"{len(times)}/{len(subset)}", (eta, np.median(times)))
        ax.axvline(2/(2+mu), linestyle="--", label="local spectral optimum")
        ax.set(xlabel="Damping eta", ylabel="Median steps among target-reaching runs",
               title=f"Gaussian starts: k={k}, mu={mu}; labels = reached / attempted")
        ax.legend(fontsize=8)
        _save(fig, out/f"gaussian-eta-k{k}-mu{mu}")
        (out/f"gaussian-eta-k{k}-mu{mu}.json").write_text(json.dumps(stats, indent=2)+"\n")
    tied = [c for c in cases if c["study"] == "minimizer_manifold" and c["parameters"]["splitting"] == 0
            and c["result"].get("records")]
    if tied:
        fig, ax = plt.subplots()
        angles = [c["result"]["records"][-1]["selected_angle"] for c in tied]
        ax.hist(angles, bins=12, range=(0, math.pi))
        ax.set(xlabel="Final observed selected line angle (modulo pi)", ylabel="Runs",
               title="Finite-run endpoints; not an assumed uniform law")
        _save(fig, out/"tied-endpoints")


def plot_study(input_dir: Path, out: Path, max_cases: int = 8) -> list[dict]:
    manifest = json.loads((input_dir/"manifest.json").read_text())
    cases = [json.loads((input_dir/v["file"]).read_text()) for v in manifest["cases"]]
    for case in cases[:max_cases]:
        plot_case(case, out)
    # Always include the optional reference even if it falls beyond the figure cap.
    for case in cases[max_cases:]:
        if case["result"].get("backend"):
            plot_case(case, out)
    plot_aggregate(cases, out)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=8, help="cap individual figures; aggregate uses all cases")
    args = parser.parse_args()
    if args.max_cases < 0:
        parser.error("max-cases must be nonnegative")
    plot_study(args.input, args.out, args.max_cases)


if __name__ == "__main__":
    main()
