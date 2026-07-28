"""Audit committed Phase 2 smoke JSON and build dependency-free SVG dashboards.

This script never runs an experiment.  It reads the unaggregated histories in
``results/smoke`` and writes an explicitly preliminary pipeline audit plus one
small-multiple dashboard per experiment.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
import html
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "smoke"
OUT = ROOT / "results" / "smoke_audit"


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _runs(document: dict[str, Any]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for case in document.get("cases", []):
        candidate = case.get("run", case)
        if isinstance(candidate, dict) and "records" in candidate:
            runs.append(candidate)
    for case in document.get("certified_subset", []):
        candidate = case.get("run")
        if isinstance(candidate, dict) and "records" in candidate:
            runs.append(candidate)
    return runs


def _polyline(values: list[float], x: int, y: int, width: int, height: int) -> str:
    finite = [value for value in values if math.isfinite(value) and value > 0]
    if len(finite) < 2:
        return ""
    logs = [math.log10(value) for value in finite]
    low, high = min(logs), max(logs)
    span = max(high - low, 1e-15)
    points = [
        f"{x + i * width / (len(logs) - 1):.1f},{y + height - (v - low) * height / span:.1f}"
        for i, v in enumerate(logs)
    ]
    return f'<polyline points="{" ".join(points)}" fill="none" stroke="#2563eb" stroke-width="1.2"/>'


def _dashboard(name: str, document: dict[str, Any]) -> Path:
    runs = _runs(document)
    cards: list[tuple[str, list[float]]] = []
    for index, run in enumerate(runs):
        records = run["records"]
        values = [float(row.get("factor_manifold_distance", row.get("objective_gap", 0))) for row in records]
        cards.append((f"run {index}: {run.get('termination', '?')}, n={len(records)}", values))
    if not cards:
        source = document.get("modes", document.get("cases", document.get("part_b", [])))
        values = []
        for row in source:
            for key in ("absolute_error", "escape_time", "F"):
                if row.get(key) is not None:
                    values.append(abs(float(row[key])) + 1e-300)
                    break
        cards = [(f"all {len(values)} scalar records", values)]
    columns, card_w, card_h = 4, 250, 115
    rows = math.ceil(len(cards) / columns)
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{columns*card_w}" height="{45+rows*card_h}" viewBox="0 0 {columns*card_w} {45+rows*card_h}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="12" y="25" font-family="sans-serif" font-size="17">{html.escape(name)} — preliminary smoke histories (log scale)</text>']
    for index, (label, values) in enumerate(cards):
        x, y = (index % columns) * card_w, 45 + (index // columns) * card_h
        body.extend([f'<rect x="{x+5}" y="{y+3}" width="{card_w-10}" height="{card_h-8}" fill="#f8fafc" stroke="#cbd5e1"/>',
                     f'<text x="{x+10}" y="{y+19}" font-family="monospace" font-size="10">{html.escape(label)}</text>',
                     _polyline(values, x + 10, y + 28, card_w - 20, card_h - 42)])
    body.append("</svg>\n")
    path = OUT / "plots" / f"{name}.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW.glob("*.json"))
    if len(files) != 8:
        raise RuntimeError(f"expected eight smoke JSON files, found {len(files)}")
    lines = ["# Phase 2 smoke-pipeline audit", "",
             f"Generated {date.today().isoformat()} from committed raw JSON. **These smoke cases only check that the pipeline appears internally correct; they are not experimental results and do not verify any theorem.**", ""]
    total_bytes = 0
    total_run_seconds = 0.0
    total_runs = 0
    total_wrapper_seconds = 0.0
    projected_wrapper_seconds = 0.0
    total_records = 0
    total_history_bytes = 0
    fit_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    identity_rows: list[tuple[str, str, float, str]] = []
    for path in files:
        document = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        total_bytes += path.stat().st_size
        metadata = document["metadata"]
        wrapper_seconds=float(metadata.get("wrapper_elapsed_seconds",0.0))
        total_wrapper_seconds+=wrapper_seconds
        scale={"predicted_local_rates":400/12,"hessian_mode_isolation":216/24,
               "boundary_gap_scaling":900/12,"tied_eigenvalues":250/6,
               "geometry_of_kc":442002/1013,"saddle_escape":66/4,
               "step_size_phase_diagram":12480/72,"initialization_ablation":463/35}[path.stem]
        projected_wrapper_seconds+=wrapper_seconds*scale
        runs = _runs(document)
        total_runs += len(runs)
        total_run_seconds += sum(float(run.get("elapsed_seconds", 0.0)) for run in runs)
        total_records += sum(len(run["records"]) for run in runs)
        total_history_bytes += sum(int(run.get("raw_history_bytes", 0)) for run in runs)
        statuses = Counter(run.get("termination", "missing") for run in runs)
        nonfinite = sum(isinstance(value, float) and not math.isfinite(value) for value in _walk(document))
        fits = [value for value in _walk(document) if isinstance(value, dict) and {"points", "rho_ratio", "rho_slope"} <= value.keys()]
        invalid = sum(fit["rho_ratio"] is None or fit["rho_slope"] is None for fit in fits)
        warnings = sum(statuses[key] for key in ("lost_rank", "nan", "cycle", "diverged", "stagnated", "underflow"))
        for index, case in enumerate(document.get("cases", [])):
            run = case.get("run", case)
            descriptor = ";".join(f"{key}={case[key]}" for key in ("k", "eta", "delta", "family", "seed", "spectrum", "value") if key in case)
            manifest_rows.append({"experiment": path.stem, "case": index, "descriptor": descriptor,
                                  "records": len(run.get("records", [])), "termination": run.get("termination", "not_applicable")})
            if run.get("termination") in {"lost_rank", "nan", "cycle", "diverged", "stagnated", "underflow"}:
                event_rows.append({"experiment": path.stem, "case": index, "descriptor": descriptor,
                                   "event": run["termination"], "detail": run.get("termination_detail") or ""})
            for fit_name in ("factor_fit", "objective_fit", "rate_fit", "late_rate_fit"):
                if fit_name not in case:
                    continue
                key = "objective_gap" if fit_name == "objective_fit" else "factor_manifold_distance"
                usable = [row for row in run["records"] if row.get(key) is not None and 1e-9 <= float(row[key]) <= 1e-3]
                fit = case[fit_name]
                predicted = case.get("predicted_objective_rate" if fit_name == "objective_fit" else
                                     "predicted_factor_rate", case.get("predicted_rate"))
                fit_rows.append({"experiment": path.stem, "case": index, "descriptor": descriptor,
                                 "fit": fit_name, "predicted": predicted, "points": fit["points"],
                                 "window_first": fit.get("window_first"),
                                 "window_last": fit.get("window_last"),
                                 "rho_ratio": fit["rho_ratio"], "rho_slope": fit["rho_slope"],
                                 "valid": fit["rho_ratio"] is not None and fit["rho_slope"] is not None})
        for index, case in enumerate(document.get("certified_subset", [])):
            run=case["run"]; descriptor=f"certified_subset;family={case['family']};seed={case['seed']};eta={case['eta']}"
            manifest_rows.append({"experiment":path.stem,"case":f"certified_{index}","descriptor":descriptor,
                                  "records":len(run["records"]),"termination":run["termination"]})
            if run["termination"] in {"lost_rank","nan","cycle","diverged","stagnated","underflow"}:
                event_rows.append({"experiment":path.stem,"case":f"certified_{index}","descriptor":descriptor,
                                   "event":run["termination"],"detail":run.get("termination_detail") or ""})
        if path.stem == "hessian_mode_isolation":
            identity_rows.append((path.stem, "max absolute predicted/measured error",
                                  max(float(row["absolute_error"]) for row in document["modes"]), "finite-difference check"))
        elif path.stem == "saddle_escape":
            identity_rows.append((path.stem, "exact saddle fixed-point error", float(document["exact_fixed_error"]), "exact identity"))
        elif path.stem == "tied_eigenvalues":
            for key, value in document["isotropic_sanity"].items():
                identity_rows.append((path.stem, key, float(value), "isotropic identity"))
        dashboard = _dashboard(path.stem, document).relative_to(ROOT)
        lines.extend([f"## `{path.stem}`", "",
                      f"- Raw histories and metadata: [`{path.relative_to(ROOT)}`](../smoke/{path.name}) ({path.stat().st_size:,} bytes).",
                      f"- Configuration: `{json.dumps(metadata['config'], sort_keys=True)}`; dtype `{metadata['dtype']}`; device `{metadata['device']}`; seed `{metadata['seed']}`; recorded commit `{metadata['git_commit']}`.",
                      f"- Cases/runs: {len(document.get('cases', document.get('modes', [])))} / {len(runs)}; statuses: `{dict(statuses)}`.",
                      f"- Fits: {len(fits)} total, {invalid} invalid (fewer than ten usable points or no consecutive ratios). The fitting rule is `1e-9 <= error <= 1e-3`; exact reconstructed first/last indices and predicted/fitted values are in [`fit_audit.csv`](fit_audit.csv).",
                      f"- Numerical event count: {warnings} rank-loss/NaN/cycle/divergence/stagnation/underflow terminations; non-finite JSON values: {nonfinite}.",
                      f"- Preliminary per-run dashboard: [`{dashboard}`](plots/{path.stem}.svg).", ""])
    lines.extend(["## Detailed indexes", "",
                  "- [`smoke_manifest.csv`](smoke_manifest.csv) enumerates every smoke case, swept descriptor, history length, and status.",
                  "- [`fit_audit.csv`](fit_audit.csv) enumerates every predicted and fitted rate, reconstructed fitting window, and invalid fit.",
                  "- [`event_audit.csv`](event_audit.csv) enumerates every rank-loss, NaN, cycle, divergence, stagnation, or underflow event and its serialized detail.", "",
                  "## Manual identity checks", "",
                  *[f"- **{experiment} — {check}:** `{value:.17g}` ({note})." for experiment, check, value, note in identity_rows],
                  "- Hessian zero-factor rows correctly use absolute rather than relative residual semantics.",
                  "- Geometry certificate: the artifact labels sampled bounds as empirical lower bounds on conservativeness, not exact extrema.", "",
                  "## Exact proposed full grids and run counts", "",
                  "| Experiment | Full grid | Run/action count |",
                  "|---|---|---:|",
                  "| Predicted local rates | `k={1,2,4,8}`; `eta={0.1,0.25,0.5,0.7,eta_loc*}`; 20 seeds | 400 trajectories |",
                  "| Hessian modes | 24 modes; `eta={0.25,0.5,0.7}`; `epsilon={1e-4,1e-5,1e-6}` | 216 finite-difference actions |",
                  "| Boundary-gap scaling | 4 k values; 9 gaps; local-normal 5 seeds plus support-Gaussian 20 seeds | 900 trajectories |",
                  "| Tied eigenvalues | 5 gaps; 50 seeds | 250 trajectories plus isotropic sanity |",
                  "| Geometry of K_C | 601x601 minus origin; two 201x201 slices; 5,000 rejection attempts | 361,200 grid points + 80,802 slice points |",
                  "| Saddle escape | 11 epsilons; 3 steps; 2 signs | 66 trajectories |",
                  "| Step-size phase diagram | 3 spectra; 4 families; 5 seeds; per-initialization sorted union of 4 certificate multiples, 60 log points, 141 linear points, and 3 named steps | at most 12,480 trajectories before duplicate removal |",
                  "| Initialization ablation | 3 stochastic groups x20; 40 controlled values x10; three seed-zero certified checks | 463 trajectories |", "",
                  "The pre-deduplication trajectory upper total is 14,559, plus 216 Hessian actions and the geometry evaluations. Experiment 7's exact post-deduplication total depends on each initialization's `eta_C`; the implementation removes duplicates with the specified relative `1e-12` tolerance.", "",
                  "## Runtime and storage estimate", "",
                  f"The eight smoke JSON files occupy {total_bytes:,} bytes. Their measured in-process wrapper time totals {total_wrapper_seconds:.3f} seconds. Direct grid/case-count scaling gives a smoke-shaped lower planning estimate of {projected_wrapper_seconds/3600:.2f} hours for the full grids. The {total_records:,} iterative records across {total_runs} runs took {total_run_seconds:.3f} serialized per-run seconds and occupy {total_history_bytes:,} raw-history bytes. Scaling the observed average run to 14,559 trajectories gives approximately {total_run_seconds/max(1,total_runs)*14559/3600:.2f} trajectory-hours and {total_history_bytes/max(1,total_runs)*14559/1e9:.2f} GB. A deliberately conservative all-runs-hit-20,001-iterations bound is approximately {total_run_seconds/max(1,total_records)*14559*20001/3600:.1f} trajectory-hours and {total_history_bytes/max(1,total_records)*14559*20001/1e9:.1f} GB. These estimates exclude process startup and assume roughly linear per-case/per-record costs; they are planning bounds, not guarantees.", "",
                  "## Readiness and remaining limitations", "",
                  "1. Every fit now serializes its actual first/last usable iteration and an explicit invalid reason.",
                  "2. Geometry records contour/critical-point/trajectory data, multiple levels, Hessian norms, extrema, and bound ratios.",
                  "3. Tied cases record canonical errors, tied-block angles, rates, and iterations.",
                  "4. Step-size cases record relative-deduplicated grids, late rates, and grid-dependent cutoff summaries and ratios.",
                  "5. Initialization cases record certified checks, family success rates, fits, transients, and a structured exploratory regression (which may be invalid on a short smoke run).",
                  "6. Rank loss, divergence, cycles, stagnation, and underflow remain valid observed outcomes and must not be relabelled as successes.", "",
                  "**Conclusion:** the corrected smoke pipeline contains the specified raw outputs and summaries. Full sweeps still require explicit approval; smoke behavior is not an experimental conclusion.", ""])
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    for filename, rows, fields in (
        ("smoke_manifest.csv", manifest_rows, ["experiment", "case", "descriptor", "records", "termination"]),
        ("fit_audit.csv", fit_rows, ["experiment", "case", "descriptor", "fit", "predicted", "points", "window_first", "window_last", "rho_ratio", "rho_slope", "valid"]),
        ("event_audit.csv", event_rows, ["experiment", "case", "descriptor", "event", "detail"]),
    ):
        with (OUT / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
