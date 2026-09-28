"""Read-only rank-one reporting: verified control deduplication and hitting times.

Pure standard library: this module never imports or runs an optimizer. Gaussian
runs are never deduplicated. First hits and persistent tails refer to SAVED
observations, not unobserved iterates or infinite-time guarantees.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
from typing import Callable

VERSION = "rankone-reporting-v1"
FAILURES = {"failed", "nonfinite", "numerical_failure", "norm_limit"}
EVENT_NAMES = ("annulus", "strict_band", "fixed_margin_band", "factor_accuracy", "product_accuracy")
EXECUTION_KEYS = ("source_sha256", "git_commit", "torch", "numpy", "dtype", "device", "threads", "platform", "python")


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _terminal_step(result: dict, records: list[dict]) -> int | None:
    flag = result.get("final_state_evaluated")
    if flag is False or (result.get("status") in FAILURES and flag is not True):
        return None
    return result.get("last_evaluated_step", records[-1]["t"] if records else None)


def event_summary(records: list[dict], metric: str, predicate: Callable[[float], bool] | None,
                  terminal_step: int | None) -> dict:
    """Summarize a threshold without extrapolating beyond recorded samples."""
    samples = [(row["t"], row[metric]) for row in records if _number(row.get(metric))]
    observations = [(t, v, bool(predicate(v))) for t, v in samples] if predicate else []
    hits = [(t, v) for t, v, met in observations if met]
    first = hits[0][0] if hits else None
    last_exit = max((t for t, _, met in observations if not met), default=-1)
    sustained = next((t for t, _, met in observations if met and t > last_exit), None)
    final = next(((v, met) for t, v, met in observations if t == terminal_step), None)
    prefix = [t for t, _, _ in observations if first is not None and t <= first]
    exact_prefix = first is not None and prefix == list(range(first+1))
    return {
        "metric": metric, "criterion_available": predicate is not None,
        "observation_scope": "saved_records_only", "observations": len(observations),
        "missing_metric_records": len(records)-len(samples),
        "first_hit_step": first, "first_hit_value": hits[0][1] if hits else None,
        "first_hit_has_complete_observed_prefix": exact_prefix,
        "initially_met": observations[0][2] if observations and observations[0][0] == 0 else None,
        "last_observed_value": samples[-1][1] if samples else None,
        "last_observed_step": samples[-1][0] if samples else None,
        "first_hit_with_no_later_observed_exit": sustained,
        "no_later_observed_exit_after_first_hit": (all(met for t, _, met in observations if t >= first)
                                                    if first is not None else None),
        "final_value": final[0] if final else None, "final_met": final[1] if final else None,
    }


def case_events(case: dict, *, fixed_margin: float = .5, factor_tolerance: float = 1e-8,
                product_tolerance: float = 1e-8) -> dict:
    if not _number(fixed_margin) or not 0 < fixed_margin < 1:
        raise ValueError("fixed_margin must be strictly between zero and one")
    if any(not _number(v) or v < 0 for v in (factor_tolerance, product_tolerance)):
        raise ValueError("accuracy tolerances must be finite and nonnegative")
    result = case["result"]
    records = result.get("records", [])
    steps = [row["t"] for row in records]
    if any(type(t) is not int or t < 0 for t in steps) or any(a >= b for a, b in zip(steps, steps[1:])):
        raise ValueError("records must have strictly increasing nonnegative integer steps")
    for row in records:
        for key in ("normalized_norm", "trace_over_gap", "factor_error", "product_error"):
            if _number(row.get(key)) and row[key] < 0:
                raise ValueError(f"{key} must be nonnegative; raw errors are not silently clipped")
    theory = result.get("theory", {})
    a, b, tau = theory.get("a_js"), theory.get("b_js"), theory.get("tau_js")
    annulus_valid = _number(a) and _number(b) and 0 < a <= b
    criteria = {
        "annulus": ("normalized_norm", (lambda v: a <= v <= b) if annulus_valid else None),
        "strict_band": ("trace_over_gap", lambda v: v < 1.),
        "fixed_margin_band": ("trace_over_gap", lambda v: v <= fixed_margin),
        "factor_accuracy": ("factor_error", lambda v: v <= factor_tolerance),
        "product_accuracy": ("product_error", lambda v: v <= product_tolerance),
    }
    terminal = _terminal_step(result, records)
    events = {name: event_summary(records, key, pred, terminal) for name, (key, pred) in criteria.items()}
    for event in events.values():
        row = next((r for r in records if r["t"] == event["first_hit_step"]), {})
        event["at_first_hit"] = {key: row.get(key) for key in
                                 ("leading_alignment", "normalized_norm", "trace_over_gap", "factor_error", "product_error")}
    later = [row for row in records if type(tau) is int and tau >= 0 and row["t"] >= tau]
    valid = [row for row in later if _number(row.get("normalized_norm"))]
    return {
        "events": events, "annulus_bounds": {"lower": a, "upper": b}, "tau_js": tau,
        "annulus_at_all_observed_steps_from_tau": (all(a <= row["normalized_norm"] <= b for row in valid)
                                                    if annulus_valid and valid else None),
        "annulus_samples_from_tau": len(valid), "missing_norm_samples_from_tau": len(later)-len(valid),
        "saved_steps_complete_from_zero": bool(steps) and steps == list(range(steps[-1]+1)),
        "criterion_definitions": {
            "annulus": "a_js <= ||x||/sqrt(lambda_1) <= b_js (closed interval; saved bounds)",
            "strict_band": "e(x)/Delta < 1 (strict)",
            "fixed_margin_band": f"e(x)/Delta <= {fixed_margin:g} (inclusive)",
            "factor_accuracy": f"min_+/- ||x +/- sqrt(lambda_1) u_1||_2 <= {factor_tolerance:g} (absolute)",
            "product_accuracy": f"||xx^T-lambda_1 u_1u_1^T||_F/lambda_1 <= {product_tolerance:g} (relative)",
        },
    }


def _control_fingerprint(case: dict) -> str | None:
    result, params, meta = case["result"], case.get("parameters", {}), case.get("metadata", {})
    # Missing evidence or failures are retained, not compressed into one outcome.
    if (params.get("start") != "controlled" or not _number(params.get("overlap"))
            or result.get("status") in FAILURES or not result.get("records")
            or not result.get("initial_state") or not result.get("problem") or not meta.get("source_sha256")):
        return None
    return _hash({"study": case["study"], "parameters": {k: v for k, v in params.items() if k != "seed"},
                  "result": result, "execution": {k: meta.get(k) for k in EXECUTION_KEYS}})


def deduplicate_controls(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep Gaussian trials and distinct/failed controls; merge only exact stored replays."""
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("duplicate case identifiers")
    kept, mapping, representatives = [], [], {}
    for case in sorted(cases, key=lambda c: c["id"]):
        fingerprint = _control_fingerprint(case)
        representative = representatives.get(fingerprint) if fingerprint is not None else None
        included = representative is None
        if included:
            representative = case["id"]
            kept.append(case)
            if fingerprint is not None:
                representatives[fingerprint] = representative
        mapping.append({"id": case["id"], "representative_id": representative,
                        "included_in_analysis": included, "nominal_seed": case.get("parameters", {}).get("seed"),
                        "start": case.get("parameters", {}).get("start"), "evidence_sha256": fingerprint,
                        "reason": "exact_control_replay" if not included else "retained"})
    return kept, mapping


def _csv(path: Path, rows: list[dict], first: tuple[str, ...]) -> None:
    fields = list(first) + sorted(set().union(*(r.keys() for r in rows))-set(first))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True, allow_nan=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def build_report(cases: list[dict], *, fixed_margin: float = .5,
                 factor_tolerance: float = 1e-8, product_tolerance: float = 1e-8) -> dict:
    # Validate policy even for an empty/incomplete collection.
    case_events({"result": {}}, fixed_margin=fixed_margin, factor_tolerance=factor_tolerance,
                product_tolerance=product_tolerance)
    if any(c.get("study") != "jedra_shah_rank1" for c in cases):
        raise ValueError("this reporter accepts jedra_shah_rank1 only")
    kept, mapping = deduplicate_controls(cases)
    entries = []
    for case in kept:
        aliases = [m for m in mapping if m["representative_id"] == case["id"]]
        entries.append({"id": case["id"], "parameters": case.get("parameters", {}),
                        "source_ids": [m["id"] for m in aliases],
                        "source_nominal_seeds": [m["nominal_seed"] for m in aliases],
                        "source_file_count": len(aliases), "analysis_weight": 1,
                        "termination_status": case["result"].get("status"),
                        "saved_rate_fit": case["result"].get("rate_fit"),
                        "initial_overlap": case["result"].get("theory", {}).get("initial_overlap"),
                        "predicted_rho": case["result"].get("theory", {}).get("rho"),
                        **case_events(case, fixed_margin=fixed_margin, factor_tolerance=factor_tolerance,
                                      product_tolerance=product_tolerance)})
    grouped = defaultdict(list)
    for entry in entries:
        key = json.dumps({k: v for k, v in entry["parameters"].items() if k != "seed"}, sort_keys=True)
        grouped[key].append(entry)
    groups = []
    for key, group in grouped.items():
        params = json.loads(key)
        stats = {}
        for name in EVENT_NAMES:
            hits = [e["events"][name]["first_hit_step"] for e in group
                    if e["events"][name]["first_hit_step"] is not None]
            stats[name] = {"hit_rows": len(hits), "analysis_rows": len(group),
                           "available_rows": sum(e["events"][name]["observations"] > 0 for e in group),
                           "median_first_hit_among_hits": statistics.median(hits) if hits else None,
                           "min_first_hit_among_hits": min(hits) if hits else None,
                           "max_first_hit_among_hits": max(hits) if hits else None}
        groups.append({"parameters": params, "analysis_rows": len(group),
                       "source_files": sum(e["source_file_count"] for e in group),
                       "statistics_role": ("deterministic_controls_not_stochastic_replicates" if params.get("start") == "controlled"
                                           else "Gaussian_trials_within_this_configuration"),
                       "events": stats})
    return {"schema_version": 1, "analysis_version": VERSION, "reviewed_files": len(cases),
            "analysis_rows": len(entries), "duplicate_files_excluded": len(cases)-len(entries),
            "raw_termination_counts": dict(Counter(c["result"].get("status") for c in cases)),
            "analysis_termination_counts": dict(Counter(e["termination_status"] for e in entries)),
            "policy": {"fixed_margin": fixed_margin, "factor_tolerance": factor_tolerance,
                       "product_tolerance": product_tolerance, "deduplication": "Exact controlled replay evidence only; never Gaussian trials.",
                       "observation_scope": "saved_records_only", "comparison_tolerance": 0,
                       "interpretation": "First hits do not imply proximity or future invariance. Missing observations are not successes. Saved rate fits are retained, not recomputed. No replicate-based error bars for deterministic controls."},
            "cases": entries, "groups": groups, "duplicate_mapping": mapping}


def summarize_rank_one(input_dir: Path, output_dir: Path, **policy: float) -> dict:
    """Generate a separate report without changing any source manifest or case file."""
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if input_dir == output_dir or input_dir in output_dir.parents or output_dir in input_dir.parents:
        raise ValueError("output must be separate from, not inside or an ancestor of, raw input")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("choose an empty output directory to preserve earlier reports")
    manifest_bytes = (input_dir/"manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("study") != "jedra_shah_rank1":
        raise ValueError("this reporter accepts jedra_shah_rank1 only")
    cases, sources, seen = [], [], set()
    for item in manifest["cases"]:
        path = (input_dir/item["file"]).resolve()
        if path.parent != input_dir or path.name == "manifest.json" or path in seen:
            raise ValueError("case files must be unique direct children of the input directory")
        seen.add(path)
        raw = path.read_bytes()
        case = json.loads(raw)
        if case.get("id") != item["id"] or case.get("study") != manifest["study"]:
            raise ValueError("manifest/case identity mismatch")
        cases.append(case)
        sources.append({"id": case["id"], "file": item["file"], "sha256": hashlib.sha256(raw).hexdigest()})
    report = build_report(cases, **policy)
    report.update(source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), source_files=sources,
                  source_run_metadata=manifest.get("metadata"), source_config=manifest.get("config"),
                  planned_files=manifest.get("planned_case_count"),
                  analysis_metadata={"created_utc": datetime.now(timezone.utc).isoformat(),
                                     "python": platform.python_version(),
                                     "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    lookup = {row["id"]: row for row in sources}
    for mapping in report["duplicate_mapping"]:
        mapping.update(source_file=lookup[mapping["id"]]["file"], source_sha256=lookup[mapping["id"]]["sha256"])
    flat, event_rows = [], []
    for entry in report["cases"]:
        row = {"id": entry["id"], **entry["parameters"], "source_file_count": entry["source_file_count"],
               "source_ids": entry["source_ids"], "termination_status": entry["termination_status"],
               "a_js": entry["annulus_bounds"]["lower"], "b_js": entry["annulus_bounds"]["upper"],
               "tau_js": entry["tau_js"], "initial_overlap": entry["initial_overlap"],
               "predicted_rho": entry["predicted_rho"],
               "rho_fit": (entry["saved_rate_fit"] or {}).get("rho_fit"),
               "observation_scope": "saved_records_only", "saved_steps_complete_from_zero": entry["saved_steps_complete_from_zero"]}
        for name, event in entry["events"].items():
            for field in ("first_hit_step", "first_hit_with_no_later_observed_exit", "final_value", "final_met",
                          "no_later_observed_exit_after_first_hit", "first_hit_has_complete_observed_prefix"):
                row[f"{name}.{field}"] = event[field]
            event_rows.append({"id": entry["id"], "event": name, "criterion": entry["criterion_definitions"][name], **event})
        flat.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/"summary.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    _csv(output_dir/"hitting_times.csv", flat, ("id", "start", "mu", "overlap", "seed"))
    _csv(output_dir/"events.csv", event_rows, ("id", "event", "first_hit_step"))
    _csv(output_dir/"duplicate_mapping.csv", report["duplicate_mapping"], ("id", "representative_id", "included_in_analysis"))
    table = ["# Rank-one reporting cleanup", "", f"Input files: {len(cases)}. Analysis rows: {report['analysis_rows']}. "
             f"Exact controlled replays excluded: {report['duplicate_files_excluded']}.", "",
             "Raw inputs are unchanged. Times below are first saved hits, not guarantees about unobserved or future iterates.", "",
             "| Start | Gap | Initial overlap | Source seed labels | Norm annulus | Strict band | Fixed-margin band | Factor accuracy | Product accuracy |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for entry in report["cases"]:
        p = entry["parameters"]
        values = [p.get("start"), p.get("mu"), entry["initial_overlap"], entry["source_nominal_seeds"]]
        values += [entry["events"][name]["first_hit_step"] for name in EVENT_NAMES]
        table.append("| " + " | ".join("not observed / unavailable" if v is None else str(v) for v in values)+" |")
    table += ["", f"Fixed-margin fraction: {report['policy']['fixed_margin']}. "
              f"Absolute factor tolerance: {report['policy']['factor_tolerance']}; relative product tolerance: {report['policy']['product_tolerance']}.",
              "", "Full comparison definitions, missing-data flags and observed-tail checks are in summary.json and events.csv. "
              "duplicate_mapping.csv preserves the mapping and SHA-256 for every input file. "
              "Gaussian seeds are retained; deterministic seed aliases are not independent replicates. "
              "All statuses, including failures, are retained. Empty CSV cells mean unavailable/not observed, never zero."]
    (output_dir/"README.md").write_text("\n".join(table)+"\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixed-margin", type=float, default=.5, help="inclusive e/Delta threshold in (0,1)")
    parser.add_argument("--factor-tolerance", type=float, default=1e-8)
    parser.add_argument("--product-tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    report = summarize_rank_one(args.input, args.out, fixed_margin=args.fixed_margin,
                                factor_tolerance=args.factor_tolerance, product_tolerance=args.product_tolerance)
    print(f"{report['reviewed_files']} input files -> {report['analysis_rows']} analysis rows; "
          f"{report['duplicate_files_excluded']} exact control replays excluded. Raw inputs unchanged.")


if __name__ == "__main__":
    main()
