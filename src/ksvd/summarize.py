"""Summarize saved studies into JSON/CSV, including legacy runs, without rerunning PGD."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics

from .cli import atomic_json, metadata
from .diagnostics import (DIAGNOSTICS_VERSION, accuracy_from_records, component_rate_fits,
                          first_radial_dominance, validate_targets)

_FAILURES = {"failed", "nonfinite", "numerical_failure", "norm_limit"}


def summarize_case(case: dict, *, targets: dict | None = None, fit: dict | None = None) -> dict:
    result = case["result"]
    records = result.get("records", [])
    if targets is None and "accuracy" in result:
        accuracy = deepcopy(result["accuracy"])
    else:
        effective = dict(result.get("config", {}).get("accuracy_targets", {}))
        # Legacy files had no accuracy trackers. Only observed metrics are added.
        for key in ("product_error", "factor_error", "orbit_error_y", "trace_over_gap"):
            if any(key in row for row in records):
                effective.setdefault(key, 1e-8)
        effective.update(targets or {})
        final_step = result.get("last_evaluated_step", records[-1]["t"] if records else None)
        if result["status"] in _FAILURES and not result.get("final_state_evaluated", False):
            final_step = None
        accuracy = accuracy_from_records(records, effective, final_step)
    fits = (deepcopy(result["rate_fits"]) if fit is None and "rate_fits" in result else
            component_rate_fits(records, result.get("theory", {}), fit=fit))
    numeric = {key: case.get("metadata", {}).get(key) for key in ("dtype", "device", "decimal_digits")}
    if result.get("backend"):
        # Correct legacy high-precision reporting without modifying the raw file.
        numeric.update(dtype="mpmath.mpf", device="cpu", decimal_digits=result.get("dps"))
    report = {
        "id": case["id"], "study": case["study"], "parameters": case.get("parameters", {}),
        "termination_status": result["status"], "message": result.get("message"),
        "numeric_metadata": numeric, "problem": result.get("problem"),
        "last_evaluated_step": result.get("last_evaluated_step", records[-1]["t"] if records else None),
        "completed_updates": result.get("completed_updates"), "record_count": len(records),
        "elapsed_seconds_including_diagnostics": case.get("elapsed_seconds_including_diagnostics"),
        "accuracy": accuracy, "rate_fits": fits,
        "first_resolved_radial_dominance_step": first_radial_dominance(
            records, floor=(fit or {}).get("floor", 1e-11)),
        "certificate_diagnostics": result.get("certificate_diagnostics"),
        "initial_gram_compression_commutator_relative": result.get("initial_gram_compression_commutator_relative"),
    }
    if result.get("backend"):
        finite = [row for row in records if row.get("log_error") is not None]
        report["high_precision"] = {
            "termination_reason": result.get("termination_reason", "legacy_no_explicit_reason"),
            "final_error_decimal": records[-1].get("factor_error_decimal") if records else None,
            "last_nonzero_error_decimal": finite[-1].get("factor_error_decimal") if finite else None,
            "last_nonzero_log_error": finite[-1]["log_error"] if finite else None,
            "empirical_orders": result.get("empirical_orders"),
            "warning": "Recorded zero is finite-precision saturation, not mathematical finite termination.",
        }
    return report


def group_reports(reports: list[dict]) -> list[dict]:
    groups: dict[str, list] = defaultdict(list)
    for report in reports:
        key = {"study": report["study"],
               "parameters": {k: v for k, v in report["parameters"].items() if k != "seed"}}
        groups[json.dumps(key, sort_keys=True)].append(report)
    output = []
    for key, group in groups.items():
        entry = json.loads(key)
        entry.update(runs=len(group), termination_counts=dict(Counter(r["termination_status"] for r in group)))
        entry["accuracy"] = {}
        for metric in sorted(set().union(*(r["accuracy"] for r in group))):
            observations = [r["accuracy"][metric] for r in group if metric in r["accuracy"]]
            hits = [r["first_hit_step"] for r in observations if r["ever_met"]]
            entry["accuracy"][metric] = {
                "reported_runs": len(observations),
                "available_runs": sum(r["available"] for r in observations),
                "ever_met_runs": len(hits),
                "final_met_runs": sum(r["final_met"] is True for r in observations),
                "final_unknown_runs": sum(r["final_met"] is None for r in observations),
                "not_hit_before_end_runs": sum(r["available"] and not r["ever_met"] for r in observations),
                "thresholds": sorted(set(r["threshold"] for r in observations)),
                "median_first_hit_among_hits": statistics.median(hits) if hits else None,
                "min_first_hit_among_hits": min(hits) if hits else None,
                "max_first_hit_among_hits": max(hits) if hits else None,
                "observation_scopes": sorted(set(r["observation_scope"] for r in observations)),
            }
        output.append(entry)
    return output


def write_csv(path: Path, rows: list[dict], default_fields: tuple[str, ...]) -> None:
    fields = list(default_fields) + sorted(set().union(*(row.keys() for row in rows))-set(default_fields))
    tmp = path.with_suffix(path.suffix+".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})
    tmp.replace(path)


def summarize_study(input_dir: Path, output_dir: Path, *, targets: dict | None = None,
                    fit: dict | None = None) -> dict:
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if input_dir == output_dir or output_dir in input_dir.parents:
        raise ValueError("summary output must be separate from (not an ancestor of) raw input")
    if targets is not None:
        validate_targets(targets)
    manifest_path = input_dir/"manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    reports, source_files, flat_cases, flat_fits, power_rows = [], [], [], [], []
    seen = set()
    for item in manifest["cases"]:
        path = (input_dir/item["file"]).resolve()
        if path.parent != input_dir or path == manifest_path:
            raise ValueError("manifest case path must be a direct child of the input directory")
        if path in seen:
            raise ValueError("duplicate manifest case file")
        seen.add(path)
        raw = path.read_bytes()
        case = json.loads(raw)
        if case.get("id") != item["id"] or case.get("study") != manifest["study"]:
            raise ValueError(f"manifest/case identity mismatch: {path.name}")
        source_files.append({"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
        report = summarize_case(case, targets=targets, fit=fit)
        reports.append(report)
        flat = {key: report[key] for key in ("id", "study", "termination_status", "last_evaluated_step", "record_count")}
        flat.update({f"parameter.{key}": value for key, value in report["parameters"].items()})
        flat.update(report["numeric_metadata"])
        flat["elapsed_seconds_including_diagnostics"] = report["elapsed_seconds_including_diagnostics"]
        for metric, values in report["accuracy"].items():
            for key in ("threshold", "first_hit_step", "ever_met", "final_value", "final_met", "available", "observation_scope"):
                flat[f"{metric}.{key}"] = values[key]
        flat_cases.append(flat)
        for metric, values in report["rate_fits"].items():
            flat_fits.append({"id": report["id"], "metric": metric, **values})
        for row in case["result"].get("summary", []):
            if "q" in row:
                power_rows.append({"id": report["id"], **report["parameters"], **row})
    output = {
        "schema_version": 2, "analysis_version": DIAGNOSTICS_VERSION,
        "analysis_metadata": metadata(), "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_files": source_files, "input_directory": str(input_dir), "study": manifest["study"],
        "source_run_metadata": manifest.get("metadata"), "source_config": manifest.get("config"),
        "target_overrides": targets, "fit_overrides": fit,
        "planned_cases": manifest.get("planned_case_count"), "reviewed_cases": len(reports),
        "termination_counts": dict(Counter(r["termination_status"] for r in reports)),
        "cases": reports, "groups": group_reports(reports),
        "interpretation": "All statuses retained; hit-time statistics are conditional on observed attainment. Legacy first hits use saved samples only.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir/"summary.json", output)
    write_csv(output_dir/"cases.csv", flat_cases, ("id", "study", "termination_status"))
    write_csv(output_dir/"rate_fits.csv", flat_fits, ("id", "metric", "status", "rho_fit", "reference_factor"))
    if power_rows:
        write_csv(output_dir/"power_entry.csv", power_rows, ("id", "q", "entry_probability"))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", action="append", default=[], metavar="METRIC=TOLERANCE",
                        help="reanalyze accuracy using saved records; raw data is never modified")
    parser.add_argument("--fit-floor", type=float)
    parser.add_argument("--fit-ceiling", type=float)
    parser.add_argument("--fit-window", type=int)
    args = parser.parse_args()
    targets = {}
    for value in args.target:
        try:
            metric, tolerance = value.split("=", 1)
            targets[metric] = float(tolerance)
        except ValueError:
            parser.error("--target requires METRIC=TOLERANCE")
    fit = {key: value for key, value in {"floor": args.fit_floor, "ceiling": args.fit_ceiling,
                                       "window": args.fit_window}.items() if value is not None}
    result = summarize_study(args.input, args.out, targets=targets or None, fit=fit or None)
    print(f"Reviewed {result['reviewed_cases']} cases: {result['termination_counts']}")
    print(f"JSON/CSV summaries written to {args.out}; raw inputs unchanged.")


if __name__ == "__main__":
    main()
