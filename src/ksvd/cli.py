"""Config-driven study execution, per-case atomic files, and safe resumption."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import version, PackageNotFoundError
import json
import platform
from pathlib import Path
import subprocess
import time
import torch
import numpy as np
import matplotlib
from .theory import FORMULA_VERSION
from .diagnostics import DIAGNOSTICS_VERSION

STUDIES = ("trace_contraction", "power_sketch_entry", "eventual_rates",
           "minimizer_manifold", "jedra_shah_rank1", "damping_stress")
ROOT = Path(__file__).resolve().parents[2]


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_digest() -> str:
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        h.update(str(path.relative_to(Path(__file__).parent)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def metadata() -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = None, None
    try:
        precision_version = version("mpmath")
    except PackageNotFoundError:
        precision_version = None
    return {"mpmath": precision_version, "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": sha,
            "git_dirty": dirty, "source_sha256": source_digest(), "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
            "platform": platform.platform(), "dtype": "float64",
            "threads": torch.get_num_threads(), "formula_version": FORMULA_VERSION,
            "diagnostics_version": DIAGNOSTICS_VERSION}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    tmp.replace(path)


def deep_update(base: dict, changes: dict) -> dict:
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def planned_jobs(name: str, cfg: dict) -> list:
    if name not in STUDIES:
        raise ValueError(f"unknown study {name}")
    jobs = list(importlib.import_module(f"ksvd.studies.{name}").jobs(cfg))
    names = [job.name for job in jobs]
    if len(names) != len(set(names)):
        raise ValueError("duplicate case identifiers; refusing to overwrite measurements")
    if any(Path(name).name != name or name in {".", ".."} for name in names):
        raise ValueError("case identifiers must be plain filenames")
    return jobs


def run_study(name: str, cfg: dict, out: Path, *, resume: bool = False) -> dict:
    jobs = planned_jobs(name, cfg)
    out = Path(out)
    meta = metadata()
    signature = digest({"study": name, "config": cfg, "source": meta["source_sha256"],
                        "torch": meta["torch"], "numpy": meta["numpy"], "mpmath": meta["mpmath"], "threads": meta["threads"]})
    manifest_path = out/"manifest.json"
    previous = None
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if not resume:
            raise FileExistsError(f"{manifest_path} exists; use --resume or a new output directory")
        if previous["signature"] != signature:
            raise ValueError("refusing resume: configuration, source, environment, or threads changed")
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError("nonempty output directory has no compatible manifest")
    manifest = {"schema_version": 2, "study": name, "config": cfg,
                "metadata": previous["metadata"] if previous else meta,
                "signature": signature, "planned_case_count": len(jobs),
                "cases": list(previous.get("cases", [])) if previous else []}
    completed = {row["id"]: row for row in manifest["cases"]}
    atomic_json(manifest_path, manifest)
    for index, job in enumerate(jobs, 1):
        path = out/(job.name+".json")
        case_sig = digest({"signature": signature, "parameters": job.parameters})
        reused = False
        if resume and path.exists():
            case = json.loads(path.read_text())
            if case.get("signature") != case_sig:
                raise ValueError(f"refusing incompatible case {path}")
            reused = case.get("result", {}).get("status") != "failed"
        if not reused:
            started = time.perf_counter()
            try:
                result = job.run()
            except Exception as exc:
                result = {"status": "failed", "exception": type(exc).__name__, "message": str(exc)}
            numeric = result.get("numeric_metadata", {"dtype": "float64", "device": cfg.get("device", "cpu")})
            case = {"schema_version": 2, "study": name, "id": job.name, "signature": case_sig,
                    "parameters": job.parameters, "metadata": {**meta, **numeric}, "result": result,
                    "elapsed_seconds_including_diagnostics": time.perf_counter()-started}
            atomic_json(path, case)
        completed[job.name] = {"id": job.name, "file": path.name,
                               "status": case["result"]["status"], "reused": reused}
        manifest["cases"] = [completed[j.name] for j in jobs if j.name in completed]
        atomic_json(manifest_path, manifest)
        accuracy = case["result"].get("accuracy", {}).get("product_error", {})
        attained = ""
        if accuracy.get("final_value") is not None:
            attained = (f"; product={accuracy['final_value']:.3g}, "
                        f"first <= {accuracy['threshold']:g}: {accuracy['first_hit_step']}")
        print(f"[{index}/{len(jobs)}] {job.name}: {case['result']['status']}" + attained
              + (" (reused)" if reused else ""), flush=True)
    return manifest


def main(study: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    if study is None:
        parser.add_argument("--experiment", choices=STUDIES, required=True)
    parser.add_argument("--preset", choices=("smoke", "collection", "paper"), default="smoke")
    parser.add_argument("--config", type=Path, help="JSON overrides for the selected study")
    parser.add_argument("--out", type=Path, help="new output directory (required unless --dry-run)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="show effective config and case counts; do not run trajectories")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--high-precision", action="store_true", help="optional mpmath quadratic reference")
    args = parser.parse_args()
    name = study or args.experiment
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.out is None and not args.dry_run:
        parser.error("--out is required unless --dry-run is set")
    torch.set_num_threads(args.threads)
    preset_path = Path(__file__).parent/"presets"/(args.preset+".json")
    cfg = json.loads(preset_path.read_text())[name]
    if args.config:
        deep_update(cfg, json.loads(args.config.read_text()))
    cfg["device"] = args.device
    cfg["preset"] = args.preset
    if args.high_precision:
        if name != "eventual_rates":
            parser.error("--high-precision only applies to eventual_rates")
        cfg["high_precision"] = True
    if args.dry_run:
        from collections import Counter
        jobs = planned_jobs(name, cfg)
        print(json.dumps({"study": name, "case_count": len(jobs), "effective_config": cfg,
                          "cases_by_start": dict(Counter(j.parameters.get("start", "other") for j in jobs))}, indent=2))
        return
    manifest = run_study(name, cfg, args.out, resume=args.resume)
    if any(c["status"] in {"failed", "nonfinite", "numerical_failure", "norm_limit"} for c in manifest["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
