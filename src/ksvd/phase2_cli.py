"""Command-line entry point shared by the eight Phase 2 wrappers."""

from __future__ import annotations

import argparse
import json
import time

from .phase2 import EXPERIMENTS, save_raw_result


def main(experiment: str | None = None) -> None:
    """Run one experiment in smoke mode by default and serialize raw JSON."""
    parser = argparse.ArgumentParser()
    if experiment is None:
        parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--full", action="store_true", help="explicitly opt into the full specified sweep")
    args = parser.parse_args()
    selected = experiment if experiment is not None else args.experiment
    started=time.perf_counter()
    result=EXPERIMENTS[selected](smoke=not args.full)
    result["metadata"]["wrapper_elapsed_seconds"]=time.perf_counter()-started
    result["metadata"]["serialized_bytes"]=0
    for _ in range(3):
        result["metadata"]["serialized_bytes"]=len((json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8"))
    save_raw_result(result,args.output)
