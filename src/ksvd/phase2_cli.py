"""Command-line entry point shared by the eight Phase 2 wrappers."""

from __future__ import annotations

import argparse
import inspect
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
    parser.add_argument("--workers",type=int,default=1,help="parallel case workers when supported")
    args = parser.parse_args()
    selected = experiment if experiment is not None else args.experiment
    if args.workers<1:
        parser.error("--workers must be positive")
    function=EXPERIMENTS[selected]
    supports_workers="workers" in inspect.signature(function).parameters
    if supports_workers and args.workers>1:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    started=time.perf_counter()
    result=function(smoke=not args.full,**({"workers":args.workers} if supports_workers else {}))
    result["metadata"]["requested_workers"]=args.workers
    result["metadata"]["effective_workers"]=args.workers if supports_workers else 1
    result["metadata"]["wrapper_elapsed_seconds"]=time.perf_counter()-started
    result["metadata"]["serialized_bytes"]=0
    for _ in range(3):
        result["metadata"]["serialized_bytes"]=len((json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8"))
    save_raw_result(result,args.output)
