"""Command-line entry point shared by the eight Phase 2 wrappers."""

from __future__ import annotations

import argparse

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
    save_raw_result(EXPERIMENTS[selected](smoke=not args.full), args.output)
