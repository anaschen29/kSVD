"""Shared job definitions; generators allow case-by-case resumable execution."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch
from ..problem import SpectralProblem


@dataclass
class Job:
    name: str
    parameters: dict
    run: Callable[[], dict]


def spectrum(k: int, r: int, mu: float) -> list[float]:
    if not 1 <= k < r or not 0 < mu < 1:
        raise ValueError("need 1<=k<r, 0<mu<1")
    top = torch.linspace(1.5, 1., k, dtype=torch.float64).tolist() if k > 1 else [1.]
    return top + [(1-mu)*.8**j for j in range(r-k)]


def problem(cfg: dict, k: int, mu: float, *, seed: int | None = None) -> SpectralProblem:
    return SpectralProblem.make(spectrum(k, cfg["r"], mu), n=cfg["n"], seed=seed,
                                device=cfg.get("device", "cpu"))
