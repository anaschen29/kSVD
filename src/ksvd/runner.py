"""Generic deterministic trajectory runner with structured results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import platform
import subprocess
from typing import Callable

import torch
from torch import Tensor

from .dynamics import reduced_y_update
from .problem import SpectralProblem, _check_factor
from .quantities import potential
from .theory import CertifiedStep


@dataclass(frozen=True)
class RunConfig:
    """Configuration for a reduced-coordinate deterministic run."""

    eta: float
    max_steps: int
    seed: int
    divergence_threshold: float = 1e12
    rank_tolerance: float | None = None
    cycle_tolerance: float = 1e-12
    cycle_history: int = 8

    def __post_init__(self) -> None:
        if self.eta <= 0 or self.max_steps < 0 or self.divergence_threshold <= 0:
            raise ValueError("eta/divergence threshold must be positive and max_steps nonnegative")
        if self.cycle_tolerance < 0 or self.cycle_history < 1:
            raise ValueError("invalid cycle controls")


@dataclass
class RunResult:
    """Raw trajectory, diagnostics, termination reason, and reproducibility metadata."""

    states: list[Tensor]
    potentials: list[float]
    diagnostics: list[dict[str, float]]
    termination: str
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Convert tensors to nested lists for structured serialization."""
        return {
            "states": [state.detach().cpu().tolist() for state in self.states],
            "potentials": self.potentials,
            "diagnostics": self.diagnostics,
            "termination": self.termination,
            "metadata": self.metadata,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the complete raw result to JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_deterministic(
    problem: SpectralProblem,
    initial_y: Tensor,
    config: RunConfig,
    *,
    diagnostic: Callable[[Tensor], dict[str, float]] | None = None,
    certified_step: CertifiedStep | None = None,
) -> RunResult:
    """Run reduced ``Y (r,k)`` dynamics and explicitly guard failure modes.

    Termination is one of ``max_steps``, ``nonfinite``, ``diverged``,
    ``rank_loss``, or ``cycle``. A fixed point is not labelled a cycle. When a
    certified threshold is supplied, its flat value/log mapping is stored at
    ``metadata["theory"]["certified_step"]``.
    """
    _check_factor(initial_y, problem.r, "initial_y")
    y = initial_y.detach().clone()
    metadata: dict[str, object] = {
        "seed": config.seed,
        "dtype": str(y.dtype),
        "device": str(y.device),
        "config": asdict(config),
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    if certified_step is not None:
        metadata["theory"] = {"certified_step": certified_step.to_dict()}
    states, potentials, diagnostics = [y.clone()], [], []
    termination = "max_steps"
    history: list[Tensor] = []

    for step in range(config.max_steps + 1):
        if not bool(torch.isfinite(y).all()):
            termination = "nonfinite"
            break
        norm = float(torch.linalg.norm(y))
        if norm > config.divergence_threshold:
            termination = "diverged"
            break
        singular = torch.linalg.svdvals(y)
        rank_tol = config.rank_tolerance
        if rank_tol is None:
            rank_tol = torch.finfo(y.dtype).eps * max(y.shape) * float(singular[0])
        if float(singular[-1]) <= rank_tol:
            termination = "rank_loss"
            break
        try:
            potentials.append(float(potential(problem, y)))
        except (ValueError, torch.linalg.LinAlgError):
            termination = "rank_loss"
            break
        diagnostics.append({} if diagnostic is None else diagnostic(y))
        if step == config.max_steps:
            break
        next_y = reduced_y_update(problem, y, config.eta)
        # Compare with older states only; equality to the current state is a
        # converged fixed point, not a nontrivial cycle.
        for old in history[-config.cycle_history :]:
            scale = max(1.0, float(torch.linalg.norm(old)))
            if float(torch.linalg.norm(next_y - old)) <= config.cycle_tolerance * scale:
                termination = "cycle"
                y = next_y
                states.append(y.clone())
                break
        if termination == "cycle":
            break
        history.append(y.clone())
        y = next_y
        states.append(y.clone())

    return RunResult(states, potentials, diagnostics, termination, metadata)
