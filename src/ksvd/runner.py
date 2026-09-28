"""Trajectory recording with separate stopping reasons and attained accuracy."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Callable
import math
import torch
from torch import Tensor
from .problem import SpectralProblem
from .dynamics import ambient_step, reduced_step, potential_f, quotient
from .metrics import snapshot
from .diagnostics import AccuracyTracker, DEFAULT_TARGETS, validate_targets


@dataclass(frozen=True)
class RunConfig:
    eta: float = .5
    max_steps: int = 500
    coordinates: str = "x"
    record_every: int = 1
    stop_metric: str | None = None
    stop_tolerance: float = 1e-10
    norm_limit: float = 1e100
    cycle_tolerance: float = 1e-12
    accuracy_targets: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TARGETS))

    def __post_init__(self) -> None:
        if not math.isfinite(self.eta) or self.eta <= 0 or self.max_steps < 0:
            raise ValueError("invalid damping or step limit")
        if not isinstance(self.max_steps, int) or not isinstance(self.record_every, int):
            raise ValueError("step limit and record interval must be integers")
        if self.coordinates not in {"x", "y"} or self.record_every < 1:
            raise ValueError("invalid coordinates or record interval")
        if (not all(math.isfinite(v) for v in (self.stop_tolerance, self.cycle_tolerance, self.norm_limit))
                or self.stop_tolerance < 0 or self.cycle_tolerance < 0 or self.norm_limit <= 0):
            raise ValueError("invalid tolerances")
        validate_targets(self.accuracy_targets)


@torch.no_grad()
def run_trajectory(p: SpectralProblem, initial: Tensor, config: RunConfig, *,
                   extra: Callable[[Tensor, Tensor, int], dict] | None = None) -> dict:
    state = initial.detach().clone()
    records, history = [], []
    length = 0.
    status, message = "max_steps", None
    last_step = 0.
    last_row = None
    completed_updates, update_attempts = 0, 0
    targets = dict(config.accuracy_targets)
    if config.stop_metric is not None:
        targets.setdefault(config.stop_metric, config.stop_tolerance)
    accuracy = AccuracyTracker(targets)
    for t in range(config.max_steps+1):
        try:
            if not bool(torch.isfinite(state).all()):
                status = "nonfinite"
                break
            if float(state.norm()) > config.norm_limit:
                status = "norm_limit"
                break
            x = p.to_x(state) if config.coordinates == "y" else state
            row = snapshot(p, x, y=state if config.coordinates == "y" else None)
            if config.coordinates == "y":
                row.update(potential=float(potential_f(p, state)), quotient=float(quotient(p, state)))
            row.update(t=t, step_norm=last_step, cumulative_length=length)
            if extra:
                row.update(extra(state, x, t))
            if not all(not isinstance(v, float) or math.isfinite(v) for v in row.values()):
                raise ArithmeticError("nonfinite diagnostic")
            new_hit = accuracy.observe(row)
            last_row = row
            reached = (config.stop_metric is not None and row[config.stop_metric] <= config.stop_tolerance)
            if t % config.record_every == 0 or t == config.max_steps or reached or new_hit:
                records.append(row)
            if reached:
                status = "target_reached"
                break
            if t == config.max_steps:
                break
            step = reduced_step if config.coordinates == "y" else ambient_step
            update_attempts += 1
            nxt = step(p, state, config.eta)
            candidate_length = float((nxt-state).norm())
            scale = max(1., float(state.norm()))
            if candidate_length <= 10*torch.finfo(state.dtype).eps*scale:
                # Accuracy is decided by errors, not this stagnation criterion.
                status = "stagnated"
                break
            if config.cycle_tolerance and candidate_length > 10*config.cycle_tolerance*scale:
                if any(torch.equal(nxt, old) for old in history[-4:]):
                    status = "cycle_detected"
                    break
            history.append(state.clone())
            history = history[-4:]
            state = nxt
            # Only count accepted increments, not a discarded terminal candidate.
            completed_updates += 1
            last_step = candidate_length
            length += candidate_length
        except (ArithmeticError, torch.linalg.LinAlgError) as exc:
            status, message = "numerical_failure", str(exc)
            break
    if last_row is not None and (not records or records[-1]["t"] != last_row["t"]):
        records.append(last_row)
    final = state.cpu().tolist() if bool(torch.isfinite(state).all()) else None
    evaluated_final = last_row is not None and last_row["t"] == completed_updates
    return {"config": asdict(config), "status": status, "message": message,
            "records": records, "last_evaluated_step": last_row["t"] if last_row else None,
            "completed_updates": completed_updates, "update_attempts": update_attempts,
            "final_state_evaluated": evaluated_final,
            "last_evaluated_metrics": last_row,
            "final_metrics": last_row if evaluated_final else None,
            "accuracy": accuracy.summary(completed_updates),
            "initial_state": initial.cpu().tolist() if bool(torch.isfinite(initial).all()) else None,
            "final_state": final,
            "problem": {"n": p.n, "r": p.r, "k": state.shape[1],
                        "positive_eigenvalues": p.lam.cpu().tolist()}}
