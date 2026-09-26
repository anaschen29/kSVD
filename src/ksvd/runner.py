"""Trajectory recording, explicit failure states, and no convergence-by-small-step shortcut."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Callable
import math
import torch
from torch import Tensor
from .problem import SpectralProblem
from .dynamics import ambient_step, reduced_step, potential_f, quotient
from .metrics import snapshot


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

    def __post_init__(self) -> None:
        if not math.isfinite(self.eta) or self.eta <= 0 or self.max_steps < 0:
            raise ValueError("invalid damping or step limit")
        if self.coordinates not in {"x", "y"} or self.record_every < 1:
            raise ValueError("invalid coordinates or record interval")
        if self.stop_tolerance < 0 or self.cycle_tolerance < 0 or self.norm_limit <= 0:
            raise ValueError("invalid tolerances")


@torch.no_grad()
def run_trajectory(p: SpectralProblem, initial: Tensor, config: RunConfig, *,
                   extra: Callable[[Tensor, Tensor, int], dict] | None = None) -> dict:
    state = initial.detach().clone()
    records, history = [], []
    length = 0.
    status, message = "max_steps", None
    last_step = 0.
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
            reached = (config.stop_metric is not None and row[config.stop_metric] <= config.stop_tolerance)
            if t % config.record_every == 0 or t == config.max_steps or reached:
                records.append(row)
            if reached:
                status = "target_reached"
                break
            if t == config.max_steps:
                break
            step = reduced_step if config.coordinates == "y" else ambient_step
            nxt = step(p, state, config.eta)
            last_step = float((nxt-state).norm())
            length += last_step
            scale = max(1., float(state.norm()))
            if last_step <= 10*torch.finfo(state.dtype).eps*scale:
                # A tiny step is NOT evidence of a global optimum.
                status = "stagnated"
                if not records or records[-1]["t"] != t:
                    records.append(row)
                break
            if config.cycle_tolerance and last_step > 10*config.cycle_tolerance*scale:
                # Exact repeated floating-point states are a cycle; approximate
                # recurrence can instead be slowly damped radial oscillation.
                if any(torch.equal(nxt, old) for old in history[-4:]):
                    status = "cycle_detected"
                    if not records or records[-1]["t"] != t:
                        records.append(row)
                    break
            history.append(state.clone())
            history = history[-4:]
            state = nxt
        except (ArithmeticError, torch.linalg.LinAlgError) as exc:
            status, message = "numerical_failure", str(exc)
            break
    # Potentially nonfinite states are not serialized into invalid JSON.
    final = state.cpu().tolist() if bool(torch.isfinite(state).all()) else None
    return {"config": asdict(config), "status": status, "message": message,
            "records": records, "last_evaluated_step": records[-1]["t"] if records else None,
            "final_state": final}
