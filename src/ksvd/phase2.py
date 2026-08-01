"""Thin, reproducible wrappers for the eight Phase 2 experiments.

The wrappers return raw, JSON-compatible records and deliberately contain no
plotting.  ``smoke=True`` selects at most two seeds and three values from every
sweep; passing ``smoke=False`` is the explicit opt-in to the full specification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from typing import Callable

import torch
from torch import Tensor

from .dynamics import ambient_update, reduced_y_update
from .initialization import minimizer_y, random_y
from .metrics import procrustes_distance, subspace_distance, tied_optimal_family_distance
from .problem import SpectralProblem, spectral_problem, xbar_to_y, y_to_xbar
from .quantities import locally_optimal_step, optimal_objective, potential, potential_gradient, predicted_local_rate, spectral_objective
from .theory import certified_step_quantities


@dataclass(frozen=True)
class Phase2Config:
    """Common stopping rules fixed by the Phase 2 specification."""

    max_steps: int = 20_000
    geometric_tolerance: float = 1e-10
    objective_tolerance: float = 1e-14
    divergence_threshold: float = 1e12


def _metadata(name: str, seed: int, config: Phase2Config, smoke: bool) -> dict[str, object]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {"experiment": name, "seed": seed, "smoke": smoke, "dtype": "torch.float64",
            "device": "cpu", "config": asdict(config), "git_commit": commit,
            "python_version": platform.python_version(), "torch_version": torch.__version__}


def save_raw_result(result: dict[str, object], path: str | Path) -> Path:
    """Atomically serialize a wrapper's complete raw result as JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def _spectrum(k: int, delta: float, *, r: int | None = None) -> list[float]:
    r = max(20, 2 * k + 2) if r is None else r
    top = ([1.0] if k == 1 else torch.linspace(2.0, 1.0, k, dtype=torch.float64).tolist())
    return top + [(1.0 - delta) * 0.8**j for j in range(r - k)]


def _fixed_condition_spectrum(k: int, delta: float, condition: float = 20.0, r: int = 20) -> list[float]:
    """Construct the Phase 2 spectrum with prescribed gap and condition number."""
    top=([1.0] if k==1 else torch.linspace(2.,1.,k,dtype=torch.float64).tolist())
    first=1.-delta; last=top[0]/condition
    return top+torch.logspace(math.log10(first),math.log10(last),r-k,dtype=torch.float64).tolist()


def _normal_local(problem: SpectralProblem, k: int, seed: int, magnitude: float = 1e-2) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    h = torch.randn(problem.r, k, generator=generator, dtype=torch.float64)
    h[:k] = 0  # normal cross-subspace component; excludes tangent rotations
    h /= torch.linalg.norm(h)
    return minimizer_y(problem, k) + magnitude * h


def _record(problem: SpectralProblem, y: Tensor, previous: Tensor | None, k: int,
            *, tied: bool = False, error_previous: float | None = None) -> tuple[dict[str, object], float]:
    xbar, target = y_to_xbar(problem, y), y_to_xbar(problem, minimizer_y(problem, k))
    objective = spectral_objective(problem, xbar)
    gap = max(0.0, float(objective - optimal_objective(problem, k)))
    factor = float(procrustes_distance(xbar, target))
    subspace = float(subspace_distance(y, minimizer_y(problem, k)))
    singular = torch.linalg.svdvals(y)
    geometric = float(tied_optimal_family_distance(problem, y, k)) if tied else factor
    row: dict[str, object] = {
        "potential": float(potential(problem, y)), "objective": float(objective), "objective_gap": gap,
        "factor_manifold_distance": factor, "subspace_error": subspace,
        "tied_optimal_family_error": geometric if tied else None,
        "gradient_norm": float(torch.linalg.norm(potential_gradient(problem, y))),
        "step_norm": None if previous is None else float(torch.linalg.norm(y - previous)),
        "sigma_min_y": float(singular[-1]), "sigma_max_y": float(singular[0]),
        "condition_y": float(singular[0] / singular[-1]),
        "error_ratio": None if error_previous in (None, 0.0) else geometric / error_previous,
        "potential_decreased": None if previous is None else bool(potential(problem, y) <= potential(problem, previous)),
    }
    return row, geometric


def _trajectory(problem: SpectralProblem, y: Tensor, k: int, eta: float, config: Phase2Config,
                *, tied: bool = False) -> dict[str, object]:
    records: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    started = time.perf_counter()
    previous = None
    previous_error = None
    termination = "max_steps"
    termination_detail: str | None = None
    for iteration in range(config.max_steps + 1):
        if not bool(torch.isfinite(y).all()):
            termination = "nan"; warnings.append({"iteration": iteration, "kind": "nonfinite", "detail": "Y contains NaN or infinity"}); break
        if float(torch.linalg.norm(y)) > config.divergence_threshold:
            termination = "diverged"; warnings.append({"iteration": iteration, "kind": "divergence", "detail": "factor norm exceeded threshold"}); break
        singular = torch.linalg.svdvals(y)
        if float(singular[-1]) <= 100 * torch.finfo(y.dtype).eps * float(singular[0]):
            termination = "lost_rank"; warnings.append({"iteration": iteration, "kind": "rank_loss", "detail": "factor rank tolerance crossed"}); break
        try:
            row, error = _record(problem, y, previous, k, tied=tied, error_previous=previous_error)
        except (ValueError, torch.linalg.LinAlgError) as error:
            # The factor-level SVD can pass the prescribed rank tolerance even
            # when the weighted Gram matrix has already become numerically
            # singular (its conditioning is approximately squared).  Treat a
            # rejected diagnostic solve as rank loss rather than allowing a
            # controlled initialization sweep to abort.
            termination = "lost_rank"
            termination_detail = str(error)
            warnings.append({"iteration": iteration, "kind": "rank_loss", "detail": str(error)})
            break
        row["iteration"] = iteration
        records.append(row)
        numeric_values=[value for value in row.values() if isinstance(value,(float,int)) and not isinstance(value,bool)]
        if not all(math.isfinite(float(value)) for value in numeric_values):
            termination="nan"; warnings.append({"iteration":iteration,"kind":"nonfinite","detail":"a recorded diagnostic is non-finite"}); break
        if abs(float(row["objective"]))>config.divergence_threshold or abs(float(row["potential"]))>config.divergence_threshold:
            termination="diverged"; warnings.append({"iteration":iteration,"kind":"divergence","detail":"objective or potential exceeded threshold"}); break
        if error <= config.geometric_tolerance or float(row["objective_gap"]) <= config.objective_tolerance:
            termination = "converged"; break
        try:
            next_y = reduced_y_update(problem, y, eta)
        except (ValueError, torch.linalg.LinAlgError) as error:
            termination = "lost_rank"
            termination_detail = str(error)
            warnings.append({"iteration": iteration, "kind": "rank_loss", "detail": str(error)})
            break
        movement = float(torch.linalg.norm(next_y - y))
        movement_floor = 100 * torch.finfo(y.dtype).eps * max(1.0, float(torch.linalg.norm(y)))
        if movement <= movement_floor:
            underflow=eta>0 and movement==0
            termination = "underflow" if underflow else "stagnated"
            warnings.append({"iteration": iteration, "kind": "update_underflow" if underflow else "stagnation",
                             "detail": "update rounded to zero" if underflow else "update fell below the float64-scaled movement threshold"})
            y = next_y
            break
        if previous is not None and float(procrustes_distance(next_y, previous)) <= 1e-12 * max(1.0, float(torch.linalg.norm(previous))):
            termination = "cycle"
            warnings.append({"iteration": iteration, "kind": "period_two_cycle", "detail": "next iterate returned to the Procrustes-aligned t-1 iterate"})
            y = next_y; break
        previous, previous_error, y = y, error, next_y
    elapsed = time.perf_counter() - started
    history_bytes = len(json.dumps(records, allow_nan=False, separators=(",", ":")).encode("utf-8"))
    return {"eta": eta, "termination": termination,
            "termination_detail": termination_detail, "records": records,
            "warnings": warnings, "elapsed_seconds": elapsed,
            "iterations_recorded": len(records), "raw_history_bytes": history_bytes,
            "final_y": y.detach().cpu().tolist()}


def _fit(records: list[dict[str, object]], key: str) -> dict[str, object]:
    points = [(int(r["iteration"]), float(r[key])) for r in records if r[key] is not None and 1e-9 <= float(r[key]) <= 1e-3]
    bounds = {"window_first": points[0][0] if points else None, "window_last": points[-1][0] if points else None}
    if len(points) < 10: return {"points": len(points), **bounds, "rho_ratio": None, "rho_slope": None, "valid": False, "invalid_reason": "fewer_than_ten_usable_points"}
    ratios = [points[i + 1][1] / points[i][1] for i in range(len(points) - 1) if points[i + 1][0] == points[i][0] + 1]
    t = torch.tensor([p[0] for p in points], dtype=torch.float64); logs = torch.log(torch.tensor([p[1] for p in points], dtype=torch.float64))
    slope = float(torch.sum((t - t.mean()) * (logs - logs.mean())) / torch.sum((t - t.mean()).square()))
    ratio = float(torch.median(torch.tensor(ratios))) if ratios else None
    return {"points": len(points), **bounds, "rho_ratio": ratio, "rho_slope": math.exp(slope),
            "valid": ratio is not None, "invalid_reason": None if ratio is not None else "no_consecutive_usable_points"}


def _relative_unique(values: list[float], tolerance: float = 1e-12) -> list[float]:
    """Return a sorted grid with relative-``tolerance`` near-duplicates removed."""
    unique: list[float] = []
    for value in sorted(float(item) for item in values if float(item) > 0):
        if not unique or abs(value - unique[-1]) > tolerance * max(abs(value), abs(unique[-1])):
            unique.append(value)
    return unique


def _ordered_parallel_map(function: Callable[[object], object], tasks: list[object], workers: int,
                          label: str) -> list[object]:
    """Run independent tasks concurrently while preserving input order."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        return [function(task) for task in tasks]
    results: list[object | None] = [None] * len(tasks)
    report_every=max(1,len(tasks)//20)
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix="ksvd") as executor:
        futures={executor.submit(function,task):index for index,task in enumerate(tasks)}
        for completed,future in enumerate(as_completed(futures),start=1):
            results[futures[future]]=future.result()
            if completed%report_every==0 or completed==len(tasks):
                print(f"{label}: {completed}/{len(tasks)} cases complete",flush=True)
    return [result for result in results if result is not None]


def _potential_hessian_norm(problem: SpectralProblem, y: Tensor) -> float:
    """Return the exact small-problem Hessian operator norm used in Experiment 5."""
    shape = y.shape
    flat = y.detach().clone().requires_grad_(True).reshape(-1)
    def differentiable_potential(value: Tensor) -> Tensor:
        factor=value.reshape(shape)
        return .5*torch.sum(factor.square())-.5*torch.linalg.slogdet(factor.T@(problem.eigenvalues[:,None]*factor)).logabsdet
    hessian = torch.autograd.functional.hessian(differentiable_potential, flat)
    return float(torch.linalg.matrix_norm(hessian, ord=2))


def _principal_angles(left: Tensor, right: Tensor) -> list[float]:
    """Return principal angles between the column spaces of full-rank factors."""
    q_left, q_right = torch.linalg.qr(left).Q, torch.linalg.qr(right).Q
    cosines = torch.linalg.svdvals(q_left.T @ q_right).clamp(0.0, 1.0)
    return torch.arccos(cosines).detach().cpu().tolist()


def _exploratory_regression(rows: list[dict[str, object]]) -> dict[str, object]:
    """Fit the descriptive ``T_lin`` model requested by Experiment 8."""
    usable=[row for row in rows if row["T_lin"] is not None and math.isfinite(float(row["initial"]["condition"]))]
    names=["intercept","minus_log_alpha_0","log_condition","abs_log_scale","initial_F"]
    if len(usable)<len(names):
        return {"label":"exploratory, not theoretical","n":len(usable),"features":names,"coefficients":None,
                "invalid_reason":"insufficient_complete_rows"}
    design=[]; target=[]
    for row in usable:
        initial=row["initial"]; value=max(float(row["value"]),torch.finfo(torch.float64).tiny)
        design.append([1.,-math.log(max(float(initial["alpha_0"]),torch.finfo(torch.float64).tiny)),
                       math.log(float(initial["condition"])),abs(math.log(value)),float(initial["F"])])
        target.append(float(row["T_lin"]))
    matrix=torch.tensor(design,dtype=torch.float64); response=torch.tensor(target,dtype=torch.float64)
    coefficients=torch.linalg.lstsq(matrix,response).solution
    return {"label":"exploratory, not theoretical","n":len(usable),"features":names,
            "coefficients":coefficients.detach().cpu().tolist(),"invalid_reason":None}


def predicted_local_rates(*, smoke: bool = True, config: Phase2Config | None = None) -> dict[str, object]:
    """Experiment 1: predicted factor and squared objective-gap rates."""
    config = config or Phase2Config(max_steps=80 if smoke else 20_000)
    cases = []
    for k in ([1, 4] if smoke else [1, 2, 4, 8]):
        p = spectral_problem(_spectrum(k, 0.3)); optimal_eta, _ = locally_optimal_step(p, k)
        for eta in ([0.25, 0.5, optimal_eta] if smoke else [0.1, 0.25, 0.5, 0.7, optimal_eta]):
            mu = 0.3; rho = max(abs(1 - 2 * eta), 1 - eta * mu)
            for seed in (range(2) if smoke else range(20)):
                run = _trajectory(p, _normal_local(p, k, seed), k, eta, config)
                run.update({"k": k, "seed": seed, "mu_k": mu, "predicted_factor_rate": rho,
                            "predicted_objective_rate": rho**2, "factor_fit": _fit(run["records"], "factor_manifold_distance"),
                            "objective_fit": _fit(run["records"], "objective_gap")}); cases.append(run)
    return {"metadata": _metadata("predicted_local_rates", 0, config, smoke), "cases": cases}


def hessian_mode_isolation(*, smoke: bool = True) -> dict[str, object]:
    """Experiment 2: centred-difference Jacobian actions for all mode families."""
    p = spectral_problem([2, 1.5, 1, .7, .5, .35, .2, .1]); k = 3; e = minimizer_y(p, k); rows = []
    etas = [.5] if smoke else [.25, .5, .7]; epsilons = [1e-5] if smoke else [1e-4, 1e-5, 1e-6]
    modes: list[tuple[str, int, int, Tensor, float]] = []
    for i in range(k):
        for j in range(i, k):
            h = torch.zeros_like(e); h[i, j] = 1; h[j, i] += 1 if i != j else 0; h /= torch.linalg.norm(h); modes.append(("symmetric", i, j, h, 2.0))
    for i in range(k):
        for j in range(k, p.r):
            h = torch.zeros_like(e); h[j, i] = 1; modes.append(("cross", i, j, h, 1 - float(p.eigenvalues[j] / p.eigenvalues[i])))
    for i in range(k):
        for j in range(i + 1, k):
            h = torch.zeros_like(e); h[i, j], h[j, i] = 1, -1; h /= torch.linalg.norm(h); modes.append(("tangent", i, j, h, 0.0))
    for eta in etas:
        for eps in epsilons:
            for family, i, j, h, eigenvalue in modes:
                action = (reduced_y_update(p, e + eps*h, eta) - reduced_y_update(p, e - eps*h, eta))/(2*eps)
                measured = float(torch.sum(action*h)); predicted = 1 - eta*eigenvalue
                absolute_error = abs(measured - predicted)
                # A relative error is undefined for the symmetric modes when
                # eta=1/2 makes their predicted factor exactly zero.  Use the
                # absolute finite-difference residual in that case instead of
                # dividing by machine epsilon and manufacturing a huge value.
                error = absolute_error if predicted == 0.0 else absolute_error / abs(predicted)
                rows.append({"family": family, "i": i, "j": j, "eta": eta, "epsilon": eps, "predicted": predicted,
                             "measured": measured, "absolute_error": absolute_error,
                             "relative_error": error,
                             "error_kind": "absolute" if predicted == 0.0 else "relative"})
    return {"metadata": _metadata("hessian_mode_isolation", 0, Phase2Config(max_steps=0), smoke), "modes": rows}


def boundary_gap_scaling(*, smoke: bool = True, workers: int = 1) -> dict[str, object]:
    """Experiment 3: eigengap-dependent transients and asymptotic rates."""
    config = Phase2Config(max_steps=120 if smoke else 20_000); tasks=[]
    for k in ([1] if smoke else [1,2,4,8]):
        for delta in ([1e-2, .1, .5] if smoke else [1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,.1,.3,.5]):
            for family in ["local_normal","support_gaussian"]:
                for seed in (range(2) if smoke else range(5 if family=="local_normal" else 20)):
                    tasks.append((k,delta,family,seed))
    def run_case(task: object) -> dict[str, object]:
        k,delta,family,seed=task
        p=spectral_problem(_spectrum(k,delta))
        y=_normal_local(p,k,seed) if family=="local_normal" else xbar_to_y(p,p.eigenvalues[:,None]*random_y(p,k,seed))
        run=_trajectory(p,y,k,.5,config); run.update({"k":k,"delta":delta,"family":family,"seed":seed,
            "predicted_rate":1-delta/2,"rate_fit":_fit(run["records"],"factor_manifold_distance")})
        return run
    cases=_ordered_parallel_map(run_case,tasks,workers,"boundary_gap_scaling")
    result={"metadata":_metadata("boundary_gap_scaling",0,config,smoke),"cases":cases}
    result["metadata"]["workers"]=workers
    return result


def tied_eigenvalues(*, smoke: bool = True) -> dict[str, object]:
    """Experiment 4: cutoff ties and isotropic column-space sanity check."""
    config=Phase2Config(max_steps=100 if smoke else 20_000); cases=[]
    for delta in ([0,.01,.1] if smoke else [0,1e-4,1e-3,1e-2,.1]):
        p=spectral_problem([2,1.5,1,1,1-delta,1-delta,.6,.4,.2,.1])
        for seed in (range(2) if smoke else range(50)):
            y=xbar_to_y(p,p.eigenvalues[:,None]*random_y(p,4,seed)); run=_trajectory(p,y,4,.5,config,tied=delta==0)
            final=torch.tensor(run["final_y"],dtype=torch.float64); canonical=minimizer_y(p,4)
            tied_block=final[2:6]
            selected=torch.linalg.qr(tied_block).Q[:, :2]
            canonical_tied=torch.eye(4,dtype=torch.float64)[:, :2]
            run.update({"delta":delta,"seed":seed,
                        "canonical_projector_error":float(subspace_distance(final,canonical)),
                        "final_tied_block_principal_angles":_principal_angles(selected,canonical_tied),
                        "iterations":len(run["records"])-1 if run["records"] else 0,
                        "rate_fit":_fit(run["records"],"tied_optimal_family_error" if delta==0 else "factor_manifold_distance")})
            cases.append(run)
    iso=spectral_problem(torch.ones(10,dtype=torch.float64)); y=random_y(iso,4,99); updated=reduced_y_update(iso,y,.5)
    identity = torch.eye(4, dtype=y.dtype, device=y.device)
    sanity={"column_space_error":float(subspace_distance(y,updated)),"gram_error_before":float(torch.linalg.norm(y.T@y-identity)),
            "gram_error_after":float(torch.linalg.norm(updated.T@updated-identity))}
    return {"metadata":_metadata("tied_eigenvalues",0,config,smoke),"cases":cases,"isotropic_sanity":sanity}


def geometry_of_kc(*, smoke: bool = True) -> dict[str, object]:
    """Experiment 5: exact scalar/block slices and empirical certificate bounds."""
    p=spectral_problem([2,.5]); grid=torch.linspace(-3,3,31 if smoke else 601,dtype=torch.float64); values=[]
    for a in grid:
        for b in grid:
            if a != 0 or b != 0: values.append([float(a),float(b),float(potential(p,torch.tensor([[a],[b]],dtype=torch.float64)))])
    p2=spectral_problem([2,1,.4]); slice_rows=[]
    for theta in torch.linspace(0,math.pi/2,5 if smoke else 201, dtype=torch.float64):
        q=torch.zeros(3, 2, dtype=torch.float64)
        q[0, 0], q[1, 1], q[2, 1] = 1.0, torch.cos(theta), torch.sin(theta)
        for s in torch.linspace(-3,3,5 if smoke else 201, dtype=torch.float64):
            scales = torch.stack((torch.exp(s), torch.exp(-s)))
            slice_rows.append({"theta":float(theta),"kind":"balanced","parameter":float(s),"F":float(potential(p2,q@torch.diag(scales)))})
        for a in torch.logspace(-2,2,5 if smoke else 201,dtype=torch.float64): slice_rows.append({"theta":float(theta),"kind":"scale","parameter":float(a),"F":float(potential(p2,a*q))})
    c_star=.5-.5*math.log(2); contour_offsets=[.05,.2,.5] if smoke else [.05,.2,.5,1.,2.]
    starts=[[-2.,-2.],[0.5,2.],[2.,-1.]] if smoke else [[a,b] for a in (-2.,-1.,1.,2.) for b in (-2.,-1.,1.,2.)]
    trajectories=[]
    for start in starts:
        y=torch.tensor(start,dtype=torch.float64).reshape(2,1); history=[]
        for iteration in range(16 if smoke else 101):
            history.append({"iteration":iteration,"y":[float(y[0,0]),float(y[1,0])],"F":float(potential(p,y))})
            y=reduced_y_update(p,y,.5)
        trajectories.append({"start":start,"records":history})
    levels=[]
    for offset in contour_offsets:
        C=float(potential(p2,minimizer_y(p2,2)))+offset; cert=certified_step_quantities(p2,2,C); samples=[]
        accepted=[("minimizer",minimizer_y(p2,2))]
        for seed in range(20 if smoke else 5000):
            base=random_y(p2,2,seed); source=("scale","condition","haar")[seed%3]
            if source=="scale": y=base*10**(-1+2*(seed%7)/6)
            elif source=="condition": y=torch.linalg.qr(base).Q@torch.diag(torch.tensor([1.,10.**(-(seed%7))],dtype=torch.float64))
            else: y=torch.linalg.qr(base).Q
            if float(potential(p2,y))<=C:
                accepted.append((source,y))
        for source,y in accepted:
            gram=y.T@(p2.eigenvalues[:,None]*y)
            samples.append({"source":source,"norm_squared":float(torch.sum(y*y)),"lambda_min_A":float(torch.linalg.eigvalsh(gram)[0]),
                            "hessian_operator_norm":_potential_hessian_norm(p2,y)})
        extrema={"max_norm_squared":max((row["norm_squared"] for row in samples),default=None),
                 "min_lambda_min_A":min((row["lambda_min_A"] for row in samples),default=None),
                 "max_hessian_operator_norm":max((row["hessian_operator_norm"] for row in samples),default=None)}
        ratios={"S_C_over_empirical_max_norm_squared":None if not samples else cert.S_C/extrema["max_norm_squared"],
                "empirical_min_lambda_min_A_over_a_C":None if not samples or cert.a_C==0 else extrema["min_lambda_min_A"]/cert.a_C,
                "hessian_bound_C_over_empirical_max":None if not samples else cert.hessian_bound_C/extrema["max_hessian_operator_norm"]}
        levels.append({"C":C,"certificate":cert.to_dict(),"samples":samples,"empirical_extrema":extrema,"bound_to_empirical_ratios":ratios})
    return {"metadata":_metadata("geometry_of_kc",0,Phase2Config(max_steps=0),smoke),
            "part_a":{"grid_size":len(grid),"values":values,"C_star":c_star,
                      "contour_levels":[c_star+offset for offset in contour_offsets],
                      "global_minima":[[-1.,0.],[1.,0.]],"nonglobal_critical_points":[[0.,-1.],[0.,1.]],
                      "trajectories":trajectories},
            "part_b":slice_rows,"part_c":{"levels":levels,"label":"empirical lower bounds on conservativeness, not exact extrema"}}


def saddle_escape(*, smoke: bool = True) -> dict[str, object]:
    """Experiment 6: escape from the specified nonglobal fixed point."""
    p=spectral_problem([2,1.5,1,.7,.5,.35,.2,.1]); saddle=torch.eye(8,dtype=torch.float64)[:,[0,1,3]]; h=torch.zeros_like(saddle); h[2,2]=1; gamma=1/.7-1; rows=[]
    for eps in ([1e-8,1e-4] if smoke else [10.**i for i in range(-12,-1)]):
        for eta in ([.5] if smoke else [.25,.5,.75]):
            for sign in [-1,1]:
                y=saddle+sign*eps*h; escape=None
                for step in range(101 if smoke else 20_001):
                    if float(procrustes_distance(y,saddle))>.1: escape=step; break
                    y=reduced_y_update(p,y,eta)
                rows.append({"epsilon":eps,"eta":eta,"sign":sign,"escape_time":escape,
                    "predicted_escape_time":math.log(.1/eps)/math.log(1+eta*gamma)})
    return {"metadata":_metadata("saddle_escape",0,Phase2Config(max_steps=100 if smoke else 20_000),smoke),"exact_fixed_error":float(torch.linalg.norm(reduced_y_update(p,saddle,.5)-saddle)),"cases":rows}


def step_size_phase_diagram(*, smoke: bool = True) -> dict[str, object]:
    """Experiment 7: certified versus grid-dependent empirical cutoffs."""
    config=Phase2Config(max_steps=50 if smoke else 20_000); rows=[]
    top=torch.linspace(2.,1.,4,dtype=torch.float64).tolist()
    def tail(first: float, last: float) -> list[float]:
        return torch.logspace(math.log10(first),math.log10(last),16,dtype=torch.float64).tolist()
    spectra={"well_separated":top+tail(.5,.1),"small_gap":top+tail(.99,.1),
             "ill_conditioned":top+tail(.8,2e-4)}
    families=["support_gaussian","reduced_gaussian","orthonormal","ill_conditioned"]
    for name,lam in spectra.items():
        p=spectral_problem(lam)
        for family in families:
            for seed in (range(1) if smoke else range(5)):
                base=random_y(p,4,seed)
                if family=="support_gaussian": y=xbar_to_y(p,p.eigenvalues[:,None]*base)
                elif family=="orthonormal": y=torch.linalg.qr(base).Q
                elif family=="ill_conditioned": y=base@torch.diag(torch.logspace(0,-6,4,dtype=torch.float64))
                else: y=base
                C=float(potential(p,y)); cert=certified_step_quantities(p,4,C); local,_=locally_optimal_step(p,4)
                etas=_relative_unique([.1*cert.eta_C,cert.eta_C,10*cert.eta_C,.5,1.,local] if smoke else [.1*cert.eta_C,cert.eta_C,10*cert.eta_C,100*cert.eta_C,.5,1.,local]+torch.logspace(-8,-1,60).tolist()+torch.linspace(.1,1.5,141).tolist())
                for eta in etas:
                    if eta<=0: continue
                    run=_trajectory(p,y,4,eta,config); decreased=all(r["potential_decreased"] is not False for r in run["records"])
                    if run["termination"]=="converged": classification="monotone_convergence" if decreased else "nonmonotone_convergence"
                    elif run["termination"] in {"max_steps","cycle","stagnated","underflow"}: classification="bounded_nonconvergence_or_cycle"
                    elif run["termination"] in {"nan","diverged"}: classification="divergence_or_nonfinite"
                    else: classification=run["termination"]
                    _,predicted=predicted_local_rate(p,4,eta)
                    rows.append({"spectrum":name,"family":family,"seed":seed,"eta":eta,"eta_C":cert.eta_C,"eta_local_star":local,
                                 "classification":classification,"monotone":decreased,"predicted_late_rate":predicted,
                                 "late_rate_fit":_fit(run["records"],"factor_manifold_distance"),"run":run})
    summaries=[]
    for name in spectra:
        for family in families:
            for seed in (range(1) if smoke else range(5)):
                group=[row for row in rows if row["spectrum"]==name and row["family"]==family and row["seed"]==seed]
                monotone=[row["eta"] for row in group if row["monotone"]]
                converged=[row["eta"] for row in group if row["classification"] in {"monotone_convergence","nonmonotone_convergence"}]
                eta_c=group[0]["eta_C"]
                desc=max(monotone) if monotone else None; conv=max(converged) if converged else None
                summaries.append({"spectrum":name,"family":family,"seed":seed,"eta_C":eta_c,
                                  "empirical_monotone_cutoff":desc,"empirical_convergence_cutoff":conv,
                                  "monotone_cutoff_over_eta_C":None if desc is None or eta_c==0 else desc/eta_c,
                                  "convergence_cutoff_over_eta_C":None if conv is None or eta_c==0 else conv/eta_c})
    return {"metadata":_metadata("step_size_phase_diagram",0,config,smoke),"cases":rows,"cutoff_summaries":summaries,
            "grid_duplicate_relative_tolerance":1e-12,"cutoffs_are_grid_dependent":True}


def initialization_ablation(*, smoke: bool = True) -> dict[str, object]:
    """Experiment 8: initialization-family descriptors, transients, and outcomes."""
    config=Phase2Config(max_steps=80 if smoke else 20_000); p=spectral_problem(_fixed_condition_spectrum(4,.2)); rows=[]
    controlled=[("scale",s) for s in ([1e-2,1,1e2] if smoke else [10.**i for i in range(-4,5)])]
    controlled += [("condition",v) for v in ([1,1e4,1e8] if smoke else [1,10,1e2,1e3,1e4,1e5,1e6,1e7,1e8])]
    controlled += [("overlap",v) for v in ([1e-8,1e-4,1.] if smoke else [10.**i for i in range(-8,1)])]
    controlled += [("near_saddle",v) for v in ([1e-8,1e-4] if smoke else [10.**i for i in range(-12,-1)])]
    controlled += [("ambient_vs_support",v) for v in ([0.,1.] if smoke else [0.,1.])]
    certified_subset=[]
    for family,value in [("support_gaussian",1.),("reduced_gaussian",1.),("orthonormal",1.),*controlled]:
        for seed in (range(2) if smoke else range(20 if family in {"support_gaussian","reduced_gaussian","orthonormal"} else 10)):
            base=random_y(p,4,seed)
            if family=="support_gaussian": y=xbar_to_y(p,p.eigenvalues[:,None]*base)
            elif family=="orthonormal": y=torch.linalg.qr(base).Q
            elif family=="scale": y=value*base
            elif family=="condition":
                q=torch.linalg.qr(base).Q; singular=torch.logspace(0,-math.log10(value),4,dtype=torch.float64) if value>1 else torch.ones(4,dtype=torch.float64); y=q@torch.diag(singular); y*=float(torch.linalg.norm(base)/torch.linalg.norm(y))
            elif family=="overlap":
                y=minimizer_y(p,4).clone(); angle=math.acos(min(1.,value)); y[3,3]=math.cos(angle); y[4,3]=math.sin(angle)
            elif family=="near_saddle":
                y=torch.eye(p.r,dtype=torch.float64)[:,[0,1,2,4]]; y[3,3]=value
            elif family=="ambient_vs_support":
                # Reduced support trajectory; the paired ambient null component
                # and its exact decay are recorded below without applying M^-1.
                y=base
            else: y=base
            singular=torch.linalg.svdvals(y); q=torch.linalg.qr(y).Q; alpha=float(torch.linalg.svdvals(q[:4])[0 if False else -1]); C=float(potential(p,y)); cert=certified_step_quantities(p,4,C)
            run=_trajectory(p,y.clone(),4,.5,config); ratios=[r for r in run["records"] if r["error_ratio"] is not None and 1e-9<=float(r["factor_manifold_distance"])<=1e-3]
            rho=max(0.,1-.5*.2); T_lin=None
            for start in range(max(0,len(ratios)-19)):
                window=ratios[start:start+20]
                if len(window)==20 and all(abs(float(r["error_ratio"])-rho)<=.1*(1-rho) for r in window): T_lin=int(window[0]["iteration"]); break
            null_decay=None
            if family=="ambient_vs_support":
                ambient=torch.cat((y_to_xbar(p,y),value*torch.ones(2,4,dtype=torch.float64))); matrix=torch.diag(torch.cat((p.eigenvalues,torch.zeros(2,dtype=torch.float64)))); updated=ambient_update(matrix,ambient,.5)
                null_decay=float(torch.linalg.norm(updated[-2:]-.5*ambient[-2:]))
            saddle=torch.eye(p.r,dtype=torch.float64)[:,[0,1,2,4]]
            row={"family":family,"value":value,"seed":seed,
                 "initial":{"F":C,"norm":float(torch.linalg.norm(y)),"sigma_min":float(singular[-1]),
                            "condition":float(singular[0]/singular[-1]),"alpha_0":alpha,
                            "nearest_tested_saddle_distance":float(procrustes_distance(y,saddle)),"eta_C":cert.eta_C},
                 "T_lin":T_lin,"total_iterations":len(run["records"])-1 if run["records"] else 0,
                 "rate_fit":_fit(run["records"],"factor_manifold_distance"),
                 "ambient_null_decay_error":null_decay,"run":run}
            rows.append(row)
            if seed==0 and family in {"support_gaussian","reduced_gaussian","orthonormal"}:
                certified_run=_trajectory(p,y.clone(),4,.5*cert.eta_C,config)
                certified_subset.append({"family":family,"seed":seed,"eta":.5*cert.eta_C,"eta_C":cert.eta_C,
                                         "strictly_below_certificate":.5*cert.eta_C<cert.eta_C,
                                         "run":certified_run})
    family_summaries=[]
    for family in sorted({row["family"] for row in rows}):
        group=[row for row in rows if row["family"]==family]; successes=sum(row["run"]["termination"]=="converged" for row in group)
        family_summaries.append({"family":family,"runs":len(group),"successes":successes,"failures":len(group)-successes,
                                 "success_rate":successes/len(group)})
    return {"metadata":_metadata("initialization_ablation",0,config,smoke),"cases":rows,
            "certified_subset":certified_subset,"family_summaries":family_summaries,
            "regression":_exploratory_regression(rows)}


EXPERIMENTS: dict[str, Callable[..., dict[str, object]]] = {
    "predicted_local_rates": predicted_local_rates, "hessian_mode_isolation": hessian_mode_isolation,
    "boundary_gap_scaling": boundary_gap_scaling, "tied_eigenvalues": tied_eigenvalues,
    "geometry_of_kc": geometry_of_kc, "saddle_escape": saddle_escape,
    "step_size_phase_diagram": step_size_phase_diagram, "initialization_ablation": initialization_ablation,
}
