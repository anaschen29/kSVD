"""Thin, reproducible wrappers for the eight Phase 2 experiments.

The wrappers return raw, JSON-compatible records and deliberately contain no
plotting.  ``smoke=True`` selects at most two seeds and three values from every
sweep; passing ``smoke=False`` is the explicit opt-in to the full specification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Callable

import torch
from torch import Tensor

from .dynamics import ambient_update, reduced_y_update
from .initialization import minimizer_y, random_y
from .metrics import procrustes_distance, subspace_distance, tied_optimal_family_distance
from .problem import SpectralProblem, spectral_problem, xbar_to_y, y_to_xbar
from .quantities import locally_optimal_step, optimal_objective, potential, potential_gradient, spectral_objective
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
    previous = None
    previous_error = None
    termination = "max_steps"
    termination_detail: str | None = None
    for iteration in range(config.max_steps + 1):
        if not bool(torch.isfinite(y).all()): termination = "nan"; break
        if float(torch.linalg.norm(y)) > config.divergence_threshold: termination = "diverged"; break
        singular = torch.linalg.svdvals(y)
        if float(singular[-1]) <= 100 * torch.finfo(y.dtype).eps * float(singular[0]): termination = "lost_rank"; break
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
            break
        row["iteration"] = iteration
        records.append(row)
        if error <= config.geometric_tolerance or float(row["objective_gap"]) <= config.objective_tolerance:
            termination = "converged"; break
        try:
            next_y = reduced_y_update(problem, y, eta)
        except (ValueError, torch.linalg.LinAlgError) as error:
            termination = "lost_rank"
            termination_detail = str(error)
            break
        if previous is not None and float(procrustes_distance(next_y, previous)) <= 1e-12 * max(1.0, float(torch.linalg.norm(previous))):
            termination = "cycle"; y = next_y; break
        previous, previous_error, y = y, error, next_y
    return {"eta": eta, "termination": termination,
            "termination_detail": termination_detail, "records": records,
            "final_y": y.detach().cpu().tolist()}


def _fit(records: list[dict[str, object]], key: str) -> dict[str, float | int | None]:
    points = [(int(r["iteration"]), float(r[key])) for r in records if r[key] is not None and 1e-9 <= float(r[key]) <= 1e-3]
    if len(points) < 10: return {"points": len(points), "rho_ratio": None, "rho_slope": None}
    ratios = [points[i + 1][1] / points[i][1] for i in range(len(points) - 1) if points[i + 1][0] == points[i][0] + 1]
    t = torch.tensor([p[0] for p in points], dtype=torch.float64); logs = torch.log(torch.tensor([p[1] for p in points], dtype=torch.float64))
    slope = float(torch.sum((t - t.mean()) * (logs - logs.mean())) / torch.sum((t - t.mean()).square()))
    return {"points": len(points), "rho_ratio": float(torch.median(torch.tensor(ratios))) if ratios else None, "rho_slope": math.exp(slope)}


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


def boundary_gap_scaling(*, smoke: bool = True) -> dict[str, object]:
    """Experiment 3: eigengap-dependent transients and asymptotic rates."""
    config = Phase2Config(max_steps=120 if smoke else 20_000); cases=[]
    for k in ([1] if smoke else [1,2,4,8]):
        for delta in ([1e-2, .1, .5] if smoke else [1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,.1,.3,.5]):
            p=spectral_problem(_spectrum(k,delta))
            for family in ["local_normal","support_gaussian"]:
                for seed in (range(2) if smoke else range(5 if family=="local_normal" else 20)):
                    y=_normal_local(p,k,seed) if family=="local_normal" else xbar_to_y(p,p.eigenvalues[:,None]*random_y(p,k,seed))
                    run=_trajectory(p,y,k,.5,config); run.update({"k":k,"delta":delta,"family":family,"seed":seed,
                        "predicted_rate":1-delta/2,"rate_fit":_fit(run["records"],"factor_manifold_distance")}); cases.append(run)
    return {"metadata":_metadata("boundary_gap_scaling",0,config,smoke),"cases":cases}


def tied_eigenvalues(*, smoke: bool = True) -> dict[str, object]:
    """Experiment 4: cutoff ties and isotropic column-space sanity check."""
    config=Phase2Config(max_steps=100 if smoke else 20_000); cases=[]
    for delta in ([0,.01,.1] if smoke else [0,1e-4,1e-3,1e-2,.1]):
        p=spectral_problem([2,1.5,1,1,1-delta,1-delta,.6,.4,.2,.1])
        for seed in (range(2) if smoke else range(50)):
            y=xbar_to_y(p,p.eigenvalues[:,None]*random_y(p,4,seed)); run=_trajectory(p,y,4,.5,config,tied=delta==0)
            run.update({"delta":delta,"seed":seed}); cases.append(run)
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
    C=float(potential(p2,minimizer_y(p2,2)))+.5; cert=certified_step_quantities(p2,2,C); samples=[]
    for seed in range(20 if smoke else 5000):
        y=random_y(p2,2,seed,scale=10**(-1+2*(seed%7)/6))
        if float(potential(p2,y))<=C:
            gram=y.T@(p2.eigenvalues[:,None]*y); samples.append({"norm_squared":float(torch.sum(y*y)),"lambda_min_A":float(torch.linalg.eigvalsh(gram)[0])})
    return {"metadata":_metadata("geometry_of_K_C",0,Phase2Config(max_steps=0),smoke),"part_a":{"grid_size":len(grid),"values":values},
            "part_b":slice_rows,"part_c":{"C":C,"certificate":cert.to_dict(),"samples":samples,"label":"empirical lower bounds on conservativeness, not exact extrema"}}


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
    spectra={"well_separated":_spectrum(4,.5),"small_gap":_spectrum(4,.01),"ill_conditioned":list(torch.logspace(0,-4,20,dtype=torch.float64))}
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
                etas=sorted(set([.1*cert.eta_C,cert.eta_C,10*cert.eta_C,.5,1.,local])) if smoke else sorted(set([.1*cert.eta_C,cert.eta_C,10*cert.eta_C,100*cert.eta_C,.5,1.,local]+torch.logspace(-8,-1,60).tolist()+torch.linspace(.1,1.5,141).tolist()))
                for eta in etas:
                    if eta<=0: continue
                    run=_trajectory(p,y,4,eta,config); decreased=all(r["potential_decreased"] is not False for r in run["records"])
                    classification=("monotone_convergence" if decreased else "nonmonotone_convergence") if run["termination"]=="converged" else run["termination"]
                    rows.append({"spectrum":name,"family":family,"seed":seed,"eta":eta,"eta_C":cert.eta_C,"eta_local_star":local,"classification":classification,"run":run})
    return {"metadata":_metadata("step_size_phase_diagram",0,config,smoke),"cases":rows,"cutoffs_are_grid_dependent":True}


def initialization_ablation(*, smoke: bool = True) -> dict[str, object]:
    """Experiment 8: initialization-family descriptors, transients, and outcomes."""
    config=Phase2Config(max_steps=80 if smoke else 20_000); p=spectral_problem(_spectrum(4,.2)); rows=[]
    controlled=[("scale",s) for s in ([1e-2,1,1e2] if smoke else [10.**i for i in range(-4,5)])]
    controlled += [("condition",v) for v in ([1,1e4,1e8] if smoke else [1,10,1e2,1e3,1e4,1e5,1e6,1e7,1e8])]
    controlled += [("overlap",v) for v in ([1e-8,1e-4,1.] if smoke else [10.**i for i in range(-8,1)])]
    controlled += [("near_saddle",v) for v in ([1e-8,1e-4] if smoke else [10.**i for i in range(-12,-1)])]
    controlled += [("ambient_vs_support",v) for v in ([0.,1.] if smoke else [0.,1.])]
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
            run=_trajectory(p,y,4,.5,config); ratios=[r for r in run["records"] if r["error_ratio"] is not None and 1e-9<=float(r["factor_manifold_distance"])<=1e-3]
            rho=max(0.,1-.5*.2); T_lin=None
            for start in range(max(0,len(ratios)-19)):
                window=ratios[start:start+20]
                if len(window)==20 and all(abs(float(r["error_ratio"])-rho)<=.1*(1-rho) for r in window): T_lin=int(window[0]["iteration"]); break
            null_decay=None
            if family=="ambient_vs_support":
                ambient=torch.cat((y_to_xbar(p,y),value*torch.ones(2,4,dtype=torch.float64))); matrix=torch.diag(torch.cat((p.eigenvalues,torch.zeros(2,dtype=torch.float64)))); updated=ambient_update(matrix,ambient,.5)
                null_decay=float(torch.linalg.norm(updated[-2:]-.5*ambient[-2:]))
            rows.append({"family":family,"value":value,"seed":seed,"initial":{"F":C,"norm":float(torch.linalg.norm(y)),"sigma_min":float(singular[-1]),"condition":float(singular[0]/singular[-1]),"alpha_0":alpha,"nearest_tested_saddle_distance":None,"eta_C":cert.eta_C},"T_lin":T_lin,"ambient_null_decay_error":null_decay,"run":run})
    return {"metadata":_metadata("initialization_ablation",0,config,smoke),"cases":rows,"regression_label":"exploratory, not theoretical"}


EXPERIMENTS: dict[str, Callable[..., dict[str, object]]] = {
    "predicted_local_rates": predicted_local_rates, "hessian_mode_isolation": hessian_mode_isolation,
    "boundary_gap_scaling": boundary_gap_scaling, "tied_eigenvalues": tied_eigenvalues,
    "geometry_of_kc": geometry_of_kc, "saddle_escape": saddle_escape,
    "step_size_phase_diagram": step_size_phase_diagram, "initialization_ablation": initialization_ablation,
}
