"""Core Phase 1 API for preconditioned PSD factorization experiments."""

from .dynamics import ambient_update, reduced_xbar_update, reduced_y_update
from .initialization import aligned_y, minimizer_y, perturbed_minimizer_y, random_y
from .metrics import procrustes_distance, subspace_distance, tied_optimal_family_distance
from .problem import SpectralProblem, spectral_problem, x_to_xbar, xbar_to_x, xbar_to_y, y_to_xbar
from .quantities import (
    dense_objective,
    locally_optimal_step,
    objective_gradient,
    objective_hessian_action,
    optimal_objective,
    potential,
    potential_gradient,
    predicted_local_rate,
    spectral_objective,
)
from .runner import RunConfig, RunResult, run_deterministic
from .theory import CertifiedStep, certified_step_quantities
from .phase2 import (
    Phase2Config, boundary_gap_scaling, geometry_of_kc, hessian_mode_isolation,
    initialization_ablation, predicted_local_rates, saddle_escape,
    save_raw_result, step_size_phase_diagram, tied_eigenvalues,
)

__all__ = [
    "CertifiedStep", "RunConfig", "RunResult", "SpectralProblem", "aligned_y",
    "ambient_update", "certified_step_quantities", "dense_objective",
    "locally_optimal_step", "minimizer_y", "objective_gradient",
    "objective_hessian_action", "optimal_objective", "perturbed_minimizer_y",
    "potential", "potential_gradient", "predicted_local_rate",
    "procrustes_distance", "random_y", "reduced_xbar_update",
    "reduced_y_update", "run_deterministic", "spectral_objective",
    "spectral_problem", "subspace_distance", "tied_optimal_family_distance",
    "x_to_xbar", "xbar_to_x", "xbar_to_y", "y_to_xbar", "Phase2Config",
    "boundary_gap_scaling", "geometry_of_kc", "hessian_mode_isolation",
    "initialization_ablation", "predicted_local_rates", "saddle_escape",
    "save_raw_result", "step_size_phase_diagram", "tied_eigenvalues",
]
