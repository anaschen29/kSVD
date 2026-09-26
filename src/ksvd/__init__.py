"""Fixed-damping preconditioned gradient descent, not dictionary-learning K-SVD."""

from .problem import SpectralProblem
from .dynamics import ambient_step, reduced_step
from .runner import RunConfig, run_trajectory

__all__ = ["SpectralProblem", "ambient_step", "reduced_step", "RunConfig", "run_trajectory"]
