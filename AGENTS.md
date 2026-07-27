# AGENTS.md

## Project objective

This repository implements and experimentally evaluates preconditioned gradient
descent for low-rank positive-semidefinite matrix approximation:

    g(X) = 1/4 ||M - XX^T||_F^2

with update

    X_{t+1}
      = X_t - eta * grad g(X_t) * (X_t^T X_t)^{-1}.

The main implementation specification is:

- `docs/experiment_implementation_spec.md`

The mathematical notation, transformations, and theoretical predictions are:

- `docs/mathematical_reference.md`

Read both documents completely before designing or modifying the experimental
code.

## Planning requirement

For this project, use an ExecPlan as described in `.agent/PLANS.md`.

Before implementation:

1. Inspect the existing repository.
2. Map the requested components onto the existing architecture.
3. Create or update `docs/plans/pgd_experiments.md`.
4. Record design decisions, assumptions, commands, progress, and validation
   results in that plan.
5. Do not replace an existing structure merely to match the proposed directory
   layout. Adapt the design to the repository.

## Scope discipline

- Preserve the mathematics in `docs/mathematical_reference.md`.
- Do not silently modify formulas, metrics, theoretical thresholds, or
  experiment definitions.
- Do not claim a theorem is numerically verified merely because a small run
  succeeds.
- Distinguish exact theoretical quantities from empirical estimates.
- Do not run full experimental sweeps unless the user explicitly requests it.
- During initial implementation, run only unit tests and minimal smoke tests.
- Do not add production dependencies without explaining why they are needed.

## Numerical requirements

- Use `torch.float64` by default for mathematical experiments.
- Never explicitly form a matrix inverse in the algorithm.
- Use `torch.linalg.solve`, Cholesky solves, or another numerically appropriate
  linear solve.
- Use `torch.linalg.slogdet` for log determinants.
- Make all stochastic experiments reproducible through explicit seeds.
- Record dtype, device, seed, configuration, git commit, and relevant package
  versions with every run.
- Detect NaNs, divergence, numerical rank loss, and cycles explicitly.
- Avoid measuring asymptotic rates after errors reach floating-point noise.

## Coordinate systems

Keep the following variables distinct:

- `X`: ambient coordinate in R^{n x k}.
- `Xbar`: support coordinate in R^{r x k}, with X = U_r Xbar.
- `Y`: Euclidean-gradient coordinate, with Xbar = Lambda^{1/2} Y.

The reduced dynamics are

    A(Y) = Y^T Lambda Y

    Y_{t+1}
      = (1 - eta) Y_t
        + eta Lambda Y_t A(Y_t)^{-1}.

Do not apply `M^{-1}` or `M^{-1/2}` to a singular ambient matrix.

## Core implementation requirements

The reusable core must include:

- spectral problem generation;
- reduced `Y`, reduced `Xbar`, and ambient update implementations;
- objective and potential evaluation;
- certified step-size constants;
- predicted local convergence rates;
- minimizer-manifold and subspace metrics;
- tied-eigenspace metrics;
- initialization families;
- generic experiment runner;
- structured result serialization;
- deterministic unit tests.

Experiment scripts should be thin wrappers around the tested core.

## Required validation tests

At minimum, test:

1. Ambient and reduced-coordinate update equivalence.
2. Potential equivalence under the coordinate transformation.
3. Objective equivalence between dense and spectral formulas.
4. Full-rank preservation for admissible steps.
5. Gradient and Hessian finite differences.
6. Hessian spectrum at a global minimizer.
7. Rank-one Heron iteration for eta = 1/2.
8. Column-space preservation for an isotropic spectrum.
9. Exact null-space decay for singular M.
10. Orthogonal-Procrustes metric invariance.
11. Correctness of the tied-optimal-family metric.
12. Certified descent below the theoretical threshold.

Use tolerances appropriate for `float64`; document each tolerance.

## Implementation quality

- Add type annotations to public functions.
- Add docstrings containing mathematical shapes and conventions.
- Validate tensor ranks, dimensions, ordering of eigenvalues, positivity, and
  full-rank assumptions.
- Keep plotting separate from numerical computation.
- Store raw numerical results before producing figures.
- Prefer small, composable functions over experiment-specific monoliths.
- Preserve existing linting, formatting, testing, and packaging conventions.
- Run the repository’s relevant tests after every milestone.

## Completion standard

A milestone is complete only when:

- its code is implemented;
- its unit tests pass;
- formulas have been checked against the mathematical reference;
- failure modes are handled explicitly;
- the ExecPlan records what changed and how it was validated;
- no full experiment sweep has been run without explicit permission.
