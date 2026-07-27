# Phase 1 PGD experiments ExecPlan

This is a living plan. It follows the project `AGENTS.md`; the referenced
`.agent/PLANS.md` is not present in commit `d434e00`, so the conventional
self-contained ExecPlan sections below are used. This document must remain
usable by a contributor who has only this checkout.

## Purpose and scope

Phase 1 will provide a tested Python/PyTorch numerical core for the spectral
positive-semidefinite approximation problem. A user will be able to construct a
spectral problem, move without ambiguity among ambient `X`, support `Xbar`, and
Euclidean-gradient `Y` coordinates, take any of the three equivalent PGD
updates, evaluate theory quantities and metrics, construct reproducible
initializations, and execute a deterministic, guarded trajectory. Scientific
sweeps, plotting, and algorithm comparisons remain out of scope.

## Repository map and design decisions

At the start, the repository contains only `AGENTS.md` and two documentation
files; there is no package, dependency manifest, test framework configuration,
or existing code/style convention. We therefore add a conventional `src`
package (`src/ksvd`) with a minimal `pyproject.toml`, using the already required
PyTorch numerical stack and pytest for tests. Public functions are re-exported
from `ksvd.__init__`; computation stays in small modules rather than scripts.

The spectral problem stores descending strictly-positive support eigenvalues
and an orthonormal support basis. Transformations never apply an inverse of the
singular ambient matrix. All right multiplication by a Gram inverse is
implemented as a transposed `torch.linalg.solve`. Calculations default to
`torch.float64`, preserve the input device, and reject incompatible shapes or
rank-deficient factors.

The supplied mathematical reference requests exact paper definitions of
`S_C`, `a_C`, `Gamma_C`, `L_C`, `d_C`, and `eta_C`, but does not actually state
them or identify the paper. Phase 1 uses explicit conservative certified
sublevel bounds derived directly from the displayed potential: `S_C` is an
upper bound on squared Frobenius norm, `a_C` a lower Gram-eigenvalue bound,
`Gamma_C` a gradient bound, `L_C` a Hessian/operator Lipschitz bound, `d_C` the
distance-to-boundary safeguard, and `eta_C` the minimum descent/safeguard step.
The formulas and derivation are documented in code and are not presented as
the unnamed paper's unavailable exact constants.

## Coordinate and formula invariants

For `eigenvalues = (lambda_1,...,lambda_r)`, `X = U_r Xbar` and
`Xbar = diag(sqrt(lambda)) Y`. The right solve `B A^{-1}` is evaluated as
`solve(A, B.T).T`. The spectral objective includes the ambient null-space only
through zero eigenvalues (and hence no extra term). The tied-family metric
splits eigenvectors strictly above the cutoff from the entire eigenvalue block
equal to the cutoff and measures both mandatory-subspace omission and excess
energy outside the eligible block.

## Milestones and progress

- [x] Read all present project instructions and both mathematical documents in
  full; inventory the complete repository and git history. (`.agent/PLANS.md`
  was confirmed absent.)
- [x] Record repository mapping, assumptions, and formula invariants here.
- [x] Add packaging and the spectral problem/coordinate/dynamics core.
- [x] Add objectives, potential, theory constants/rates, metrics, and
  initialization families.
- [x] Add deterministic runner, structured serialization, and explicit NaN,
  divergence, numerical-rank-loss, and cycle termination.
- [x] Add all twelve required deterministic validation tests with documented
  float64 tolerances.
- [x] Attempt unit tests and run syntax/static smoke checks only; execution is
  blocked because PyTorch is absent and the package index is unreachable.
- [x] Perform and record the seven-item adversarial audit.

## Validation plan

Use `python -m pytest` for the twelve required mathematical validations. Use a
single short runner invocation as the smoke test. No grids, sweeps, or claims of
theorem verification are permitted. Tests will cover coordinate/update and
potential transformations, dense/spectral objective identity, admissible-step
rank preservation, finite-difference derivatives, minimizer Hessian spectrum,
the scalar Heron map, isotropic column-space preservation, ambient null decay,
Procrustes invariance, tied-cutoff geometry, and one-step certified descent.

## Discoveries and risks

- 2026-07-27: `.agent/PLANS.md` is absent from the repository and all visible
  history. This plan records its own conventions so work can proceed.
- 2026-07-27: The mathematical reference names but omits the certified-sublevel
  formulas and does not cite the paper. Conservative independently derived
  certificates are isolated and clearly labelled pending a source-of-truth.
- Near-rank-deficient Gram systems are checked before solves using singular
  values scaled by matrix size and machine epsilon; tests must exercise this.

## Results and retrospective

Phase 1 core and its deterministic tests are implemented. No scientific sweep
was run. Validation commands and outcomes:

- 2026-07-27 follow-up: added root `README.md` setup instructions for standard
  `venv`/pip and `uv`, dependency requirements, installation verification,
  complete and targeted pytest invocations, and the syntax-only check.

- `python -m pytest`: failed during collection because the base Python 3.14
  environment has no `torch` module.
- `uv venv --python /usr/bin/python3 .venv && uv pip install --python
  .venv/bin/python 'torch>=2.0' 'pytest>=7'`: created a local Python 3.12
  environment, then failed after three retries because the PyPI tunnel is
  unavailable. The untracked `.venv` is excluded from delivery.
- `python -m compileall -q src tests`: passed.
- `git diff --check`: passed.

Adversarial audit (2026-07-27):

1. **Coordinates:** substitution of `Xbar=sqrt(Lambda)Y` in both support
   dynamics and the potential gives the implemented maps; tests compare all
   three update paths.
2. **Solve orientation:** every expression `B A^-1` uses
   `solve(A, B.T).T`; no inverse is formed.
3. **Indexing:** Python entries `k-1` and `k` implement mathematical
   `lambda_k` and `lambda_(k+1)`; full-rank `k=r` is rejected by the gap-rate
   function and separately documented by the reference's `|1-2 eta|` rule.
4. **Procrustes:** for `min_Q ||X-target Q||`, the SVD is taken of
   `target.T@X` and `Q=U@Vh`; a nontrivial right rotation is tested.
5. **Ties:** the metric separately penalizes omission of directions strictly
   above the cutoff and mass outside the whole cutoff eigenspace. Its zero set
   is precisely the eligible optimal family of `k`-planes.
6. **Near rank loss:** both update solves and metrics use a scale-aware
   singular-value threshold; the runner terminates with `rank_loss`, and a
   nearly dependent factor test checks pre-solve rejection.
7. **Objectives:** expansion of `||Lambda-Xbar Xbar.T||_F^2/4` matches all
   three spectral terms and is compared directly with the dense ambient
   calculation in float64.

The execution limitation means the completion criterion “unit tests pass”
cannot be empirically established in this container. The tests are present but
must be run in an environment providing the declared PyTorch dependency. Also,
the source-of-truth paper formulas and `.agent/PLANS.md` remain external inputs
that were absent from this checkout; the conservative certificate is clearly
identified rather than misrepresented as the missing paper definition.
