# Phase 1 and Phase 2 PGD experiments ExecPlan

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

The current milestone adds the eight Phase 2 experiment wrappers.  Each wrapper
owns the construction and metrics fixed by `docs/phase2_experiment_spec.md`,
accepts an explicit smoke/full mode, and writes its complete raw numerical
payload before any future plotting step.  Only smoke mode will be exercised in
this milestone; full sweeps remain an explicit user action.

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

The certified constants use the authoritative manuscript-v1 definitions from
the section “A sufficient step-size bound.” The code names the manuscript's
`mathcal L_C` as `hessian_bound_C` to distinguish it from `L=lambda_1`, and
labels `eta_C` as a sufficient, non-sharp threshold requiring `eta < eta_C`.

## Coordinate and formula invariants

For `eigenvalues = (lambda_1,...,lambda_r)`, `X = U_r Xbar` and
`Xbar = diag(sqrt(lambda)) Y`. The right solve `B A^{-1}` is evaluated as
`solve(A, B.T).T`. The spectral objective includes the ambient null-space only
through zero eigenvalues (and hence no extra term). The tied-family metric
splits eigenvectors strictly above the cutoff from the entire eigenvalue block
equal to the cutoff and measures both mandatory-subspace omission and excess
energy outside the eligible block.

## Milestones and progress

### Phase 2 (current)

- [x] Re-read the convergence manuscript and Phase 2 specification completely,
  inventory the Phase 1 API, and run the requested baseline test command.
- [x] Audit the certified-step equations against the manuscript, with special
  attention to the competing `Gamma_C` definitions.
- [x] Extend the reusable runner records and implement shared Phase 2 helpers.
- [x] Implement thin wrappers for all eight specified experiments.
- [x] Add deterministic wrapper tests; attempt only minimal smoke configurations.
- [x] Record the environment block on numerical raw-result inspection, validate
  all available static checks, commit, and open a pull request.

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

- 2026-07-27 Phase 2 audit: equations (6)--(12) of the manuscript define
  `Gamma_C = sqrt(S_C) * (1 + L/a_C)`.  This is the gradient-norm bound obtained
  directly from `||Y|| <= sqrt(S_C)`, `||Lambda|| = L`, and
  `||A(Y)^-1|| <= 1/a_C`.  The checked-in Phase 1 implementation and tests use
  this definition.  The alternative historical expression is not authoritative
  and must not be mixed into `eta_C = min(1, d_C/Gamma_C, 1/L_C)`.
- 2026-07-27: the required `python -m pytest -vv` currently fails at collection
  because the active Python 3.14 environment has no PyTorch.  A CPU-wheel install
  was attempted but the package tunnel returned HTTP 403; implementation will
  still include syntax checks and another final test attempt.

- 2026-07-27: `.agent/PLANS.md` is absent from the repository and all visible
  history. This plan records its own conventions so work can proceed.
- 2026-07-27: Replaced the provisional certified quantities with the supplied
  manuscript-v1 formulas, larger-root condition, log-space calculations, and
  namespaced value/log10 metadata.
- Near-rank-deficient Gram systems are checked before solves using singular
  values scaled by matrix size and machine epsilon; tests must exercise this.

## Results and retrospective

Phase 2 validation (2026-07-27):

- 2026-07-28 smoke follow-up: the condition-`1e8` initialization ablation
  exposed a weighted Gram matrix whose conditioning had squared beyond the
  float64 solve threshold even though the factor-level rank check passed.  The
  Phase 2 trajectory now catches rejected diagnostic and update solves,
  serializes `lost_rank` plus a termination detail, and continues the sweep
  instead of raising out of the wrapper.

- Follow-up test execution in a PyTorch-enabled environment reported 19 passing
  tests and two Phase 2 failures.  The Hessian wrapper had divided the residual
  of an exactly-zero predicted factor by machine epsilon, and the geometry slice
  had constructed its diagonal scales with PyTorch's default float32 dtype.
  The wrapper now records an absolute residual for zero-factor modes and builds
  every slice tensor explicitly in float64.

- `python -m pytest -vv` was run before and after implementation.  Collection
  remains blocked because `torch` is absent.  The second run also exposed that
  pytest did not add the src-layout package to `sys.path`, so `pythonpath =
  ["src"]` was added to the repository configuration.
- `python -m pip install torch --index-url
  https://download.pytorch.org/whl/cpu` exhausted all retries with HTTP 403 from
  the environment's package tunnel.  No alternate installed interpreter or
  cached wheel was found.
- `python -m compileall -q src experiments tests` and `git diff --check` pass.
- Because PyTorch cannot be imported, executing even the minimal numerical
  smoke wrappers and inspecting their serialized values is environment-blocked.
  The atomic JSON serializer and representative raw schema are covered by the
  new Phase 2 test, but this plan deliberately does not misstate a syntax check
  as a numerical experiment.

Phase 1 core and its deterministic tests are implemented. No scientific sweep
was run. Validation commands and outcomes:

- 2026-07-27 follow-up: added root `README.md` setup instructions for standard
  `venv`/pip and `uv`, dependency requirements, installation verification,
  complete and targeted pytest invocations, and the syntax-only check.
- 2026-07-27 validation follow-up: an external PyTorch run reported 14 passing
  tests and exposed cancellation in the tied-family metric: subtracting two
  nearly equal projector traces produced an `O(eps)` squared residual and an
  `O(sqrt(eps))` distance. Replaced those trace differences with direct
  orthogonal residual norms, preserving the formula while making its zero set
  numerically stable.

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
