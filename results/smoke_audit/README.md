# Phase 2 smoke-pipeline audit

Generated 2026-07-28 from committed raw JSON. **These smoke cases only check that the pipeline appears internally correct; they are not experimental results and do not verify any theorem.**

## `boundary_gap_scaling`

- Raw histories and metadata: [`results/smoke/boundary_gap_scaling.json`](../smoke/boundary_gap_scaling.json) (740,167 bytes).
- Configuration: `{"divergence_threshold": 1000000000000.0, "geometric_tolerance": 1e-10, "max_steps": 120, "objective_tolerance": 1e-14}`; dtype `torch.float64`; device `cpu`; seed `0`; recorded commit `8fd0f10f651ef01cf39e1d904d7a9ef3903a2b0e`.
- Cases/runs: 12 / 12; statuses: `{'max_steps': 8, 'converged': 4}`.
- Fits: 12 total, 3 invalid (fewer than ten usable points or no consecutive ratios). The fitting rule is `1e-9 <= error <= 1e-3`; exact reconstructed first/last indices and predicted/fitted values are in [`fit_audit.csv`](fit_audit.csv).
- Numerical event count: 0 rank-loss/NaN/cycle/divergence/stagnation/underflow terminations; non-finite JSON values: 0.
- Preliminary per-run dashboard: [`results/smoke_audit/plots/boundary_gap_scaling.svg`](plots/boundary_gap_scaling.svg).

## `geometry_of_kc`

- Raw histories and metadata: [`results/smoke/geometry_of_kc.json`](../smoke/geometry_of_kc.json) (104,796 bytes).
- Configuration: `{"divergence_threshold": 1000000000000.0, "geometric_tolerance": 1e-10, "max_steps": 0, "objective_tolerance": 1e-14}`; dtype `torch.float64`; device `cpu`; seed `0`; recorded commit `8fd0f10f651ef01cf39e1d904d7a9ef3903a2b0e`.
- Cases/runs: 0 / 0; statuses: `{}`.
- Fits: 0 total, 0 invalid (fewer than ten usable points or no consecutive ratios). The fitting rule is `1e-9 <= error <= 1e-3`; exact reconstructed first/last indices and predicted/fitted values are in [`fit_audit.csv`](fit_audit.csv).
- Numerical event count: 0 rank-loss/NaN/cycle/divergence/stagnation/underflow terminations; non-finite JSON values: 0.
- Preliminary per-run dashboard: [`results/smoke_audit/plots/geometry_of_kc.svg`](plots/geometry_of_kc.svg).

## `hessian_mode_isolation`

- Raw histories and metadata: [`results/smoke/hessian_mode_isolation.json`](../smoke/hessian_mode_isolation.json) (7,774 bytes).
- Configuration: `{"divergence_threshold": 1000000000000.0, "geometric_tolerance": 1e-10, "max_steps": 0, "objective_tolerance": 1e-14}`; dtype `torch.float64`; device `cpu`; seed `0`; recorded commit `8fd0f10f651ef01cf39e1d904d7a9ef3903a2b0e`.
- Cases/runs: 24 / 0; statuses: `{}`.
- Fits: 0 total, 0 invalid (fewer than ten usable points or no consecutive ratios). The fitting rule is `1e-9 <= error <= 1e-3`; exact reconstructed first/last indices and predicted/fitted values are in [`fit_audit.csv`](fit_audit.csv).
- Numerical event count: 0 rank-loss/NaN/cycle/divergence/stagnation/underflow terminations; non-finite JSON values: 0.
- Preliminary per-run dashboard: [`results/smoke_audit/plots/hessian_mode_isolation.svg`](plots/hessian_mode_isolation.svg).

## `initialization_ablation`

- Raw histories and metadata: [`results/smoke/initialization_ablation.json`](../smoke/initialization_ablation.json) (1,667,320 bytes).
- Configuration: `{"divergence_threshold": 1000000000000.0, "geometric_tolerance": 1e-10, "max_steps": 80, "objective_tolerance": 1e-14}`; dtype `torch.float64`; device `cpu`; seed `0`; recorded commit `8fd0f10f651ef01cf39e1d904d7a9ef3903a2b0e`.
- Cases/runs: 32 / 35; statuses: `{'max_steps': 28, 'lost_rank': 2, 'converged': 2, 'underflow': 2, 'stagnated': 1}`.
- Fits: 32 total, 12 invalid (fewer than ten usable points or no consecutive ratios). The fitting rule is `1e-9 <= error <= 1e-3`; exact reconstructed first/last indices and predicted/fitted values are in [`fit_audit.csv`](fit_audit.csv).
- Numerical event count: 5 rank-loss/NaN/cycle/divergence/stagnation/underflow terminations; non-finite JSON values: 0.
- Preliminary per-run dashboard: [`results/smoke_audit/plots/initialization_ablation.svg`](plots/initialization_ablation.svg).

## `predicted_local_rates`

- Raw histories and metadata: [`results/smoke/predicted_local_rates.json`](../smoke/predicted_local_rates.json) (453,933 bytes).
- Configuration: `{"divergence_threshold": 1000000000000.0, "geometric_tolerance": 1e-10, "max_steps": 80, "objective_tolerance": 1e-14}`; dtype `torch.float64`; device `cpu`; seed `0`; recorded commit `8fd0f10f651ef01cf39e1d904d7a9ef3903a2b0e`.
- Cases/runs: 12 / 12; statuses: `{'max_steps': 4, 'converged': 8}`.
- Fits: 24 total, 3 invalid (fewer than ten usable points or no consecutive ratios). The fitting rule is `1e-9 <= error <= 1e-3`; exact reconstructed first/last indices and predicted/fitted values are in [`fit_audit.csv`](fit_audit.csv).
- Numerical event count: 0 rank-loss/NaN/cycle/divergence/stagnation/underflow terminations; non-finite JSON values: 0.
- Preliminary per-run dashboard: [`results/smoke_audit/plots/predicted_local_rates.svg`](plots/predicted_local_rates.svg).

## `saddle_escape`

- Raw histories and metadata: [`results/smoke/saddle_escape.json`](../smoke/saddle_escape.json) (1,152 bytes).
- Configuration: `{"divergence_threshold": 1000000000000.0, "geometric_tolerance": 1e-10, "max_steps": 100, "objective_tolerance": 1e-14}`; dtype `torch.float64`; device `cpu`; seed `0`; recorded commit `8fd0f10f651ef01cf39e1d904d7a9ef3903a2b0e`.
- Cases/runs: 4 / 0; statuses: `{}`.
- Fits: 0 total, 0 invalid (fewer than ten usable points or no consecutive ratios). The fitting rule is `1e-9 <= error <= 1e-3`; exact reconstructed first/last indices and predicted/fitted values are in [`fit_audit.csv`](fit_audit.csv).
- Numerical event count: 0 rank-loss/NaN/cycle/divergence/stagnation/underflow terminations; non-finite JSON values: 0.
- Preliminary per-run dashboard: [`results/smoke_audit/plots/saddle_escape.svg`](plots/saddle_escape.svg).

## `step_size_phase_diagram`

- Raw histories and metadata: [`results/smoke/step_size_phase_diagram.json`](../smoke/step_size_phase_diagram.json) (1,206,844 bytes).
- Configuration: `{"divergence_threshold": 1000000000000.0, "geometric_tolerance": 1e-10, "max_steps": 50, "objective_tolerance": 1e-14}`; dtype `torch.float64`; device `cpu`; seed `0`; recorded commit `8fd0f10f651ef01cf39e1d904d7a9ef3903a2b0e`.
- Cases/runs: 72 / 72; statuses: `{'underflow': 33, 'converged': 6, 'cycle': 3, 'stagnated': 3, 'diverged': 9, 'max_steps': 18}`.
- Fits: 72 total, 63 invalid (fewer than ten usable points or no consecutive ratios). The fitting rule is `1e-9 <= error <= 1e-3`; exact reconstructed first/last indices and predicted/fitted values are in [`fit_audit.csv`](fit_audit.csv).
- Numerical event count: 48 rank-loss/NaN/cycle/divergence/stagnation/underflow terminations; non-finite JSON values: 0.
- Preliminary per-run dashboard: [`results/smoke_audit/plots/step_size_phase_diagram.svg`](plots/step_size_phase_diagram.svg).

## `tied_eigenvalues`

- Raw histories and metadata: [`results/smoke/tied_eigenvalues.json`](../smoke/tied_eigenvalues.json) (386,416 bytes).
- Configuration: `{"divergence_threshold": 1000000000000.0, "geometric_tolerance": 1e-10, "max_steps": 100, "objective_tolerance": 1e-14}`; dtype `torch.float64`; device `cpu`; seed `0`; recorded commit `8fd0f10f651ef01cf39e1d904d7a9ef3903a2b0e`.
- Cases/runs: 6 / 6; statuses: `{'converged': 2, 'max_steps': 4}`.
- Fits: 6 total, 4 invalid (fewer than ten usable points or no consecutive ratios). The fitting rule is `1e-9 <= error <= 1e-3`; exact reconstructed first/last indices and predicted/fitted values are in [`fit_audit.csv`](fit_audit.csv).
- Numerical event count: 0 rank-loss/NaN/cycle/divergence/stagnation/underflow terminations; non-finite JSON values: 0.
- Preliminary per-run dashboard: [`results/smoke_audit/plots/tied_eigenvalues.svg`](plots/tied_eigenvalues.svg).

## Detailed indexes

- [`smoke_manifest.csv`](smoke_manifest.csv) enumerates every smoke case, swept descriptor, history length, and status.
- [`fit_audit.csv`](fit_audit.csv) enumerates every predicted and fitted rate, reconstructed fitting window, and invalid fit.
- [`event_audit.csv`](event_audit.csv) enumerates every rank-loss, NaN, cycle, divergence, stagnation, or underflow event and its serialized detail.

## Manual identity checks

- **hessian_mode_isolation — max absolute predicted/measured error:** `4.4408920985006255e-11` (finite-difference check).
- **saddle_escape — exact saddle fixed-point error:** `0` (exact identity).
- **tied_eigenvalues — column_space_error:** `1.2082440073477922e-15` (isotropic identity).
- **tied_eigenvalues — gram_error_after:** `5.1418816566284482` (isotropic identity).
- **tied_eigenvalues — gram_error_before:** `22.075843090749594` (isotropic identity).
- Hessian zero-factor rows correctly use absolute rather than relative residual semantics.
- Geometry certificate: the artifact labels sampled bounds as empirical lower bounds on conservativeness, not exact extrema.

## Exact proposed full grids and run counts

| Experiment | Full grid | Run/action count |
|---|---|---:|
| Predicted local rates | `k={1,2,4,8}`; `eta={0.1,0.25,0.5,0.7,eta_loc*}`; 20 seeds | 400 trajectories |
| Hessian modes | 24 modes; `eta={0.25,0.5,0.7}`; `epsilon={1e-4,1e-5,1e-6}` | 216 finite-difference actions |
| Boundary-gap scaling | 4 k values; 9 gaps; local-normal 5 seeds plus support-Gaussian 20 seeds | 900 trajectories |
| Tied eigenvalues | 5 gaps; 50 seeds | 250 trajectories plus isotropic sanity |
| Geometry of K_C | 601x601 minus origin; two 201x201 slices; 5,000 rejection attempts | 361,200 grid points + 80,802 slice points |
| Saddle escape | 11 epsilons; 3 steps; 2 signs | 66 trajectories |
| Step-size phase diagram | 3 spectra; 4 families; 5 seeds; per-initialization sorted union of 4 certificate multiples, 60 log points, 141 linear points, and 3 named steps | at most 12,480 trajectories before duplicate removal |
| Initialization ablation | 3 stochastic groups x20; 40 controlled values x10; three seed-zero certified checks | 463 trajectories |

The pre-deduplication trajectory upper total is 14,559, plus 216 Hessian actions and the geometry evaluations. Experiment 7's exact post-deduplication total depends on each initialization's `eta_C`; the implementation removes duplicates with the specified relative `1e-12` tolerance.

## Runtime and storage estimate

The eight smoke JSON files occupy 4,568,402 bytes. Their measured in-process wrapper time totals 15.697 seconds. Direct grid/case-count scaling gives a smoke-shaped lower planning estimate of 0.32 hours for the full grids. The 5,958 iterative records across 137 runs took 14.844 serialized per-run seconds and occupy 2,713,697 raw-history bytes. Scaling the observed average run to 14,559 trajectories gives approximately 0.44 trajectory-hours and 0.29 GB. A deliberately conservative all-runs-hit-20,001-iterations bound is approximately 201.5 trajectory-hours and 132.6 GB. These estimates exclude process startup and assume roughly linear per-case/per-record costs; they are planning bounds, not guarantees.

## Readiness and remaining limitations

1. Every fit now serializes its actual first/last usable iteration and an explicit invalid reason.
2. Geometry records contour/critical-point/trajectory data, multiple levels, Hessian norms, extrema, and bound ratios.
3. Tied cases record canonical errors, tied-block angles, rates, and iterations.
4. Step-size cases record relative-deduplicated grids, late rates, and grid-dependent cutoff summaries and ratios.
5. Initialization cases record certified checks, family success rates, fits, transients, and a structured exploratory regression (which may be invalid on a short smoke run).
6. Rank loss, divergence, cycles, stagnation, and underflow remain valid observed outcomes and must not be relabelled as successes.

**Conclusion:** the corrected smoke pipeline contains the specified raw outputs and summaries. Full sweeps still require explicit approval; smoke behavior is not an experimental conclusion.
