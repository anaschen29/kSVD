# Phase 2 smoke-pipeline audit

Generated 2026-07-28 from committed raw JSON. **These smoke cases only check that the pipeline appears internally correct; they are not experimental results and do not verify any theorem.**

## `boundary_gap_scaling`

- Raw histories and metadata: [`results/smoke/boundary_gap_scaling.json`](../smoke/boundary_gap_scaling.json) (737,064 bytes).
- Configuration: `{"divergence_threshold": 1000000000000.0, "geometric_tolerance": 1e-10, "max_steps": 120, "objective_tolerance": 1e-14}`; dtype `torch.float64`; device `cpu`; seed `0`; recorded commit `4713d4c77a6173ab811b4d3f5d1aa1bbe4baa656`.
- Cases/runs: 12 / 12; statuses: `{'max_steps': 8, 'converged': 4}`.
- Fits: 12 total, 3 invalid (fewer than ten usable points or no consecutive ratios). The fitting rule is `1e-9 <= error <= 1e-3`; exact reconstructed first/last indices and predicted/fitted values are in [`fit_audit.csv`](fit_audit.csv).
- Numerical event count: 0 rank-loss/NaN/cycle/divergence terminations; non-finite JSON values: 0.
- Preliminary per-run dashboard: [`results/smoke_audit/plots/boundary_gap_scaling.svg`](plots/boundary_gap_scaling.svg).

## `geometry_of_kc`

- Raw histories and metadata: [`results/smoke/geometry_of_kc.json`](../smoke/geometry_of_kc.json) (89,918 bytes).
- Configuration: `{"divergence_threshold": 1000000000000.0, "geometric_tolerance": 1e-10, "max_steps": 0, "objective_tolerance": 1e-14}`; dtype `torch.float64`; device `cpu`; seed `0`; recorded commit `4713d4c77a6173ab811b4d3f5d1aa1bbe4baa656`.
- Cases/runs: 0 / 0; statuses: `{}`.
- Fits: 0 total, 0 invalid (fewer than ten usable points or no consecutive ratios). The fitting rule is `1e-9 <= error <= 1e-3`; exact reconstructed first/last indices and predicted/fitted values are in [`fit_audit.csv`](fit_audit.csv).
- Numerical event count: 0 rank-loss/NaN/cycle/divergence terminations; non-finite JSON values: 0.
- Preliminary per-run dashboard: [`results/smoke_audit/plots/geometry_of_kc.svg`](plots/geometry_of_kc.svg).

## `hessian_mode_isolation`

- Raw histories and metadata: [`results/smoke/hessian_mode_isolation.json`](../smoke/hessian_mode_isolation.json) (7,693 bytes).
- Configuration: `{"divergence_threshold": 1000000000000.0, "geometric_tolerance": 1e-10, "max_steps": 0, "objective_tolerance": 1e-14}`; dtype `torch.float64`; device `cpu`; seed `0`; recorded commit `4713d4c77a6173ab811b4d3f5d1aa1bbe4baa656`.
- Cases/runs: 24 / 0; statuses: `{}`.
- Fits: 0 total, 0 invalid (fewer than ten usable points or no consecutive ratios). The fitting rule is `1e-9 <= error <= 1e-3`; exact reconstructed first/last indices and predicted/fitted values are in [`fit_audit.csv`](fit_audit.csv).
- Numerical event count: 0 rank-loss/NaN/cycle/divergence terminations; non-finite JSON values: 0.
- Preliminary per-run dashboard: [`results/smoke_audit/plots/hessian_mode_isolation.svg`](plots/hessian_mode_isolation.svg).

## `initialization_ablation`

- Raw histories and metadata: [`results/smoke/initialization_ablation.json`](../smoke/initialization_ablation.json) (1,638,656 bytes).
- Configuration: `{"divergence_threshold": 1000000000000.0, "geometric_tolerance": 1e-10, "max_steps": 80, "objective_tolerance": 1e-14}`; dtype `torch.float64`; device `cpu`; seed `0`; recorded commit `4713d4c77a6173ab811b4d3f5d1aa1bbe4baa656`.
- Cases/runs: 32 / 32; statuses: `{'max_steps': 28, 'lost_rank': 2, 'converged': 2}`.
- Fits: 0 total, 0 invalid (fewer than ten usable points or no consecutive ratios). The fitting rule is `1e-9 <= error <= 1e-3`; exact reconstructed first/last indices and predicted/fitted values are in [`fit_audit.csv`](fit_audit.csv).
- Numerical event count: 2 rank-loss/NaN/cycle/divergence terminations; non-finite JSON values: 0.
- Preliminary per-run dashboard: [`results/smoke_audit/plots/initialization_ablation.svg`](plots/initialization_ablation.svg).

## `predicted_local_rates`

- Raw histories and metadata: [`results/smoke/predicted_local_rates.json`](../smoke/predicted_local_rates.json) (449,645 bytes).
- Configuration: `{"divergence_threshold": 1000000000000.0, "geometric_tolerance": 1e-10, "max_steps": 80, "objective_tolerance": 1e-14}`; dtype `torch.float64`; device `cpu`; seed `0`; recorded commit `4713d4c77a6173ab811b4d3f5d1aa1bbe4baa656`.
- Cases/runs: 12 / 12; statuses: `{'max_steps': 4, 'converged': 8}`.
- Fits: 24 total, 3 invalid (fewer than ten usable points or no consecutive ratios). The fitting rule is `1e-9 <= error <= 1e-3`; exact reconstructed first/last indices and predicted/fitted values are in [`fit_audit.csv`](fit_audit.csv).
- Numerical event count: 0 rank-loss/NaN/cycle/divergence terminations; non-finite JSON values: 0.
- Preliminary per-run dashboard: [`results/smoke_audit/plots/predicted_local_rates.svg`](plots/predicted_local_rates.svg).

## `saddle_escape`

- Raw histories and metadata: [`results/smoke/saddle_escape.json`](../smoke/saddle_escape.json) (1,071 bytes).
- Configuration: `{"divergence_threshold": 1000000000000.0, "geometric_tolerance": 1e-10, "max_steps": 100, "objective_tolerance": 1e-14}`; dtype `torch.float64`; device `cpu`; seed `0`; recorded commit `4713d4c77a6173ab811b4d3f5d1aa1bbe4baa656`.
- Cases/runs: 4 / 0; statuses: `{}`.
- Fits: 0 total, 0 invalid (fewer than ten usable points or no consecutive ratios). The fitting rule is `1e-9 <= error <= 1e-3`; exact reconstructed first/last indices and predicted/fitted values are in [`fit_audit.csv`](fit_audit.csv).
- Numerical event count: 0 rank-loss/NaN/cycle/divergence terminations; non-finite JSON values: 0.
- Preliminary per-run dashboard: [`results/smoke_audit/plots/saddle_escape.svg`](plots/saddle_escape.svg).

## `step_size_phase_diagram`

- Raw histories and metadata: [`results/smoke/step_size_phase_diagram.json`](../smoke/step_size_phase_diagram.json) (1,444,117 bytes).
- Configuration: `{"divergence_threshold": 1000000000000.0, "geometric_tolerance": 1e-10, "max_steps": 50, "objective_tolerance": 1e-14}`; dtype `torch.float64`; device `cpu`; seed `0`; recorded commit `4713d4c77a6173ab811b4d3f5d1aa1bbe4baa656`.
- Cases/runs: 72 / 72; statuses: `{'cycle': 40, 'converged': 11, 'max_steps': 21}`.
- Fits: 0 total, 0 invalid (fewer than ten usable points or no consecutive ratios). The fitting rule is `1e-9 <= error <= 1e-3`; exact reconstructed first/last indices and predicted/fitted values are in [`fit_audit.csv`](fit_audit.csv).
- Numerical event count: 40 rank-loss/NaN/cycle/divergence terminations; non-finite JSON values: 0.
- Preliminary per-run dashboard: [`results/smoke_audit/plots/step_size_phase_diagram.svg`](plots/step_size_phase_diagram.svg).

## `tied_eigenvalues`

- Raw histories and metadata: [`results/smoke/tied_eigenvalues.json`](../smoke/tied_eigenvalues.json) (382,195 bytes).
- Configuration: `{"divergence_threshold": 1000000000000.0, "geometric_tolerance": 1e-10, "max_steps": 100, "objective_tolerance": 1e-14}`; dtype `torch.float64`; device `cpu`; seed `0`; recorded commit `4713d4c77a6173ab811b4d3f5d1aa1bbe4baa656`.
- Cases/runs: 6 / 6; statuses: `{'converged': 2, 'max_steps': 4}`.
- Fits: 0 total, 0 invalid (fewer than ten usable points or no consecutive ratios). The fitting rule is `1e-9 <= error <= 1e-3`; exact reconstructed first/last indices and predicted/fitted values are in [`fit_audit.csv`](fit_audit.csv).
- Numerical event count: 0 rank-loss/NaN/cycle/divergence terminations; non-finite JSON values: 0.
- Preliminary per-run dashboard: [`results/smoke_audit/plots/tied_eigenvalues.svg`](plots/tied_eigenvalues.svg).

## Detailed indexes

- [`smoke_manifest.csv`](smoke_manifest.csv) enumerates every smoke case, swept descriptor, history length, and status.
- [`fit_audit.csv`](fit_audit.csv) enumerates every predicted and fitted rate, reconstructed fitting window, and invalid fit.
- [`event_audit.csv`](event_audit.csv) enumerates every rank-loss, NaN, cycle, or divergence event and its serialized detail.

## Manual identity checks

- **hessian_mode_isolation — max absolute predicted/measured error:** `4.4408920985006255e-11` (finite-difference check).
- **saddle_escape — exact saddle fixed-point error:** `0` (exact identity).
- **tied_eigenvalues — column_space_error:** `9.1871430648557113e-16` (isotropic identity).
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
| Initialization ablation | 3 stochastic groups x20; 40 controlled values x10 | 460 primary trajectories; required certified secondary subset is not implemented |

The trajectory upper total is 14,556, plus 216 Hessian actions and the geometry evaluations. Experiment 7's exact post-deduplication total depends on each initialization's `eta_C`; the implementation currently uses exact set equality rather than the specification's relative `1e-12` duplicate tolerance.

## Runtime and storage estimate

The eight smoke JSON files occupy 4,750,359 bytes. Runtime was not serialized, so a defensible wall-clock estimate cannot be derived from these artifacts. A case-count-only storage extrapolation is also unsafe because full runs allow 20,000 iterations versus 50–120 in most smoke trajectories. Instrumenting elapsed time and serialized bytes per completed trajectory is required before approving a resource estimate.

## Audit findings blocking full sweeps

1. Rate fits serialize the number of usable points but not the first/last iteration of the fitting window.
2. Geometry Part A omits the specified contour levels, critical-point overlays, and fixed-grid trajectories; Part C samples only one level and does not record Hessian norms or bound-to-empirical ratios.
3. Tied-eigenvalue output omits canonical projector errors, final tied-block principal angles, and explicit rate/iteration summaries.
4. Step-size output does not aggregate empirical monotone/convergence cutoffs or their ratios to `eta_C`, and duplicate removal is not relative-tolerance based.
5. Initialization ablation omits the certified-step secondary subset, family success-rate summaries, and regression data (only its label is present).
6. Run timing and output byte accounting are absent, preventing evidence-based runtime/storage estimates.
7. Several smoke runs terminate through rank loss; those are useful pipeline events, not successful convergence cases.

**Conclusion:** serialization and basic diagnostics execute, but the smoke artifacts expose specification-completeness gaps. Do not launch full sweeps until these gaps are implemented, retested, and re-smoked.
