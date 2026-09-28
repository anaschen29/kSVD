# Collecting results after the smoke review

This update changes data collection and diagnostics, not the PGD update or
plotting code. The original manuscript trace certificate is unchanged.

## Run the next, targeted collection

Install the updated branch with `python -m pip install -e '.[test,precision]'`.
`collection` is a new intermediate preset; it does not replace `smoke` or
silently enlarge `paper`. Inspect its full effective configuration first:

```bash
ksvd-run --experiment eventual_rates --preset collection --dry-run
ksvd-run --experiment trace_contraction --preset collection --dry-run
```

Then collect the two current priorities, without generating plots:

```bash
ksvd-run --experiment eventual_rates --preset collection \
  --out results/rates-collection-v2 --high-precision
ksvd-run --experiment trace_contraction --preset collection \
  --out results/trace-collection-v2

ksvd-summarize --input results/rates-collection-v2 \
  --out reports/rates-collection-v2
ksvd-summarize --input results/trace-collection-v2 \
  --out reports/trace-collection-v2
```

The rate preset has **253 cases**, or **254** with `--high-precision`:
72 isolated radial/angular/mixed cases, 60 coupled block cases, 120 Gaussian
starts, and one float64 quadratic case. It uses k=1,4, relative gaps
0.05,0.2,0.5, and damping 1/2,2/3,0.95 plus the spectral optimum for each gap.
Each coupled and Gaussian configuration has five paired seeds. Rate budgets
are 1600 updates per trajectory; the quadratic reference has its own budget.

The trace preset has **144 cases**: k=2,4, gaps 0.1,0.2,0.5, initial band
fractions 0.2,0.8, damping 0.5,0.85, initial Gram condition numbers 1,100,
and three seeds. Each start mixes positive-tail and nullspace directions.
The budget is 1200 updates per trajectory. These are collection budgets, not
promises of convergence before the cap. Timing includes diagnostics.

An optional next initialization study is available through:

```bash
ksvd-run --experiment power_sketch_entry --preset collection \
  --out results/power-collection-v2
ksvd-summarize --input results/power-collection-v2 \
  --out reports/power-collection-v2
```

It uses six spectral configurations with 256 paired Gaussian trials each.
Its q range extends five powers beyond the theorem's sufficient q. The
power measurements are QR-stabilized subspace measurements, not a modified
PGD initialization.

Use `--resume` only with the same output directory, effective config, source,
package versions checked by the runner, and thread count. Interrupted runs
keep per-case atomic files and the manifest's completed entries. A completed
case is not rerun except when its status is `failed`; numerical failures and
censored outcomes remain data. Use a new directory to deliberately retry a
previous numerical failure or change the budget. Reports can be generated
from a partial manifest. Output records remain valid inputs to existing plots.

## Reuse existing smoke data without rerunning it

```bash
ksvd-summarize --input results/rates-smoke --out reports/rates-smoke-v2
ksvd-summarize --input results/trace-smoke --out reports/trace-smoke-v2
```

This reads the old JSON files, adds component fits and accuracy summaries to
a SEPARATE report, and leaves the original measurements untouched. The report
records hashes of its inputs and the analysis source. Do not use `--resume`
on pre-update directories: the code/config signature intentionally differs.

Every report contains:

- `summary.json`: per-case details, numerical metadata, input hashes, and
  seed-group aggregates including all stopping outcomes;
- `cases.csv`: parameters, stopping reason, final errors, threshold attainment,
  and first observed hit steps;
- `rate_fits.csv`: separate component fits, fitting windows, sign changes,
  numerical floors, and labeled theoretical references;
- `power_entry.csv` for power-sketch studies.

There are no plots in this reporting path. Reports/results are gitignored.
For transfer, archive `results/...` and `reports/...`, not only figures.

## Accuracy is independent of termination

Each configured nonnegative error metric has a threshold and an `accuracy`
entry: `first_hit_step`, `ever_met`, `final_value`, `final_met`, and the
minimum observed error. `stagnated` and `max_steps` are still stopping
reasons, not failures or successes by themselves. A stationary saddle does
not pass the product-error criterion. A run that once met a tolerance but
later leaves it has `ever_met=true` and `final_met=false`.

New runs track accuracy on **every evaluated iterate**, even with sparse
`record_every`; first-hit rows are retained. Legacy reanalysis explicitly
uses `saved_records_only`: a first saved hit need not be the first true hit.
A failed terminal evaluation has unknown final accuracy, not the last good
value mislabeled as a terminal result. `last_evaluated_metrics` remains
available for diagnosis. Missing metrics (e.g. unique-product error at a
cutoff tie) are unavailable, not zero. Group medians are labeled as medians
among hitting runs and reported alongside attempted/available/hitting counts.

Threshold overrides can be supplied in the run's JSON config through
`accuracy_targets` for the trace/rate studies, or applied retrospectively:

```bash
ksvd-summarize --input results/rates-smoke --out reports/rates-smoke-1e9 \
  --target factor_error=1e-9
```

A retrospective override uses saved samples and is labeled accordingly.
Only an explicit `stop_metric` terminates a trajectory on tolerance; tracking
additional criteria does not alter the optimizer or introduce new stopping.

## Component rates and genuine block starts

The isolated experiment still records `radial_signed` and `angular_signed`.
Fits now use their magnitudes, so negative radial samples are not dropped.
The signed reference multiplier and observed sign changes are saved. The
linearized radial multiplier is 1-2 eta, and the selected angular multiplier
is 1-eta*mu. **An angular-only initial perturbation can generate radial error
at second order.** Therefore the total error can eventually follow a slower
radial mode. Reference kinds explicitly distinguish a spectral upper bound
from a linearized component multiplier. Zero linear derivative is labeled
as such; insufficient samples do not produce invented linear fits.

The rule remains the last 30 eligible samples above 1e-11 and below 1e-2,
with at least eight points. The effective policy and exact window are saved.
Fitting does not use the theoretical factor to select favorable samples.
`--fit-floor`, `--fit-ceiling`, and `--fit-window` are explicit retrospective
analysis overrides, recorded in report metadata.

The new `coupled` family perturbs all columns with a dense symmetric top
block and dense omitted block. Each block has Frobenius norm amplitude/sqrt(2).
The initial Gram/compression commutator is saved to diagnose coupling.
The actual limit's right rotation is not known: use `orbit_error_y`, not a
fabricated point error to E_star or the last numerical iterate. Diagnostic
Procrustes alignment also gives radial/omitted block norms and a selected
slow angular coordinate. This alignment NEVER replaces the evolving state.
Gaussian starts now record these orbit diagnostics as well.

High-precision results have dtype `mpmath.mpf`, device CPU and explicit
precision digits. They record nonzero error-based empirical orders and flag
precision saturation; a stored zero is not a claim of exact finite termination.

## Validation for this update

- All **97 tests passed** locally (the original 71 plus 26 new tests).
- Editable installation and compile checks passed.
- All six smoke families ran: **68 cases**, including the new coupled cases
  and the optional high-precision reference. None had a failed, nonfinite,
  numerical-failure, or norm-limit status. The expected eta=1 cycle remains.
- A separate five-case pilot covered coupled k=4 at gaps 0.05 and 0.5 with
  half-step and spectrally optimal damping, plus the quadratic control.
- The supplied 42 archived smoke cases were summarized without rerunning the
  trajectories or modifying their raw files.
- No full collection/paper preset or CUDA run has been executed. Local pilot
  observations are not a proof or a publication-scale empirical conclusion.
