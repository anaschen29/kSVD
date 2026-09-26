# kSVD PGD experiments

Reproducible experiments for **preconditioned gradient descent for low-rank
positive-semidefinite matrix approximation**, not dictionary-learning K-SVD:

```text
g(X) = 1/4 ||M - XX^T||_F^2
X_next = (1 - eta) X + eta M X (X^T X)^(-1)
```

The code implements the supplied manuscript's fixed-damping algorithm and
quantitative predictions. It does not normalize the evolving factor, add a
ridge to its Gram matrix, or fit theoretical certificate constants to data.

## Install and test

Python 3.10 or newer:

```bash
python -m pip install -e '.[test]'
pytest -q
```

Optional high-precision diagonal Heron reference:

```bash
python -m pip install -e '.[precision]'
```

PyTorch float64 and CPU are the defaults. `--device cuda` is available where
supported; CPU validation does not establish CUDA reproducibility or performance.
Dependencies are deliberately small: torch, NumPy, matplotlib, plus pytest for
tests and optional mpmath. The validation environment is recorded in
[docs/validation.md](docs/validation.md).

## Run a study (smoke by default)

```bash
ksvd-run --experiment trace_contraction --out results/trace-smoke
ksvd-plot --input results/trace-smoke --out figures/trace-smoke
```

Equivalent script entry points:

```bash
python experiments/trace_contraction.py --preset smoke --out results/trace-smoke
python plotting/plot_results.py --input results/trace-smoke --out figures/trace-smoke
```

Use a new output directory per configuration. `--resume` reuses completed cases
only when the effective config, source digest, torch/NumPy versions, and thread
count match. Incompatible results are never overwritten silently.

| Study | Question | Main outputs |
| --- | --- | --- |
| `trace_contraction` | How conservative is the certified ambient trace bound? | Trace/envelope, resolved ratios, product error, Gram conditioning |
| `power_sketch_entry` | How reliably does `M^q Omega` enter the trace band? | Paired samples across q, entry probabilities, pointwise Wilson intervals, quantiles, theorem/sample-specific bounds |
| `eventual_rates` | Do normal modes and damping agree with the spectrum? | Radial/angular/mixed modes, fitted factors, eta curve, Gaussian-start iteration counts, quadratic case |
| `minimizer_manifold` | What changes at a tied cutoff? | Full-family residual, distance to one reference orbit, selected angles, tie-splitting transients |
| `jedra_shah_rank1` | How do annulus entry and the eventual rate differ? | One rank-one optimizer, JS norm-annulus diagnostics, trace-band entry, eventual spectral slope |
| `damping_stress` | What happens near the permitted damping endpoints? | Ill-conditioned supported starts and an explicitly excluded scalar eta=1 two-cycle |

All six:

```bash
for study in trace_contraction power_sketch_entry eventual_rates minimizer_manifold jedra_shah_rank1 damping_stress; do
  ksvd-run --experiment "$study" --out "results/smoke/$study"
  ksvd-plot --input "results/smoke/$study" --out "figures/smoke/$study" --max-cases 2
done
```

High precision is a separate optional reference, not a silent change of backend:

```bash
ksvd-run --experiment eventual_rates --high-precision --out results/rates-mp
ksvd-plot --input results/rates-mp --out figures/rates-mp
```

## Full sweeps are explicit opt-in

```bash
ksvd-run --experiment power_sketch_entry --preset paper --out results/power-paper
ksvd-run --experiment power_sketch_entry --preset paper --out results/power-paper --resume
```

The `paper` presets can take substantial time and are **not** run by tests or
CI. Start with smoke runs, inspect diagnostics, then choose a budget. Override
any study parameters with a JSON file (`--config configs/my_study.json`).
Packaged presets live in `src/ksvd/presets/`. The complete effective config is
saved, including device and preset name.

## What is saved

Each run directory contains a `manifest.json` and one atomic JSON file per
case. Files include seeds, all effective parameters, source SHA-256, git commit
and dirty flag, versions, dtype/device, measured trajectories, theoretical
constants, fitting windows, and termination status. Power experiments retain
individual trial errors; dynamical runs retain measurements and a final factor,
not every full matrix. Timing includes diagnostics and is not a kernel benchmark.
Plotting reads these files without executing the optimizer and writes PNG/SVG.
The figure cap only limits individual plots; aggregate plots use every saved case.
Raw results and figures are gitignored.

Termination status is not a theorem verdict. In particular, `stagnated` means a
finite-precision small step, **not** a proven global optimum; `max_steps` means
censoring at the configured budget. `cycle_detected` is expected in the scalar
eta=1 control. A numerical Cholesky/rank failure is recorded, not regularized
away. Smoke runs validate code paths, not almost-sure mathematical statements.

## Numerical and mathematical conventions

See [docs/mathematical_reference.md](docs/mathematical_reference.md) for formulas,
assumptions, metrics, and source mapping, and
[docs/experiments.md](docs/experiments.md) for the experimental protocol.

Important distinctions:

- The trace certificate uses the **current manuscript beta**, not a later
  suggested tightening. A formula-version identifier is saved with each run.
- Supported local rates exclude nullspace modes and handle repeated eigenvalues
  and `k=r`. They are not advertised as ambient factor-rate guarantees.
- A QR basis is fine for projector metrics. The exact trace identity paired with
  `G=X.T@X` requires the **polar** frame; tests enforce this distinction.
- QR-stabilized power sketches are used for subspace-entry measurements only.
  Gaussian PGD starts use the raw `M Omega` factor.
- The tied-family diagnostic is a **residual**, not an exact nearest-point
  distance. A fixed-reference Procrustes distance is labeled separately.
- Rate fits use predeclared error floors/windows and known targets in controlled
  tests. Missing windows are reported, not fabricated. A displayed anchored
  spectral slope is not a certified finite-time envelope.
- The Jedra--Shah rank-one experiment has one optimizer: the half-step iteration
  is the same algorithm. This is not a speed comparison between two solvers.
