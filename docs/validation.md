# Implementation validation

Validation performed on CPU with Python 3.13.5, PyTorch 2.10.0+cpu,
NumPy 2.3.5, matplotlib 3.10.8, pytest 9.0.2, and mpmath 1.3.0.
PyTorch used one thread and float64 for numerical studies.

- `pytest -q`: **71 passed**.
- `python -m compileall -q src experiments plotting`: passed.
- Editable installation with test and precision extras: passed using the
  already installed dependencies (no network dependency upgrades).
- All six packaged smoke presets executed; eventual rates also exercised the
  optional 100-digit quadratic reference. **60 study cases** in total:
  8 trace, 4 power-entry configurations (32 trials each), 34 rate/precision,
  6 manifold, 4 rank-one comparison, and 4 damping cases.
- Independent PNG/SVG plotting paths exercised for each study; representative
  plots inspected. Regression tests protect decimal-valued filenames from
  accidental overwrite.
- No failed/nonfinite/numerical-failure case in those smoke runs. `max_steps`
  and `stagnated` statuses remain in the data rather than being called successes.
  The scalar eta=1 control reports its expected exact two-cycle.
- No full `paper` preset, CUDA run, or sequential-deflation runtime benchmark
  has been executed. Smoke behavior is not evidence of an almost-sure theorem.

The numerical source SHA-256 for this validation is
`b6263c102987fcc0f65d6ca0b4f1bb25584109b06d9320fef30b78b2286386bd`.
Run output records additionally contain the effective config and source digest.

Important regression checks include the noncommuting-Gram trace identity using
Q from the polar factorization, exact nullspace decay, stable gap identities,
finite-difference derivatives, tied Hessian modes, k=r without a mixing mode,
raw-vs-QR power-subspace equivalence, manuscript certificate constants,
rank-one half-step slope, exact cycles versus damped oscillations, and safe
config/source-checked resume. Numerical tolerances do not constitute proofs.
