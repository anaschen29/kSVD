# kSVD PGD experiments

This repository contains a PyTorch implementation of preconditioned gradient
descent for low-rank positive-semidefinite matrix approximation:

```text
g(X) = 1/4 ||M - XX^T||_F^2.
```

The reusable numerical package lives in `src/ksvd`, and the mathematical and
experimental specifications live in `docs/`.

## Requirements

- Python 3.10 or newer
- PyTorch 2.0 or newer
- pytest 7 or newer for running the test suite

PyTorch is the only runtime dependency. The `test` optional dependency installs
pytest. Experiments use `torch.float64` by default; a GPU is not required.

## Environment setup

Run the following commands from the repository root.

### Standard `venv` and pip

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

The editable install makes changes under `src/ksvd` immediately available in
the environment. On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1` instead.

### Using `uv`

If [`uv`](https://docs.astral.sh/uv/) is installed, the equivalent setup is:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[test]'
```

Python 3.12 is an example supported interpreter; any available Python version
meeting the requirement may be used.

### Platform-specific PyTorch builds

The normal editable install obtains the default PyTorch distribution. If a
specific CUDA, ROCm, or CPU-only build is needed, install the appropriate
PyTorch build for the platform first, then install this project:

```bash
# Install the appropriate torch build using the command supplied by PyTorch,
# then install this repository and its test dependency.
python -m pip install -e '.[test]'
```

## Verify the installation

Confirm that the package imports and that the expected versions and float64
default are visible:

```bash
python - <<'PY'
import torch
import ksvd

problem = ksvd.spectral_problem([4.0, 2.0, 1.0])
print("PyTorch:", torch.__version__)
print("kSVD module:", ksvd.__file__)
print("experiment dtype:", problem.eigenvalues.dtype)
PY
```

The final line should report `torch.float64`.

## Run the tests

Run the complete deterministic Phase 1 suite with:

```bash
python -m pytest
```

For more detailed test names and output, use:

```bash
python -m pytest -vv
```

To run a single validation, pass its node identifier, for example:

```bash
python -m pytest tests/test_phase1.py::test_ambient_and_reduced_coordinate_update_equivalence
```

The tests use small deterministic float64 problems; they do not execute full
experimental sweeps.

## Development checks

Syntax-check all package and test modules without running experiments:

```bash
python -m compileall -q src tests
```

## Phase 2 wrappers

The eight theorem-facing wrappers live in `experiments/` and default to the
small smoke configurations.  Every wrapper requires an explicit raw JSON
destination; for example:

```bash
python experiments/hessian_mode_isolation.py --output results/smoke/hessian.json
```

Passing `--full` is the explicit opt-in to the full sweep from
`docs/phase2_experiment_spec.md`.  Wrappers store raw numerical records and
reproducibility metadata only; plotting is intentionally separate.

### Approved staged full sweeps

The full Phase 2 grid is intentionally split into three sequential launchers.
Choose one persistent run directory and inspect each completed stage before
starting the next:

```bash
experiments/full/run_stage1.sh results/full/phase2-$(date -u +%Y%m%dT%H%M%SZ)
experiments/full/run_stage2.sh results/full/phase2-YYYYMMDDTHHMMSSZ
experiments/full/run_stage3.sh results/full/phase2-YYYYMMDDTHHMMSSZ
```

Replace the second and third example paths with the exact directory created for
Stage 1. Stage 1 runs local rates, Hessian modes, and saddle escape; Stage 2
runs boundary-gap scaling, tied eigenvalues, and initialization ablation;
Stage 3 runs geometry and the step-size phase diagram.

Each launcher runs sequentially and stops on the first failure. It records
unaggregated JSON under `raw/`, append-only console output under `logs/`, SHA256
files under `checksums/`, an environment snapshot, a tab-separated manifest,
and a completion marker. Existing raw JSON is never overwritten. To resume a
stage after a failure, rerun it with `--resume`; completed JSON is validated and
skipped before execution continues:

```bash
experiments/full/run_stage2.sh results/full/phase2-YYYYMMDDTHHMMSSZ --resume
```

Do not run two stage launchers concurrently against the same directory. The
scripts always pass `--full`; merely syntax-checking them does not launch an
experiment:

```bash
bash -n experiments/full/_common.sh experiments/full/run_stage{1,2,3}.sh
```

Before working on the numerical implementation, read `AGENTS.md`,
`docs/mathematical_reference.md`, `docs/experiment_implementation_spec.md`, and
the living plan at `docs/plans/pgd_experiments.md`.
