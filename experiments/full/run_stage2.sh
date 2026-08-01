#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

full_init "$@"
full_begin_stage stage2
full_run_experiment boundary_gap_scaling
full_run_experiment tied_eigenvalues
full_run_experiment initialization_ablation
full_finish_stage
