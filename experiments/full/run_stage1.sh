#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

full_init "$@"
full_begin_stage stage1
full_run_experiment predicted_local_rates
full_run_experiment hessian_mode_isolation
full_run_experiment saddle_escape
full_finish_stage
