#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

full_init "$@"
full_begin_stage stage3
full_run_experiment geometry_of_kc
full_run_experiment step_size_phase_diagram
full_finish_stage
