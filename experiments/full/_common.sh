#!/usr/bin/env bash
# Shared, resumable logging for the approved Phase 2 full-sweep launchers.

set -Eeuo pipefail

FULL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FULL_REPO_ROOT="$(cd -- "$FULL_SCRIPT_DIR/../.." && pwd)"

full_usage() {
  printf 'Usage: %s RUN_DIRECTORY [--resume] [--workers N]\n' "$(basename -- "$0")" >&2
}

full_init() {
  if [[ $# -lt 1 ]]; then
    full_usage
    return 2
  fi
  FULL_RUN_DIR="$(mkdir -p -- "$1" && cd -- "$1" && pwd)"
  shift
  FULL_RESUME=false
  FULL_WORKERS=${KSVD_WORKERS:-56}
  while (( $# )); do
    case $1 in
      --resume) FULL_RESUME=true; shift ;;
      --workers)
        [[ $# -ge 2 ]] || { full_usage; return 2; }
        FULL_WORKERS=$2; shift 2 ;;
      *) full_usage; return 2 ;;
    esac
  done
  [[ $FULL_WORKERS =~ ^[1-9][0-9]*$ ]] || { printf 'Worker count must be a positive integer.\n' >&2; return 2; }
  mkdir -p -- "$FULL_RUN_DIR/raw" "$FULL_RUN_DIR/logs" "$FULL_RUN_DIR/checksums"
  FULL_MANIFEST="$FULL_RUN_DIR/manifest.tsv"
  if [[ ! -e $FULL_MANIFEST ]]; then
    printf 'timestamp_utc\tstage\texperiment\tstatus\texit_code\traw_json\tconsole_log\n' >"$FULL_MANIFEST"
  fi
  if [[ ! -e $FULL_RUN_DIR/environment.txt ]]; then
    {
      printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'repository=%s\n' "$FULL_REPO_ROOT"
      printf 'git_commit=%s\n' "$(git -C "$FULL_REPO_ROOT" rev-parse HEAD)"
      printf 'git_status_begin\n'
      git -C "$FULL_REPO_ROOT" status --short --branch
      printf 'git_status_end\n'
      printf 'python_executable=%s\n' "$(command -v python)"
      python --version 2>&1
      PYTHONPATH="$FULL_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
import platform
import torch
import ksvd

print(f"platform={platform.platform()}")
print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"torch_device=cpu")
print(f"ksvd_module={ksvd.__file__}")
PY
      df -Pk "$FULL_RUN_DIR"
    } >"$FULL_RUN_DIR/environment.txt"
  fi
}

full_begin_stage() {
  FULL_STAGE="$1"
  FULL_LOCK="$FULL_RUN_DIR/.${FULL_STAGE}.running"
  if ! mkdir -- "$FULL_LOCK" 2>/dev/null; then
    printf 'Refusing to start: stage lock exists: %s\n' "$FULL_LOCK" >&2
    return 1
  fi
  trap 'rmdir -- "$FULL_LOCK" 2>/dev/null || true' EXIT
}

full_record() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$FULL_STAGE" "$1" "$2" "$3" "$4" "$5" >>"$FULL_MANIFEST"
}

full_validate_raw() {
  local raw=$1 experiment=$2
  python - "$raw" "$experiment" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
with path.open(encoding="utf-8") as stream:
    document = json.load(stream, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
metadata = document["metadata"]
if metadata["smoke"] is not False:
    raise SystemExit("artifact is not a full-sweep result")
if str(metadata["experiment"]).lower() != expected.lower():
    raise SystemExit(f"experiment mismatch: {metadata['experiment']} != {expected}")
if metadata.get("serialized_bytes") != path.stat().st_size:
    raise SystemExit("serialized byte count does not match file size")
PY
}

full_run_experiment() {
  local experiment=$1
  local raw="$FULL_RUN_DIR/raw/$experiment.json"
  local log="$FULL_RUN_DIR/logs/$experiment.log"
  if [[ -e $raw ]]; then
    if [[ $FULL_RESUME == true ]] && full_validate_raw "$raw" "$experiment"; then
      printf 'Skipping validated completed artifact: %s\n' "$raw"
      full_record "$experiment" skipped 0 "$raw" "$log"
      return 0
    fi
    printf 'Refusing to overwrite existing artifact: %s (use --resume to validate and skip it)\n' "$raw" >&2
    return 1
  fi
  full_record "$experiment" started 0 "$raw" "$log"
  printf 'Starting %s at %s\n' "$experiment" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log"
  set +e
  (
    cd -- "$FULL_REPO_ROOT"
    export PYTHONPATH="$FULL_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    time python "experiments/$experiment.py" --full --workers "$FULL_WORKERS" --output "$raw"
  ) 2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  set -e
  if (( status != 0 )); then
    full_record "$experiment" failed "$status" "$raw" "$log"
    return "$status"
  fi
  if ! full_validate_raw "$raw" "$experiment" 2>&1 | tee -a "$log"; then
    full_record "$experiment" invalid 1 "$raw" "$log"
    return 1
  fi
  sha256sum "$raw" >"$FULL_RUN_DIR/checksums/$experiment.sha256"
  full_record "$experiment" completed 0 "$raw" "$log"
  printf 'Completed %s at %s\n' "$experiment" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log"
}

full_finish_stage() {
  touch -- "$FULL_RUN_DIR/${FULL_STAGE}.complete"
  rmdir -- "$FULL_LOCK"
  trap - EXIT
  printf 'Stage %s complete. Inspect %s before launching the next stage.\n' "$FULL_STAGE" "$FULL_RUN_DIR"
}
