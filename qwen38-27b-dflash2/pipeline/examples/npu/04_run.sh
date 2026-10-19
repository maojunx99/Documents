#!/usr/bin/env bash
# CONTAINER: load one configuration, optionally smoke-test, then delegate to the existing pipeline.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
config="$PIPELINE_ROOT/config.local.npu.sh"
smoke=false
dry_run=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) [[ $# -ge 2 ]] || { echo '--config needs a path' >&2; exit 2; }; config="$2"; shift 2 ;;
    --smoke) smoke=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    --help|-h) echo 'Usage: bash examples/npu/04_run.sh [--config PATH] [--smoke] [--dry-run]'; exit 0 ;;
    *) echo 'Unknown argument; use --help' >&2; exit 2 ;;
  esac
done
[[ -f "$config" ]] || { echo 'Create/edit the configuration with step 03 first' >&2; exit 1; }
set -a
source "$config"
set +a
: "${PYTHON:?Set PYTHON in the configuration}"
: "${TARGET_MODEL:?Set TARGET_MODEL in the configuration}"
: "${RUN_ROOT:?Set RUN_ROOT in the configuration}"
mode=full
"$smoke" && mode=smoke
export RUN_DIR="${RUN_DIR:-$RUN_ROOT/${mode}_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$RUN_DIR" ]] || { echo "Run directory already exists: $RUN_DIR" >&2; exit 1; }
extra=()
if "$smoke"; then
  declare -p INPUT_FILES >/dev/null 2>&1 && [[ ${#INPUT_FILES[@]} -gt 0 ]] || {
    echo 'The smoke test requires raw INPUT_FILES in the configuration' >&2; exit 1;
  }
  export MAX_STEPS=2 EPOCHS=2 TRAIN_RATIO=0.8 WARMUP_RATIO=0
  export GENERATION_MAX_TOKENS=128 GENERATION_CONCURRENCY=4 REGENERATED_DIR=
  smoke_inputs="${RUN_DIR}_inputs"
  if ! "$dry_run"; then
    "$PYTHON" "$PIPELINE_ROOT/npu/prepare_smoke_inputs.py" --input "${INPUT_FILES[@]}" --output "$smoke_inputs"
  fi
  extra+=(--input "$smoke_inputs/prompts128.jsonl")
elif [[ -z "${REGENERATED_DIR:-}" ]]; then
  declare -p INPUT_FILES >/dev/null 2>&1 && [[ ${#INPUT_FILES[@]} -gt 0 ]] || {
    echo 'Set INPUT_FILES or a verified REGENERATED_DIR in the configuration' >&2; exit 1;
  }
  extra+=(--input "${INPUT_FILES[@]}")
fi
if "$dry_run"; then
  extra+=(--dry-run)
else
  npu_source_environment
fi
exec bash "$PIPELINE_ROOT/examples/train/npu_dflash2_27b.sh" "${extra[@]}"
