#!/usr/bin/env bash
# Shared argument forwarding; the existing one-command entry remains available.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
stage="$1"; shift
config="$PIPELINE_ROOT/config.local.npu.sh"
dry_run=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) [[ $# -ge 2 ]] || { echo '--config needs a path' >&2; exit 2; }; config="$2"; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    --help|-h) echo 'Usage: bash STAGE_SCRIPT [--config PATH] [--dry-run]'; exit 0 ;;
    *) echo 'Unknown argument; use --help' >&2; exit 2 ;;
  esac
done
[[ -f "$config" ]] || { echo 'Create/edit the configuration with 03_configure.sh first' >&2; exit 1; }
set -a
source "$config"
set +a
: "${RUN_DIR:?Set a fixed RUN_DIR in the configuration for all stage calls}"
extra=(--stage "$stage")
if [[ -z "${REGENERATED_DIR:-}" ]]; then
  declare -p INPUT_FILES >/dev/null 2>&1 && [[ ${#INPUT_FILES[@]} -gt 0 ]] || {
    echo 'Set INPUT_FILES or REGENERATED_DIR' >&2; exit 1;
  }
  extra+=(--input "${INPUT_FILES[@]}")
fi
if [[ "$stage" == evaluate ]]; then
  : "${EVAL_CASES:?Set EVAL_CASES to the five-benchmark case manifest}"
  extra+=(--eval-cases "$EVAL_CASES" --eval-python "${EVAL_PYTHON:-$PYTHON}"
    --eval-groups "${EVAL_TP_GROUPS:-0,1;2,3;4,5;6,7}" --eval-base-port "${EVAL_BASE_PORT:-8700}"
    --eval-per-benchmark "${EVAL_PER_BENCHMARK:-32}" --eval-max-tokens "${EVAL_MAX_TOKENS:-4096}"
    --eval-max-model-len "${EVAL_CONTEXT:-8192}" --eval-memory "${EVAL_MEMORY:-.8}"
    --eval-presence-penalty "${EVAL_PRESENCE_PENALTY:-0}" --eval-reasoning-effort "${EVAL_REASONING_EFFORT:-xhigh}")
  [[ -z "${EVAL_CHECKPOINT:-}" ]] || extra+=(--eval-checkpoint "$EVAL_CHECKPOINT")
  [[ -z "${EVAL_OUTPUT:-}" ]] || extra+=(--eval-output "$EVAL_OUTPUT")
  [[ -z "${EVAL_REFERENCE:-}" ]] || extra+=(--eval-reference "$EVAL_REFERENCE")
fi
if "$dry_run"; then extra+=(--dry-run); else npu_source_environment; fi
exec bash "$PIPELINE_ROOT/examples/train/npu_dflash2_27b.sh" "${extra[@]}"
