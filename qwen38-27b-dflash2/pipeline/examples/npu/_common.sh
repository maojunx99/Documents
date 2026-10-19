#!/usr/bin/env bash
# Shared only by the public setup/run scripts; preserve the inherited CANN paths.
PIPELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

npu_print_command() {
  printf '%q ' "$@"
  printf '\n'
}

npu_source_environment() {
  local cann_env="${ASCEND_ENV_FILE:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
  local atb_env="${ATB_ENV_FILE:-/usr/local/Ascend/nnal/atb/set_env.sh}"
  [[ -f "$cann_env" && -f "$atb_env" ]] || {
    printf 'Missing Ascend environment scripts: %s / %s\n' "$cann_env" "$atb_env" >&2
    return 1
  }
  # Vendor environment scripts may read unset optional variables.
  set +u
  source "$cann_env"
  source "$atb_env"
  set -u
}
