#!/usr/bin/env bash
# Edit these variables (or export them), then run inside each teacher container.
# CPU waits for the trainer; accelerator allocation begins only after a matching claim.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND="${BACKEND:-npu}"
PYTHON="${PYTHON:-/workspace/venvs/dflash2/bin/python}"
TARGET_MODEL="${TARGET_MODEL:-/models/Qwen3.8-27B}"
TEACHER_DEVICES="${TEACHER_DEVICES:-0,1}"
SHARED_RUN="${SHARED_RUN:-/shared/dflash2/teacher_a_job01}" # Fresh per-host directory, same mount on trainer.
API_KEY_FILE="${API_KEY_FILE:-/run/secrets/dflash2_teacher.key}"
BIND_HOST="${BIND_HOST:-127.0.0.1}" # Set reachable private IP, or keep loopback behind an SSH tunnel.
GATEWAY_PORT="${GATEWAY_PORT:-8395}"
NATIVE_PORT="${NATIVE_PORT:-8394}"
DISTRIBUTED_PORT="${DISTRIBUTED_PORT:-29300}"
TEACHER_CONTEXT="${TEACHER_CONTEXT:-16384}"
TEACHER_MAX_NUM_SEQS="${TEACHER_MAX_NUM_SEQS:-4}"
TEACHER_MEMORY="${TEACHER_MEMORY:-0.8}"
LEASE_SECONDS="${LEASE_SECONDS:-120}"
if [[ "$BACKEND" == npu && " ${*} " != *' --dry-run '* ]]; then
  source "$ROOT/examples/npu/_common.sh"
  npu_source_environment
fi
exec "$PYTHON" "$ROOT/teachers/service.py" --backend "$BACKEND" --model "$TARGET_MODEL" \
  --devices "$TEACHER_DEVICES" --run "$SHARED_RUN" --api-key-file "$API_KEY_FILE" --bind "$BIND_HOST" \
  --port "$GATEWAY_PORT" --native-port "$NATIVE_PORT" --distributed-port "$DISTRIBUTED_PORT" \
  --max-model-len "$TEACHER_CONTEXT" --max-num-seqs "$TEACHER_MAX_NUM_SEQS" \
  --memory-utilization "$TEACHER_MEMORY" --lease-seconds "$LEASE_SECONDS" "$@"
