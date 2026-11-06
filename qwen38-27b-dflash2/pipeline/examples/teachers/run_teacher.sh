#!/usr/bin/env bash
# Optional prestarted native teacher; lifecycle stays with its operator.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND="${BACKEND:-npu}"
PYTHON="${PYTHON:-python}"
TARGET_MODEL="${TARGET_MODEL:-/models/Qwen3.8-27B}"
TEACHER_DEVICES="${TEACHER_DEVICES:-0,1}"
HIDDEN_CACHE="${HIDDEN_CACHE:?Set the same fresh shared HIDDEN_CACHE as the trainer}"
TEACHER_PORT="${TEACHER_PORT:-8395}"
PROVENANCE_DIR="${PROVENANCE_DIR:?Set a new provenance directory}"
if [[ "${1:-}" == --dry-run ]]; then
  printf 'Native %s teacher: model=%s cards=%s port=%s shared_cache=%s\n' "$BACKEND" "$TARGET_MODEL" "$TEACHER_DEVICES" "$TEACHER_PORT" "$HIDDEN_CACHE"
  exit 0
fi
TARGET_MODEL="$(readlink -f "$TARGET_MODEL")"
mkdir "$PROVENANCE_DIR"
export PYTHONPATH="$ROOT/$BACKEND/source/src:$ROOT/$BACKEND/source/hs_connectors/src:${PYTHONPATH:-}"
if [[ "$BACKEND" == npu ]]; then
  source "$ROOT/examples/npu/_common.sh"
  npu_source_environment
  export ASCEND_RT_VISIBLE_DEVICES="$("$PYTHON" -c 'import sys;sys.path.insert(0,sys.argv[1]);from teachers.launch import npu_runtime_devices;print(npu_runtime_devices([int(x) for x in sys.argv[2].split(",")]))' "$ROOT" "$TEACHER_DEVICES")"
  export PYTHONPATH="$ROOT/npu/scripts:$PYTHONPATH"
else
  export CUDA_VISIBLE_DEVICES="$TEACHER_DEVICES"
fi
IFS=, read -ra cards <<< "$TEACHER_DEVICES"
printf '%q ' "$PYTHON" "$ROOT/$BACKEND/source/scripts/launch_vllm.py" "$TARGET_MODEL" "$@" > "$PROVENANCE_DIR/launch_command.txt"
extra=()
[[ "$BACKEND" != gpu ]] || extra+=(--provenance-dir "$PROVENANCE_DIR/native")
exec "$PYTHON" "$ROOT/$BACKEND/source/scripts/launch_vllm.py" "$TARGET_MODEL"  --target-layer-ids 6 20 34 48 62 --hidden-states-backend shared_file --hidden-states-path "$HIDDEN_CACHE"  "${extra[@]}" -- --host 0.0.0.0 --port "$TEACHER_PORT" --tensor-parallel-size "${#cards[@]}"  --dtype bfloat16 --enforce-eager --no-enable-chunked-prefill --max-model-len 8192 --max-num-seqs 4 "$@"
