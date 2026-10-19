#!/usr/bin/env bash
# HOST: edit these host paths/card IDs, or set the corresponding environment variables.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.26.0rc2}" # A2/910B; A3 needs the -a3 image.
CONTAINER_NAME="${CONTAINER_NAME:-dflash2-27b-train}"
HOST_WORK_ROOT="${HOST_WORK_ROOT:-/data/dflash2_workspace}"
HOST_MODEL_ROOT="${HOST_MODEL_ROOT:-/data/models}"
HOST_PROMPT_ROOT="${HOST_PROMPT_ROOT:-/data/dflash2_prompts}"
HOST_RUN_ROOT="${HOST_RUN_ROOT:-/data/dflash2_runs}"
NPU_IDS="${NPU_IDS:-0,1,2,3,4,5,6,7}"
SHM_SIZE="${SHM_SIZE:-128g}"

dry_run=false
case "${1:-}" in
  --dry-run) dry_run=true ;;
  --help|-h) echo 'Host usage: bash examples/npu/01_container.sh [--dry-run]'; exit 0 ;;
  '') ;;
  *) echo 'Unknown argument; use --help' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Too many arguments' >&2; exit 2; }
for directory in "$HOST_WORK_ROOT" "$HOST_MODEL_ROOT" "$HOST_PROMPT_ROOT" "$HOST_RUN_ROOT"; do
  [[ "$directory" == /* && "$directory" != *:* ]] || {
    echo 'Host mount paths must be absolute Linux paths without colons' >&2; exit 1;
  }
done
[[ "$NPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid NPU_IDS' >&2; exit 1; }
declare -A seen=()
device_args=()
IFS=',' read -r -a card_ids <<< "$NPU_IDS"
for card in "${card_ids[@]}"; do
  [[ -z "${seen[$card]:-}" ]] || { echo "Duplicate NPU card: $card" >&2; exit 1; }
  seen[$card]=1
  device_args+=(--device "/dev/davinci${card}")
done
mount_args=()
[[ ! -f /etc/hccn.conf ]] || mount_args+=(-v /etc/hccn.conf:/etc/hccn.conf:ro)
command=(docker run -it --name "$CONTAINER_NAME" --network host
  --shm-size="$SHM_SIZE" --ulimit memlock=-1 "${device_args[@]}"
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
  -v /usr/local/dcmi:/usr/local/dcmi:ro
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro "${mount_args[@]}"
  -v "$HOST_WORK_ROOT:/workspace" -v "$PIPELINE_ROOT:/workspace/pipeline"
  -v "$HOST_MODEL_ROOT:/models:ro" -v "$HOST_PROMPT_ROOT:/data/prompts:ro"
  -v "$HOST_RUN_ROOT:/data/runs" -w /workspace/pipeline "$IMAGE" bash)
if "$dry_run"; then
  npu_print_command docker pull "$IMAGE"
  npu_print_command "${command[@]}"
  exit 0
fi
command -v docker >/dev/null
npu-smi info
[[ -d "$HOST_MODEL_ROOT" && -d "$HOST_PROMPT_ROOT" ]] || {
  echo 'Prepare the host model and raw prompt directories first' >&2; exit 1;
}
for card in "${card_ids[@]}"; do
  [[ -c "/dev/davinci${card}" ]] || { echo "Missing device: /dev/davinci${card}" >&2; exit 1; }
done
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  printf 'Container already exists. Re-enter with: docker start -ai %q\n' "$CONTAINER_NAME" >&2
  exit 1
fi
mkdir -p "$HOST_WORK_ROOT" "$HOST_RUN_ROOT"
docker pull "$IMAGE"
exec "${command[@]}"
