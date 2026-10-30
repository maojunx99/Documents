#!/usr/bin/env bash
# Run inside the configured NPU training container. Edit these variables, then:
# bash examples/train/npu_dflash2_27b.sh --dry-run
# bash examples/train/npu_dflash2_27b.sh
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# 1. Model and data. Qwen3.6-27B is usable when it has the same27B text/tokenizer structure.
PYTHON="${PYTHON:-python}"
TARGET_MODEL="${TARGET_MODEL:-/models/Qwen3.8-27B}"
RUN_DIR="${RUN_DIR:-/data/runs/dflash2_27b_$(date +%Y%m%d_%H%M%S)}"
INPUT_FILES=(/data/prompts/*.jsonl)
# Optional: reuse verified answers from this same target instead of regenerating.
REGENERATED_DIR="${REGENERATED_DIR:-}"
DRAFT_CONFIG="${DRAFT_CONFIG:-$ROOT/npu/draft_config.json}"
FROM_PRETRAINED="${FROM_PRETRAINED:-}" # Optional native/official DFlash2 weights; new optimizer.
TARGET_LAYERS=(6 20 34 48 62)
MASK_TOKEN_ID="${MASK_TOKEN_ID:-248070}"

# 2. Parallelism: training cards = FSDP_REPLICAS * FSDP_SHARD_SIZE.
# 1*4 -> FSDP4; 2*4 -> HSDP8; 4*4 -> HSDP16; 2*2 -> HSDP4.
FSDP_REPLICAS="${FSDP_REPLICAS:-2}"
FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE:-4}"
TRAIN_DEVICES="${TRAIN_DEVICES:-8,9,10,11,12,13,14,15}"
# Semicolons separate independent TP services; commas are cards within each service.
REGEN_TP_GROUPS="${REGEN_TP_GROUPS:-8,9;10,11;12,13;14,15}"
TEACHER_TP_GROUPS="${TEACHER_TP_GROUPS:-2,3}"
# Generation services are stopped before training; those cards may be reused.
GENERATION_PORT="${GENERATION_PORT:-8500}"
TEACHER_PORT="${TEACHER_PORT:-8360}"
ROUTER_PORT="${ROUTER_PORT:-8350}"

# 3. Target response regeneration. No extra system prompt is added.
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"
GENERATION_MAX_TOKENS="${GENERATION_MAX_TOKENS:-8192}"
GENERATION_CONCURRENCY="${GENERATION_CONCURRENCY:-34}"
GENERATION_CONTEXT="${GENERATION_CONTEXT:-32768}"
GENERATION_MAX_NUM_SEQS="${GENERATION_MAX_NUM_SEQS:-64}"
GENERATION_MEMORY="${GENERATION_MEMORY:-0.85}"
# Leave empty to retain native unseeded response sampling.
GENERATION_SEED="${GENERATION_SEED:-}"

# 4. Training recipe. Anchors are a per-card packed-batch budget.
SAMPLE_MAX_LENGTH="${SAMPLE_MAX_LENGTH:-4096}"
PACKED_TOKENS="${PACKED_TOKENS:-4096}"
MAX_ANCHORS="${MAX_ANCHORS:-512}"
EPOCHS="${EPOCHS:-5}"
MAX_STEPS="${MAX_STEPS:-}"   # Empty = full epochs; a positive value = finite global-update smoke.
STOP_AFTER_EPOCH="${STOP_AFTER_EPOCH:-}" # Inclusive zero-based boundary; keeps EPOCHS for the LR schedule.
SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-}"
TRAIN_RATIO="${TRAIN_RATIO:-0.99}"
LR="${LR:-0.0006}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
SELECTOR_ALPHA="${SELECTOR_ALPHA:-0.1}"
CE_WEIGHT="${CE_WEIGHT:-0.1}"
TV_WEIGHT="${TV_WEIGHT:-0.9}"
GAMMA="${GAMMA:-4}"
NOISE_STD="${NOISE_STD:-0.05}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PREPARE_WORKERS="${PREPARE_WORKERS:-8}"
TEACHER_MAX_NUM_SEQS="${TEACHER_MAX_NUM_SEQS:-4}"
TEACHER_MEMORY="${TEACHER_MEMORY:-0.80}"
HIDDEN_CACHE="${HIDDEN_CACHE:-}"  # Empty = RUN_DIR/hidden_states; explicit path must not exist.
STAGE_TIMEOUT="${STAGE_TIMEOUT:-86400}"

data_args=()
if [[ -n "$REGENERATED_DIR" ]]; then
  data_args=(--regenerated-dir "$REGENERATED_DIR")
else
  data_args=(--input "${INPUT_FILES[@]}")
fi
extra=()
[[ -z "$FROM_PRETRAINED" ]] || extra+=(--from-pretrained "$FROM_PRETRAINED")
[[ -z "$MAX_STEPS" ]] || extra+=(--max-steps "$MAX_STEPS")
[[ -z "$STOP_AFTER_EPOCH" ]] || extra+=(--stop-after-epoch "$STOP_AFTER_EPOCH")
[[ -z "$SCHEDULER_TOTAL_STEPS" ]] || extra+=(--scheduler-total-steps "$SCHEDULER_TOTAL_STEPS")
[[ -z "$SCHEDULER_WARMUP_STEPS" ]] || extra+=(--scheduler-warmup-steps "$SCHEDULER_WARMUP_STEPS")
[[ -z "$GENERATION_SEED" ]] || extra+=(--generation-seed "$GENERATION_SEED")
[[ -z "$HIDDEN_CACHE" ]] || extra+=(--hidden-cache "$HIDDEN_CACHE")

exec "$PYTHON" "$ROOT/npu/run_pipeline.py" \
  --model "$TARGET_MODEL" --run "$RUN_DIR" --python "$PYTHON" "${data_args[@]}" \
  --draft-config "$DRAFT_CONFIG" --target-layer-ids "${TARGET_LAYERS[@]}" --mask-token-id "$MASK_TOKEN_ID" \
  --replicas "$FSDP_REPLICAS" --shard-size "$FSDP_SHARD_SIZE" --train-devices "$TRAIN_DEVICES" \
  --regeneration-groups "$REGEN_TP_GROUPS" --teacher-groups "$TEACHER_TP_GROUPS" \
  --generation-port "$GENERATION_PORT" --teacher-port "$TEACHER_PORT" --router-port "$ROUTER_PORT" \
  --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" --presence-penalty "$PRESENCE_PENALTY" \
  --reasoning-effort "$REASONING_EFFORT" --generation-max-tokens "$GENERATION_MAX_TOKENS" \
  --generation-concurrency "$GENERATION_CONCURRENCY" --generation-max-model-len "$GENERATION_CONTEXT" \
  --generation-max-num-seqs "$GENERATION_MAX_NUM_SEQS" --generation-memory-utilization "$GENERATION_MEMORY" \
  --sample-max-length "$SAMPLE_MAX_LENGTH" --packed-tokens "$PACKED_TOKENS" --max-anchors "$MAX_ANCHORS" \
  --epochs "$EPOCHS" --train-ratio "$TRAIN_RATIO" --lr "$LR" --weight-decay "$WEIGHT_DECAY" \
  --warmup-ratio "$WARMUP_RATIO" --selector-alpha "$SELECTOR_ALPHA" --ce-weight "$CE_WEIGHT" --tv-weight "$TV_WEIGHT" \
  --gamma "$GAMMA" --noise-std "$NOISE_STD" --seed "$SEED" --workers "$WORKERS" --prefetch-factor "$PREFETCH_FACTOR" \
  --prepare-workers "$PREPARE_WORKERS" --teacher-max-num-seqs "$TEACHER_MAX_NUM_SEQS" --teacher-memory-utilization "$TEACHER_MEMORY" \
  --stage-timeout "$STAGE_TIMEOUT" "${extra[@]}" "$@"
