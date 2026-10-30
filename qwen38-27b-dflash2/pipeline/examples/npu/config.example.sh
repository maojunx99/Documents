#!/usr/bin/env bash
# Copy with 03_configure.sh. Edit config.local.npu.sh; step 04 exports these variables.
# Container paths: /models and /data/prompts are read-only host mounts.
PYTHON="${PYTHON:-/workspace/venvs/dflash2/bin/python}"
TARGET_MODEL="${TARGET_MODEL:-/models/Qwen3.8-27B}"
INPUT_FILES=(/data/prompts/open_perfectblend.jsonl /data/prompts/nemotron_v2.jsonl)
RUN_ROOT="${RUN_ROOT:-/data/runs}"
RUN_DIR="${RUN_DIR:-}" # Empty: choose a fresh timestamped directory under RUN_ROOT.
REGENERATED_DIR="${REGENERATED_DIR:-}" # Optional same-target verified shards; skips regeneration.

# Training cards = replicas * shard size. Default: 8-card host, FSDP4 + TP2 teacher.
# For 2*FSDP4, edit x/y and card lists together; live teachers need separate cards.
FSDP_REPLICAS="${FSDP_REPLICAS:-1}"
FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE:-4}"
TRAIN_DEVICES="${TRAIN_DEVICES:-0,1,2,3}"
TEACHER_TP_GROUPS="${TEACHER_TP_GROUPS:-4,5}"
# Regeneration runs first, then releases its devices before training.
REGEN_TP_GROUPS="${REGEN_TP_GROUPS:-0,1;2,3;4,5;6,7}"
GENERATION_PORT="${GENERATION_PORT:-8500}"
TEACHER_PORT="${TEACHER_PORT:-8360}"
ROUTER_PORT="${ROUTER_PORT:-8350}"

# Target response regeneration: native chat template; no manual extra system prompt.
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
GENERATION_SEED="${GENERATION_SEED:-}" # Empty: native unseeded sampling.

# Same 27B structure with different target weights: change TARGET_MODEL and regenerate.
# Different architectures require deliberate draft/layer/mask changes and new validation.
DRAFT_CONFIG="${DRAFT_CONFIG:-$PIPELINE_ROOT/npu/draft_config.json}"
FROM_PRETRAINED="${FROM_PRETRAINED:-}" # Optional official/native DFlash2 weight directory; empty = random draft.
MASK_TOKEN_ID="${MASK_TOKEN_ID:-248070}"
SAMPLE_MAX_LENGTH="${SAMPLE_MAX_LENGTH:-4096}"
PACKED_TOKENS="${PACKED_TOKENS:-4096}"
MAX_ANCHORS="${MAX_ANCHORS:-512}" # Shared budget per rank's packed batch.
EPOCHS="${EPOCHS:-5}"
MAX_STEPS="${MAX_STEPS:-}" # Empty: full epochs; positive: finite cumulative updates.
STOP_AFTER_EPOCH="${STOP_AFTER_EPOCH:-}" # Set 0 to stop after epoch0 without shortening EPOCHS' LR schedule.
SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-}" # Optional exact horizon from the reference run.
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
HIDDEN_CACHE="${HIDDEN_CACHE:-}" # Empty: under RUN_DIR; explicit path must not exist.
STAGE_TIMEOUT="${STAGE_TIMEOUT:-604800}" # Per-stage upper bound; adjust to your budget.

# Standalone real-inference evaluation (T0 + T1, five benchmarks, no extra system prompt).
# Set a fixed RUN_DIR above when using the separate stage scripts.
EVAL_CASES="${EVAL_CASES:-/data/prompts/five_benchmark_cases.json}"
EVAL_OUTPUT="${EVAL_OUTPUT:-}" # Empty: a fresh RUN_DIR/evaluations/<timestamp> directory per call.
EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-}" # Empty: last completed native epoch checkpoint.
EVAL_REFERENCE="${EVAL_REFERENCE:-}" # Optional official/community serving checkpoint.
EVAL_PYTHON="${EVAL_PYTHON:-$PYTHON}" # Must support DFlash2; may differ from training Python.
EVAL_TP_GROUPS="${EVAL_TP_GROUPS:-0,1;2,3;4,5;6,7}"
EVAL_BASE_PORT="${EVAL_BASE_PORT:-8700}"
EVAL_PER_BENCHMARK="${EVAL_PER_BENCHMARK:-32}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-4096}"
EVAL_CONTEXT="${EVAL_CONTEXT:-8192}"
EVAL_MEMORY="${EVAL_MEMORY:-0.80}"
EVAL_PRESENCE_PENALTY="${EVAL_PRESENCE_PENALTY:-0}"
EVAL_REASONING_EFFORT="${EVAL_REASONING_EFFORT:-xhigh}"
