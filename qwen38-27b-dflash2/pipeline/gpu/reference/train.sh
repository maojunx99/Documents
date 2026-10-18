# Publication change: existing smoke directories are rejected, never deleted.
set -euo pipefail
# Stage 5: training. Full run by default; SMOKE_STEPS=<n> runs a bounded smoke.
source "$(dirname "$0")/env.sh"
cd "$REPO"
EXTRA=()
SAVE_PATH="$ROOT/run/checkpoints"
RUN_NAME=qwen38_27b_pb860k_5ep
if [ -n "${SMOKE_STEPS:-}" ]; then
  EXTRA+=(--max-steps "$SMOKE_STEPS")
  SAVE_PATH="$ROOT/run/smoke_checkpoints"
  RUN_NAME=qwen38_27b_pb860k_smoke
  test ! -e "$SAVE_PATH" || { echo "Refusing to overwrite existing smoke output" >&2; exit 1; }
fi
mkdir -p "$ROOT/run/logs"
CUDA_VISIBLE_DEVICES=4,5,6,7 setsid nohup "$VENV_PY" -m torch.distributed.run \
  --standalone --nproc_per_node 4 \
  -m speculators.train \
  --verifier-name-or-path "$MODEL" \
  --data-path "$ROOT/prepared" \
  --vllm-endpoint "http://127.0.0.1:$TEACHER_PORT/v1" \
  --save-path "$SAVE_PATH" \
  --log-dir "$ROOT/run/logs" --run-name "$RUN_NAME" --log-freq 20 \
  --speculator-type dflash2 --block-size 8 --max-anchors "$MAX_ANCHORS" \
  --draft-config "$DRAFT_CONFIG" \
  --mask-token-id "$MASK_TOKEN_ID" \
  --target-layer-ids $TARGET_LAYER_IDS \
  --loss-fn "$LOSS_FN" \
  --dflash-decay-gamma "$GAMMA" \
  --per-position-loss-weight fixed-exp-decay \
  --selector-loss-alpha "$SELECTOR_ALPHA" \
  --optimizer adamw --lr "$LR" --weight-decay 0 \
  --scheduler-type cosine --scheduler-warmup-ratio 0.04 \
  --epochs "$EPOCHS" --total-seq-len "$SEQ_LEN" \
  --train-data-ratio "$TRAIN_RATIO" \
  --noise-std "$NOISE_STD" --seed "$SEED" \
  --hidden-states-dtype bfloat16 \
  --num-workers 12 --prefetch-factor 4 \
  --on-missing generate --on-generate delete \
  --request-timeout 600 --max-retries 3 \
  --checkpoint-freq 1 --save-best \
  --no-resume-from-checkpoint \
  "${EXTRA[@]}" \
  > "$ROOT/run/trainer_launch.log" 2>&1 &
echo "TRAINER_LAUNCHING pid=$! smoke=${SMOKE_STEPS:-no}"
