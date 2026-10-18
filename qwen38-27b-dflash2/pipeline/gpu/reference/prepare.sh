set -euo pipefail
# Stage 2: prepare-data (pretokenized regen output -> HF arrow dataset).
source "$(dirname "$0")/env.sh"
cd "$REPO"
ARGS=()
for f in "$ROOT"/regen_data/*_out.jsonl; do
  ARGS+=(--data "$f")
done
"$VENV_PY" -m speculators prepare-data \
  --model "$MODEL" \
  "${ARGS[@]}" \
  --output "$ROOT/prepared" \
  --seq-length "$SEQ_LEN" \
  --seed "$SEED" \
  --num-preprocessing-workers 32
ls "$ROOT/prepared" | head -5
echo PREPARE_DONE
