# Publication change: workspace paths anonymized; see pipeline/PROVENANCE.md.
set -euo pipefail
# Manual DP4 switch: launch 4x TP1 teachers on GPUs 0-3
ROOT=/workspace/dflash2_work/public_reproduction/qwen38_27b_pb860k_20260926
REPO=/workspace/speculators
VENV_PY=/workspace/dflash2_work/venv/bin/python
MODEL=/workspace/dflash2_work/models/Qwen3.8-27B

mkdir -p "$ROOT/run/teacher_dp4" "$ROOT/run/hidden_states"

for GPU in 0 1 2 3; do
    PORT=$((8260 + GPU))
    LOG="$ROOT/run/teacher_dp4/gpu${GPU}.log"
    PROV="$ROOT/run/teacher_dp4/gpu${GPU}_provenance"
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup "$VENV_PY" "$REPO/scripts/launch_vllm.py" "$MODEL" \
        --target-layer-ids 6 20 34 48 62 \
        --hidden-states-path "$ROOT/run/hidden_states" \
        --provenance-dir "$PROV" \
        -- --tensor-parallel-size 1 --max-model-len 16384 \
        --mm-processor-cache-gb 0 --port $PORT \
        > "$LOG" 2>&1 &
    echo "gpu$GPU -> port $PORT pid=$!"
done
echo "ALL_4_TP1_LAUNCHED"
