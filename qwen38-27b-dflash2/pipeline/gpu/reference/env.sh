# Publication change: workspace paths anonymized; see pipeline/PROVENANCE.md.
set -euo pipefail
# Qwen3.8-27B DFlash2 — run package environment (sourced by all stage scripts)
export ROOT=/workspace/dflash2_work/public_reproduction/qwen38_27b_pb860k_20260926
export REPO=/workspace/speculators
export VENV_BIN=/workspace/dflash2_work/venv/bin
export VENV_PY="$VENV_BIN/python"
export MODEL=/workspace/dflash2_work/models/Qwen3.8-27B
export PUBLIC_DRAFT=/workspace/dflash2_work/models/Qwen3.8-27B-DFlash2
export DRAFT_CONFIG="$REPO/examples/train/configs/dflash2_qwen38_27b_draft_config.json"
export TEACHER_PORT=8250
# aux-layer codes (HF hidden_states index, 0=embedding); launch_vllm auto-appends final layer 64
export TARGET_LAYER_IDS="6 20 34 48 62"
export MASK_TOKEN_ID=248070
# user-confirmed recipe: 4B-line recipe ported, 5 epochs
export EPOCHS=5
export LR=0.0006
export LOSS_FN='{"ce": 0.1, "tv": 0.9}'
export SELECTOR_ALPHA=0.1
export GAMMA=4.0
export SEQ_LEN=8192
export MAX_ANCHORS=1024
export NOISE_STD=0.05
export SEED=42
export TRAIN_RATIO=0.99
