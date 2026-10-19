#!/usr/bin/env bash
# CONTAINER: keep the image's matched compute stack and install supplementary packages.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
VENV_PATH="${VENV_PATH:-/workspace/venvs/dflash2}"
ENV_RECORDS="${ENV_RECORDS:-/workspace/environment_records}"
BASE_PYTHON="${BASE_PYTHON:-python3}"
[[ $# -le 1 ]] || { echo 'Too many arguments' >&2; exit 2; }
case "${1:-}" in
  --dry-run)
    npu_print_command "$BASE_PYTHON" -m venv --system-site-packages "$VENV_PATH"
    npu_print_command "$VENV_PATH/bin/python" "$PIPELINE_ROOT/npu/setup_environment.py" record-base --out "$ENV_RECORDS"
    npu_print_command "$VENV_PATH/bin/python" -m pip install -c "$ENV_RECORDS/base-stack.txt" -r "$PIPELINE_ROOT/npu/requirements-training.txt"
    npu_print_command "$VENV_PATH/bin/python" "$PIPELINE_ROOT/npu/setup_environment.py" verify --out "$ENV_RECORDS"
    npu_print_command "$VENV_PATH/bin/python" "$PIPELINE_ROOT/check_bundle.py"
    exit 0 ;;
  --help|-h) echo 'Container usage: bash examples/npu/02_install.sh [--dry-run]'; exit 0 ;;
  '') ;;
  *) echo 'Unknown argument; use --help' >&2; exit 2 ;;
esac
npu_source_environment
if [[ ! -e "$VENV_PATH" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV_PATH"
fi
[[ -x "$VENV_PATH/bin/python" ]] || { echo 'VENV_PATH is not a Python environment' >&2; exit 1; }
"$VENV_PATH/bin/python" "$PIPELINE_ROOT/npu/setup_environment.py" record-base --out "$ENV_RECORDS"
"$VENV_PATH/bin/python" -m pip install -c "$ENV_RECORDS/base-stack.txt" -r "$PIPELINE_ROOT/npu/requirements-training.txt"
"$VENV_PATH/bin/python" "$PIPELINE_ROOT/npu/setup_environment.py" verify --out "$ENV_RECORDS"
"$VENV_PATH/bin/python" "$PIPELINE_ROOT/check_bundle.py"
printf 'Environment ready. Training Python: %s/bin/python\n' "$VENV_PATH"
