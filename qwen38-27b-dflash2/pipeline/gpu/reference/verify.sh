set -euo pipefail
# Stage 1: verify transferred regen output (counts, errors, truncation, sanity).
source "$(dirname "$0")/env.sh"
"$VENV_PY" "$ROOT/verify_regen.py"
