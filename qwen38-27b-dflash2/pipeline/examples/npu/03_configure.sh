#!/usr/bin/env bash
# CONTAINER: create a private editable configuration, never overwrite an existing one.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
output="$PIPELINE_ROOT/config.local.npu.sh"
dry_run=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) [[ $# -ge 2 ]] || { echo '--output needs a path' >&2; exit 2; }; output="$2"; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    --help|-h) echo 'Usage: bash examples/npu/03_configure.sh [--output PATH] [--dry-run]'; exit 0 ;;
    *) echo 'Unknown argument; use --help' >&2; exit 2 ;;
  esac
done
if "$dry_run"; then
  printf 'Create config (without overwrite): %s -> %s\n' "$PIPELINE_ROOT/examples/npu/config.example.sh" "$output"
  exit 0
fi
if [[ -e "$output" ]]; then
  printf 'Kept existing config: %s\n' "$output"
else
  (set -o noclobber; cat "$PIPELINE_ROOT/examples/npu/config.example.sh" > "$output")
  printf 'Created config: %s\n' "$output"
fi
printf 'Edit this file before running step 04: %s\n' "$output"
