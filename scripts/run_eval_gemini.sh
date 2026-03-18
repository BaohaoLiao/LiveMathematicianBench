#!/usr/bin/env bash
set -euo pipefail

#######################################
# Configuration - edit these variables
#######################################
MODEL="gcp-chat-completions-chat-gemini-3-pro-preview-sandbox"
MONTHS=(202511 202512 202601 202602)
REASONING_EFFORT="high"   # low, medium, high, or leave empty for default
MAX_TOKENS=16384
CONCURRENCY=4
SEED=42
N=1
INCLUDE_THOUGHTS=true

#######################################
# Run
#######################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" && -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

ARGS=(
    --model "$MODEL"
    --month "${MONTHS[@]}"
    --max-tokens "$MAX_TOKENS"
    --concurrency "$CONCURRENCY"
    --seed "$SEED"
    --n "$N"
    --resume
)

if [[ -n "$REASONING_EFFORT" ]]; then
    ARGS+=(--reasoning-effort "$REASONING_EFFORT")
fi

if [[ "$INCLUDE_THOUGHTS" == "true" ]]; then
    ARGS+=(--include-thoughts)
else
    ARGS+=(--no-include-thoughts)
fi

python3 eval/eval_gemini.py "${ARGS[@]}" "$@"
