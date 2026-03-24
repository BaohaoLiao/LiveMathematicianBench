#!/usr/bin/env bash
set -euo pipefail

#######################################
# Configuration — edit these variables
#######################################
MODEL="claude-opus-4.6"
BASE_URL="http://localhost:4141"
API_KEY="${ANTHROPIC_API_KEY:-unused}"
MONTHS="202511 202512 202601 202602"
REASONING_EFFORT="high"    # low, medium, high, or leave empty for default
MAX_TOKENS=65000
THINKING_BUDGET=62000      # set to empty to disable extended thinking
CONCURRENCY=2
SEED=42
DEBUG=false          # set to true to only evaluate 1 question per month

#######################################
# Run
#######################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

# Activate venv if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source .venv/bin/activate
fi

ARGS=(
    --model "$MODEL"
    --base-url "$BASE_URL"
    --api-key "$API_KEY"
    --month $MONTHS
    --max-tokens "$MAX_TOKENS"
    --concurrency "$CONCURRENCY"
    --seed "$SEED"
    --resume
)

if [[ -n "$REASONING_EFFORT" ]]; then
    ARGS+=(--reasoning-effort "$REASONING_EFFORT")
fi

if [[ -n "$THINKING_BUDGET" ]]; then
    ARGS+=(--thinking-budget "$THINKING_BUDGET")
fi

if [[ "$DEBUG" == "true" ]]; then
    ARGS+=(--debug)
fi

python eval/eval_claude.py "${ARGS[@]}"
