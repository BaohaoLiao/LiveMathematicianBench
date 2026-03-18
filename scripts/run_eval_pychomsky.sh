#!/usr/bin/env bash
set -euo pipefail

#######################################
# Configuration — edit these variables
#######################################
MODEL="gcp-chat-completions-anthropic-claude-opus-4.6-sandbox"
ENDPOINT="https://chomskygw6cont.pp.vip.ebay.com/api/v1/genai"
MONTHS=(202511 202512 202601 202602)
REASONING_EFFORT="max"   # low, medium, high, max, xhigh, or leave empty for default
MAX_TOKENS=16384
TEMPERATURE=1.0
THINKING_TYPE="adaptive"
CONCURRENCY=4
SEED=42
N=1

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
    --endpoint "$ENDPOINT"
    --month "${MONTHS[@]}"
    --max-tokens "$MAX_TOKENS"
    --temperature "$TEMPERATURE"
    --thinking-type "$THINKING_TYPE"
    --concurrency "$CONCURRENCY"
    --seed "$SEED"
    --n "$N"
    --resume
)

if [[ -n "$REASONING_EFFORT" ]]; then
    ARGS+=(--reasoning-effort "$REASONING_EFFORT")
fi

python3 eval/eval_pychomsky.py "${ARGS[@]}" "$@"
