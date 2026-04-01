#!/usr/bin/env bash
set -euo pipefail

#######################################
# Configuration — edit these variables
#######################################
MODEL="deepseek-v3.2"
BASE_URL="http://localhost:8000/v1"
API_KEY="EMPTY"             # SGLang typically doesn't require auth
MONTHS="202511 202512 202601 202602"
REASONING_EFFORT=""         # low, medium, high, xhigh, or leave empty for default
THINKING=1                   # set to 1 to enable DeepSeek-V3.2 thinking mode and save raw_thinking
MAX_TOKENS=65000
TEMPERATURE=1.0
TOP_P=0.95
CONCURRENCY=4
SEED=42

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
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --concurrency "$CONCURRENCY"
    --seed "$SEED"
    --resume
)

if [[ -n "$REASONING_EFFORT" ]]; then
    ARGS+=(--reasoning-effort "$REASONING_EFFORT")
fi

if [[ "$THINKING" == "1" ]]; then
    ARGS+=(--thinking)
fi

python eval/eval_sglang.py "${ARGS[@]}"