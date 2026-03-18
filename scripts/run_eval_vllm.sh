#!/usr/bin/env bash
set -euo pipefail

#######################################
# Configuration — edit these variables
#######################################
MODEL="Qwen/Qwen2.5-72B-Instruct"
BASE_URL="http://localhost:8000/v1"
API_KEY="EMPTY"             # vLLM typically doesn't require auth
MONTHS="202511 202512 202601 202602"
REASONING_EFFORT=""         # low, medium, high, xhigh, or leave empty for default
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

python eval/eval_vllm.py "${ARGS[@]}"
