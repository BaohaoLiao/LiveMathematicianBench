#!/usr/bin/env bash
set -euo pipefail

#######################################
# Configuration — edit these variables
#######################################
MODEL="gpt-5.4"
BASE_URL="http://localhost:4141/v1"
API_KEY="dummy"
MONTHS="202511 202512 202601 202602"
REASONING_EFFORT="high"         # none, minimal, low, medium, high, xhigh, or leave empty for default
MAX_TOKENS=65000
TEMPERATURE=1.0
CONCURRENCY=4
SEED=42
DEBUG=false          # set to true to only evaluate 1 question per month
ADD_SKETCH=true      # set to true to append proof sketch as hints to the prompt

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
    --concurrency "$CONCURRENCY"
    --seed "$SEED"
    --resume
    --use-responses-api
)

if [[ "$ADD_SKETCH" == "true" ]]; then
    ARGS+=(--add-sketch)
fi

if [[ -n "$REASONING_EFFORT" ]]; then
    ARGS+=(--reasoning-effort "$REASONING_EFFORT")
fi

if [[ "$DEBUG" == "true" ]]; then
    ARGS+=(--debug)
fi

python eval/eval_vllm.py "${ARGS[@]}"
