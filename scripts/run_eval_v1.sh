#!/usr/bin/env bash
set -euo pipefail

#######################################
# Configuration — edit these variables
#######################################
MODEL="grok-4-1-fast-reasoning"
ENDPOINT="https://e0271-miptdstj-eastus2.cognitiveservices.azure.com/"
API_KEY="${AZURE_OPENAI_API_KEY:?Set AZURE_OPENAI_API_KEY environment variable}"
API_VERSION="2024-12-01-preview"
MONTHS="202511 202512 202601 202602"
REASONING_EFFORT="high"    # low, medium, high, or leave empty for default
MAX_TOKENS=128000
CONCURRENCY=1
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
    --endpoint "$ENDPOINT"
    --api-key "$API_KEY"
    --api-version "$API_VERSION"
    --month $MONTHS
    --max-tokens "$MAX_TOKENS"
    --concurrency "$CONCURRENCY"
    --seed "$SEED"
    --resume
)

if [[ -n "$REASONING_EFFORT" ]]; then
    ARGS+=(--reasoning-effort "$REASONING_EFFORT")
fi

python eval/eval.py "${ARGS[@]}"
