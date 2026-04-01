#!/usr/bin/env bash
set -euo pipefail

#######################################
# Configuration — override via env vars
#######################################
MODEL_PATH="${MODEL_PATH:-/data/agenthle/baohao/LLMs/deepseek-ai/DeepSeek-V3.2}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v3.2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# DeepSeek-V3.2 SGLang docs recommend DP attention on B200/H200.
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-8}"
ENABLE_DP_ATTENTION="${ENABLE_DP_ATTENTION:-1}"

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
REASONING_PARSER="${REASONING_PARSER:-deepseek-v3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-deepseekv32}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

#######################################
# Launch
#######################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source .venv-sglang/bin/activate
fi

if [[ ! -x .venv-sglang/bin/python ]]; then
    echo "error: .venv-sglang/bin/python not found. Install SGLang into the isolated environment first." >&2
    exit 1
fi

ARGS=(
    -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --tp-size "$TP_SIZE"
    --dp-size "$DP_SIZE"
    --reasoning-parser "$REASONING_PARSER"
    --tool-call-parser "$TOOL_CALL_PARSER"
)

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    ARGS+=(--trust-remote-code)
fi

if [[ "$ENABLE_DP_ATTENTION" == "1" ]]; then
    ARGS+=(--enable-dp-attention)
fi

if [[ -n "$EXTRA_ARGS" ]]; then
    read -r -a EXTRA_ARR <<< "$EXTRA_ARGS"
    ARGS+=("${EXTRA_ARR[@]}")
fi

echo "Launching SGLang DeepSeek-V3.2 on $HOST:$PORT"
echo "served_model_name=$SERVED_MODEL_NAME tp=$TP_SIZE dp=$DP_SIZE dp_attention=$ENABLE_DP_ATTENTION reasoning_parser=$REASONING_PARSER tool_call_parser=$TOOL_CALL_PARSER"

exec .venv-sglang/bin/python "${ARGS[@]}"