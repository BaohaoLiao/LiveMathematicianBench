#!/usr/bin/env bash
set -euo pipefail

#######################################
# Configuration — override via env vars
#######################################
MODEL="${MODEL:-/data/agenthle/baohao/LLMs/deepseek-ai/DeepSeek-V3.2}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v3.2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# The recipe prefers EP/DP on Blackwell, but this host was stable with TP8.
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-0}"

TOKENIZER_MODE="${TOKENIZER_MODE:-deepseek_v32}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-deepseek_v32}"
REASONING_PARSER="${REASONING_PARSER:-deepseek_v3}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

# DeepGEMM crashed on this CUDA 12.8 setup, so disable it by default.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"

# Optional knobs.
# vLLM serve uses --max-model-len; MAX_TOKENS is kept as a convenient alias.
MAX_TOKENS="${MAX_TOKENS:-65000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$MAX_TOKENS}"
GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}"
DEFAULT_EXTRA_ARGS="--enforce-eager"
EXTRA_ARGS="${EXTRA_ARGS:-$DEFAULT_EXTRA_ARGS}"

#######################################
# Launch
#######################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source .venv/bin/activate
fi

if [[ ! -x .venv/bin/vllm ]]; then
    echo "error: .venv/bin/vllm not found. Install vllm into the virtual environment first." >&2
    exit 1
fi

ARGS=(
    serve
    "$MODEL"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --data-parallel-size "$DATA_PARALLEL_SIZE"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --tokenizer-mode "$TOKENIZER_MODE"
    --tool-call-parser "$TOOL_CALL_PARSER"
    --reasoning-parser "$REASONING_PARSER"
)

if [[ "$ENABLE_EXPERT_PARALLEL" == "1" ]]; then
    ARGS+=(--enable-expert-parallel)
fi

if [[ "$ENABLE_AUTO_TOOL_CHOICE" == "1" ]]; then
    ARGS+=(--enable-auto-tool-choice)
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    ARGS+=(--trust-remote-code)
fi

if [[ -n "$MAX_MODEL_LEN" ]]; then
    ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi

if [[ -n "$GENERATION_CONFIG" ]]; then
    ARGS+=(--generation-config "$GENERATION_CONFIG")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
    # Intentionally split EXTRA_ARGS on shell word boundaries to allow flag passthrough.
    read -r -a EXTRA_ARR <<< "$EXTRA_ARGS"
    ARGS+=("${EXTRA_ARR[@]}")
fi

echo "Launching $MODEL on $HOST:$PORT"
echo "served_model_name=$SERVED_MODEL_NAME dp=$DATA_PARALLEL_SIZE tp=$TENSOR_PARALLEL_SIZE ep=$ENABLE_EXPERT_PARALLEL deep_gemm=$VLLM_USE_DEEP_GEMM generation_config=$GENERATION_CONFIG"

exec .venv/bin/vllm "${ARGS[@]}"