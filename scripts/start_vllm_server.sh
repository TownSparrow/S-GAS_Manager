#!/usr/bin/env bash
set -euo pipefail

echo "Launching the vLLM server for S-GAS Manager..."
echo ""

# Checking virtual environment
if [ ! -d "S-GAS_Manager_env" ]; then
    echo "Virtual environment is not created. Create and set params with install_env.sh"
    exit 1
fi

# Activating environment
source S-GAS_Manager_env/bin/activate

# Setting environment variables - use local model cache without network access
export HF_HUB_CACHE="$(pwd)/models"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

prompt_choice() {
    local prompt="$1"
    local max_choice="$2"
    local choice

    while true; do
        read -r -p "$prompt" choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$max_choice" ]; then
            echo "$choice"
            return 0
        fi
        echo "Please enter a number from 1 to $max_choice."
    done
}

echo "Which model would you like to use?"
echo "  1 - Qwen/Qwen2.5-7B-Instruct-AWQ"
echo "  2 - google/gemma-4-E2B-it"
MODEL_CHOICE=$(prompt_choice "Select model [1-2]: " 2)
echo ""

echo "Which mode would you like to use?"
echo "  1 - stress  (very low usage attempt, sub-4GB-style settings)"
echo "  2 - optimum (medium GPU attempt, roughly 6-12GB VRAM)"
echo "  3 - max     (RTX 4070 Ti Super 16GB profile)"
MODE_CHOICE=$(prompt_choice "Select mode [1-3]: " 3)
echo ""

case "$MODEL_CHOICE" in
    1)
        CONFIG_MODEL="Qwen__Qwen2.5-7B-Instruct-AWQ"
        MODEL_LABEL="Qwen/Qwen2.5-7B-Instruct-AWQ"
        ;;
    2)
        CONFIG_MODEL="google__gemma-4-E2B-it"
        MODEL_LABEL="google/gemma-4-E2B-it"
        ;;
esac

case "$MODE_CHOICE" in
    1) MODE="stress" ;;
    2) MODE="optimum" ;;
    3) MODE="max" ;;
esac

CONFIG="cfg/${CONFIG_MODEL}__${MODE}.json"
if [ ! -f "$CONFIG" ]; then
    echo "Config file is not detected: $CONFIG"
    exit 1
fi

CONFIG_ABS="$(pwd)/$CONFIG"
export S_GAS_CONFIG_PATH="$CONFIG_ABS"

jq_value() {
    jq -r "$1 // empty" "$CONFIG"
}

jq_json() {
    jq -c "$1 // empty" "$CONFIG"
}

is_enabled() {
    [ "$(jq_value "$1")" = "true" ]
}

# Reading config values
MODEL=$(jq_value '.vllm.model_name')
GPU_MEM=$(jq_value '.vllm.gpu_memory_utilization')
MAX_LEN=$(jq_value '.vllm.max_model_len')
MAX_SEQS=$(jq_value '.vllm.max_num_seqs')
MAX_BATCHED_TOKENS=$(jq_value '.vllm.max_num_batched_tokens')
SWAP_SPACE=$(jq_value '.vllm.swap_space')
CPU_OFFLOAD_GB=$(jq_value '.vllm.cpu_offload_gb')
KV_CACHE_DTYPE=$(jq_value '.vllm.kv_cache_dtype')
KV_OFFLOADING_SIZE=$(jq_value '.vllm.kv_offloading_size')
QUANT=$(jq_value '.vllm.quantization')
DTYPE=$(jq_value '.vllm.dtype')
LIMIT_MM_PER_PROMPT=$(jq_json '.vllm.limit_mm_per_prompt')
API_HOST=$(jq_value '.api.host')
API_PORT=$(jq_value '.api.port')
PROFILE_DESCRIPTION=$(jq_value '.profile.description')

API_HOST=${API_HOST:-"0.0.0.0"}
API_PORT=${API_PORT:-8080}
DTYPE=${DTYPE:-"auto"}

echo "Configuration:"
echo "   Preset: $CONFIG"
echo "   Model choice: $MODEL_LABEL"
echo "   Mode: $MODE"
echo "   Model: $MODEL"
echo "   GPU Memory Utilization: ${GPU_MEM:-default}"
echo "   Max Model Length: ${MAX_LEN:-default}"
echo "   Max Sequences: ${MAX_SEQS:-default}"
echo "   Max Batched Tokens: ${MAX_BATCHED_TOKENS:-default}"
echo "   Swap Space: ${SWAP_SPACE:-default}"
echo "   CPU Offload GB: ${CPU_OFFLOAD_GB:-0}"
echo "   KV Cache DType: ${KV_CACHE_DTYPE:-default}"
echo "   KV Offloading Size: ${KV_OFFLOADING_SIZE:-0}"
echo "   Quantization: ${QUANT:-none}"
echo "   Data Type: $DTYPE"
if [ -n "$PROFILE_DESCRIPTION" ]; then
    echo "   Note: $PROFILE_DESCRIPTION"
fi
echo ""

if [[ "$MODEL" == google/gemma-4-E2B-it* ]]; then
    echo "Gemma 4 E2B note: this preset is text-only and disables image/audio/video inputs where supported."
    echo "Gemma low/medium VRAM profiles are experimental and may require a recent vLLM with CPU/KV offload support."
    echo ""
fi

if [ "$MODE" = "stress" ]; then
    echo "Stress mode note: this is a very low usage attempt, not a guarantee that the exact model fits under 4GB VRAM."
    echo ""
fi

# Creation of model directory
mkdir -p models

# Checking if model exists in local cache
MODEL_CACHE="models/models--$(echo "$MODEL" | sed 's|/|--|g')"
if [ -d "$MODEL_CACHE" ]; then
    echo "Model found in local cache: $MODEL_CACHE"
else
    echo "Model not found locally: $MODEL_CACHE"
    echo "Download this exact model into ./models first, or run an install/download step with:"
    echo "   $MODEL"
    exit 1
fi

# Kill any existing S-GAS API on port 8080 so we start fresh
fuser -k 8080/tcp 2>/dev/null || true

# Launch S-GAS API in a separate terminal so logs are visible
echo ""
echo "Launching S-GAS API in a separate terminal (port $API_PORT)..."
PROJECT_DIR="$(pwd)"
gnome-terminal --title="S-GAS API" -- bash -c "
  cd '$PROJECT_DIR'
  source S-GAS_Manager_env/bin/activate
  export S_GAS_CONFIG_PATH='$CONFIG_ABS'
  unset HF_HUB_CACHE
  unset HF_HUB_OFFLINE
  unset TRANSFORMERS_OFFLINE
  echo '=== S-GAS API Server ==='
  echo 'Using config: $CONFIG_ABS'
  echo 'Waiting for vLLM to start...'
  while ! curl -s http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 2; done
  echo 'vLLM is ready. Starting S-GAS API...'
  echo ''
  uvicorn run:app --host $API_HOST --port $API_PORT
  exec bash
" 2>/dev/null || xterm -title 'S-GAS API' -e bash -c "
  cd '$PROJECT_DIR'
  source S-GAS_Manager_env/bin/activate
  export S_GAS_CONFIG_PATH='$CONFIG_ABS'
  unset HF_HUB_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  while ! curl -s http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 2; done
  uvicorn run:app --host $API_HOST --port $API_PORT
  exec bash
" 2>/dev/null || {
  echo "No GUI terminal found. Starting S-GAS API in background..."
  (
    cd "$PROJECT_DIR"
    source S-GAS_Manager_env/bin/activate
    export S_GAS_CONFIG_PATH="$CONFIG_ABS"
    unset HF_HUB_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    while ! curl -s http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 2; done
    uvicorn run:app --host "$API_HOST" --port "$API_PORT"
  ) &
  echo "S-GAS API will start on port $API_PORT once vLLM is ready (PID: $!)"
}

echo ""
echo "Starting vLLM server..."
echo "   vLLM endpoint: http://0.0.0.0:8000"
echo "   S-GAS API:     http://$API_HOST:$API_PORT"
echo "   Benchmark UI:  http://$API_HOST:$API_PORT/benchmark"
echo "   Config path:   $CONFIG_ABS"
echo "   Press Ctrl+C to stop vLLM (S-GAS API will also stop)"
echo ""

VLLM_CMD=(vllm serve "$MODEL" --host 0.0.0.0 --port 8000)
VLLM_HELP="$(vllm serve --help 2>&1 || true)"

flag_supported() {
    local flag="$1"
    grep -q -- "$flag" <<< "$VLLM_HELP"
}

append_value_arg() {
    local flag="$1"
    local value="$2"
    if [ -n "$value" ] && [ "$value" != "null" ]; then
        VLLM_CMD+=("$flag" "$value")
    fi
}

append_positive_value_arg() {
    local flag="$1"
    local value="$2"
    if [ -n "$value" ] && [ "$value" != "null" ] && [ "$value" != "0" ] && [ "$value" != "0.0" ]; then
        VLLM_CMD+=("$flag" "$value")
    fi
}

append_optional_value_arg() {
    local flag="$1"
    local value="$2"
    if [ -n "$value" ] && [ "$value" != "null" ]; then
        if flag_supported "$flag"; then
            VLLM_CMD+=("$flag" "$value")
        else
            echo "Warning: installed vLLM does not support $flag; skipping it."
        fi
    fi
}

append_optional_positive_value_arg() {
    local flag="$1"
    local value="$2"
    if [ -n "$value" ] && [ "$value" != "null" ] && [ "$value" != "0" ] && [ "$value" != "0.0" ]; then
        if flag_supported "$flag"; then
            VLLM_CMD+=("$flag" "$value")
        else
            echo "Warning: installed vLLM does not support $flag; skipping it."
        fi
    fi
}

append_value_arg "--gpu-memory-utilization" "$GPU_MEM"
append_value_arg "--max-model-len" "$MAX_LEN"
append_value_arg "--max-num-seqs" "$MAX_SEQS"
append_value_arg "--max-num-batched-tokens" "$MAX_BATCHED_TOKENS"
append_value_arg "--swap-space" "$SWAP_SPACE"
append_positive_value_arg "--cpu-offload-gb" "$CPU_OFFLOAD_GB"
append_value_arg "--kv-cache-dtype" "$KV_CACHE_DTYPE"
append_optional_positive_value_arg "--kv-offloading-size" "$KV_OFFLOADING_SIZE"
append_value_arg "--dtype" "$DTYPE"

if [ -n "$QUANT" ] && [ "$QUANT" != "null" ] && [ "$QUANT" != "none" ]; then
    VLLM_CMD+=("--quantization" "$QUANT")
fi

if is_enabled '.vllm.trust_remote_code'; then
    VLLM_CMD+=("--trust-remote-code")
fi

if is_enabled '.vllm.language_model_only'; then
    if flag_supported "--language-model-only"; then
        VLLM_CMD+=("--language-model-only")
    else
        echo "Warning: installed vLLM does not support --language-model-only; using limit-mm-per-prompt only."
    fi
fi

if [ -n "$LIMIT_MM_PER_PROMPT" ] && [ "$LIMIT_MM_PER_PROMPT" != "null" ]; then
    append_optional_value_arg "--limit-mm-per-prompt" "$LIMIT_MM_PER_PROMPT"
fi

PREFIX_CACHING=$(jq_value '.vllm.enable_prefix_caching')
if [ "$PREFIX_CACHING" = "true" ]; then
    VLLM_CMD+=("--enable-prefix-caching")
elif [ "$PREFIX_CACHING" = "false" ]; then
    if flag_supported "--no-enable-prefix-caching"; then
        VLLM_CMD+=("--no-enable-prefix-caching")
    else
        echo "Warning: installed vLLM does not support --no-enable-prefix-caching; skipping it."
    fi
fi

printf 'vLLM command:'
printf ' %q' "${VLLM_CMD[@]}"
printf '\n\n'

# Starting vLLM server (foreground - logs visible in this terminal)
"${VLLM_CMD[@]}"
