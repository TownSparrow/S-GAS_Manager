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

choose_custom_config() {
    local max_choice choice file label model
    mapfile -t CONFIG_FILES < <(find cfg -maxdepth 1 -type f -name '*.json' | sort)
    if [ "${#CONFIG_FILES[@]}" -eq 0 ]; then
        echo "No JSON configs found in cfg/."
        exit 1
    fi

    echo "Available cfg/*.json files:"
    for i in "${!CONFIG_FILES[@]}"; do
        file="${CONFIG_FILES[$i]}"
        label="$(jq -r '.profile.name // .vllm.model_name // empty' "$file" 2>/dev/null || true)"
        model="$(jq -r '.vllm.model_name // empty' "$file" 2>/dev/null || true)"
        if [ -n "$label" ] && [ "$label" != "$model" ]; then
            echo "  $((i + 1)) - $file  ($label; $model)"
        elif [ -n "$model" ]; then
            echo "  $((i + 1)) - $file  ($model)"
        else
            echo "  $((i + 1)) - $file"
        fi
    done
    max_choice="${#CONFIG_FILES[@]}"
    choice=$(prompt_choice "Select config [1-$max_choice]: " "$max_choice")
    CONFIG="${CONFIG_FILES[$((choice - 1))]}"
}

echo "Which configuration would you like to use?"
echo "  1 - Qwen/Qwen2.5-7B-Instruct-AWQ presets"
echo "  2 - google/gemma-4-E2B-it presets"
echo "  3 - custom cfg/*.json file"
CONFIG_SOURCE=$(prompt_choice "Select source [1-3]: " 3)
echo ""

if [ "$CONFIG_SOURCE" = "3" ]; then
    choose_custom_config
    CONFIG_MODEL="custom"
    MODE="custom"
    MODEL_LABEL="$(jq -r '.profile.name // .vllm.model_name // "custom"' "$CONFIG" 2>/dev/null || echo custom)"
else
    case "$CONFIG_SOURCE" in
        1)
            CONFIG_MODEL="Qwen__Qwen2.5-7B-Instruct-AWQ"
            MODEL_LABEL="Qwen/Qwen2.5-7B-Instruct-AWQ"
            ;;
        2)
            CONFIG_MODEL="google__gemma-4-E2B-it"
            MODEL_LABEL="google/gemma-4-E2B-it"
            ;;
    esac

    echo "Which mode would you like to use?"
    echo "  1 - stress  (4k context window, lowest usage preset)"
    echo "  2 - optimum (8k context window, balanced preset)"
    echo "  3 - max     (16k context window, high usage preset)"
    MODE_CHOICE=$(prompt_choice "Select mode [1-3]: " 3)
    echo ""

    case "$MODE_CHOICE" in
        1) MODE="stress" ;;
        2) MODE="optimum" ;;
        3) MODE="max" ;;
    esac

    CONFIG="cfg/${CONFIG_MODEL}__${MODE}.json"
fi

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

is_positive_value() {
    local value="$1"
    [ -n "$value" ] && [ "$value" != "null" ] && [ "$value" != "0" ] && [ "$value" != "0.0" ]
}

latest_snapshot_path() {
    local model_cache="$1"
    if [ ! -d "$model_cache/snapshots" ]; then
        return 1
    fi
    find "$model_cache/snapshots" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR == 1 {print $2}'
}

has_usable_hf_snapshot() {
    local snapshot_path="$1"
    if [ -z "$snapshot_path" ] || [ ! -d "$snapshot_path" ]; then
        return 1
    fi
    if ! { [ -e "$snapshot_path/config.json" ] || [ -e "$snapshot_path/params.json" ]; }; then
        return 1
    fi
    find -L "$snapshot_path" -type f \( \
        -name '*.safetensors' -o \
        -name '*.bin' -o \
        -name '*.pt' -o \
        -name '*.pth' -o \
        -name '*.gguf' \
    \) -print -quit | grep -q .
}

print_invalid_model_cache_message() {
    local model="$1"
    local model_cache="$2"
    local snapshot_path="$3"
    echo "Model cache is present but incomplete for vLLM:"
    echo "   Model: $model"
    echo "   Cache: $model_cache"
    if [ -n "$snapshot_path" ]; then
        echo "   Snapshot: $snapshot_path"
    fi
    echo "   Required: config.json or params.json plus at least one weight file (*.safetensors, *.bin, *.pt, *.pth, *.gguf)."
    echo "   This usually means the selected Hugging Face repo is metadata-only or the download did not fetch model weights."
}

remove_torch_allocator_expandable_segments() {
    local current="${PYTORCH_CUDA_ALLOC_CONF:-}"
    local cleaned=""
    local part

    if [ -z "$current" ]; then
        return 0
    fi

    IFS=',' read -ra parts <<< "$current"
    for part in "${parts[@]}"; do
        if [[ "$part" =~ ^[[:space:]]*expandable_segments:[Tt]rue[[:space:]]*$ ]]; then
            continue
        fi
        if [ -n "$cleaned" ]; then
            cleaned+=","
        fi
        cleaned+="$part"
    done

    if [ -n "$cleaned" ]; then
        export PYTORCH_CUDA_ALLOC_CONF="$cleaned"
    else
        unset PYTORCH_CUDA_ALLOC_CONF
    fi
}

# Reading config values
MODEL=$(jq_value '.vllm.model_name')
SERVED_MODEL_NAME=$(jq_value '.vllm.served_model_name')
TOKENIZER=$(jq_value '.vllm.tokenizer')
HF_CONFIG_PATH=$(jq_value '.vllm.hf_config_path')
LOAD_FORMAT=$(jq_value '.vllm.load_format')
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
ENFORCE_EAGER=$(jq_value '.vllm.enforce_eager')
LIMIT_MM_PER_PROMPT=$(jq_json '.vllm.limit_mm_per_prompt')
ALLOW_DOWNLOAD=$(jq_value '.vllm.allow_download')
API_HOST=$(jq_value '.api.host')
API_PORT=$(jq_value '.api.port')
PROFILE_DESCRIPTION=$(jq_value '.profile.description')

API_HOST=${API_HOST:-"0.0.0.0"}
API_PORT=${API_PORT:-8080}
DTYPE=${DTYPE:-"auto"}

if is_positive_value "$KV_OFFLOADING_SIZE"; then
    remove_torch_allocator_expandable_segments
    echo "KV offloading note: disabled PyTorch expandable_segments allocator because vLLM KV offload is incompatible with it."
else
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi

echo "Configuration:"
echo "   Preset: $CONFIG"
echo "   Model choice: $MODEL_LABEL"
echo "   Mode: $MODE"
echo "   Model: $MODEL"
echo "   Served Model Name: ${SERVED_MODEL_NAME:-$MODEL}"
echo "   Tokenizer: ${TOKENIZER:-default}"
echo "   HF Config Path: ${HF_CONFIG_PATH:-default}"
echo "   Load Format: ${LOAD_FORMAT:-auto}"
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
echo "   Enforce Eager: ${ENFORCE_EAGER:-false}"
if [ -n "$PROFILE_DESCRIPTION" ]; then
    echo "   Note: $PROFILE_DESCRIPTION"
fi
echo ""

if [[ "$MODEL" == *gemma-4-E2B-it* ]]; then
    echo "Gemma 4 E2B note: this preset is text-only and disables image/audio/video inputs where supported."
    echo "Gemma quantization note: this preset uses the original unquantized checkpoint."
    echo ""
fi

if [ "$MODE" = "stress" ]; then
    echo "Stress mode note: this preset uses the minimum configured 4k context window."
    echo ""
fi

# Creation of model directory
mkdir -p models

# Checking if model exists in local cache
MODEL_CACHE="models/models--$(echo "$MODEL" | sed 's|/|--|g')"
if [ -e "$MODEL" ]; then
    echo "Using local model path: $MODEL"
elif [ -d "$MODEL_CACHE" ]; then
    SNAPSHOT_PATH="$(latest_snapshot_path "$MODEL_CACHE" || true)"
    if has_usable_hf_snapshot "$SNAPSHOT_PATH"; then
        echo "Model found in local cache: $MODEL_CACHE"
    elif [ "$ALLOW_DOWNLOAD" = "true" ]; then
        print_invalid_model_cache_message "$MODEL" "$MODEL_CACHE" "$SNAPSHOT_PATH"
        echo "Config allows online download; vLLM may try to refresh the model from Hugging Face."
        unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    else
        print_invalid_model_cache_message "$MODEL" "$MODEL_CACHE" "$SNAPSHOT_PATH"
        echo ""
        echo "Run scripts/install_env.sh after fixing the model repo, or choose a known-good preset such as Qwen."
        exit 1
    fi
else
    echo "Model not found locally: $MODEL_CACHE"
    if [ "$ALLOW_DOWNLOAD" = "true" ]; then
        echo "Config allows online download; vLLM may download the model from Hugging Face."
        unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    else
        read -r -p "Allow vLLM to download this model now? [y/N]: " DOWNLOAD_CHOICE
        if [[ "$DOWNLOAD_CHOICE" =~ ^[Yy]$ ]]; then
            unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
        else
            echo "Download this exact model into ./models first, or set .vllm.allow_download=true in the custom config:"
            echo "   $MODEL"
            exit 1
        fi
    fi
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
VLLM_SOURCE_FLAGS="$(python - <<'PY'
from pathlib import Path
import importlib.util
import re

spec = importlib.util.find_spec("vllm")
if not spec or not spec.origin:
    raise SystemExit(0)

root = Path(spec.origin).parent
candidate_files = [
    root / "engine" / "arg_utils.py",
    root / "entrypoints" / "openai" / "cli_args.py",
    root / "entrypoints" / "cli" / "serve.py",
]

flags = set()
for path in candidate_files:
    if not path.exists():
        continue
    flags.update(
        re.findall(
            r'["\'](--[A-Za-z0-9][A-Za-z0-9-]*)["\']',
            path.read_text(encoding="utf-8", errors="ignore"),
        )
    )

for flag in sorted(flags):
    print(flag)
PY
)"

if ! grep -q -- "usage:" <<< "$VLLM_HELP"; then
    echo "Warning: unable to read 'vllm serve --help'; using installed package metadata for flag checks."
fi

flag_supported() {
    local flag="$1"
    grep -q -- "$flag" <<< "$VLLM_HELP" || grep -qx -- "$flag" <<< "$VLLM_SOURCE_FLAGS"
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
append_optional_value_arg "--served-model-name" "$SERVED_MODEL_NAME"
append_optional_value_arg "--tokenizer" "$TOKENIZER"
append_optional_value_arg "--hf-config-path" "$HF_CONFIG_PATH"
append_optional_value_arg "--load-format" "$LOAD_FORMAT"
append_optional_value_arg "--swap-space" "$SWAP_SPACE"
append_optional_positive_value_arg "--cpu-offload-gb" "$CPU_OFFLOAD_GB"
append_value_arg "--kv-cache-dtype" "$KV_CACHE_DTYPE"
append_optional_positive_value_arg "--kv-offloading-size" "$KV_OFFLOADING_SIZE"
append_value_arg "--dtype" "$DTYPE"

if [ -n "$QUANT" ] && [ "$QUANT" != "null" ] && [ "$QUANT" != "none" ]; then
    VLLM_CMD+=("--quantization" "$QUANT")
fi

if [ "$ENFORCE_EAGER" = "true" ]; then
    if flag_supported "--enforce-eager"; then
        VLLM_CMD+=("--enforce-eager")
    else
        echo "Warning: installed vLLM does not support --enforce-eager; skipping it."
    fi
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
