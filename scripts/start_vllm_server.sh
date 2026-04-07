set -euo pipefail

echo "Launching the vLLM server for S-GAS Manager..."

# Checking virtual environment
if [ ! -d "S-GAS_Manager_env" ]; then
    echo "❌ Virtual environment is not created. Create and set params with install_env.sh"
    exit 1
fi

# Activating environment
source S-GAS_Manager_env/bin/activate

# Setting environment variables — use local model cache without network access
export HF_HUB_CACHE="$(pwd)/models"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Checking config file
CONFIG="cfg/system_params.json"
if [ ! -f "$CONFIG" ]; then
    echo "❌ Config file is not dectected: $CONFIG"
    exit 1
fi

# Reading config values
MODEL=$(jq -r '.vllm.model_name' "$CONFIG")
GPU_MEM=$(jq -r '.vllm.gpu_memory_utilization' "$CONFIG")
MAX_LEN=$(jq -r '.vllm.max_model_len' "$CONFIG")
QUANT=$(jq -r '.vllm.quantization // "awq"' "$CONFIG")
DTYPE=$(jq -r '.vllm.dtype // "auto"' "$CONFIG")
MAX_SEQS=$(jq -r '.vllm.max_num_seqs // 4' "$CONFIG")

echo "Configuration:"
echo "   Model: $MODEL"
echo "   GPU Memory Utilization: ${GPU_MEM}"
echo "   Max Model Length: $MAX_LEN"
echo "   Quantization: $QUANT"
echo "   Data Type: $DTYPE"
echo "   Max Sequences: $MAX_SEQS"
echo ""

# Creation of model directory
mkdir -p models

# Checking if model exists in local cache
MODEL_CACHE="models/models--$(echo $MODEL | sed 's|/|--|g')"
if [ -d "$MODEL_CACHE" ]; then
    echo "Model found in local cache: $MODEL_CACHE"
else
    echo "Model not found locally: $MODEL_CACHE"
    echo "Run install_env.sh first to download the model."
    exit 1
fi

# Kill any existing S-GAS API on port 8080 so we start fresh
fuser -k 8080/tcp 2>/dev/null || true

# Read API config
API_HOST=$(jq -r '.api.host // "0.0.0.0"' "$CONFIG")
API_PORT=$(jq -r '.api.port // 8080' "$CONFIG")

# Launch S-GAS API in a separate terminal so logs are visible
echo ""
echo "Launching S-GAS API in a separate terminal (port $API_PORT)..."
PROJECT_DIR="$(pwd)"
gnome-terminal --title="S-GAS API" -- bash -c "
  cd '$PROJECT_DIR'
  source S-GAS_Manager_env/bin/activate
  unset HF_HUB_CACHE
  unset HF_HUB_OFFLINE
  unset TRANSFORMERS_OFFLINE
  echo '=== S-GAS API Server ==='
  echo 'Waiting for vLLM to start...'
  while ! curl -s http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 2; done
  echo 'vLLM is ready. Starting S-GAS API...'
  echo ''
  uvicorn run:app --host $API_HOST --port $API_PORT
  exec bash
" 2>/dev/null || xterm -title 'S-GAS API' -e bash -c "
  cd '$PROJECT_DIR'
  source S-GAS_Manager_env/bin/activate
  unset HF_HUB_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  while ! curl -s http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 2; done
  uvicorn run:app --host $API_HOST --port $API_PORT
  exec bash
" 2>/dev/null || {
  echo "No GUI terminal found. Starting S-GAS API in background..."
  (
    cd "$PROJECT_DIR"
    source S-GAS_Manager_env/bin/activate
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
echo "   Press Ctrl+C to stop vLLM (S-GAS API will also stop)"
echo ""

# Starting vLLM server (foreground — logs visible in this terminal)
vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --quantization "$QUANT" \
  --dtype "$DTYPE" \
  --trust-remote-code