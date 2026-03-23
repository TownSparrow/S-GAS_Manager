set -euo pipefail

echo "Launching the vLLM server for S-GAS Manager..."

# Checking virtual environment
if [ ! -d "S-GAS_Manager_env" ]; then
    echo "❌ Virtual environment is not created. Create and set params with install_env.sh"
    exit 1
fi

# Activating environment
source S-GAS_Manager_env/bin/activate

# Setting envirionment variables
export HF_HOME="$(pwd)/models/.cache"
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

# Checking if model exists
if [ -d "models/$MODEL" ]; then
    echo "✅ Model already downloaded: models/$MODEL"
else
    echo "Downloading model: $MODEL"
    echo "This may take several minutes..."
fi

echo ""
echo "Starting vLLM server..."
echo "   API endpoint: http://0.0.0.0:8000"
echo "   Press Ctrl+C to stop"
echo ""

# Starting vLLM server
vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --quantization "$QUANT" \
  --dtype "$DTYPE" \
  --trust-remote-code \
  --download-dir models/