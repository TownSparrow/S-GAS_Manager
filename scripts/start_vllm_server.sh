set -euo pipefail

echo "🚀 Launching the vLLM server for S-GAS Manager..."

if [ ! -d "S-GAS_Manager_env" ]; then
    echo "❌ Virtual environment is not created. Create and set params with install_env.sh"
    exit 1
fi

source S-GAS_Manager_env/bin/activate

export HF_HOME="$(pwd)/models/.cache"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="configs/system_params.json"

if [ ! -f "$CONFIG" ]; then
    echo "❌ Config file is not dectected: $CONFIG"
    exit 1
fi

MODEL=$(jq -r '.vllm.model_name' "$CONFIG")
GPU_MEM=$(jq -r '.vllm.gpu_memory_utilization' "$CONFIG")
MAX_LEN=$(jq -r '.vllm.max_model_len' "$CONFIG")
QUANT=$(jq -r '.vllm.quantization // "awq"' "$CONFIG")
DTYPE=$(jq -r '.vllm.dtype // "auto"' "$CONFIG")

echo "📋 Config settings:"
echo "   Model: $MODEL"
echo "   GPU Memory: ${GPU_MEM}"
echo "   Max Length: $MAX_LEN"
echo "   Quantization: $QUANT"
echo ""

mkdir -p models

echo "⏳ Model loading and launching the server..."
echo "   This may take a few minutes the first time you run it..."
echo ""

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs 4 \
  --quantization "$QUANT" \
  --dtype "$DTYPE" \
  --trust-remote-code \
  --download-dir models/ #\
  #--disable-log-requests