#!/bin/bash

# Activate the environment
source S-GAS_Manager_env/bin/activate

CONFIG_FILE="configs/system_params.json"

VLLM_MODEL=$(python -c "import json;print(json.load(open('$CONFIG_FILE'))['vllm']['model_name'])")
VLLM_GPU_MEM=$(python -c "import json;print(json.load(open('$CONFIG_FILE'))['vllm']['gpu_memory_utilization'])")
VLLM_MAX_LEN=$(python -c "import json;print(json.load(open('$CONFIG_FILE'))['vllm']['max_model_len'])")

echo "Starting vLLM server with model: $VLLM_MODEL"
echo "GPU memory utilization: $VLLM_GPU_MEM"
echo "Max model length: $VLLM_MAX_LEN"

# Running server with reduced warm-up batch
vllm serve $VLLM_MODEL \
    --gpu-memory-utilization $VLLM_GPU_MEM \
    --max-model-len $VLLM_MAX_LEN \
    --max-num-seqs 32 \
    --host 0.0.0.0 \
    --port 8000