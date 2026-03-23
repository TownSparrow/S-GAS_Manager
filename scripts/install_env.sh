#!/usr/bin/env bash
set -euo pipefail

echo "Installing of S-GAS Manager.."
echo ""

# Checking the Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $PYTHON_VERSION"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Python 3.10 or newer is required"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

# Checking CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️ NVIDIA drivers not detected. CUDA may not work."
else
    echo "✅ NVIDIA drivers detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# 1) Virtual environment
echo ""
echo "1) Creating the virtual environment..."
python3 -m venv S-GAS_Manager_env

# 2) Activation of the environment
echo "2) Activation of the virutal environment..."
source S-GAS_Manager_env/bin/activate

# 3) Upgrade of the pip and tools
echo "3) Upgrading of pip, setup tools and wheel..."
pip install --upgrade pip setuptools wheel

# 4) Installing the PyTorch with CUDA Support
echo "4) Installation of PyTorch and CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5) Installing vLLM
echo "5) Installation of vLLM..."
pip install vllm

# 6. Installing of the dependences
echo "6) Installation of dependence form requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "❌ File requirements.txt is not detected"
    exit 1
fi

# 7a. spaCy models
echo "7a) Downloading the spaCy models..."
python -m spacy download en_core_web_sm
python -m spacy download ru_core_news_lg

# 7b. Natasha model for NER
echo "7b) Installing Natasha for Russian NER..."
pip install natasha yargy

# 7c. YAKE
echo "7c) Installing YAKE for keyword extraction..."
pip install yake

# 8. Creating the directories
echo "8) Creating the directories..."
mkdir -p data/uploads
mkdir -p data/chroma_db
mkdir -p models/.cache
mkdir -p logs
mkdir -p tests/scenarios
mkdir -p tests/documents

# 9. Checking the installs
echo ""
echo "9) Checking the installations..."
python3 -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'✅ CUDA is available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'✅ CUDA version: {torch.version.cuda}')" || true
python3 -c "import vllm; print(f'✅ vLLM: {vllm.__version__}')"
python3 -c "import transformers; print(f'✅ Transformers: {transformers.__version__}')"
python3 -c "import chromadb; print(f'✅ ChromaDB: {chromadb.__version__}')"
python3 -c "import pymorphy3; print('✅ PyMorphy3: available')" 2>/dev/null || echo "⚠️ PyMorphy3: not found"
python3 -c "import natasha; print('✅ Natasha: available')" 2>/dev/null || echo "⚠️ Natasha: not found"
python3 -c "import yake; print('✅ YAKE: available')" 2>/dev/null || echo "⚠️ YAKE: not found"

# 10. Installing the benchmark dependencies
echo ""
echo "10) Installing benchmark dependencies..."
pip install rouge-score scipy seaborn jinja2 pytest pytest-asyncio pytest-cov evaluate

echo ""
echo "✅ Installation is completed!"
echo ""
echo "Next steps:"
echo "   1. Set the params in cfg/system_params.json"
echo "   2. Activate environment: source S-GAS_Manager_env/bin/activate"
echo "   3. Launch vLLM server: ./scripts/start_vllm_server.sh"
echo "   4. Launch API: uvicorn run:app --reload --host 0.0.0.0 --port 8080"
echo ""