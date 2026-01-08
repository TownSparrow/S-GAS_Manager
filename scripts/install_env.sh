#!/usr/bin/env bash
set -euo pipefail

echo "Installing of S-GAS Manager.."
echo ""

# Checking the Python version
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "Detected version of Python: $PYTHON_VERSION"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Python 3.10 version and newer is required"
    exit 1
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

# 4) Install the PyTorch with CUDA Support
echo "4) Installation of PyTorch and CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5) Install vLLM
echo "5) Installation of vLLM..."
pip install vllm

# 6. Install of the dependences
echo "6) Installation of dependence form requirements.txt..."
if [ -f "other/requirements.txt" ]; then
    pip install -r other/requirements.txt
elif [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "❌ File requirements.txt is not detected"
    exit 1
fi

# 7. spaCy models
echo "7) Downloading the spaCy models..."
python -m spacy download en_core_web_sm
python -m spacy download ru_core_news_md

# 8. Creating the directories
echo "8) Creating the directories..."
mkdir -p data/uploads
mkdir -p data/chroma_db
mkdir -p models/.cache
mkdir -p logs

# 9. Checking the installs
echo ""
echo "9) Checking the installations..."
python3 -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'✅ CUDA is available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'✅ CUDA version: {torch.version.cuda}')" || true
python3 -c "import vllm; print(f'✅ vLLM: {vllm.__version__}')"
python3 -c "import transformers; print(f'✅ Transformers: {transformers.__version__}')"
python3 -c "import chromadb; print(f'✅ ChromaDB: {chromadb.__version__}')"

echo ""
echo "✅ Installation is completed!"
echo ""
echo "Next steps:"
echo "   1. Set the params in configs/system_params.json"
echo "   2. Activate environment: source S-GAS_Manager_env/bin/activate"
echo "   3. Launch vLLM server: ./scripts/start_vllm_server.sh"
echo "   4. Launch API: python src/web/api.py"
echo ""