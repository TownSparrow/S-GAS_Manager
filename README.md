# S-GAS Manager

Enables small language models (SLMs) with retrieval-augmented generation (RAG) to control massive contexts by adaptive swapping chunks between GPU and CPU memory. Applies semantic-graph relevance assessment before each query to dynamically manage information, allowing to go beyond VRAM on consumer hardware.

## Overview

S-GAS Manager is a research project that combines:
- **Adaptive Swapping**: Dynamic memory management between GPU and CPU
- **Semantic + Graph Relevance Assessment**: Hybrid chunk ranking using cosine similarity and graph distances
- **RAG for Small Language Models**: Extending context windows for SLMs on consumer hardware
- **Web-based Testing Interface**: Interactive chat client for comfortable testing

## Project Architecture

```
S-GAS_MANAGER/
├── src/
│   ├── core/                    # Core algorithm modules
│   ├── modules/
│   │   ├── retrieval/           # Embedding and retrieval logic
│   │   ├── graph/               # Knowledge graph construction
│   │   └── swap/                # Memory swap management
│   ├── web/
│   │   ├── api.py               # FastAPI REST endpoints
│   │   └── static/              # Web client files
│   │       ├── index.html       # Main interface
│   │       ├── styles.css       # UI styling
│   │       └── script.js        # Client-side logic
│   ├── config.py                # Configuration management
│   └── main.py                  # Main testing script
├── configs/
│   ├── models.json              # Model configurations
│   └── system_params.json       # System parameters
├── scripts/
│   ├── install_env.sh           # Environment setup
│   └── start_vllm_server.sh     # vLLM server launcher
├── other/
│   └── requirements.txt         # Python dependencies
└── data/                        # Data storage (created automatically)
```

## How to Set Up and Launch

### 1. Prepare the Environment and Install Dependencies

**a) Prerequisites:**
- Python 3.10+ installed (Python 3.12 recommended)
- NVIDIA GPU with CUDA support (for optimal performance)
- At least 16GB system RAM and 12GB VRAM

**b) Use the setup script to create and configure the Python virtual environment:**
```bash
chmod +x ./scripts/install_env.sh
./scripts/install_env.sh
```

This script will:
- Create a virtual environment (`S-GAS_Manager_env`)
- Activate it automatically
- Upgrade pip and essential tools
- Install all Python dependencies from `other/requirements.txt`
- Download the Russian spaCy model for entity extraction

**c) Manually activate the environment when needed:**
```bash
source S-GAS_Manager_env/bin/activate
```

### 2. Verify and Adjust Configuration

Check the main configuration file:
```
configs/system_params.json
```

Key settings to verify:
- `vllm.model_name`: Path or HuggingFace model name
- `vllm.gpu_memory_utilization`: GPU memory usage (0.8 = 80%)
- `vllm.max_model_len`: Maximum context length
- `embeddings.model`: Embedding model for semantic analysis

### 3. Prepare Static Files for the Web Client

Ensure the following files are in `src/web/static/`:
- `index.html` (main web interface)
- `styles.css` (styling and animations)
- `script.js` (client-side chat logic)

Expected directory structure:
```
src/web/static/
 ├── index.html
 ├── styles.css
 └── script.js
```

### 4. Start the Language Model Server (vLLM)

**Option A: Use the provided script:**
```bash
chmod +x ./scripts/start_vllm_server.sh
./scripts/start_vllm_server.sh
```

**Option B: Manual launch:**
```bash
source S-GAS_Manager_env/bin/activate

# Extract config values
CONFIG_FILE="configs/system_params.json"
VLLM_MODEL=$(python -c "import json;print(json.load(open('$CONFIG_FILE'))['vllm']['model_name'])")
VLLM_GPU_MEM=$(python -c "import json;print(json.load(open('$CONFIG_FILE'))['vllm']['gpu_memory_utilization'])")
VLLM_MAX_LEN=$(python -c "import json;print(json.load(open('$CONFIG_FILE'))['vllm']['max_model_len'])")

# Start vLLM server
vllm serve $VLLM_MODEL \
    --gpu-memory-utilization $VLLM_GPU_MEM \
    --max-model-len $VLLM_MAX_LEN \
    --max-num-seqs 32 \
    --host 0.0.0.0 \
    --port 8000
```

> **Important:** Keep this server running in a separate terminal. The API server depends on it.

### 5. Start the Main API Server (FastAPI/Uvicorn)

In another terminal, from the project root:
```bash
source S-GAS_Manager_env/bin/activate
uvicorn src.web.api:app --reload --host 0.0.0.0 --port 8080
```

This launches:
- REST API endpoints at `/api/chat`, `/health`
- Static file server for the web client
- Automatic reload on code changes (development mode)

### 6. Access the Web Client

Open your browser and navigate to: **[http://localhost:8080](http://localhost:8080)**

**Web Client Features:**
- 🎨 Modern chat interface with animations
- 📊 Real-time status monitoring (server health, model info)
- 🔧 RAG toggle (enable/disable retrieval-augmented generation)
- 📈 Session statistics (message count, embedding dimensions)
- 💾 Export chat history to JSON
- 🧹 Clear chat functionality
- ⌨️ Keyboard shortcuts (Ctrl+Enter to send)

### 7. Testing & Diagnostics

**Successful setup indicators:**
- Green status dot: "Server online • vLLM: healthy"
- Model name displayed in the info panel
- Embedding dimensions shown after first query

**Troubleshooting:**
- **Red status dot**: Check vLLM server is running on port 8000
- **No response**: Check terminal logs for both servers
- **JavaScript errors**: Open browser console (F12) for client-side issues
- **API errors**: Check FastAPI logs in the terminal

**Health check endpoints:**
- API status: `GET http://localhost:8080/health`
- vLLM models: `GET http://localhost:8000/v1/models`

### 8. Development and Testing

**Run basic component tests:**
```bash
source S-GAS_Manager_env/bin/activate
python src/main.py
```

This will test:
- Configuration loading
- Embedding generation
- Basic system connectivity

**API testing with curl:**
```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?", "use_rag": true}'
```

### 9. Stopping the Servers

1. Press `Ctrl+C` in each terminal running servers
2. Deactivate the virtual environment:
```bash
deactivate
```

## Research Goals

This project implements and tests the S-GAS algorithm for:
- Reducing swap operations by 30-50% compared to existing algorithms
- Improving retrieval relevance using semantic + graph ranking
- Maintaining latency within 1.1-1.3x of baseline performance
- Enabling context windows up to 200K tokens on consumer hardware

## Contributing

This is a research prototype. For issues or suggestions:
1. Check logs for error details
2. Verify all configuration files
3. Ensure both servers are running correctly
4. Test with different model configurations

---

**Ready to test your S-GAS Manager! 🚀**

The web interface makes it easy to experiment with different queries, monitor system performance, and evaluate the adaptive swapping algorithm in real-time.