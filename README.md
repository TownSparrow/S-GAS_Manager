# Semantic-Graph Adaptive Swapping Manager (S-GAS Manager)

## ⚠️ WARNING! ⚠️
Current version is `v0.1.0-alpha.1`. Project is in development process! I don't recommend to use it in production-tasks yet!

## About the project
This project is a prototype system that applies an adaptive data distribution algorithm with semantic-graph scoring of text fragments (chunks) when working with small language models (SLM). It allows language models with Retrieval-Augmented Generation (RAG) capabilities to handle large contexts by adaptively swapping chunks between GPU memory and system RAM. The system applies a semantic-graph evaluation of the value of each text fragment before every request in order to dynamically manage information on consumer-grade hardware.

The project and algorithm were developed by student Leonid Vorobyev as part of research work within the master’s program “Game Development Technologies” at the Game Development School of ITMO University.

### Scientific problem
Small language models are limited by a fixed context window, which creates critical issues in:
- multi-turn dialogues, where the model loses early context and generates contradictory answers;
- RAG systems, where naive accumulation of chunks leads to GPU memory overflow;
- working with long documents, where the model loses information in the middle of the context.

### Key components of the algorithm
The algorithm consists of three main mechanisms:
1. Semantic similarity between embeddings of the query and candidate fragments  
2. Graph-based scoring through analysis of entities and their relations in a knowledge graph  
3. Dynamic movement of fragments between VRAM and RAM based on the resulting scores  

### Core technologies
| Component         | Technology             |
|------------------|------------------------|
| Inference engine | vLLM (PagedAttention)  |
| Embeddings       | Sentence-Transformers  |
| Vector store     | ChromaDB (HNSW)        |
| Knowledge graph  | NetworkX + spaCy       |
| API              | FastAPI + Uvicorn      |
| Memory management| PyTorch + CUDA         |

### Target outcomes
1. Reduce VRAM consumption by 15–20% while preserving system quality  
2. Improve Recall@K by 5–10% in multi-turn dialogue scenarios  
3. Maintain interactive latency (<200 ms per token)  
4. Enable operation with context sizes at least 2× larger than the physical GPU memory capacity  

## Quick start

### Requirements
- **OS**: Ubuntu 22.04 LTS (or any other Linux distribution supported by vLLM)  
- **GPU**: NVIDIA GPU with CUDA support (12GB+ VRAM recommended)  
- **CPU**: At least 8 cores  
- **RAM**: 16GB+  
- **Python**: 3.10+  

### System installation
To install the system, download the latest version of the repository, go to the project root directory and run the installation script:
```bash
chmod +x ./scripts/install_env.sh
./scripts/install_env.sh
```

### System launch
1. Check the configuration file and adjust runtime parameters in `configs/system_params.json` if needed.
2. In the project root directory, open a terminal and run the launch script:
```bash
chmod +x ./scripts/start_vllm_server.sh
./scripts/start_vllm_server.sh
```
3. Wait for the system to start successfully.
4. In the project root directory, open an additional terminal (without closing the previous one) and start the local API server:
```bash
source S-GAS_Manager_env/bin/activate
uvicorn src.web.api:app --reload --host 0.0.0.0 --port 8080
```
5. After the API is successfully started, open the system page in your browser: **http://localhost:8080**

## How to use the systen

### Via web UI
1. To create a request, use the message input form. After entering your text, click the corresponding button to send it.
2. RAG mode is automatically enabled when documents are used. A dedicated button is provided for document upload. Successful upload and processing of a document is indicated by a corresponding message in the chat.
3. When the chat page is opened, a session with a unique ID is created. This ID is required to access the statistics page for a specific session.

### Main endpoints for system inspection
- `/api/session/{session_id}/info` - information about the current session
- `/api/session/{session_id}/documents` - information about documents used in the system
- `/api/sgas-statistics` - overall state of the algorithm and system

### How to inspect and collect system logs
1. Use the corresponding endpoints to view system usage statistics.
2. The main log file with system state is stored in the  `logs`. The default file name is `session_metrics.jsonl`
3. During system operation, service messages are also printed to the terminal.


### Possible issues and their solutions
| Issue | Soluation |
|----------|---------|
| vLLM does not start | Check CUDA support: `nvidia-smi` |
| API does not respond | Make sure vLLM is using the correct port (8000) |
| GPU out-of-memory | Decreese `gpu_memory_utilization` in `configs/system_params.json` |
| Graph is not built | Ensure the correct version of spaCy model is installed: `python -m spacy download ru_core_news_sm` (for Russian) |

## Architecture of system
```
S-GAS_MANAGER/
├── src/
│   ├── core/
│   │   └── scoring.py                # Core: semantic-graph scoring of text fragments
│   ├── modules/
│   │   ├── retrieval/
│   │   │   ├── chunking.py           # Splitting text into semantic chunks
│   │   │   ├── document_loader.py    # Document loading and initialization
│   │   │   ├── document_processor.py # Preprocessing of text
│   │   │   ├── embedder.py           # Converting text into embeddings
│   │   │   ├── vector_store.py       # Interaction with ChromaDB
│   │   │   └── retrieval_models.py   # Data models for RAG modules
│   │   ├── graph/
│   │   │   └── graph_builder.py      # Knowledge graph construction and entity extraction
│   │   └── swap/
│   │       └── swap_manager.py       # Data swapping manager 
│   ├── web/
│   │   ├── api.py                    # FastAPI REST endpoints
│   │   └── static/
│   │       ├── index.html            # Main web client page
│   │       ├── styles.css            # Web client styles
│   │       └── script.js             # Web client logic
│   ├── config.py                     # Configuration deserialization
│   └── main.py                       # Test script for startup validation
├── configs/
│   └── system_params.json            # System runtime parameters (model, GPU mem, batch size, etc.)
├── scripts/
│   ├── install_env.sh                # Automatic dependency installation
│   └── start_vllm_server.sh          # vLLM server startup
├── logs/
│   └── session_metrics.jsonl         # Per-session metrics logging
├── other/
│   ├── requirements.txt              # Main dependencies
│   └── requirements_dev.txt          # Development dependencies
└── data/                             # Document and embedding storage (created automatically)

```

## Contributing
This is a research prototype and is under active development. If you encounter any issues:
1. Check the logs for detailed error information.
2. Verify that all configuration files are correct.
3. Make sure both servers are running correctly.
4. Test with different model configurations.

Any feedback or reports about discovered issues are highly appreciated. Your support helps the project evolve!

[🔗 Author's Telegram](https://t.me/TownSparrow)