# Semantic-Graph Adaptive Scoring Manager (S-GAS Manager)
[RU Version of README](https://github.com/TownSparrow/S-GAS_Manager/blob/main/README_ru.md)

## ⚠️ WARNING! ⚠️
Current version is `v0.2.0-alpha.1`. Project is in active development! I don't recommend using it in production tasks yet!

## About the project
This project is a prototype system that applies a hybrid semantic-graph scoring algorithm for Retrieval-Augmented Generation (RAG) with small language models (SLM). The system combines vector search, knowledge graph analysis, and adaptive query classification to deliver high-quality retrieval results on consumer-grade hardware.

The project and algorithm were developed by student Leonid Vorobyev as part of research work within the master's program "Game Development Technologies" at the Game Development School of ITMO University.

### Scientific problem
Small language models are limited by a fixed context window, which creates critical issues in:
- **Multi-turn dialogues**: the model loses early context and generates contradictory answers;
- **RAG systems**: naive chunk accumulation leads to GPU memory overflow and retrieval quality degradation;
- **Long documents**: the model loses information in the middle of the context (lost-in-the-middle effect);
- **Query type adaptation**: uniform retrieval strategy ignores the difference between factual and analytical queries.

### Key components of the algorithm
The algorithm consists of three main mechanisms:
1. **Hybrid scoring formula**: s_i(q) = α·cos(e_q, e_i) + β·[1/(1+d_graph(c_i, q))]
2. **Dynamic weight classification**: adaptive α/β weights based on query type (factual vs. analytical)
3. **Multi-stage retrieval pipeline**: dense retrieval → graph expansion → cross-encoder reranking → hybrid fusion

### Core technologies
| Component         | Technology             |
|------------------|------------------------|
| Inference engine | vLLM (PagedAttention)  |
| Embeddings       | Sentence-Transformers  |
| Vector store     | ChromaDB (HNSW)        |
| Knowledge graph  | NetworkX + spaCy/Natasha |
| NER (Russian)    | Natasha + spaCy ru_core_news_md |
| Keyword extraction | YAKE                   |
| Reranking        | Cross-Encoder (ms-marco-MiniLM-L-6-v2) |
| Hybrid retrieval | BM25 + RRF fusion      |
| API              | FastAPI + Uvicorn      |
| Dependency injection | Manual DI container  |

### Target outcomes
1. Improve Recall@K by 5–10% through hybrid semantic-graph scoring
2. Achieve +2–6% recall gain on multi-turn dialogue scenarios with dynamic weights
3. Maintain interactive latency (<200 ms per token for retrieval + generation)
4. Enable operation with context sizes exceeding physical GPU memory capacity

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
1. Check the configuration file and adjust runtime parameters in `cfg/system_params.json` if needed.
2. In the project root directory, open a terminal and run the launch script:
```bash
chmod +x ./scripts/start_vllm_server.sh
./scripts/start_vllm_server.sh
```
3. Wait for the system to start successfully.
4. An additional terminal is automatically opened in the project root folder without closing the previous one, and a script is launched to launch the local server API.
5. After the API is successfully started, open the system page in your browser: **http://localhost:8080**

## How to use the system

### Via web UI
1. To create a request, use the message input form. After entering your text, click the corresponding button to send it.
2. RAG mode is automatically enabled when documents are used. A dedicated button is provided for document upload. Successful upload and processing of a document is indicated by a corresponding message in the chat.
3. When the chat page is opened, a session with a unique ID is created. This ID is required to access the statistics page for a specific session.

### Main endpoints for system inspection
- `/api/session/{session_id}/info` - information about the current session
- `/api/session/{session_id}/documents` - information about documents used in the system
- `/api/sgas-statistics` - overall state of the algorithm and system
- `/api/benchmark/run/{dataset}` - run benchmark on specified dataset

### How to inspect and collect system logs
1. Use the corresponding endpoints to view system usage statistics.
2. The main log file with system state is stored in the `logs` directory. The default file name is `session_metrics.jsonl`
3. During system operation, service messages are also printed to the terminal.


### Possible issues and their solutions
| Issue | Solution |
|----------|---------|
| vLLM does not start | Check CUDA support: `nvidia-smi` |
| API does not respond | Make sure vLLM is using the correct port (8000) |
| GPU out-of-memory | Decrease `gpu_memory_utilization` in `cfg/system_params.json` |
| Graph is not built | Ensure the correct version of spaCy model is installed: `python -m spacy download ru_core_news_md` (for Russian) |
| NER fails for Russian | Install Natasha: `pip install natasha` |

## Architecture of system
```
S-GAS_MANAGER/
├── run.py                            # Entry point: DI wiring, FastAPI routes, startup
├── config.py                         # Configuration loading from JSON
├── requirements.txt                  # Main dependencies
├── requirements_dev.txt              # Development dependencies
├── app/
│   ├── consts/                       # Constants and prompt templates
│   ├── models/                       # Data models (Chunk, Document, Session, API schemas)
│   ├── interfaces/                   # Service interfaces (contracts)
│   ├── services/                     # Business logic (Embedding, VectorStore, Graph, Scoring, Chat)
│   │   ├── _processors/              # NER and keyword extraction processors
│   │   ├── monitoring/               # KV-cache monitoring
│   │   └── testing/                  # Benchmark runner, metrics, evaluators
│   ├── controllers/                  # Thin HTTP handlers (Session, Document, Search, Chat, Benchmark)
│   ├── loaders/                      # Document loaders (PDF, Text, DOCX)
│   └── utils/                        # Utilities (GPU, serialization, validation)
├── cfg/
│   └── system_params.json            # System runtime parameters
├── static/                           # Web client (HTML, CSS, JS)
├── scripts/
│   ├── install_env.sh                # Automatic dependency installation
│   └── start_vllm_server.sh          # vLLM server startup
├── logs/                             # Per-session metrics logging
└── data/                             # Document and embedding storage (created automatically)

```

## Benchmarking & Evaluation

The project includes a comprehensive benchmarking system for evaluating retrieval quality:

### Metrics
- **Recall@K**: Ability to retrieve relevant chunks in top-K results
- **Coverage**: Percentage of queries with at least one relevant chunk retrieved
- **Text Recall**: Exact match quality of retrieved content
- **Semantic Similarity**: Cosine similarity between query and retrieved embeddings
- **Multi-turn Accuracy**: Consistency across dialogue turns
- **Latency**: End-to-end response time (retrieval + generation)

### Benchmark usability scenarios that are recommended to be configured manually in system_params.json
- **Stress**: Minimal resources (gpu_memory_utilization: 0.5, max_tokens: 256)
- **Optimal**: Balanced configuration (gpu_memory_utilization: 0.7, max_tokens: 512)
- **Maximal**: Full resource utilization (gpu_memory_utilization: 0.85, max_tokens: 1024)

### Datasets
- **Baldur's Gate 3**: Multi-turn dialogue dataset based on the game narrative

## Contributing
This is a research prototype and is under active development. If you encounter any issues:
1. Check the logs for detailed error information.
2. Verify that all configuration files are correct.
3. Make sure both servers are running correctly.
4. Test with different model configurations.

Any feedback or reports about discovered issues are highly appreciated. Your support helps the project evolve!

[🔗 Author's Telegram](https://t.me/TownSparrow)
