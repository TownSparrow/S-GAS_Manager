from pathlib import Path

# Directories
UPLOADS_DIR = Path("data/uploads")
STATIC_DIR = Path(__file__).parent.parent.parent / "static"

# API
API_VERSION = "v0.1.0-alpha.2"

# Embeddings
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

# GPU
GPU_SAFETY_MARGIN = 1.5
ESTIMATED_BYTES_PER_CHUNK = EMBEDDING_DIM * 4 + 2000

# Chunks
MAX_CHUNK_TEXT_LEN = 5000

# Graph
GRAPH_FALLBACK_DISTANCE = 0.5
GRAPH_INFINITY_DISTANCE = 999.0

# Swap
SWAP_LRU_OFFLOAD_COUNT = 3
SWAP_GPU_PRELOAD_COUNT = 5
SWAP_TIMEOUT_SEC = 5

# Metrics
METRICS_FILE = Path("logs/session_metrics.jsonl")
