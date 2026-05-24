from pathlib import Path

# Directories
UPLOADS_DIR = Path("data/uploads")
STATIC_DIR = Path(__file__).parent.parent.parent / "static"

# API
API_VERSION = "v0.2.1-alpha.1"

# Embeddings
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

# GPU
GPU_SAFETY_MARGIN = 1.5
ESTIMATED_BYTES_PER_CHUNK = EMBEDDING_DIM * 4 + 2000

# Chunks
MAX_CHUNK_TEXT_LEN = 5000

# Graph
GRAPH_FALLBACK_DISTANCE = 0.8
GRAPH_INFINITY_DISTANCE = 999.0

# Swap
SWAP_LRU_OFFLOAD_COUNT = 3
SWAP_GPU_PRELOAD_COUNT = 5
SWAP_TIMEOUT_SEC = 5

# Scoring
SEMANTIC_ANCHOR_THRESHOLD = 0.70
MIN_RETURN_CHUNKS = 5
LOW_SEMANTIC_WARNING_THRESHOLD = 0.3

# Metrics
METRICS_FILE = Path("logs/session_metrics.jsonl")
