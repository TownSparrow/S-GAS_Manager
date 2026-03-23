import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Settings:
    """Loads and provides access to configuration from a JSON file."""

    def __init__(self, config_path=None):
        if config_path is None:
            current_file = Path(__file__)
            project_root = current_file.parent
            self.config_path = project_root / "cfg" / "system_params.json"
        else:
            self.config_path = Path(config_path)
        self._data = self._load_config()

    def _load_config(self) -> dict:
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        return {
            "vllm": {"model_name": "Qwen/Qwen3-8B-AWQ", "gpu_memory_utilization": 0.8, "max_model_len": 4096, "api_base": "http://localhost:8000/v1", "temperature": 0.7, "top_p": 0.9, "max_tokens": 256},
            "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2", "similarity_metric": "cosine"},
            "rag": {"top_k": 5},
            "chunking": {"max_chunk_size": 512, "overlap_size": 50},
            "database": {"chroma_persist_dir": "data/chroma_db", "collection_name": "documents"},
            "graph": {"alpha": 0.6, "beta": 0.4, "spacy_model": "ru_core_news_md"},
            "swap": {"threshold": 0.3, "prefetch_count": 5, "memory_check_interval": 50, "debug_mode": False, "force_offload_on_iteration": -1},
            "prompt": {"enable_context_limit": True, "max_context_tokens": 5000},
            "api": {"host": "0.0.0.0", "port": 8080},
        }

    def __getitem__(self, key):
        if key not in self._data:
            raise KeyError(f"Section '{key}' missing in config!")
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        return self._data.get(key, default)
