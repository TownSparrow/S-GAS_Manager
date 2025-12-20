import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Settings:
    """ Class for loading settings from JSON-file """
    
    def __init__(self, config_path=None):
        if config_path is None:
            # Getting the path to the root folder of the project (one level above src)
            current_file = Path(__file__)  # src/config.py
            project_root = current_file.parent.parent  # root folder of project
            self.config_path = project_root / "configs" / "system_params.json"
        else:
            self.config_path = Path(config_path)
            
        self.settings = self._load_config()
        self._create_compatibility_attributes()
        
    def _load_config(self) -> dict:
        """ Loading configuration from JSON-file, fallback if file missing/bitten """
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except FileNotFoundError as e:
            logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """ Returns the default configuration """
        logger.warning("Default configuration is used")
        return {
            "vllm": {
                "model_name": "Qwen/Qwen3-8B-AWQ",
                "gpu_memory_utilization": 0.8,
                "max_model_len": 4096,
                "api_base": "http://localhost:8000/v1",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 256
            },
            "embeddings": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "similarity_metric": "cosine"
            },
            "rag": {
                "top_k": 5
            },
            "chunking": {
                "max_chunk_size": 512,
                "overlap_size": 50
            },
            "database": {
                "chroma_persist_dir": "data/chroma_db",
                "collections": "documents"
            },
            "graph": {
                "alpha": 0.6,
                "beta": 0.4
            },
            "swap": {
                "threshold": 0.3,
                "prefetch_count": 5,
                "memory_check_interval": 50
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8080
            }
        }
    
    def _create_compatibility_attributes(self):
        """ Creating attributes for backward compatibility with existing code """
        self.VLLM_MODEL_NAME = self.settings['vllm']['model_name']
        self.VLLM_API_URL = self.settings['vllm']['api_base']
        self.EMBEDDING_MODEL_NAME = self.settings['embeddings']['model']
        self.API_HOST = self.settings['api']['host']
        self.API_PORT = self.settings['api']['port']
        self.CHROMA_PERSIST_DIR = self.settings['database']['chroma_persist_dir']
        self.COLLECTION_NAME = self.settings['database'].get('collection_name', 'documents')
        self.PROMPT_ENABLE_CONTEXT_LIMIT = self.settings['prompt'].get('enable_context_limit', True)
        self.PROMPT_MAX_CONTEXT_TOKENS = self.settings['prompt'].get('max_context_tokens', 3000)
        self.SWAP_DEBUG_MODE = self.settings['swap'].get('debug_mode', False)
        self.SWAP_FORCE_OFFLOAD_ON_ITERATION = self.settings['swap'].get('force_offload_on_iteration', -1)
        self.vllm_config = VLLMConfig(
            gpu_memory_utilization=self.settings['vllm']['gpu_memory_utilization'],
            max_model_len=self.settings['vllm']['max_model_len'],
            temperature=self.settings['vllm']['temperature'],
            top_p=self.settings['vllm']['top_p'],
            max_tokens=self.settings['vllm']['max_tokens']
        )
    
    def __getitem__(self, key):
        """ Allows access as settings['vllm'] """
        if key not in self.settings:
            raise KeyError(f"Section '{key}' missing in config! Check your system_params.json.")
        return self.settings[key]
    
    def __contains__(self, key):
        """ Checking for a key """
        return key in self.settings

class VLLMConfig:
    """ Configuration for vLLM """
    
    def __init__(self, gpu_memory_utilization=0.6, max_model_len=4096, temperature=0.8, top_p=0.9, max_tokens=256):
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

# Global object of seetings
settings = Settings()

# Function to get settings (compatibility with old code)
def get_settings():
    """ Returns a settings instance """
    return settings