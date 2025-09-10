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
        """ Loading configuration from JSON-file"""
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
                "max_model_len": 8192,
                "api_base": "http://localhost:8000/v1",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 256
            },
            "embeddings": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "similarity_metric": "cosine"
            },
            "database": {
                "chroma_persist_dir": "data/chroma_db"
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
        # vLLM settings
        self.VLLM_MODEL_NAME = self.get('vllm.model_name', 'Qwen/Qwen3-8B-AWQ')
        self.VLLM_API_URL = self.get('vllm.api_base', 'http://localhost:8000/v1')
        
        # Embedding settings
        self.EMBEDDING_MODEL_NAME = self.get('embeddings.model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        # API settings
        self.API_HOST = self.get('api.host', '0.0.0.0')
        self.API_PORT = self.get('api.port', 8080)
        
        # Chroma DB settings
        self.CHROMA_PERSIST_DIR = self.get('database.chroma_persist_dir', 'data/chroma_db')
        
        # Creating a vllm_config object for compatibility
        self.vllm_config = VLLMConfig(
            gpu_memory_utilization=self.get('vllm.gpu_memory_utilization', 0.8),
            max_model_len=self.get('vllm.max_model_len', 8192),
            temperature=self.get('vllm.temperature', 0.7),
            top_p=self.get('vllm.top_p', 0.9),
            max_tokens=self.get('vllm.max_tokens', 256)
        )

    def get(self, key_path: str, default=None):
        """
        Gets the value by key path (eg 'vllm.model_name')
        
        Args:
            key_path: Path to the key using a dot
            default: Default value
        """
        keys = key_path.split('.')
        value = self.settings
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.warning(f"Key '{key_path}' not found, using default value:: {default}")
            return default
    
    def __getitem__(self, key):
        """ Allows access as settings['vllm'] """
        return self.settings[key]
    
    def __contains__(self, key):
        """ Checking for a key """
        return key in self.settings

class VLLMConfig:
    """ Configuration for vLLM """
    
    def __init__(self, gpu_memory_utilization=0.8, max_model_len=8192, 
                 temperature=0.7, top_p=0.9, max_tokens=256):
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