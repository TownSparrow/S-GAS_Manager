import numpy as np
import httpx
from sentence_transformers import SentenceTransformer
import asyncio
from typing import List
import logging


logger = logging.getLogger(__name__)


class EmbeddingManager:
    """ Manager for working with embeddings """
    
    def __init__(self, config=None):
        self.config = config
        self.local_model = None
        self._load_local_model()
    
    def _load_local_model(self):
        """ Loading local embedding model """
        
        try:
            # Late import to avoid cyclical import
            if self.config is None:
                from src.config import settings
                model_name = settings['embeddings']['model']
            else:
                model_name = self.config.get('embeddings.model', 'sentence-transformers/all-MiniLM-L6-v2')

            self.local_model = SentenceTransformer(model_name)
            logger.info(f"✅ Local embedding model loaded: {model_name}")

        except Exception as e:
            logger.warning(f"❌ Failed to load local model: {e}")
    
    async def get_embeddings_via_api(self, texts: List[str]) -> np.ndarray:
        """ Getting Embeddings via vLLM API """

        # Late import
        if self.config is None:
            from src.config import settings
            api_url = settings.get('vllm.api_base', 'http://localhost:8000/v1')
            model_name = settings.get('embeddings.model', 'sentence-transformers/all-MiniLM-L6-v2')
        else:
            api_url = self.config.get('vllm.api_base', 'http://localhost:8000/v1')
            model_name = self.config.get('embeddings.model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        logger.debug(f"Requesting embeddings from {api_url}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{api_url}/embeddings",
                    json={
                        "model": model_name,
                        "input": texts
                    }
                )

                response.raise_for_status()
                data = response.json()

                embeddings = np.array([item["embedding"] for item in data["data"]])

                logger.debug(f"✅ Got {len(embeddings)} embeddings from API")

                return embeddings
            
        except Exception as e:
            logger.warning(f"⚠️ API embeddings are not available: {e}")
            raise
    
    def get_embeddings_local(self, texts: List[str]) -> np.ndarray:
        """ Getting Embeddings via Local Model """

        if self.local_model is None:
            raise RuntimeError("❌ Local embedding model not loaded")
        
        logger.debug(f"Computing embeddings locally for {len(texts)} texts")

        embeddings = self.local_model.encode(texts, convert_to_numpy=True)
        
        return embeddings
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """ Getting Embeddings (API first, then local model) """

        if not texts:
            return np.array([])
        
        try:
            logger.info(f"Getting embeddings for {len(texts)} texts (trying API)...")
            return asyncio.run(self.get_embeddings_via_api(texts))
        
        except Exception as api_e:
            logger.warning(f"⚠️ API failed, falling back to local: {api_e}")
            try:
                return self.get_embeddings_local(texts)
            except Exception as local_e:
                logger.error(f"❌ Both API and local embeddings failed: {local_e}")
                raise RuntimeError(f"Cannot get embeddings: {local_e}")


# Global instance
embedding_manager = EmbeddingManager()


def get_embeddings(texts: List[str]) -> np.ndarray:
    """ Wrapper function for getting embeddings """
    
    return embedding_manager.get_embeddings(texts)