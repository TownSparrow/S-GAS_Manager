import asyncio
from typing import List
import logging

import numpy as np
import httpx
from sentence_transformers import SentenceTransformer

from app.interfaces.embedding import IEmbeddingService

logger = logging.getLogger(__name__)


class EmbeddingService(IEmbeddingService):
    """
    Embedding service with local model (primary) and optional API fallback.
    """

    def __init__(self, model_name: str, api_base: str, embedding_api_url: str = ""):
        self._model_name = model_name
        self._api_base = api_base
        self._embedding_api_url = embedding_api_url  # optional dedicated embedding endpoint
        self._local_model = None
        self._api_available = False
        self._load_local_model()

        if self._embedding_api_url:
            self._api_available = True
            logger.info(f"Embedding API endpoint configured: {self._embedding_api_url}")

    def _load_local_model(self) -> None:
        try:
            self._local_model = SentenceTransformer(self._model_name)
            logger.info(f"Local embedding model loaded: {self._model_name}")
        except Exception as e:
            logger.warning(f"Failed to load local embedding model: {e}")

    async def _get_embeddings_via_api(self, texts: List[str]) -> np.ndarray:
        url = self._embedding_api_url or f"{self._api_base}/embeddings"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                json={"model": self._model_name, "input": texts},
            )
            response.raise_for_status()
            data = response.json()
            return np.array([item["embedding"] for item in data["data"]])

    def _get_embeddings_local(self, texts: List[str]) -> np.ndarray:
        if self._local_model is None:
            raise RuntimeError("Local embedding model not loaded")
        return self._local_model.encode(texts, convert_to_numpy=True)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])

        # Primary: local model (fast, no network overhead)
        if self._local_model is not None:
            try:
                return self._get_embeddings_local(texts)
            except Exception as e:
                logger.warning(f"Local embedding failed: {e}")

        # Fallback: API (only if explicitly configured)
        if self._api_available:
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, self._get_embeddings_via_api(texts))
                        return future.result(timeout=30)
                else:
                    return asyncio.run(self._get_embeddings_via_api(texts))
            except Exception as e:
                logger.warning(f"API embedding failed: {e}")

        raise RuntimeError("No embedding method available (local model failed, API not configured)")
