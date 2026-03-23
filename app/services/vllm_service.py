import logging
from typing import Dict, Any

import httpx
from fastapi import HTTPException

from app.interfaces.vllm_client import IVLLMClient
from app.consts.prompts import VLLM_STOP_SEQUENCES

logger = logging.getLogger(__name__)


class VLLMService(IVLLMClient):
    def __init__(self, api_base: str, model_name: str, default_temperature: float = 0.7, default_top_p: float = 0.9, default_max_tokens: int = 512):
        self._api_base = api_base
        self._model_name = model_name
        self._default_temperature = default_temperature
        self._default_top_p = default_top_p
        self._default_max_tokens = default_max_tokens

    async def chat(self, prompt: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": overrides.get("temperature") if overrides.get("temperature") is not None else self._default_temperature,
            "max_tokens": overrides.get("max_tokens") if overrides.get("max_tokens") is not None else self._default_max_tokens,
            "top_p": overrides.get("top_p") if overrides.get("top_p") is not None else self._default_top_p,
            "stop": VLLM_STOP_SEQUENCES,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{self._api_base}/chat/completions", json=payload)

                if resp.status_code != 200:
                    raise HTTPException(
                        status_code=502,
                        detail=f"vLLM error {resp.status_code}: {resp.text}",
                    )

                data = resp.json()
                choices = data.get("choices")

                if not isinstance(choices, list) or not choices or "message" not in choices[0]:
                    raise HTTPException(
                        status_code=502,
                        detail=f"vLLM malformed response: {data}",
                    )

                if "content" not in choices[0]["message"]:
                    raise HTTPException(
                        status_code=502,
                        detail=f"vLLM response without content: {data}",
                    )

                return data

        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503,
                detail=f"vLLM unavailable: {str(e)}",
            )

    async def check_health(self) -> str:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._api_base}/models")
                return "healthy" if response.status_code == 200 else "unhealthy"
        except Exception:
            return "unavailable"

    async def flush_cache(self) -> bool:
        logger.debug("vLLM cache flush skipped: isolation is handled by run_id prompt prefix")
        return False
