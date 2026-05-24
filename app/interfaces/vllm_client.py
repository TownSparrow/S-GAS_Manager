from typing import Dict, Any


class IVLLMClient:
    async def chat(self, prompt: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    async def check_health(self) -> str:
        raise NotImplementedError

    async def flush_cache(self) -> bool:
        raise NotImplementedError

    async def get_runtime_metrics(self) -> Dict[str, Any]:
        raise NotImplementedError
