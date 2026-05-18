import logging
import os
import re
from typing import Dict, Any, Optional

import httpx
from fastapi import HTTPException

from app.interfaces.vllm_client import IVLLMClient
from app.consts.prompts import VLLM_STOP_SEQUENCES

logger = logging.getLogger(__name__)


class VLLMService(IVLLMClient):
    def __init__(
        self,
        api_base: str,
        model_name: str,
        default_temperature: float = 0.7,
        default_top_p: float = 0.9,
        default_max_tokens: int = 2048,
        max_model_len: int = 8192,
        request_timeout: float = 300.0,
    ):
        self._api_base = api_base
        self._model_name = model_name
        self._default_temperature = default_temperature
        self._default_top_p = default_top_p
        self._default_max_tokens = default_max_tokens
        self._max_model_len = max_model_len
        self._request_timeout = request_timeout
        self._metrics_base = self._derive_metrics_base(api_base)

    async def chat(self, prompt: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        requested_max_tokens = (
            overrides.get("max_tokens")
            if overrides.get("max_tokens") is not None
            else self._default_max_tokens
        )
        max_tokens = self._fit_max_tokens_to_context(prompt, requested_max_tokens)
        payload = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": overrides.get("temperature") if overrides.get("temperature") is not None else self._default_temperature,
            "max_tokens": max_tokens,
            "top_p": overrides.get("top_p") if overrides.get("top_p") is not None else self._default_top_p,
            "stop": VLLM_STOP_SEQUENCES,
            "stream": False,
        }

        try:
            timeout = httpx.Timeout(
                timeout=self._request_timeout,
                connect=10.0,
                read=self._request_timeout,
                write=30.0,
                pool=10.0,
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{self._api_base}/chat/completions", json=payload)
                if resp.status_code == 400:
                    retry_tokens = self._max_tokens_from_context_error(resp.text)
                    if retry_tokens is not None and retry_tokens < payload["max_tokens"]:
                        logger.warning(
                            "Retrying vLLM request with max_tokens=%s after context-limit error",
                            retry_tokens,
                        )
                        payload["max_tokens"] = retry_tokens
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

        except httpx.TimeoutException as e:
            raise HTTPException(
                status_code=504,
                detail=f"vLLM request timed out after {self._request_timeout:.0f}s: {str(e)}",
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503,
                detail=f"vLLM unavailable: {str(e)}",
            )

    def _fit_max_tokens_to_context(self, prompt: str, requested_max_tokens: int) -> int:
        """Keep output budget inside the configured context window using a conservative estimate."""
        if not self._max_model_len:
            return requested_max_tokens
        estimated_input_tokens = max(1, len(prompt) // 4)
        available = self._max_model_len - estimated_input_tokens - 16
        if available <= 0:
            logger.warning(
                "Prompt is likely too large for context window: estimated_input_tokens=%s, max_model_len=%s",
                estimated_input_tokens,
                self._max_model_len,
            )
            return min(requested_max_tokens, 16)
        fitted = max(16, min(requested_max_tokens, available))
        if fitted < requested_max_tokens:
            logger.info(
                "Reduced vLLM max_tokens from %s to %s to fit max_model_len=%s",
                requested_max_tokens,
                fitted,
                self._max_model_len,
            )
        return fitted

    def _max_tokens_from_context_error(self, response_text: str) -> Optional[int]:
        """Extract vLLM's actual token counts from a context-limit 400 and compute retry budget."""
        max_len_match = re.search(r"maximum context length is (\d+) tokens", response_text)
        input_match = re.search(r"prompt contains at least (\d+) input tokens", response_text)
        if not max_len_match or not input_match:
            return None
        max_len = int(max_len_match.group(1))
        input_tokens = int(input_match.group(1))
        available = max_len - input_tokens - 16
        if available < 16:
            return None
        return available

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

    @staticmethod
    def _derive_metrics_base(api_base: str) -> str:
        base = api_base.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return base.rstrip("/")

    @staticmethod
    def _metric_key(name: str) -> str:
        return name.replace("vllm:", "").replace("vllm_", "")

    @staticmethod
    def _parse_labels(raw_labels: str) -> Dict[str, str]:
        labels = {}
        for key, value in re.findall(r'([^=,\s]+)="([^"]*)"', raw_labels or ""):
            labels[key] = value
        return labels

    def _parse_prometheus_metrics(self, text: str) -> Dict[str, Any]:
        """Parse the vLLM /metrics text into a compact snapshot for benchmarks."""
        values: Dict[str, float] = {}
        cache_config: Dict[str, str] = {}

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+eE0-9.]+)$", line)
            if not match:
                continue
            raw_name, raw_labels, raw_value = match.groups()
            try:
                value = float(raw_value)
            except ValueError:
                continue

            name = self._metric_key(raw_name)
            labels = self._parse_labels(raw_labels or "")
            if name == "cache_config_info":
                cache_config.update(labels)
                continue

            # Sum counters/histogram values across labels (model, finish_reason, etc.).
            values[name] = values.get(name, 0.0) + value

        def get(*names: str) -> Optional[float]:
            for name in names:
                if name in values:
                    return values[name]
            return None

        def avg(sum_key: str, count_key: str) -> Optional[float]:
            total = get(sum_key)
            count = get(count_key)
            if total is None or not count:
                return None
            return total / count

        prefix_queries = get("prefix_cache_queries")
        prefix_hits = get("prefix_cache_hits")
        prefix_hit_rate = (
            prefix_hits / prefix_queries
            if prefix_queries and prefix_hits is not None else None
        )

        kv_usage = get("kv_cache_usage_perc", "gpu_cache_usage_perc")
        return {
            "available": True,
            "kv_cache_usage_perc": kv_usage,
            "gpu_cache_usage_perc": get("gpu_cache_usage_perc"),
            "prefix_cache_queries_total": prefix_queries,
            "prefix_cache_hits_total": prefix_hits,
            "prefix_cache_hit_rate": prefix_hit_rate,
            "prompt_tokens_total": get("prompt_tokens_total"),
            "generation_tokens_total": get("generation_tokens_total"),
            "num_requests_running": get("num_requests_running"),
            "num_requests_waiting": get("num_requests_waiting"),
            "num_requests_swapped": get("num_requests_swapped"),
            "num_preemptions_total": get("num_preemptions_total"),
            "request_success_total": get("request_success_total"),
            "e2e_latency_avg_s": avg("e2e_request_latency_seconds_sum", "e2e_request_latency_seconds_count"),
            "ttft_avg_s": avg("time_to_first_token_seconds_sum", "time_to_first_token_seconds_count"),
            "inter_token_latency_avg_s": avg("inter_token_latency_seconds_sum", "inter_token_latency_seconds_count"),
            "prefill_time_avg_s": avg("request_prefill_time_seconds_sum", "request_prefill_time_seconds_count"),
            "decode_time_avg_s": avg("request_decode_time_seconds_sum", "request_decode_time_seconds_count"),
            "queue_time_avg_s": avg("request_queue_time_seconds_sum", "request_queue_time_seconds_count"),
            "cache_config": cache_config,
        }

    async def get_runtime_metrics(self) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self._metrics_base}/metrics")
                if response.status_code != 200:
                    return {"available": False, "error": f"HTTP {response.status_code}"}
                return self._parse_prometheus_metrics(response.text)
        except Exception as e:
            return {"available": False, "error": str(e)}

    async def force_restart(self) -> bool:
        """Restart the vLLM server process to guarantee a clean KV-cache.

        If S_GAS_VLLM_RESTART_CMD is configured, execute it and wait for vLLM
        health. Without an explicit restart command, do not kill the foreground
        vLLM process that may be supervising this API; just wait for recovery.
        """
        import asyncio
        import subprocess

        logger.info("Force-restarting vLLM server for clean KV-cache...")
        try:
            restart_cmd = os.getenv("S_GAS_VLLM_RESTART_CMD")
            allow_pkill = os.getenv("S_GAS_ALLOW_VLLM_PKILL_RESTART") == "1"
            if restart_cmd:
                subprocess.Popen(restart_cmd, shell=True)
                await asyncio.sleep(3)
            elif allow_pkill:
                subprocess.run(["pkill", "-f", "vllm.entrypoints"], timeout=10)
                await asyncio.sleep(3)
            else:
                logger.warning(
                    "vLLM restart command is not configured; waiting for existing server recovery. "
                    "Set S_GAS_VLLM_RESTART_CMD to enable managed restarts."
                )

            # Wait for vLLM to come back up (it may be managed by systemd/docker)
            for attempt in range(30):
                health = await self.check_health()
                if health == "healthy":
                    logger.info(f"vLLM server restarted and healthy (attempt {attempt+1})")
                    return True
                await asyncio.sleep(2)

            logger.warning("vLLM server did not come back after restart")
            return False
        except Exception as e:
            logger.warning(f"vLLM force restart failed: {e}")
            return False
