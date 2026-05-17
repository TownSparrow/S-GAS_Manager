from typing import List, Dict, Any


class ISwapManager:
    def initialize_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        raise NotImplementedError

    def decide_swap_action(self, context_chunks: List[Dict[str, Any]], current_context_tokens: int, iteration: int = 0) -> Dict[str, Any]:
        raise NotImplementedError

    def execute_swap_decision(self, decision: Dict[str, Any]) -> None:
        raise NotImplementedError

    def update_prefetch_buffer(self, chunks: List[Dict[str, Any]]) -> None:
        raise NotImplementedError

    def record_compute_time(self, compute_time_ms: float) -> None:
        raise NotImplementedError

    def get_statistics(self) -> Dict[str, Any]:
        raise NotImplementedError

    def cleanup(self) -> None:
        raise NotImplementedError
