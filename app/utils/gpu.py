import logging
import torch
import torch.cuda

logger = logging.getLogger(__name__)


def log_gpu_memory_detailed(stage: str, iteration: int) -> None:
    if not torch.cuda.is_available():
        return
    try:
        gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        gpu_free = gpu_total - gpu_allocated
        gpu_max = torch.cuda.max_memory_allocated() / 1024 / 1024
        logger.info(
            f"[Iteration {iteration}] {stage} GPU STATE: "
            f"Allocated={gpu_allocated:.2f}MB/{gpu_total:.2f}MB ({gpu_allocated/gpu_total*100:.1f}%), "
            f"Reserved={gpu_reserved:.2f}MB, Free={gpu_free:.2f}MB, Peak={gpu_max:.2f}MB"
        )
    except Exception as e:
        logger.warning(f"Failed to log GPU memory: {e}")


def get_gpu_memory_stats() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        'allocated_mb': round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
        'reserved_mb': round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
    }
