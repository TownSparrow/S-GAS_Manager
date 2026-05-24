import logging
import subprocess
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


def get_nvidia_smi_stats() -> dict:
    """Collect whole-GPU utilization/memory stats for the vLLM process window.

    This observes the physical GPU via nvidia-smi, so it works even when vLLM
    runs in a separate process from the FastAPI app.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            return {}
        util_pct, mem_used_mb, mem_total_mb, power_w, temp_c = parts[:5]
        mem_used = float(mem_used_mb)
        mem_total = float(mem_total_mb)
        return {
            "gpu_utilization_pct": float(util_pct),
            "gpu_memory_used_mb": mem_used,
            "gpu_memory_total_mb": mem_total,
            "gpu_memory_used_pct": round(mem_used / mem_total * 100, 2) if mem_total else 0.0,
            "gpu_power_w": float(power_w),
            "gpu_temperature_c": float(temp_c),
        }
    except Exception as e:
        logger.debug(f"nvidia-smi stats unavailable: {e}")
        return {}


def summarize_nvidia_smi_samples(samples: list) -> dict:
    if not samples:
        return {}

    def avg(key):
        vals = [s.get(key) for s in samples if s.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def peak(key):
        vals = [s.get(key) for s in samples if s.get(key) is not None]
        return round(max(vals), 4) if vals else 0.0

    return {
        "gpu_sample_count": len(samples),
        "gpu_utilization_avg_pct": avg("gpu_utilization_pct"),
        "gpu_utilization_peak_pct": peak("gpu_utilization_pct"),
        "gpu_memory_used_avg_mb": avg("gpu_memory_used_mb"),
        "gpu_memory_used_peak_mb": peak("gpu_memory_used_mb"),
        "gpu_memory_used_peak_pct": peak("gpu_memory_used_pct"),
        "gpu_power_avg_w": avg("gpu_power_w"),
        "gpu_power_peak_w": peak("gpu_power_w"),
        "gpu_temperature_peak_c": peak("gpu_temperature_c"),
    }
