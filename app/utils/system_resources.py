"""System resource monitoring: RAM, disk, and process-level memory."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None
    logger.warning("psutil not installed — system resource monitoring disabled")


def get_system_resources() -> Dict[str, Any]:
    """Collect current RAM, disk, and process memory usage."""
    if psutil is None:
        return {}

    try:
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        proc = psutil.Process()
        proc_mem = proc.memory_info()

        return {
            'ram': {
                'total_gb': round(vm.total / (1024 ** 3), 2),
                'used_gb': round(vm.used / (1024 ** 3), 2),
                'available_gb': round(vm.available / (1024 ** 3), 2),
                'percent': vm.percent,
            },
            'disk': {
                'total_gb': round(disk.total / (1024 ** 3), 2),
                'used_gb': round(disk.used / (1024 ** 3), 2),
                'free_gb': round(disk.free / (1024 ** 3), 2),
                'percent': disk.percent,
            },
            'process': {
                'rss_mb': round(proc_mem.rss / (1024 ** 2), 2),
                'vms_mb': round(proc_mem.vms / (1024 ** 2), 2),
            },
        }
    except Exception as e:
        logger.warning(f"Failed to collect system resources: {e}")
        return {}
