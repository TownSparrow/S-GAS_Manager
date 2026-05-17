from typing import Dict, Any
import numpy as np


def serialize_chunk_safe(chunk: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in chunk.items():
        if key == 'embedding':
            continue
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            result[key] = value.item()
        elif isinstance(value, dict):
            result[key] = serialize_chunk_safe(value)
        elif isinstance(value, (list, tuple)):
            result[key] = [item.item() if isinstance(item, (np.integer, np.floating)) else item for item in value]
        else:
            result[key] = value
    return result
