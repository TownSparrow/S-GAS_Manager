import math
from typing import Any, Dict, Iterable, Optional

import numpy as np


def _serialize_value(value: Any, omit_keys: Optional[Iterable[str]] = None) -> Any:
    if isinstance(value, np.ndarray):
        return _serialize_value(value.tolist(), omit_keys=omit_keys)
    if isinstance(value, np.generic):
        return _serialize_value(value.item(), omit_keys=omit_keys)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        omitted = set(omit_keys or ())
        return {
            key: _serialize_value(item, omit_keys=omit_keys)
            for key, item in value.items()
            if key not in omitted
        }
    if isinstance(value, (list, tuple, set)):
        return [_serialize_value(item, omit_keys=omit_keys) for item in value]
    return value


def serialize_json_safe(value: Any) -> Any:
    return _serialize_value(value)


def serialize_chunk_safe(chunk: Dict[str, Any]) -> Dict[str, Any]:
    return _serialize_value(chunk, omit_keys={'embedding'})
