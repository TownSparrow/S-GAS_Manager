from typing import Dict, Any


def validate_chunk_structure(chunk_dict: Dict[str, Any]) -> bool:
    required_keys = {'id', 'text', 'metadata'}
    if not all(key in chunk_dict for key in required_keys):
        return False
    required_metadata_keys = {'session_id', 'document_uuid', 'chunk_index', 'chunk_size'}
    metadata = chunk_dict.get('metadata', {})
    if not all(key in metadata for key in required_metadata_keys):
        return False
    return True
