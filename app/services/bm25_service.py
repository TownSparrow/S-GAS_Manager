import logging
import re
from typing import List, Dict, Any, Optional

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

logger = logging.getLogger(__name__)

_SPLIT_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    """Simple unicode-aware lowercased tokenizer."""
    return _SPLIT_RE.findall(text.lower())


class BM25Service:
    """Per-session BM25 index over chunk texts."""

    def __init__(self) -> None:
        # session_id → (BM25Okapi, list[chunk_dict])
        self._indices: Dict[str, tuple] = {}

    def build_index(self, session_id: str, chunks: List[Dict[str, Any]]) -> None:
        """Building (or rebuilding) the BM25 index for a session."""
        if not chunks:
            return
        if BM25Okapi is None:
            logger.warning("rank_bm25 not installed — BM25 index skipped")
            return
        corpus = [_tokenize(c.get("text", "")) for c in chunks]
        bm25 = BM25Okapi(corpus)
        self._indices[session_id] = (bm25, list(chunks))
        logger.info(f"BM25 index built for session {session_id}: {len(chunks)} chunks")

    def search(self, session_id: str, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Returning top-k chunks ranked by BM25 score."""
        entry = self._indices.get(session_id)
        if entry is None:
            return []

        bm25, chunks = entry
        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = bm25.get_scores(tokens)

        # Pair (index, score), sort descending
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            chunk = dict(chunks[idx])  # shallow copy
            chunk["bm25_score"] = float(score)
            results.append(chunk)
        return results

    def delete_session(self, session_id: str) -> None:
        self._indices.pop(session_id, None)

    def reset_state(self) -> None:
        self._indices.clear()
