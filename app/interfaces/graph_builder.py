from typing import List, Dict, Any
import numpy as np
import networkx as nx


class IGraphBuilder:
    def build_graph(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> nx.DiGraph:
        raise NotImplementedError

    def compute_graph_distances(self, query_text: str, chunk_ids: List[str]) -> Dict[str, float]:
        raise NotImplementedError

    def get_graph_statistics(self) -> Dict[str, Any]:
        raise NotImplementedError

    def export_graph_info(self) -> Dict[str, Any]:
        raise NotImplementedError

    def update_graph(self, new_chunks: List[Dict[str, Any]], new_embeddings: np.ndarray) -> nx.DiGraph:
        raise NotImplementedError

    def get_neighboring_chunk_ids(self, chunk_ids: List[str], top_k: int = 5) -> List[str]:
        raise NotImplementedError

    def get_neighboring_chunks_data(self, chunk_ids: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError
