from typing import List, Dict, Set
import logging
import math


logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """ Evaluator for retrieval metrics. Measures how well the system retrieves relevant chunks from the knowledge base """
    
    def __init__(self):
        self.recall_history = []
        self.precision_history = []
        self.coverage_history = []
    
    def calculate_recall_at_k(
        self,
        retrieved_chunks: List[str],
        relevant_chunks: List[str],
        k: int = 5
    ) -> float:
        """ Calculating the Recall@K. Recall@K = (number of relevant chunks in top-K) / (total number of relevant chunks) """
        if not isinstance(retrieved_chunks, list):
            raise ValueError("retrieved_chunks must be a list")
        if not isinstance(relevant_chunks, list):
            raise ValueError("relevant_chunks must be a list")
        if k <= 0:
            raise ValueError("k must be positive")
        
        if not relevant_chunks:
            return 0.0
        
        retrieved_top_k = retrieved_chunks[:k]
        relevant_set = set(relevant_chunks)
        retrieved_set = set(retrieved_top_k)
        
        relevant_in_top_k = len(relevant_set & retrieved_set)
        total_relevant = len(relevant_set)
        
        recall = relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0
        return recall
    
    def calculate_precision(
        self,
        retrieved_chunks: List[str],
        relevant_chunks: List[str],
        k: int = 5
    ) -> float:
        """ Calculating the Precision@K. Precision@K = (number of relevant chunks in top-K) / K """
        if not relevant_chunks:
            return 0.0
        
        retrieved_top_k = retrieved_chunks[:k]
        relevant_set = set(relevant_chunks)
        retrieved_set = set(retrieved_top_k)
        
        relevant_in_top_k = len(relevant_set & retrieved_set)
        
        precision = relevant_in_top_k / k if k > 0 else 0.0
        return precision

    def calculate_f1_at_k(
        self,
        retrieved_chunks: List[str],
        relevant_chunks: List[str],
        k: int = 5
    ) -> float:
        """Calculating F1@K from Precision@K and Recall@K."""
        precision = self.calculate_precision(retrieved_chunks, relevant_chunks, k)
        recall = self.calculate_recall_at_k(retrieved_chunks, relevant_chunks, k)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def calculate_hit_at_k(
        self,
        retrieved_chunks: List[str],
        relevant_chunks: List[str],
        k: int = 5
    ) -> float:
        """Hit@K: 1 if at least one relevant chunk appears in top-K, else 0."""
        if not relevant_chunks:
            return 0.0
        return 1.0 if set(retrieved_chunks[:k]) & set(relevant_chunks) else 0.0

    def calculate_mrr(
        self,
        retrieved_chunks: List[str],
        relevant_chunks: List[str]
    ) -> float:
        """Mean Reciprocal Rank for one query."""
        relevant_set = set(relevant_chunks)
        if not relevant_set:
            return 0.0
        for rank, chunk_id in enumerate(retrieved_chunks, start=1):
            if chunk_id in relevant_set:
                return 1.0 / rank
        return 0.0

    def calculate_ndcg_at_k(
        self,
        retrieved_chunks: List[str],
        relevant_chunks: List[str],
        k: int = 5
    ) -> float:
        """nDCG@K with binary chunk relevance."""
        relevant_set = set(relevant_chunks)
        if not relevant_set:
            return 0.0

        def _dcg(items):
            score = 0.0
            for idx, chunk_id in enumerate(items[:k], start=1):
                rel = 1.0 if chunk_id in relevant_set else 0.0
                if rel:
                    score += rel / math.log2(idx + 1)
            return score

        ideal_hits = min(len(relevant_set), k)
        idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
        if idcg == 0:
            return 0.0
        return _dcg(retrieved_chunks) / idcg

    def calculate_average_precision_at_k(
        self,
        retrieved_chunks: List[str],
        relevant_chunks: List[str],
        k: int = 5
    ) -> float:
        """AP@K with binary relevance."""
        relevant_set = set(relevant_chunks)
        if not relevant_set:
            return 0.0

        hits = 0
        precision_sum = 0.0
        for rank, chunk_id in enumerate(retrieved_chunks[:k], start=1):
            if chunk_id in relevant_set:
                hits += 1
                precision_sum += hits / rank
        return precision_sum / min(len(relevant_set), k)
    
    def calculate_coverage(
        self,
        used_chunks: Set[str],
        total_chunks: int
    ) -> float:
        """ Calculating the Coverage Score. Coverage = (number of unique chunks used) / (total number of chunks) """
        if total_chunks == 0:
            return 0.0
        
        coverage = len(used_chunks) / total_chunks
        return coverage
    
    def evaluate_retrieval(
        self,
        retrieved_chunks: List[str],
        relevant_chunks: List[str],
        used_chunks: Set[str],
        total_chunks: int,
        top_k_values: List[int] = None,
        record: bool = True
    ) -> Dict:
        """ Evaluating the retrieval performance comprehensively """
        if top_k_values is None:
            top_k_values = [5, 10, 20]
        
        # Calculating Recall@K for each K
        recall_at_k = {}
        precision_at_k = {}
        f1_at_k = {}
        hit_at_k = {}
        ndcg_at_k = {}
        average_precision_at_k = {}
        
        for k in top_k_values:
            recall_at_k[f'recall@{k}'] = self.calculate_recall_at_k(
                retrieved_chunks, relevant_chunks, k
            )
            precision_at_k[f'precision@{k}'] = self.calculate_precision(
                retrieved_chunks, relevant_chunks, k
            )
            f1_at_k[f'f1@{k}'] = self.calculate_f1_at_k(
                retrieved_chunks, relevant_chunks, k
            )
            hit_at_k[f'hit@{k}'] = self.calculate_hit_at_k(
                retrieved_chunks, relevant_chunks, k
            )
            ndcg_at_k[f'ndcg@{k}'] = self.calculate_ndcg_at_k(
                retrieved_chunks, relevant_chunks, k
            )
            average_precision_at_k[f'ap@{k}'] = self.calculate_average_precision_at_k(
                retrieved_chunks, relevant_chunks, k
            )
        
        # Calculating coverage
        coverage = self.calculate_coverage(used_chunks, total_chunks)
        
        # Calculating average metrics
        avg_recall = sum(recall_at_k.values()) / len(recall_at_k) if recall_at_k else 0.0
        avg_precision = sum(precision_at_k.values()) / len(precision_at_k) if precision_at_k else 0.0
        avg_f1 = sum(f1_at_k.values()) / len(f1_at_k) if f1_at_k else 0.0
        avg_hit = sum(hit_at_k.values()) / len(hit_at_k) if hit_at_k else 0.0
        avg_ndcg = sum(ndcg_at_k.values()) / len(ndcg_at_k) if ndcg_at_k else 0.0
        mean_average_precision = (
            sum(average_precision_at_k.values()) / len(average_precision_at_k)
            if average_precision_at_k else 0.0
        )
        mrr = self.calculate_mrr(retrieved_chunks, relevant_chunks)
        
        result = {
            'recall_at_k': recall_at_k,
            'precision_at_k': precision_at_k,
            'f1_at_k': f1_at_k,
            'hit_at_k': hit_at_k,
            'ndcg_at_k': ndcg_at_k,
            'average_precision_at_k': average_precision_at_k,
            'coverage': coverage,
            'avg_recall': avg_recall,
            'avg_precision': avg_precision,
            'avg_f1': avg_f1,
            'avg_hit': avg_hit,
            'avg_ndcg': avg_ndcg,
            'mean_average_precision': mean_average_precision,
            'mrr': mrr,
            'retrieved_count': len(retrieved_chunks),
            'relevant_count': len(relevant_chunks),
            'used_chunks_count': len(used_chunks)
        }
        
        # Storing in history
        if record:
            self.recall_history.append(avg_recall)
            self.precision_history.append(avg_precision)
            self.coverage_history.append(coverage)
        
        return result
    
    def get_summary(self) -> Dict:
        """ Getting summary statistics """
        if not self.recall_history:
            return {}
        
        return {
            'avg_recall_overall': sum(self.recall_history) / len(self.recall_history),
            'avg_precision_overall': sum(self.precision_history) / len(self.precision_history),
            'avg_coverage_overall': sum(self.coverage_history) / len(self.coverage_history),
            'final_coverage': self.coverage_history[-1] if self.coverage_history else 0.0,
            'total_evaluations': len(self.recall_history)
        }
    
    def reset(self):
        """ Resetting all statistics """
        self.recall_history = []
        self.precision_history = []
        self.coverage_history = []
