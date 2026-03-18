from typing import List, Dict, Set
import logging


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
            return 1.0
        
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
            return 1.0
        
        retrieved_top_k = retrieved_chunks[:k]
        relevant_set = set(relevant_chunks)
        retrieved_set = set(retrieved_top_k)
        
        relevant_in_top_k = len(relevant_set & retrieved_set)
        
        precision = relevant_in_top_k / k if k > 0 else 0.0
        return precision
    
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
        top_k_values: List[int] = None
    ) -> Dict:
        """ Evaluating the retrieval performance comprehensively """
        if top_k_values is None:
            top_k_values = [5, 10, 20]
        
        # Calculating Recall@K for each K
        recall_at_k = {}
        precision_at_k = {}
        
        for k in top_k_values:
            recall_at_k[f'recall@{k}'] = self.calculate_recall_at_k(
                retrieved_chunks, relevant_chunks, k
            )
            precision_at_k[f'precision@{k}'] = self.calculate_precision(
                retrieved_chunks, relevant_chunks, k
            )
        
        # Calculating coverage
        coverage = self.calculate_coverage(used_chunks, total_chunks)
        
        # Calculating average metrics
        avg_recall = sum(recall_at_k.values()) / len(recall_at_k) if recall_at_k else 0.0
        avg_precision = sum(precision_at_k.values()) / len(precision_at_k) if precision_at_k else 0.0
        
        result = {
            'recall_at_k': recall_at_k,
            'precision_at_k': precision_at_k,
            'coverage': coverage,
            'avg_recall': avg_recall,
            'avg_precision': avg_precision,
            'retrieved_count': len(retrieved_chunks),
            'relevant_count': len(relevant_chunks),
            'used_chunks_count': len(used_chunks)
        }
        
        # Storing in history
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