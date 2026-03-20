from typing import List, Dict, Optional
import logging
from sentence_transformers import util
import numpy as np


logger = logging.getLogger(__name__)


class GenerationEvaluator:
    """ Evaluator for generation quality. Uses semantic similarity metrics instead of LLM-as-a-judge. """
    
    def __init__(
        self,
        use_bertscore: bool = True,
        use_rouge: bool = True,
        bertscore_threshold: float = 0.85
    ):
        self.use_bertscore = use_bertscore
        self.use_rouge = use_rouge
        self.bertscore_threshold = bertscore_threshold
        
        # Lazy import to avoid dependency issues
        self.sentence_transformer = None
        self.rouge_scorer = None
    
    def _load_models(self):
        """ Lazy load of required models """
        if self.use_bertscore and self.sentence_transformer is None:
            from sentence_transformers import SentenceTransformer
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded SentenceTransformer for BERTScore")
        
        if self.use_rouge and self.rouge_scorer is None:
            try:
                from rouge_score import rouge_scorer as rs
                self.rouge_scorer = rs.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                logger.info("Loaded ROUGE scorer")
            except ImportError:
                logger.warning("ROUGE scorer not available. Install rouge-score package.")
                self.use_rouge = False
    
    def calculate_bertscore(
        self,
        generated_answer: str,
        reference_answer: str
    ) -> float:
        """ Calculating BERTScore using semantic similarity. BERTScore = cosine similarity between sentence embeddings """
        if not self.use_bertscore:
            return 0.0
        
        self._load_models()
        
        if self.sentence_transformer is None:
            return 0.0
        
        # Encoding both answers
        gen_emb = self.sentence_transformer.encode(generated_answer, convert_to_tensor=True)
        ref_emb = self.sentence_transformer.encode(reference_answer, convert_to_tensor=True)
        
        # Calculating cosine similarity
        similarity = util.cos_sim(gen_emb, ref_emb).item()
        
        return similarity
    
    def calculate_rouge(
        self,
        generated_answer: str,
        reference_answer: str
    ) -> Dict:
        """ Calculating ROUGE scores (n-gram overlap) """
        if not self.use_rouge or self.rouge_scorer is None:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        scores = self.rouge_scorer.score(reference_answer, generated_answer)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def evaluate_generation(
        self,
        generated_answer: str,
        reference_answer: str
    ) -> Dict:
        """ Evaluating generation quality comprehensively """
        
        # Calculating BERTScore
        bertscore = self.calculate_bertscore(generated_answer, reference_answer)
        
        # Calculating ROUGE
        rouge_scores = self.calculate_rouge(generated_answer, reference_answer)
        
        # Determine if answer is correct based on threshold
        is_correct = bertscore >= self.bertscore_threshold
        
        result = {
            'bertscore': bertscore,
            'is_correct': is_correct,
            'bertscore_threshold': self.bertscore_threshold,
            'rouge_scores': rouge_scores,
            'generated_answer_preview': generated_answer[:100],
            'reference_answer_preview': reference_answer[:100]
        }
        
        return result
    
    def evaluate_multi_turn(
        self,
        generated_answers: List[str],
        reference_answers: List[str]
    ) -> Dict:
        """ Evaluating multi-turn dialogue accuracy """
        if len(generated_answers) != len(reference_answers):
            raise ValueError("Number of generated and reference answers must match")
        
        results = []
        correct_count = 0
        
        for gen, ref in zip(generated_answers, reference_answers):
            result = self.evaluate_generation(gen, ref)
            results.append(result)
            
            if result['is_correct']:
                correct_count += 1
        
        accuracy = correct_count / len(generated_answers) if generated_answers else 0.0
        
        # Calculating average scores
        avg_bertscore = sum(r['bertscore'] for r in results) / len(results) if results else 0.0
        avg_rouge1 = sum(r['rouge_scores']['rouge1'] for r in results) / len(results) if results else 0.0
        avg_rouge2 = sum(r['rouge_scores']['rouge2'] for r in results) / len(results) if results else 0.0
        avg_rougeL = sum(r['rouge_scores']['rougeL'] for r in results) / len(results) if results else 0.0
        
        return {
            'multi_turn_accuracy': accuracy,
            'total_turns': len(generated_answers),
            'correct_turns': correct_count,
            'avg_bertscore': avg_bertscore,
            'avg_rouge1': avg_rouge1,
            'avg_rouge2': avg_rouge2,
            'avg_rougeL': avg_rougeL,
            'per_turn_results': results
        }