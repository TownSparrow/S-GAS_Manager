from typing import List, Dict, Optional
import logging
from sentence_transformers import util
import numpy as np
import re


logger = logging.getLogger(__name__)


class GenerationEvaluator:
    """ Evaluator for generation quality. Uses semantic similarity metrics instead of LLM-as-a-judge. """
    
    def __init__(
        self,
        use_bertscore: bool = True,
        use_rouge: bool = True,
        bertscore_threshold: float = 0.65
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

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Language-agnostic-ish tokenizer for local answer overlap metrics."""
        return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)

    def calculate_token_f1(
        self,
        generated_answer: str,
        reference_answer: str
    ) -> float:
        """Token-F1 between generated and reference answers, no external judge."""
        gen_tokens = self._tokenize(generated_answer)
        ref_tokens = self._tokenize(reference_answer)
        if not gen_tokens or not ref_tokens:
            return 0.0

        ref_counts = {}
        for token in ref_tokens:
            ref_counts[token] = ref_counts.get(token, 0) + 1

        overlap = 0
        for token in gen_tokens:
            if ref_counts.get(token, 0) > 0:
                overlap += 1
                ref_counts[token] -= 1

        if overlap == 0:
            return 0.0
        precision = overlap / len(gen_tokens)
        recall = overlap / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    def calculate_exact_match(
        self,
        generated_answer: str,
        reference_answer: str
    ) -> float:
        """Normalized exact match for deterministic answer checks."""
        gen = " ".join(self._tokenize(generated_answer))
        ref = " ".join(self._tokenize(reference_answer))
        if not ref:
            return 0.0
        return 1.0 if gen == ref else 0.0
    
    def evaluate_generation(
        self,
        generated_answer: str,
        reference_answer: str
    ) -> Dict:
        """ Evaluating generation quality comprehensively """
        self._load_models()

        # Calculating BERTScore
        bertscore = self.calculate_bertscore(generated_answer, reference_answer)

        # Calculating ROUGE
        rouge_scores = self.calculate_rouge(generated_answer, reference_answer)
        rougeL = rouge_scores.get('rougeL', 0.0)
        token_f1 = self.calculate_token_f1(generated_answer, reference_answer)
        exact_match = self.calculate_exact_match(generated_answer, reference_answer)
        is_correct = bertscore >= self.bertscore_threshold or rougeL >= 0.40

        result = {
            'bertscore': bertscore,
            'token_f1': token_f1,
            'exact_match': exact_match,
            'is_correct': is_correct,
            'bertscore_threshold': self.bertscore_threshold,
            'rouge_scores': rouge_scores,
            'generated_answer_preview': generated_answer[:500],
            'reference_answer_preview': reference_answer[:500]
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
        avg_token_f1 = sum(r['token_f1'] for r in results) / len(results) if results else 0.0
        avg_exact_match = sum(r['exact_match'] for r in results) / len(results) if results else 0.0
        
        return {
            'multi_turn_accuracy': accuracy,
            'total_turns': len(generated_answers),
            'correct_turns': correct_count,
            'avg_bertscore': avg_bertscore,
            'avg_token_f1': avg_token_f1,
            'avg_exact_match': avg_exact_match,
            'avg_rouge1': avg_rouge1,
            'avg_rouge2': avg_rouge2,
            'avg_rougeL': avg_rougeL,
            'per_turn_results': results
        }
