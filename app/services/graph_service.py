import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

import numpy as np
import networkx as nx
import spacy
import pymorphy3
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

from app.interfaces.graph_builder import IGraphBuilder

logger = logging.getLogger(__name__)


class GraphService(IGraphBuilder):
    def __init__(
        self,
        priority_model_for_en: str = "en_core_web_sm",
        priority_model_for_ru: str = "natasha",
        priority_kw_extractor_for_ru: str = "yake",
        priority_kw_extractor_for_en: str = "yake",
        use_gpu: bool = False,
    ):
        self._priority_model_for_en = priority_model_for_en
        self._priority_model_for_ru = priority_model_for_ru
        self._priority_kw_extractor_for_ru = priority_kw_extractor_for_ru
        self._priority_kw_extractor_for_en = priority_kw_extractor_for_en

        self._morph_analyzer = None
        try:
            self._morph_analyzer = pymorphy3.MorphAnalyzer()
            logger.info("PyMorphy3 lemmatizer loaded")
        except ImportError:
            logger.warning("PyMorphy3 not found. Lemmatization might be skipped.")

        self._nlp_en_fallback = None
        self._nlp_ru_fallback = None

        try:
            self._nlp_en_fallback = spacy.load(priority_model_for_en)
            logger.info(f"spaCy English fallback model loaded: {priority_model_for_en}")
        except OSError:
            logger.error(f"Failed to load spaCy English fallback model: {priority_model_for_en}")

        spacy_model_ru_fallback = priority_model_for_ru if priority_model_for_ru != "natasha" else "ru_core_news_lg"
        try:
            self._nlp_ru_fallback = spacy.load(spacy_model_ru_fallback)
            logger.info(f"spaCy Russian fallback model loaded: {spacy_model_ru_fallback}")
        except OSError:
            logger.error(f"Failed to load spaCy Russian fallback model: {spacy_model_ru_fallback}")

        self._processor_en_ner = None
        self._processor_ru_ner = None
        self._processor_en_kw = None
        self._processor_ru_kw = None

        self._graph = nx.DiGraph()
        self._entity_to_chunks = defaultdict(set)
        self._distance_cache = {}

        logger.info("KnowledgeGraphBuilder initialized with multi-language support")

    def _initialize_processor(self, lang: str, processor_type: str, priority_model_or_extractor: str):
        attr_name = f'_processor_{lang}_{processor_type}'
        if getattr(self, attr_name, None) is not None:
            return

        if processor_type == 'ner':
            if lang == 'ru' and priority_model_or_extractor == "natasha":
                from app.services._processors.ner_natasha_processor import NerNatashaProcessor
                setattr(self, attr_name, NerNatashaProcessor(morph_analyzer=self._morph_analyzer))
            elif lang in ['ru', 'en']:
                from app.services._processors.ner_spacy_processor import NerSpacyProcessor
                setattr(self, attr_name, NerSpacyProcessor(model_name=priority_model_or_extractor, morph_analyzer=self._morph_analyzer))
        elif processor_type == 'kw':
            if priority_model_or_extractor == "yake":
                lang_code = 'ru' if lang == 'ru' else 'en'
                from app.services._processors.keyword_yake_processor import KeywordYakeProcessor
                setattr(self, attr_name, KeywordYakeProcessor(lang_code=lang_code))

    def _detect_language(self, text: str) -> str:
        try:
            detected_lang = detect(text)
            if detected_lang in ['ru']:
                return 'ru'
            elif detected_lang in ['en']:
                return 'en'
            else:
                return 'en'
        except LangDetectException:
            return 'en'

    def _extract_concepts_universal(self, text: str) -> List[Dict[str, Any]]:
        if not text.strip():
            return []

        lang = self._detect_language(text)
        concepts = []

        # NER Extraction
        priority_model = self._priority_model_for_ru if lang == 'ru' else self._priority_model_for_en
        self._initialize_processor(lang, 'ner', priority_model)
        processor_ner = getattr(self, f'_processor_{lang}_ner', None)

        if processor_ner:
            entities_ner = processor_ner.process(text)
            concepts.extend(entities_ner)
        else:
            nlp_fallback = self._nlp_en_fallback if lang == 'en' else self._nlp_ru_fallback
            if nlp_fallback:
                doc_spacy = nlp_fallback(text)
                for ent in doc_spacy.ents:
                    if len(ent.text.strip()) > 2:
                        word_normalized = ent.text.strip().lower()
                        if self._morph_analyzer:
                            try:
                                parsed = self._morph_analyzer.parse(word_normalized)[0]
                                word_normalized = parsed.normal_form.lower()
                            except Exception:
                                pass
                        concepts.append({
                            'word': word_normalized, 'label': ent.label_,
                            'type': 'NER', 'start': ent.start_char, 'end': ent.end_char,
                        })

        # Keywords Extraction
        priority_kw = self._priority_kw_extractor_for_ru if lang == 'ru' else self._priority_kw_extractor_for_en
        self._initialize_processor(lang, 'kw', priority_kw)
        processor_kw = getattr(self, f'_processor_{lang}_kw', None)

        if processor_kw:
            keywords_raw = processor_kw.process(text, morph_analyzer=self._morph_analyzer)
            concepts.extend(keywords_raw)

        # Removing duplicates
        unique_concepts = []
        seen = set()
        for concept in concepts:
            key = (concept['word'], concept['type'])
            if key not in seen:
                seen.add(key)
                unique_concepts.append(concept)

        return unique_concepts

    def build_graph(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> nx.DiGraph:
        self._graph = nx.DiGraph()
        self._entity_to_chunks.clear()
        self._distance_cache.clear()

        # Step 1: Add chunk nodes
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get('id', f'chunk_{i}')
            self._graph.add_node(
                chunk_id, type='chunk', text=chunk.get('text', ''),
                embedding=embeddings[i] if i < len(embeddings) else None,
                metadata=chunk.get('metadata', {}),
            )

        # Step 2: Extract concepts (NER + Keywords) and add as nodes
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get('id', f'chunk_{i}')
            chunk_concepts = self._extract_concepts_universal(chunk.get('text', ''))

            for concept in chunk_concepts:
                concept_word = concept['word']
                concept_label = concept['label']
                concept_type = concept['type']
                concept_id = f"concept_{concept_type}_{concept_label}_{concept_word}"

                if concept_id not in self._graph:
                    self._graph.add_node(
                        concept_id, type='concept', word=concept_word,
                        label=concept_label, concept_type=concept_type,
                    )

                self._graph.add_edge(chunk_id, concept_id, weight=0.3, relation=f'contains_{concept_type.lower()}')
                self._graph.add_edge(concept_id, chunk_id, weight=0.3, relation='found_in_chunk')
                self._entity_to_chunks[concept_id].add(chunk_id)

        # Step 3: Link chunks via common concepts
        chunk_ids = [chunk.get('id', f'chunk_{i}') for i, chunk in enumerate(chunks)]
        for i, ci in enumerate(chunk_ids):
            for j, cj in enumerate(chunk_ids):
                if i >= j:
                    continue
                concepts_i = set(self._graph.neighbors(ci))
                concepts_j = set(self._graph.neighbors(cj))
                common = concepts_i & concepts_j
                if common:
                    weight = max(0.1, 1.0 / (1.0 + len(common)))
                    self._graph.add_edge(ci, cj, weight=weight, relation='related_via_common_concepts')
                    self._graph.add_edge(cj, ci, weight=weight, relation='related_via_common_concepts')

        # Step 4: Link chunks via semantic similarity
        for i, ci in enumerate(chunk_ids):
            for j, cj in enumerate(chunk_ids):
                if i >= j:
                    continue
                if i < len(embeddings) and j < len(embeddings):
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                    )
                    if similarity > 0.3:
                        distance = 1.0 - similarity
                        if distance < 0:
                            distance = 0.0

                        for s, t in [(ci, cj), (cj, ci)]:
                            if self._graph.has_edge(s, t):
                                self._graph[s][t]['weight'] = min(self._graph[s][t]['weight'], distance)
                            else:
                                self._graph.add_edge(s, t, weight=distance, relation='semantic_similarity')

        logger.info(f"Graph built: {self._graph.number_of_nodes()} nodes, {self._graph.number_of_edges()} edges")
        return self._graph

    def update_graph(self, new_chunks: List[Dict[str, Any]], new_embeddings: np.ndarray) -> nx.DiGraph:
        """
        Adds new chunks to the existing graph
        """
        existing_chunk_ids = {
            n for n, d in self._graph.nodes(data=True) if d.get('type') == 'chunk'
        }
        truly_new = [
            (i, c) for i, c in enumerate(new_chunks)
            if c.get('id', f'chunk_{i}') not in existing_chunk_ids
        ]
        if not truly_new:
            return self._graph

        # Step 1: Add new chunk nodes
        for i, chunk in truly_new:
            chunk_id = chunk.get('id', f'chunk_{i}')
            self._graph.add_node(
                chunk_id, type='chunk', text=chunk.get('text', ''),
                embedding=new_embeddings[i] if i < len(new_embeddings) else None,
                metadata=chunk.get('metadata', {}),
            )

        # Step 2: Extract concepts and link new chunks to concept nodes
        new_chunk_ids = []
        for i, chunk in truly_new:
            chunk_id = chunk.get('id', f'chunk_{i}')
            new_chunk_ids.append(chunk_id)
            chunk_concepts = self._extract_concepts_universal(chunk.get('text', ''))
            for concept in chunk_concepts:
                concept_id = f"concept_{concept['type']}_{concept['label']}_{concept['word']}"
                if concept_id not in self._graph:
                    self._graph.add_node(
                        concept_id, type='concept', word=concept['word'],
                        label=concept['label'], concept_type=concept['type'],
                    )
                self._graph.add_edge(chunk_id, concept_id, weight=0.3,
                                     relation=f"contains_{concept['type'].lower()}")
                self._graph.add_edge(concept_id, chunk_id, weight=0.3, relation='found_in_chunk')
                self._entity_to_chunks[concept_id].add(chunk_id)

        # Step 3: Link new chunks to ALL existing chunks via shared concepts
        all_chunk_ids_in_graph = [
            n for n, d in self._graph.nodes(data=True) if d.get('type') == 'chunk'
        ]
        for new_cid in new_chunk_ids:
            concepts_new = set(self._graph.neighbors(new_cid))
            for existing_cid in all_chunk_ids_in_graph:
                if existing_cid == new_cid:
                    continue
                concepts_existing = set(self._graph.neighbors(existing_cid))
                common = concepts_new & concepts_existing
                if common:
                    weight = max(0.1, 1.0 / (1.0 + len(common)))
                    for s, t in [(new_cid, existing_cid), (existing_cid, new_cid)]:
                        if not self._graph.has_edge(s, t):
                            self._graph.add_edge(s, t, weight=weight,
                                                 relation='related_via_common_concepts')
                        else:
                            self._graph[s][t]['weight'] = min(
                                self._graph[s][t]['weight'], weight)

        # Step 4: Link new chunks to all existing chunks via semantic similarity
        for i, chunk in truly_new:
            new_cid = chunk.get('id', f'chunk_{i}')
            if i >= len(new_embeddings):
                continue
            new_emb = new_embeddings[i]
            norm_new = np.linalg.norm(new_emb)
            if norm_new == 0:
                continue
            for existing_cid, existing_data in self._graph.nodes(data=True):
                if existing_data.get('type') != 'chunk' or existing_cid == new_cid:
                    continue
                existing_emb = existing_data.get('embedding')
                if existing_emb is None:
                    continue
                norm_ex = np.linalg.norm(existing_emb)
                if norm_ex == 0:
                    continue
                similarity = float(np.dot(new_emb, existing_emb) / (norm_new * norm_ex + 1e-8))
                if similarity > 0.3:
                    distance = max(0.0, 1.0 - similarity)
                    for s, t in [(new_cid, existing_cid), (existing_cid, new_cid)]:
                        if self._graph.has_edge(s, t):
                            self._graph[s][t]['weight'] = min(
                                self._graph[s][t]['weight'], distance)
                        else:
                            self._graph.add_edge(s, t, weight=distance,
                                                 relation='semantic_similarity')

        logger.info(
            f"Graph updated: +{len(new_chunk_ids)} new chunks → "
            f"{self._graph.number_of_nodes()} nodes, {self._graph.number_of_edges()} edges"
        )
        return self._graph

    def compute_graph_distances(self, query_text: str, chunk_ids: List[str]) -> Dict[str, float]:
        if self._graph is None:
            raise ValueError("The graph is not built. Call build_graph() first.")

        query_concepts = self._extract_concepts_universal(query_text)
        query_concept_ids = [f"concept_{c['type']}_{c['label']}_{c['word']}" for c in query_concepts]

        # Keep only concepts that exist in the graph
        valid_concepts = [cid for cid in query_concept_ids if cid in self._graph]

        if not valid_concepts:
            return {chunk_id: 0.5 for chunk_id in chunk_ids}

        # Cache single-source Dijkstra per concept
        for concept_id in valid_concepts:
            if concept_id not in self._distance_cache:
                try:
                    self._distance_cache[concept_id] = dict(
                        nx.single_source_dijkstra_path_length(
                            self._graph, concept_id, cutoff=4.0, weight='weight'
                        )
                    )
                except Exception as e:
                    logger.warning(f"Dijkstra failed for concept {concept_id}: {e}")
                    self._distance_cache[concept_id] = {}

        # For each chunk, pick the minimum distance across all query concepts
        raw_distances = {}
        for chunk_id in chunk_ids:
            if chunk_id not in self._graph:
                raw_distances[chunk_id] = 999.0
                continue

            min_dist = float('inf')
            for concept_id in valid_concepts:
                d = self._distance_cache[concept_id].get(chunk_id, float('inf'))
                if d < min_dist:
                    min_dist = d

            raw_distances[chunk_id] = min_dist if min_dist != float('inf') else 999.0

        return self._normalize_distances(raw_distances)

    def _normalize_distances(self, distances: Dict[str, float]) -> Dict[str, float]:
        """
        Normalizes raw graph distances for consumption by ScoringService.
        """
        if not distances:
            return {}

        # Separate reachable chunks (real path found) from unreachable ones
        reachable = {k: v for k, v in distances.items() if v < 900.0}

        if not reachable:
            return {k: 0.5 for k in distances}

        min_d = min(reachable.values())
        max_d = max(reachable.values())

        UNREACHABLE = 1.5

        normalized = {}

        # Check the distance and make desicion of chunks
        for chunk_id, dist in distances.items():
            if dist >= 900.0:
                normalized[chunk_id] = UNREACHABLE
            elif max_d == min_d:
                normalized[chunk_id] = 0.0
            else:
                normalized[chunk_id] = (dist - min_d) / (max_d - min_d)

        return normalized

    def get_graph_statistics(self) -> Dict[str, Any]:
        if self._graph is None:
            return {}
        chunk_nodes = [n for n, d in self._graph.nodes(data=True) if d.get('type') == 'chunk']
        concept_nodes = [n for n, d in self._graph.nodes(data=True) if d.get('type') == 'concept']
        concept_types = defaultdict(int)
        for node_id in concept_nodes:
            concept_types[self._graph.nodes[node_id].get('concept_type', 'unknown')] += 1
        degrees = dict(self._graph.degree())
        avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
        return {
            'total_nodes': self._graph.number_of_nodes(),
            'total_edges': self._graph.number_of_edges(),
            'chunk_nodes': len(chunk_nodes),
            'concept_nodes': len(concept_nodes),
            'concept_types_breakdown': dict(concept_types),
            'average_degree': avg_degree,
            'is_directed': self._graph.is_directed(),
            'entity_to_chunks_mapping': len(self._entity_to_chunks),
        }

    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        if node_id not in self._graph:
            return None
        node_data = self._graph.nodes[node_id]
        neighbors = list(self._graph.neighbors(node_id))
        return {
            'id': node_id, 'data': node_data, 'neighbors': neighbors,
            'in_degree': self._graph.in_degree(node_id),
            'out_degree': self._graph.out_degree(node_id),
        }

    def get_neighboring_chunks_data(self, chunk_ids: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Return full chunk dicts for graph-adjacent chunks, ranked by edge weight (closest first).
        """
        if self._graph is None or not chunk_ids:
            return []
        chunk_ids_set = set(chunk_ids)
        candidate_scores: Dict[str, float] = {}
        for cid in chunk_ids:
            if cid not in self._graph:
                continue
            for neighbor in self._graph.neighbors(cid):
                if neighbor in chunk_ids_set:
                    continue
                if self._graph.nodes.get(neighbor, {}).get('type') != 'chunk':
                    continue
                weight = self._graph[cid][neighbor].get('weight', 0.5)
                if neighbor not in candidate_scores or weight < candidate_scores[neighbor]:
                    candidate_scores[neighbor] = weight
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1])[:top_k]
        result = []
        for cid, _ in sorted_candidates:
            node_data = self._graph.nodes[cid]
            text = node_data.get('text', '')
            if not text:
                continue
            result.append({
                'id': cid,
                'text': text,
                'metadata': node_data.get('metadata', {}),
                'similarity_score': 0.0,
            })
        return result

    def get_neighboring_chunk_ids(self, chunk_ids: List[str], top_k: int = 5) -> List[str]:
        """
        Return chunk-type graph neighbors of the given chunks, ranked by edge weight (closest first).
        """
        if self._graph is None or not chunk_ids:
            return []
        chunk_ids_set = set(chunk_ids)
        candidate_scores: Dict[str, float] = {}
        for cid in chunk_ids:
            if cid not in self._graph:
                continue
            for neighbor in self._graph.neighbors(cid):
                if neighbor in chunk_ids_set:
                    continue
                if self._graph.nodes.get(neighbor, {}).get('type') != 'chunk':
                    continue
                weight = self._graph[cid][neighbor].get('weight', 0.5)
                if neighbor not in candidate_scores or weight < candidate_scores[neighbor]:
                    candidate_scores[neighbor] = weight
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1])
        return [cid for cid, _ in sorted_candidates[:top_k]]

    def export_graph_info(self) -> Dict[str, Any]:
        if self._graph is None:
            return {}
        nodes = [
            {'id': n, 'type': d.get('type'), 'label': d.get('label'), 'text': d.get('text', d.get('word', ''))[:50]}
            for n, d in self._graph.nodes(data=True)
        ]
        edges = [
            {'source': u, 'target': v, 'weight': d.get('weight'), 'relation': d.get('relation')}
            for u, v, d in self._graph.edges(data=True)
        ]
        return {'nodes': nodes, 'edges': edges, 'statistics': self.get_graph_statistics()}
