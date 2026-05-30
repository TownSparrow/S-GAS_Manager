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
        self._embedding_service = None  # set externally for semantic node weights
        self._edge_top_p = 0.7  # keep only top-p strongest edges for traversal
        self._traversal_graph = None  # filtered view for Dijkstra/PageRank
        self._edge_weight_cutoff = float('inf')  # computed from top-p

        logger.info("KnowledgeGraphBuilder initialized with multi-language support")

    def set_embedding_service(self, embedding_service):
        """Inject embedding service for semantic node weighting and fuzzy matching."""
        self._embedding_service = embedding_service

    def set_edge_top_p(self, top_p: float):
        """Set edge top-p threshold. Only the strongest top_p fraction of edges
        will be used for Dijkstra/PageRank traversal."""
        self._edge_top_p = max(0.1, min(1.0, top_p))

    def _build_traversal_graph(self):
        """Build a filtered graph view that keeps only the top-p strongest edges."""
        if self._graph.number_of_edges() == 0:
            self._traversal_graph = self._graph
            self._edge_weight_cutoff = float('inf')
            return

        weights = [d.get('weight', 0.5) for _, _, d in self._graph.edges(data=True)]
        # top_p=0.7 means keep the 70% lowest-weight (strongest) edges
        cutoff = float(np.percentile(weights, self._edge_top_p * 100))
        self._edge_weight_cutoff = cutoff

        def edge_filter(u, v):
            return self._graph[u][v].get('weight', 0.5) <= cutoff

        self._traversal_graph = nx.subgraph_view(self._graph, filter_edge=edge_filter)
        # Invalidate Dijkstra cache — paths may differ on the filtered graph
        self._distance_cache.clear()
        kept = self._traversal_graph.number_of_edges()
        total = self._graph.number_of_edges()
        logger.info(
            f"Top-p edge filter: kept {kept}/{total} edges "
            f"(cutoff weight={cutoff:.3f}, top_p={self._edge_top_p})"
        )

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

        # Step 2b: Compute semantic weights for concept nodes.
        # Semantic weight = avg cosine similarity between the concept's word embedding and the embeddings of all chunks that contain it.
        # Higher weight = concept is more semantically grounded in actual content.
        if self._embedding_service:
            try:
                concept_nodes = [
                    (nid, d) for nid, d in self._graph.nodes(data=True)
                    if d.get('type') == 'concept' and d.get('word')
                ]
                if concept_nodes:
                    concept_words = [d['word'] for _, d in concept_nodes]
                    concept_embs = self._embedding_service.get_embeddings(concept_words)
                    for idx, (nid, _) in enumerate(concept_nodes):
                        # Compute avg similarity to linked chunks
                        linked_chunks = list(self._entity_to_chunks.get(nid, set()))
                        if linked_chunks and idx < len(concept_embs):
                            c_emb = concept_embs[idx]
                            c_norm = np.linalg.norm(c_emb)
                            if c_norm > 0:
                                sims = []
                                for chunk_id in linked_chunks:
                                    ch_data = self._graph.nodes.get(chunk_id, {})
                                    ch_emb = ch_data.get('embedding')
                                    if ch_emb is not None:
                                        ch_norm = np.linalg.norm(ch_emb)
                                        if ch_norm > 0:
                                            sims.append(float(np.dot(c_emb, ch_emb) / (c_norm * ch_norm)))
                                sem_weight = float(np.mean(sims)) if sims else 0.0
                                self._graph.nodes[nid]['semantic_weight'] = sem_weight
                                self._graph.nodes[nid]['embedding'] = concept_embs[idx]
                                # Adjust edge weights: lower weight for semantically grounded concepts (makes them easier to traverse)
                                if sem_weight > 0.3:
                                    adjusted = 0.3 * (1.0 - sem_weight * 0.5)
                                    for chunk_id in linked_chunks:
                                        if self._graph.has_edge(chunk_id, nid):
                                            self._graph[chunk_id][nid]['weight'] = adjusted
                                        if self._graph.has_edge(nid, chunk_id):
                                            self._graph[nid][chunk_id]['weight'] = adjusted
            except Exception as e:
                logger.debug(f"Semantic node weighting skipped: {e}")

        # Step 3: Link chunks via common concepts (require >=2 shared concepts to avoid noisy single-keyword connections)
        chunk_ids = [chunk.get('id', f'chunk_{i}') for i, chunk in enumerate(chunks)]
        for i, ci in enumerate(chunk_ids):
            for j, cj in enumerate(chunk_ids):
                if i >= j:
                    continue
                concepts_i = set(self._graph.neighbors(ci))
                concepts_j = set(self._graph.neighbors(cj))
                common = concepts_i & concepts_j
                if len(common) >= 2:
                    weight = max(0.1, 1.0 / (1.0 + len(common)))
                    self._graph.add_edge(ci, cj, weight=weight, relation='related_via_common_concepts')
                    self._graph.add_edge(cj, ci, weight=weight, relation='related_via_common_concepts')

        # Step 4: Link chunks via semantic similarity
        # Threshold 0.65 keeps only strongly similar pairs, preventing the graph from becoming overly dense with noise connections.
        for i, ci in enumerate(chunk_ids):
            for j, cj in enumerate(chunk_ids):
                if i >= j:
                    continue
                if i < len(embeddings) and j < len(embeddings):
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                    )
                    if similarity > 0.65:
                        distance = 1.0 - similarity
                        if distance < 0:
                            distance = 0.0

                        for s, t in [(ci, cj), (cj, ci)]:
                            if self._graph.has_edge(s, t):
                                self._graph[s][t]['weight'] = min(self._graph[s][t]['weight'], distance)
                            else:
                                self._graph.add_edge(s, t, weight=distance, relation='semantic_similarity')

        logger.info(f"Graph built: {self._graph.number_of_nodes()} nodes, {self._graph.number_of_edges()} edges")
        self._build_traversal_graph()
        return self._graph

    def update_graph(self, new_chunks: List[Dict[str, Any]], new_embeddings: np.ndarray) -> nx.DiGraph:
        """Adds new chunks to the existing graph"""
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
                if len(common) >= 2:
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
                if similarity > 0.65:
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
        self._build_traversal_graph()
        return self._graph

    def _compute_personalized_pagerank(self, valid_concepts: List[str], chunk_ids: List[str]) -> Dict[str, float]:
        """Computing Personalized PageRank with query concepts as seeds."""
        if not valid_concepts or self._graph.number_of_nodes() == 0:
            return {}

        # Using top-p filtered graph for PageRank to avoid authority leaking through weak/noisy edges.
        tg = self._traversal_graph if self._traversal_graph is not None else self._graph

        # Building personalization vector: uniform weight on query concept nodes
        personalization = {node: 0.0 for node in tg.nodes()}
        weight_per_concept = 1.0 / len(valid_concepts)
        for cid in valid_concepts:
            if cid in personalization:
                personalization[cid] = weight_per_concept

        try:
            pr = nx.pagerank(
                tg, alpha=0.85, personalization=personalization,
                max_iter=50, tol=1e-4, weight='weight',
            )
            # Extracting only chunk scores, normalize to [0, 1]
            chunk_scores = {cid: pr.get(cid, 0.0) for cid in chunk_ids}
            max_pr = max(chunk_scores.values()) if chunk_scores else 0.0
            if max_pr > 0:
                chunk_scores = {cid: s / max_pr for cid, s in chunk_scores.items()}
            return chunk_scores
        except Exception as e:
            logger.warning(f"Personalized PageRank failed: {e}")
            return {}

    def _fuzzy_match_concepts(self, query_concept_ids: List[str], embedding_service=None, threshold: float = 0.55) -> List[str]:
        """Find graph concept nodes semantically similar to query concepts."""
        exact = [cid for cid in query_concept_ids if cid in self._graph]
        if not embedding_service:
            return exact

        # Collect all concept nodes with their words
        graph_concepts = [
            (nid, d.get('word', ''))
            for nid, d in self._graph.nodes(data=True)
            if d.get('type') == 'concept' and d.get('word')
        ]
        if not graph_concepts:
            return exact

        # Extract query words from concept IDs (format: concept_TYPE_LABEL_word)
        missing_words = []
        missing_ids = []
        for cid in query_concept_ids:
            if cid not in self._graph:
                parts = cid.split('_', 3)
                word = parts[3] if len(parts) > 3 else cid
                missing_words.append(word)
                missing_ids.append(cid)

        if not missing_words:
            return exact

        try:
            query_embs = embedding_service.get_embeddings(missing_words)
            graph_words = [w for _, w in graph_concepts]
            graph_embs = embedding_service.get_embeddings(graph_words)

            q_norms = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-8)
            g_norms = graph_embs / (np.linalg.norm(graph_embs, axis=1, keepdims=True) + 1e-8)
            sim_matrix = np.dot(q_norms, g_norms.T)  # [n_query x n_graph]

            fuzzy_matched = set(exact)
            for i in range(len(missing_words)):
                best_idx = int(np.argmax(sim_matrix[i]))
                if sim_matrix[i][best_idx] >= threshold:
                    fuzzy_matched.add(graph_concepts[best_idx][0])
                    logger.debug(
                        f"Fuzzy match: '{missing_words[i]}' → '{graph_words[best_idx]}' "
                        f"(sim={sim_matrix[i][best_idx]:.3f})"
                    )
            return list(fuzzy_matched)
        except Exception as e:
            logger.debug(f"Fuzzy concept matching failed: {e}")
            return exact

    def compute_graph_distances(self, query_text: str, chunk_ids: List[str],
                                embedding_service=None) -> Dict[str, float]:
        if self._graph is None:
            raise ValueError("The graph is not built. Call build_graph() first.")

        query_concepts = self._extract_concepts_universal(query_text)
        query_concept_ids = [f"concept_{c['type']}_{c['label']}_{c['word']}" for c in query_concepts]

        # Try exact match first, then fuzzy match via embeddings
        valid_concepts = self._fuzzy_match_concepts(query_concept_ids, embedding_service)

        if not valid_concepts:
            logger.warning(
                f"No valid concepts extracted from query (text={query_text[:80]!r}). "
                f"Returning empty graph_distances so the scorer falls back to "
                f"semantic-only ranking."
            )
            return {}

        # Cache single-source Dijkstra per concept — run on the top-p filtered
        # graph so weak edges are not traversed (fewer hops, better precision).
        tg = self._traversal_graph if self._traversal_graph is not None else self._graph
        for concept_id in valid_concepts:
            if concept_id not in self._distance_cache:
                try:
                    self._distance_cache[concept_id] = dict(
                        nx.single_source_dijkstra_path_length(
                            tg, concept_id, cutoff=4.0, weight='weight'
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

        normalized = self._normalize_distances(raw_distances)

        # Blend Personalized PageRank authority scores into distances.
        # High PPR score → chunk is authoritative for the query → lower distance.
        ppr_scores = self._compute_personalized_pagerank(valid_concepts, chunk_ids)
        if ppr_scores:
            ppr_weight = 0.3  # 30% PPR influence on final distance
            for cid in chunk_ids:
                ppr = ppr_scores.get(cid, 0.0)
                # PPR reduces distance: high authority → lower effective distance
                normalized[cid] = normalized.get(cid, 0.8) * (1.0 - ppr_weight) + (1.0 - ppr) * ppr_weight

        return normalized

    def _normalize_distances(self, distances: Dict[str, float]) -> Dict[str, float]:
        """
        Normalizes raw graph distances for consumption by ScoringService.
        """
        if not distances:
            return {}

        # Separate reachable chunks (real path found) from unreachable ones
        reachable = {k: v for k, v in distances.items() if v < 900.0}

        if not reachable:
            logger.warning(
                f"No chunks reachable in graph from any query concept "
                f"({len(distances)} chunks unreachable). Returning empty dict "
                f"so the scorer falls back to semantic-only ranking."
            )
            return {}

        min_d = min(reachable.values())
        max_d = max(reachable.values())

        # Unreachable chunks get a high distance (0.8) so they are ranked lower
        # than graph-connected chunks by the hybrid scorer.  The semantic score
        # can still rescue truly relevant but graph-disconnected passages.
        UNREACHABLE_NEUTRAL = 0.8

        normalized = {}

        for chunk_id, dist in distances.items():
            if dist >= 900.0:
                normalized[chunk_id] = UNREACHABLE_NEUTRAL
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

    def get_neighboring_chunks_data(self, chunk_ids: List[str], top_k: int = 10,
                                     max_edge_weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        Return full chunk dicts for graph-adjacent chunks, ranked by edge weight (closest first).
        Uses the top-p filtered traversal graph so only strong edges are followed.
        """
        tg = self._traversal_graph if self._traversal_graph is not None else self._graph
        if tg is None or not chunk_ids:
            return []
        chunk_ids_set = set(chunk_ids)
        candidate_scores: Dict[str, float] = {}
        for cid in chunk_ids:
            if cid not in tg:
                continue
            for neighbor in tg.neighbors(cid):
                if neighbor in chunk_ids_set:
                    continue
                if self._graph.nodes.get(neighbor, {}).get('type') != 'chunk':
                    continue
                weight = tg[cid][neighbor].get('weight', 0.5)
                if weight > max_edge_weight:
                    continue
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
        Uses the top-p filtered traversal graph.
        """
        tg = self._traversal_graph if self._traversal_graph is not None else self._graph
        if tg is None or not chunk_ids:
            return []
        chunk_ids_set = set(chunk_ids)
        candidate_scores: Dict[str, float] = {}
        for cid in chunk_ids:
            if cid not in tg:
                continue
            for neighbor in tg.neighbors(cid):
                if neighbor in chunk_ids_set:
                    continue
                if self._graph.nodes.get(neighbor, {}).get('type') != 'chunk':
                    continue
                weight = tg[cid][neighbor].get('weight', 0.5)
                if neighbor not in candidate_scores or weight < candidate_scores[neighbor]:
                    candidate_scores[neighbor] = weight
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1])
        return [cid for cid, _ in sorted_candidates[:top_k]]

    def get_high_centrality_chunk_ids(self, top_pct: float = 0.15) -> List[str]:
        """Return chunk IDs in the top ``top_pct`` by degree centrality.

        These are the most "associative" chunks — keeping them in hot storage
        preserves the graph's ability to quickly recall related topics.
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            return []
        chunk_nodes = [n for n, d in self._graph.nodes(data=True) if d.get('type') == 'chunk']
        if not chunk_nodes:
            return []
        centrality = nx.degree_centrality(self._graph)
        chunk_centrality = [(cid, centrality.get(cid, 0.0)) for cid in chunk_nodes]
        chunk_centrality.sort(key=lambda x: x[1], reverse=True)
        keep_count = max(1, int(len(chunk_centrality) * top_pct))
        return [cid for cid, _ in chunk_centrality[:keep_count]]

    def reset_state(self) -> None:
        """Wipe the graph completely. Used between benchmark mode runs for clean isolation."""
        self._graph = nx.DiGraph()
        self._entity_to_chunks.clear()
        self._distance_cache.clear()
        self._traversal_graph = None
        self._edge_weight_cutoff = float('inf')
        logger.info("GraphService state fully reset for new benchmark run")

    def export_graph_info(self) -> Dict[str, Any]:
        if self._graph is None:
            return {}

        def edge_weight(data: Dict[str, Any]) -> float:
            weight = data.get('weight', 0.0)
            return float(weight if weight is not None else 0.0)

        nodes = [
            {'id': n, 'type': d.get('type'), 'label': d.get('label'), 'text': d.get('text', d.get('word', ''))[:50]}
            for n, d in self._graph.nodes(data=True)
        ]
        edges = [
            {'source': u, 'target': v, 'weight': edge_weight(d), 'relation': d.get('relation')}
            for u, v, d in self._graph.edges(data=True)
        ]
        return {'nodes': nodes, 'edges': edges, 'statistics': self.get_graph_statistics()}
