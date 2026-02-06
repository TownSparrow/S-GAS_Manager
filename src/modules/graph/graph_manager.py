import networkx as nx
import spacy
from typing import List, Dict, Any, Optional, Set
import logging
import numpy as np
import pymorphy3
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from collections import defaultdict


logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    def __init__(self, priority_model_for_en: str = "en_core_web_sm", priority_model_for_ru: str = "natasha", priority_kw_extractor_for_ru: str = "yake", priority_kw_extractor_for_en: str = "yake", use_gpu: bool = False):
        """
        Initialization of KnowladgeGraphBuilder
        """
        # TODO: In future try to adapt torch to use GPU for NER just in case

        self.priority_model_for_en = priority_model_for_en
        self.priority_model_for_ru = priority_model_for_ru
        self.priority_kw_extractor_for_ru = priority_kw_extractor_for_ru
        self.priority_kw_extractor_for_en = priority_kw_extractor_for_en

        # Loading pymorphy3 analyzer
        self.morph_analyzer = None
        try:
            self.morph_analyzer = pymorphy3.MorphAnalyzer()
            logger.info("✅ PyMorphy3 lemmatizer loaded globally")
        except ImportError:
            logger.warning("⚠️ PyMorphy3 not found globally. Lemmatization might be skipped by individual processors.")

        # Loading models for different languages
        self.nlp_en_fallback = None
        self.nlp_ru_fallback = None

        # EN Fallback
        spacy_model_en_fallback = self.priority_model_for_en
        try:
            self.nlp_en_fallback = spacy.load(spacy_model_en_fallback)
            logger.info(f"✅ spaCy English fallback model loaded: {spacy_model_en_fallback}")
        except OSError:
            logger.error(f"❌ Failed to load spaCy English fallback model: {spacy_model_en_fallback}")

        # RU Fallback
        spacy_model_ru_fallback = None
        if self.priority_model_for_ru != "natasha":
             spacy_model_ru_fallback = self.priority_model_for_ru
        else:
             spacy_model_ru_fallback = "ru_core_news_lg"

        try:
            self.nlp_ru_fallback = spacy.load(spacy_model_ru_fallback)
            logger.info(f"✅ spaCy Russian fallback model loaded: {spacy_model_ru_fallback}")
        except OSError:
            logger.error(f"❌ Failed to load spaCy Russian fallback model: {spacy_model_ru_fallback}")

        # Initialize of processors
        self.processor_en_ner = None
        self.processor_ru_ner = None
        self.processor_en_kw = None
        self.processor_ru_kw = None

        # Initialize of graph
        self.graph = nx.DiGraph()
        self.entity_to_chunks = defaultdict(set)
        self._distance_cache = {}

        logger.info("✅ KnowledgeGraphBuilder initialized with multi-language support")
    

    def _initialize_processor(self, lang: str, processor_type: str, priority_model_or_extractor: str):
        """
        Generalized initialization for NER or Keyword processors
        """
        attr_name = f'processor_{lang}_{processor_type}'
        if getattr(self, attr_name, None) is not None:
            return

        if processor_type == 'ner':
            if lang == 'ru' and priority_model_or_extractor == "natasha":
                from .processors.ner.ner_natasha_processor import NerNatashaProcessor
                setattr(self, attr_name, NerNatashaProcessor(morph_analyzer=self.morph_analyzer))
            elif lang in ['ru', 'en']: # Для spaCy
                 model_name = priority_model_or_extractor
                 from .processors.ner.ner_spacy_processor import NerSpacyProcessor
                 setattr(self, attr_name, NerSpacyProcessor(model_name=model_name, morph_analyzer=self.morph_analyzer))
        elif processor_type == 'kw':
            if lang == 'ru' and priority_model_or_extractor == "yake":
                from .processors.keyword.keyword_yake_processor import KeywordYakeProcessor
                setattr(self, attr_name, KeywordYakeProcessor(lang_code='ru'))
            elif lang == 'en' and priority_model_or_extractor == "yake":
                from .processors.keyword.keyword_yake_processor import KeywordYakeProcessor
                setattr(self, attr_name, KeywordYakeProcessor(lang_code='en'))


    def _detect_language(self, text:str) -> str:
        """
        Function for detecting lanugage in text.
        """
        try:
            detected_lang = detect(text)
            logger.info(f"Detected language: {detected_lang} for text: {text[:50]}...")
            if detected_lang in ['ru']:
                return 'ru'
            elif detected_lang in ['en']:
                return 'en'
            else:
                # English by default
                return 'en'
        except LangDetectException:
            logger.warning(f"⚠️ Could not detect language for text: {text[:50]}... Using 'en' as fallback.")
            return 'en'


    def _extract_concepts_universal(self, text: str) -> List[Dict[str, Any]]:
        """
        Universal function for detecting NER and Keywords.
        Combines NER results and keyword extraction results.
        """
        if not text.strip():
            return []

        lang = self._detect_language(text)
        concepts = []

        # NER Extraction
        self._initialize_processor(lang, 'ner', getattr(self, f'priority_model_for_{lang}'))
        processor_ner = getattr(self, f'processor_{lang}_ner')
        if processor_ner:
             entities_ner = processor_ner.process(text)
             concepts.extend(entities_ner)
        else:
            # Fallback: forward using spaCy 
            logger.warning(f"⚠️ NER Processor for {lang} not available, using spaCy fallback.")
            nlp_fallback = self.nlp_en_fallback if lang == 'en' else self.nlp_ru_fallback
            if nlp_fallback:
                doc_spacy = nlp_fallback(text)
                entities_ner = []
                for ent in doc_spacy.ents:
                    if len(ent.text.strip()) > 2:
                        word_normalized = ent.text.strip().lower()
                        if self.morph_analyzer:
                            try:
                                parsed = self.morph_analyzer.parse(word_normalized)[0]
                                word_normalized = parsed.normal_form.lower()
                            except:
                                pass
                        entities_ner.append({
                            'word': word_normalized,
                            'label': ent.label_,
                            'type': 'NER',
                            'start': ent.start_char,
                            'end': ent.end_char
                        })
                concepts.extend(entities_ner)

        # Keywords Extraction
        self._initialize_processor(lang, 'kw', getattr(self, f'priority_kw_extractor_for_{lang}'))
        processor_kw = getattr(self, f'processor_{lang}_kw')
        if processor_kw:
             keywords_raw = processor_kw.process(text, morph_analyzer=self.morph_analyzer)
             concepts.extend(keywords_raw)
        else:
            logger.warning(f"⚠️ Keyword Processor for {lang} not available.")

        # Removing duplicates
        unique_concepts = []
        seen = set()
        for concept in concepts:
            key = (concept['word'], concept['type'])
            if key not in seen:
                seen.add(key)
                unique_concepts.append(concept)

        logger.info(f"Extracted {len(unique_concepts)} unique concepts (NER + Keywords) for text snippet: '{text[:30]}...' in language: {lang}")
        return unique_concepts


    def build_graph(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> nx.Graph:
        """
        Main graph build function.
        Now includes nodes for both NER entities and Keywords from chunks.
        """
        # Creating a new directed graph
        self.graph = nx.DiGraph()
        self.entity_to_chunks.clear()
        self._distance_cache.clear()

        # Step 1: Adding nodes for each chunk
        logger.info("[Graph Builder] Step 1: Adding chunk nodes...")
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get('id', f'chunk_{i}')
            chunk_text = chunk.get('text', '')

            self.graph.add_node(
                chunk_id,
                type='chunk',
                text=chunk_text,
                embedding=embeddings[i] if i < len(embeddings) else None,
                metadata=chunk.get('metadata', {})
            )

        # Step 2: Extracting concepts (NER + Keywords) from each chunk and adding them as nodes
        logger.info("[Graph Builder] Step 2: Extracting concepts (NER + Keywords) from chunks and building graph...")

        concept_count = 0
        chunk_concept_links = 0

        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get('id', f'chunk_{i}')
            chunk_text = chunk.get('text', '')

            # Extract concepts (NER + Keywords) from chunk text
            chunk_concepts = self._extract_concepts_universal(chunk_text)

            for concept in chunk_concepts:
                concept_word = concept['word']
                concept_label = concept['label']
                concept_type = concept['type'] # 'NER' or 'KEYWORD'

                concept_id = f"concept_{concept_type}_{concept_label}_{concept_word}"

                # Adding a concept node if it doesn't exist yet
                if concept_id not in self.graph:
                    self.graph.add_node(
                        concept_id,
                        type='concept',
                        word=concept_word,
                        label=concept_label,
                        concept_type=concept_type
                    )
                    concept_count += 1

                # Linking the chunk to the concept (both directions)
                self.graph.add_edge(
                    chunk_id,
                    concept_id,
                    weight=0.3,
                    relation=f'contains_{concept_type.lower()}'
                )

                # Feedback for search (concept --> chunk)
                self.graph.add_edge(
                    concept_id,
                    chunk_id,
                    weight=0.3,
                    relation=f'found_in_chunk'
                )

                # Keeping in touch for quick access
                self.entity_to_chunks[concept_id].add(chunk_id)
                chunk_concept_links += 1

        logger.info(f"Concepts extracted and linked: {concept_count} concepts, {chunk_concept_links} links created.")

        # Step 3: Adding edges between chunks based on common concepts (was Step 3 for entities)
        logger.info("[Graph Builder] Step 3: Linking chunks via common concepts (NER + Keywords)...")

        chunk_ids = [chunk.get('id', f'chunk_{i}') for i, chunk in enumerate(chunks)]
        chunk_link_count = 0

        for i, chunk_i_id in enumerate(chunk_ids):
            for j, chunk_j_id in enumerate(chunk_ids):
                if i >= j:
                    continue

                # Finding common concepts between chunks
                concepts_i = set(self.graph.neighbors(chunk_i_id))
                concepts_j = set(self.graph.neighbors(chunk_j_id))
                common_concepts = concepts_i.intersection(concepts_j)

                if common_concepts:
                    # Weight = normalized by count (more common = stronger link)
                    weight = max(0.1, 1.0 / (1.0 + len(common_concepts)))

                    # Bidirectional edges for undirected relationships
                    self.graph.add_edge(
                        chunk_i_id,
                        chunk_j_id,
                        weight=weight,
                        relation='related_via_common_concepts'
                    )
                    self.graph.add_edge(
                        chunk_j_id,
                        chunk_i_id,
                        weight=weight,
                        relation='related_via_common_concepts'
                    )
                    chunk_link_count += 1

        logger.info(f"Chunk links via common concepts added: {chunk_link_count}")

        # Step 4: Adding edges between chunks based on semantic proximity (unchanged)
        logger.info("[Graph Builder] Step 4: Linking chunks via semantic similarity...")
        sim_link_count = 0

        for i, chunk_i_id in enumerate(chunk_ids):
            for j, chunk_j_id in enumerate(chunk_ids):
                if i >= j:
                    continue

                if i < len(embeddings) and j < len(embeddings):
                    emb_i = embeddings[i]
                    emb_j = embeddings[j]

                    similarity = np.dot(emb_i, emb_j) / (
                        np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-8
                    )
                    distance = 1.0 - similarity

                    if similarity > 0.3:
                        # Calculate distance based on similarity
                        distance = 1.0 - similarity
                        
                        # Protection against negative distances
                        if distance < 0:
                            logger.warning(f"Negative distance calculated for {chunk_i_id}<->{chunk_j_id}, similarity={similarity}. Clipping to 0.")
                            distance = 0.0

                        # Forward edge: chunk_i_id -> chunk_j_id (i < j)
                        if self.graph.has_edge(chunk_i_id, chunk_j_id):
                            current_weight = self.graph[chunk_i_id][chunk_j_id]['weight']
                            self.graph[chunk_i_id][chunk_j_id]['weight'] = min(
                                current_weight, distance
                            )
                        else:
                            self.graph.add_edge(
                                chunk_i_id,
                                chunk_j_id,
                                weight=distance,
                                relation='semantic_similarity'
                            )
                            sim_link_count += 1

                        # Reverse edge: chunk_j_id -> chunk_i_id (j > i)
                        if self.graph.has_edge(chunk_j_id, chunk_i_id):
                            current_weight = self.graph[chunk_j_id][chunk_i_id]['weight'] # <-- Уже исправлено ранее
                            # Rechecking for the reverse edge
                            self.graph[chunk_j_id][chunk_i_id]['weight'] = min(
                                current_weight, distance
                            )
                        else:
                            self.graph.add_edge(
                                chunk_j_id,
                                chunk_i_id,
                                weight=distance,
                                relation='semantic_similarity'
                            )

        logger.info(f"[Graph Builder] Semantic links added: {sim_link_count}")

        logger.info(
            f"✅ [Graph Builder] The graph is built: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

        return self.graph


    def compute_graph_distances(
        self,
        query_text: str,
        chunk_ids: List[str]
    ) -> Dict[str, float]:
        """
        Computing the graph distances based on combined NER and Keyword concepts.
        Finds shortest paths from query concepts to chunk concepts.
        """
        if self.graph is None:
            raise ValueError("❌ The graph is not built. Call build_graph() again.")

        # Step 1. Extracting query concepts (NER + Keywords)
        query_concepts = self._extract_concepts_universal(query_text)
        query_concept_ids = [f"concept_{c['type']}_{c['label']}_{c['word']}" for c in query_concepts]

        logger.debug(f"Query concepts extracted: {len(query_concept_ids)} concepts: {[c['word'] for c in query_concepts]}")

        # Handle case when no concepts found in query
        if not query_concept_ids:
            logger.warning(
                f"No concepts (NER/Keywords) found in query: '{query_text[:100]}...'. "
                "Using neutral graph scoring."
            )
            return {chunk_id: 0.5 for chunk_id in chunk_ids}

        # Step 2. Calculating raw distances via Dijkstra for each chunk
        graph_distances = {}

        for chunk_id in chunk_ids:
            if chunk_id not in self.graph:
                logger.warning(f"Chunk node {chunk_id} not found in graph during distance calculation. Assigning default distance 999.0.")
                graph_distances[chunk_id] = 999.0
                continue

            # Finding all concept IDs associated with this chunk
            chunk_neighbor_concepts = [n for n in self.graph.neighbors(chunk_id) if self.graph.nodes[n].get('type') == 'concept']

            min_distance = float('inf')

            # Search and check from query concepts to any of the chunk's concepts
            for query_concept_id in query_concept_ids:
                # Checking if query_concept_id exists in the graph before searching for paths
                if query_concept_id not in self.graph:
                    logger.debug(f"Query concept node {query_concept_id} not found in graph. Skipping.")
                    continue

                # Searching for paths only from existing query_concept_id
                for chunk_concept_id in chunk_neighbor_concepts:
                    try:
                        dist = nx.shortest_path_length(
                            self.graph,
                            source=query_concept_id,
                            target=chunk_concept_id,
                            weight='weight'
                        )
                        min_distance = min(min_distance, dist)
                        logger.debug(f"Path {query_concept_id}→{chunk_concept_id} (via {chunk_id}): {dist:.3f}")
                    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                        logger.debug(f"Path {query_concept_id}→{chunk_concept_id} (via {chunk_id}) failed: {e}")
                        continue # Try next pair
                    except Exception as e: # Ловим 'Contradictory paths' и другие
                        logger.error(f"Unexpected error finding path {query_concept_id}→{chunk_concept_id} (via {chunk_id}): {e}")
                        logger.exception("Detailed error for pathfinding:") # Выводит стек
                        min_distance = 999.0
                        break

                if min_distance == 999.0:
                    break

            # Store raw distance for the chunk
            if min_distance != float('inf'):
                graph_distances[chunk_id] = min_distance
            else:
                logger.debug(f"No path found from any query concept to any concept of chunk {chunk_id}. Assigning distance 999.0.")
                graph_distances[chunk_id] = 999.0 # No path found

        # Step 3. Normalizing distances to [0, 1]
        graph_scores = self._normalize_distances(graph_distances)

        logger.info(
            f"✅ Graph scores computed: "
            f"min={min(graph_scores.values()):.3f}, "
            f"max={max(graph_scores.values()):.3f}, "
            f"mean={np.mean(list(graph_scores.values())):.3f}"
        )

        return graph_scores


    def _normalize_distances(
            self,
            distances: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Normalizing the distances in graph
        """
        if not distances:
            return {}

        values = list(distances.values())
        min_dist = min(values)
        max_dist = max(values)

        # Handle case when all distances are equal or infinite
        if max_dist == min_dist or max_dist == 999.0:
            logger.debug("All distances equal or unreachable, using neutral scores")
            return {k: 0.5 for k in distances}

        normalized = {}
        # Normalize: shortest distance -> score 1.0, longest -> score 0.0
        range_dist = max_dist - min_dist
        for chunk_id, dist in distances.items():
            if dist == 999.0: # Unreachable
                 score = 0.0 # Or keep as 0.5?
            else:
                score = 1.0 - (dist - min_dist) / range_dist
            # Clamp score to [0, 1]
            score = max(0.0, min(1.0, score))
            normalized[chunk_id] = score

        return normalized


    def _extract_query_entities(self, query_text: str) -> Set[str]:
        """
        Extract query concepts via calling universal function.
        Returns a set of concept IDs for compatibility if needed elsewhere,
        but primarily the function _extract_concepts_universal is used internally.
        """
        logger.info(f"Entity linking from query using multi-language NER + Keywords")
        logger.info(f"Query text: '{query_text[:100]}{'...' if len(query_text) > 100 else ''}'")

        # Calling the universal function with collected concepts
        query_concepts = self._extract_concepts_universal(query_text)
        query_concept_ids = {f"concept_{c['type']}_{c['label']}_{c['word']}" for c in query_concepts}

        if not query_concept_ids:
             logger.warning(f"⚠️ Universal NER + Keywords found 0 concepts in query!")
        else:
             logger.info(f"✅ Universal NER + Keywords found {len(query_concept_ids)} concepts in query:")
             for idx, concept_id in enumerate(sorted(query_concept_ids)[:5], 1): # Показываем первые 5
                 logger.info(f"   [{idx}] {concept_id}")
             if len(query_concept_ids) > 5:
                 logger.info(f"   ... and {len(query_concept_ids) - 5} more.")

        # Final result
        if query_concept_ids:
            logger.info(f"✅ Result: {len(query_concept_ids)} concepts added to query_concept_ids")
            for ent_id in sorted(query_concept_ids):
                logger.info(f"   - {ent_id}")
        else:
            logger.warning(f"⚠️  Result: no concepts added! Will use fallback scoring (0.5)")
            logger.warning(f"   Possible causes:")
            logger.warning(f"   1. Universal NER + Keywords found concepts but all < 3 chars (unlikely for keywords)")
            logger.warning(f"   2. Universal NER + Keywords found no concepts at all")
            logger.warning(f"   3. Language detection failed / model unavailable")

        return query_concept_ids


    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Receiving the graph statistics
        """
        if self.graph is None:
            return {}

        chunk_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get('type') == 'chunk'
        ]

        concept_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get('type') == 'concept'
        ]

        concept_types = defaultdict(int)
        for node_id in concept_nodes:
             node_data = self.graph.nodes[node_id]
             concept_types[node_data.get('concept_type', 'unknown')] += 1

        degrees = dict(self.graph.degree())
        avg_degree = (
            sum(degrees.values()) / len(degrees)
            if degrees else 0
        )

        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'chunk_nodes': len(chunk_nodes),
            'concept_nodes': len(concept_nodes),
            'concept_types_breakdown': dict(concept_types),
            'average_degree': avg_degree,
            'is_directed': self.graph.is_directed(),
            'entity_to_chunks_mapping': len(self.entity_to_chunks)
        }
    

    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Recieving the info about exact node in the graph
        """
        if node_id not in self.graph:
            return None
        
        node_data = self.graph.nodes[node_id]
        neighbors = list(self.graph.neighbors(node_id))
        
        return {
            'id': node_id,
            'data': node_data,
            'neighbors': neighbors,
            'in_degree': self.graph.in_degree(node_id),
            'out_degree': self.graph.out_degree(node_id)
        }


    def export_graph_info(self) -> Dict[str, Any]:
        if self.graph is None:
            return {}
        
        nodes = [
            {
                'id': node,
                'type': data.get('type'),
                'label': data.get('label'),
                'text': data.get('text', '')[:50]  # First 50 chars
            }
            for node, data in self.graph.nodes(data=True)
        ]
        
        edges = [
            {
                'source': u,
                'target': v,
                'weight': data.get('weight'),
                'relation': data.get('relation')
            }
            for u, v, data in self.graph.edges(data=True)
        ]
        
        return {
            'nodes': nodes,
            'edges': edges,
            'statistics': self.get_graph_statistics()
        }