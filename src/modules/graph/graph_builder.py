import networkx as nx
import spacy
from typing import List, Dict, Any, Optional, Set
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    def __init__(self, spacy_model: str = "ru_core_news_md", use_gpu: bool = False):
        try:
            # Loading the spaCy model to extract entities
            self.nlp = spacy.load(spacy_model)
            logger.info(f"✅ spaCy model loaded: {spacy_model}")
            
            # Attempt to use GPU if specified
            if use_gpu:
                try:
                    spacy.require_gpu()
                    logger.info("✅ spaCy uses GPU")
                except Exception as e:
                    logger.warning(f"⚠️ GPU for spaCy is unavailable: {e}")
                    
        except OSError:
            # Fallback to the English model
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.warning("⚠️ The Russian model is unavailable; en_core_web_sm is in use.")
            except OSError:
                raise RuntimeError(
                    "❌ SpaCy model not found. Please install: "
                    "python -m spacy download ru_core_news_md\n"
                    "or: python -m spacy download en_core_web_sm"
                )
        
        self.graph = nx.DiGraph()
        self.entity_to_chunks = defaultdict(set)
        self._distance_cache = {}

        logger.debug("✅ KnowledgeGraphBuilder initialized")
        
    def build_graph(
        self, 
        chunks: List[Dict[str, Any]], 
        embeddings: np.ndarray
    ) -> nx.Graph:
        # Creating a new directed graph
        self.graph = nx.DiGraph()
        self.entity_to_chunks.clear()
        self._distance_cache.clear()
        
        # Step 1: Adding nodes for each chunk
        logger.debug("Step 1: Adding chunk nodes...")
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
            
        # Step 2: Extracting entities from each chunk
        logger.debug("Step 2: Extracting entities from chunks...")

        entity_count = 0

        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get('id', f'chunk_{i}')
            chunk_text = chunk.get('text', '')
            
            # Text processing via spaCy for NER
            doc = self.nlp(chunk_text)
            
            # Extracting named entities
            for ent in doc.ents:
                entity_text = ent.text.strip().lower()
                entity_label = ent.label_
                
                # Filtering out too short entities
                if len(entity_text) < 3:
                    continue
                    
                entity_id = f"entity_{entity_label}_{entity_text}"
                
                # Adding an entity node if it doesn't exist yet
                if entity_id not in self.graph:
                    self.graph.add_node(
                        entity_id,
                        type='entity',
                        text=entity_text,
                        label=entity_label
                    )
                    entity_count += 1
                
                # Linking the chunk to the entity (both directions)
                self.graph.add_edge(
                    chunk_id,
                    entity_id,
                    weight=0.3,
                    relation='contains_entity'
                )

                # Feedback for search (entity --> chunk)
                self.graph.add_edge(
                    entity_id,
                    chunk_id,
                    weight=0.3,
                    relation='found_in_chunk'
                )
                
                # Keeping in touch for quick access
                self.entity_to_chunks[entity_id].add(chunk_id)

        logger.debug(f"Entities extracted: {entity_count}")
        
        # Step 3: Adding edges between chunks based on common entities
        logger.debug("Step 3: Linking chunks via common entities...")

        chunk_ids = [chunk.get('id', f'chunk_{i}') for i, chunk in enumerate(chunks)]
        chunk_link_count = 0
        
        for i, chunk_i_id in enumerate(chunk_ids):
            for j, chunk_j_id in enumerate(chunk_ids):
                if i >= j:
                    continue
                    
                # Finding common entities between chunks
                entities_i = set(self.graph.neighbors(chunk_i_id))
                entities_j = set(self.graph.neighbors(chunk_j_id))
                common_entities = entities_i.intersection(entities_j)
                
                if common_entities:
                    # Weight = normalized by count (more common = stronger link)
                    weight = max(0.1, 1.0 / (1.0 + len(common_entities)))
                    
                    # Bidirectional edges for undirected relationships
                    # For first direction
                    self.graph.add_edge(
                        chunk_i_id,
                        chunk_j_id,
                        weight=weight,
                        relation='related_via_entities'
                    )

                    # For another direction
                    self.graph.add_edge(
                        chunk_j_id,
                        chunk_i_id,
                        weight=weight,
                        relation='related_via_entities'
                    )

                    chunk_link_count += 1

        logger.debug(f"Chunk links added: {chunk_link_count}")
        
        # Step 4: Adding edges between chunks based on semantic proximity
        logger.debug("Step 4: Linking chunks via semantic similarity...")
        sim_link_count = 0

        for i, chunk_i_id in enumerate(chunk_ids):
            for j, chunk_j_id in enumerate(chunk_ids):
                if i >= j:
                    continue
                
                # Calculating the cosine similarity of embeddingsов
                if i < len(embeddings) and j < len(embeddings):
                    emb_i = embeddings[i]
                    emb_j = embeddings[j]
                    
                    # Cosine similarity
                    similarity = np.dot(emb_i, emb_j) / (
                        np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-8
                    )
                    
                    # Converting to distance (weight for Dijkstra's algorithm)
                    distance = 1.0 - similarity
                    
                    # Adding an edge only if the similarity is high enough
                    if similarity > 0.3:
                        # Bidirectional semantic links
                        if self.graph.has_edge(chunk_i_id, chunk_j_id):
                            # Taking the minimum weight
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

                        # Reverse direction
                        if self.graph.has_edge(chunk_j_id, chunk_i_id):
                            current_weight = self.graph[chunk_j_id][chunk_i_id]['weight']
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

        logger.debug(f"Semantic links added: {sim_link_count}")
        
        logger.info(
            f"✅ The graph is built: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        
        return self.graph
    
    def _extract_query_entities(self, query_text: str) -> Set[str]:
        doc = self.nlp(query_text)
        query_entities = set()

        # Diagnostics
        logger.info(f"Entity linking from query")
        logger.info(f"Query text: '{query_text[:100]}{'...' if len(query_text) > 100 else ''}'")
        logger.info(f"Processing with spaCy NER...")

        # Show all entitites found
        if len(doc.ents) == 0:
            logger.warning(f"⚠️ NER found 0 entities!")
        else:
            logger.info(f"✅ Found {len(doc.ents)} entities by NER:")

        for ent in doc.ents:
            entity_text = ent.text.strip().lower()
            entity_label = ent.label_
            
            # Filter short entities
            if len(entity_text) >= 3:
                entity_id = f"entity_{entity_label}_{entity_text}"
                query_entities.add(entity_id)
                logger.debug(f"Query entity found: {entity_id}")
        
        for idx, ent in enumerate(doc.ents, 1):
            entity_text = ent.text.strip().lower()
            entity_label = ent.label_
        
            # Show each entity
            logger.info(f"   [{idx}] '{ent.text}' → label={entity_label}, text='{entity_text}', len={len(entity_text)}")
        
            # Filter short entities
            if len(entity_text) < 3:
                logger.info(f"       ⚠️  Skipped: too short (len < 3)")
                continue
        
            entity_id = f"entity_{entity_label}_{entity_text}"
            query_entities.add(entity_id)
            logger.info(f"       ✅ Added to query_entities: {entity_id}")
        
        # Final result
        if query_entities:
            logger.info(f"✅ Result: {len(query_entities)} entities added to query_entities")
            for ent_id in sorted(query_entities):
                logger.info(f"   - {ent_id}")
        else:
            logger.warning(f"⚠️  Result: no entities added! Will use fallback scoring (0.5)")
            logger.warning(f"   Possible causes:")
            logger.warning(f"   1. NER found entities but all < 3 chars")
            logger.warning(f"   2. NER found no entities at all")
            logger.warning(f"   3. spaCy model problem")
        
        return query_entities

    def compute_graph_distances(
        self, 
        query_text: str, 
        chunk_ids: List[str]
    ) -> Dict[str, float]:
        if self.graph is None:
            raise ValueError("❌ The graph is not built. Call build_graph() again.")
        
        # Step 1. Extracting query entities
        query_entities = self._extract_query_entities(query_text)
        
        logger.debug(f"Query entities extracted: {query_entities}")
        
        # Handle case when no entities found in query
        if not query_entities:
            logger.warning(
                f"No entities found in query: '{query_text[:100]}...'. "
                "Using neutral graph scoring."
            )
            return {chunk_id: 0.5 for chunk_id in chunk_ids}
        
        # Step 2. Calculating raw distances via Dijkstra
        graph_distances = {}

        for chunk_id in chunk_ids:
            if chunk_id not in self.graph:
                graph_distances[chunk_id] = 999.0
                continue
            
            min_distance = float('inf')

            # Search from query entities
            for query_entity in query_entities:
                if query_entity not in self.graph:
                    continue

                # Direction 1: query_entity --> chunk_id
                try:
                    dist = nx.shortest_path_length(
                        self.graph,
                        source=query_entity,
                        target=chunk_id,
                        weight='weight'
                    )
                    min_distance = min(min_distance, dist)
                    logger.debug(f"Path {query_entity}→{chunk_id}: {dist:.3f}")
                
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

                # Direction 2: chunk_id --> query_entity (bidirectional)
                try:
                    dist = nx.shortest_path_length(
                        self.graph,
                        source=chunk_id,
                        target=query_entity,
                        weight='weight'
                    )
                    min_distance = min(min_distance, dist)
                    logger.debug(f"Path {chunk_id}→{query_entity}: {dist:.3f}")
                
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

            # Store raw distance
            if min_distance != float('inf'):
                graph_distances[chunk_id] = min_distance
            else:
                graph_distances[chunk_id] = 999.0

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
        if not distances:
            return {}
        
        values = list(distances.values())
        min_dist = min(values)
        max_dist = max(values)

        # Handle case when all distances are equal
        if max_dist == min_dist:
            logger.debug("All distances equal, using neutral scores")
            return {k: 0.5 for k in distances}
        
        normalized = {}

        for chunk_id, dist in distances.items():
            # Normalization: distance --> score
            # min_dist --> 1.0, max_dist --> 0.0
            score = 1.0 - ((dist - min_dist) / (max_dist - min_dist))
            
            # Clamping to [0, 1] just in case
            score = max(0.0, min(1.0, score))
            normalized[chunk_id] = score
        
        return normalized
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        if self.graph is None:
            return {}
        
        chunk_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get('type') == 'chunk'
        ]

        entity_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get('type') == 'entity'
        ]

        degrees = dict(self.graph.degree())
        avg_degree = (
            sum(degrees.values()) / len(degrees)
            if degrees else 0
        )

        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'chunk_nodes': len(chunk_nodes),
            'entity_nodes': len(entity_nodes),
            'average_degree': avg_degree,
            'is_directed': self.graph.is_directed(),
            'entity_to_chunks_mapping': len(self.entity_to_chunks)
        }
    
    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
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
    