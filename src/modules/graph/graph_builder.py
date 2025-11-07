import networkx as nx
import spacy
from typing import List, Dict, Any, Tuple, Optional
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    
    def __init__(self, spacy_model: str = "ru_core_news_md", use_gpu: bool = False):
        try:
            # Loading the spaCy model to extract entities
            self.nlp = spacy.load(spacy_model)
            logger.info(f"spaCy model loaded: {spacy_model}")
            
            # Attempt to use GPU if specified
            if use_gpu:
                try:
                    spacy.require_gpu()
                    logger.info("spaCy uses GPU")
                except Exception as e:
                    logger.warning(f"GPU for spaCy is unavailable: {e}")
                    
        except OSError:
            # Fallback to the English model
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.warning("The Russian model is unavailable; en_core_web_sm is in use.")
            except OSError:
                raise RuntimeError(
                    "SpaCy model not found. Please install: "
                    "python -m spacy download ru_core_news_md"
                )
        
        self.graph = None
        self.entity_to_chunks = defaultdict(set)  # entity -> set of chunk_ids
        
    def build_graph(
        self, 
        chunks: List[Dict[str, Any]], 
        embeddings: np.ndarray
    ) -> nx.Graph:
        # Creating a new directed graph
        self.graph = nx.Graph()
        self.entity_to_chunks.clear()
        
        # Step 1: Adding nodes for each chunk
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
                
                # Linking the chunk to the entity
                # Edge weight = cosine distance (1 - similarity)
                self.graph.add_edge(
                    chunk_id, 
                    entity_id,
                    weight=0.5,  # Base weight for chunk-entity relationship
                    relation='contains'
                )
                
                # Keeping in touch for quick access
                self.entity_to_chunks[entity_id].add(chunk_id)
        
        # Step 3: Adding edges between chunks based on common entities
        chunk_ids = [chunk.get('id', f'chunk_{i}') for i, chunk in enumerate(chunks)]
        
        for i, chunk_i_id in enumerate(chunk_ids):
            for j, chunk_j_id in enumerate(chunk_ids):
                if i >= j:
                    continue
                    
                # Finding common entities between chunks
                entities_i = set(self.graph.neighbors(chunk_i_id))
                entities_j = set(self.graph.neighbors(chunk_j_id))
                
                common_entities = entities_i.intersection(entities_j)
                
                if common_entities:
                    # Weight = reciprocal of number of common entities
                    # (the more common entities, the closer the chunks)
                    weight = 1.0 / len(common_entities)
                    
                    # Adding an edge between chunks if they are connected
                    self.graph.add_edge(
                        chunk_i_id,
                        chunk_j_id,
                        weight=weight,
                        relation='related_via_entities'
                    )
        
        # Step 4: Adding edges between chunks based on semantic proximity
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
                    distance = 1 - similarity
                    
                    # Adding an edge only if the similarity is high enough
                    if similarity > 0.3:  # The threshold can be adjusted
                        # Updating the weight if the edge already exists
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
        
        logger.info(
            f"The graph is built: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        
        return self.graph
    
    def compute_graph_distances(
        self, 
        query_text: str, 
        chunk_ids: List[str]
    ) -> Dict[str, float]:
        if self.graph is None:
            raise ValueError("The graph is not built. Call build_graph() again.")
        
        # Extracting entities from a query
        doc = self.nlp(query_text)
        query_entities = set()
        
        for ent in doc.ents:
            entity_text = ent.text.strip().lower()
            entity_label = ent.label_
            
            if len(entity_text) >= 3:
                entity_id = f"entity_{entity_label}_{entity_text}"
                query_entities.add(entity_id)
        
        # Returning large distances if entities are not found
        if not query_entities:
            logger.warning("The entities in the query were not found, graph distances will be large")
            return {chunk_id: 1000.0 for chunk_id in chunk_ids}
        
        # Calculating the distances from each query entity to the chunks
        graph_distances = {}
        
        for chunk_id in chunk_ids:
            if chunk_id not in self.graph:
                graph_distances[chunk_id] = 1000.0
                continue
            
            # Finding the minimum distance from any query entity to a chunk
            min_distance = float('inf')
            
            for query_entity in query_entities:
                if query_entity not in self.graph:
                    continue
                
                try:
                    # Dijkstra's algorithm for finding the shortest path
                    path_length = nx.dijkstra_path_length(
                        self.graph, 
                        source=query_entity, 
                        target=chunk_id,
                        weight='weight'
                    )
                    min_distance = min(min_distance, path_length)
                    
                except nx.NetworkXNoPath:
                    continue
                except nx.NodeNotFound:
                    continue
            
            # If the path is found from at least one entity
            if min_distance != float('inf'):
                graph_distances[chunk_id] = min_distance
            else:
                # Settting a large distance if there are no paths
                graph_distances[chunk_id] = 1000.0
        
        logger.info(f"Graph distances were calculated for {len(graph_distances)} chunks")
        
        return graph_distances
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        if self.graph is None:
            return {}
        
        chunk_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'chunk']
        entity_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'entity']
        
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'chunk_nodes': len(chunk_nodes),
            'entity_nodes': len(entity_nodes),
            'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            'is_connected': nx.is_connected(self.graph) if len(chunk_nodes) > 0 else False
        }