import uuid
import time
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Set

#from curses import meta, tparm
#import imp

#from nt import unlink
#from re import search
#from socketserver import _RequestType
#import stat

import numpy as np
import torch
import torch.cuda
import json
import httpx
import logging
import shutil
import aiofiles

#from networkx import topological_generations
#from networkx.algorithms.community.quality import require_partition

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Configs and modules
from src.config import settings
from src.modules.retrieval import vector_store
from src.modules.retrieval.vector_store import ChromaVectorStore
from src.modules.retrieval.chunking import SemanticChunker
from src.modules.retrieval.document_processor import DocumentProcessor
from src.modules.retrieval.embedder import embedding_manager, get_embeddings

from src.modules.retrieval.retrieval_models import (
    DocumentHeader,
    DocumentProcessingResult,
    SessionDocumentMetadata
)

from src.modules.graph.graph_manager import KnowledgeGraphBuilder
from src.core.scoring import HybridScorer
from src.modules.swap.swap_manager import SwapManager

from src.modules.monitoring.kv_monitor import KVCacheMonitor

# ============================================================================
# Init of app and basic services
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOADS_DIR = Path("data/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

VERSION = "v0.1.0-alpha.3"

# ============================================================================
# Memory processing
# ============================================================================

def log_gpu_memory_detailed(stage: str, iteration: int):
    """Log GPU memory state in detail for swap monitoring"""
    if not torch.cuda.is_available():
        logger.debug(f"[Iteration {iteration}] {stage}: CUDA not available, skipping")
        return
    
    try:
        gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        gpu_free = gpu_total - gpu_allocated
        gpu_max = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        logger.info(f"""
[Iteration {iteration}] {stage} GPU STATE:
├─ Allocated: {gpu_allocated:.2f} MB / {gpu_total:.2f} MB ({gpu_allocated/gpu_total*100:.1f}%)
├─ Reserved: {gpu_reserved:.2f} MB
├─ Free: {gpu_free:.2f} MB
└─ Peak: {gpu_max:.2f} MB""")
    except Exception as e:
        logger.warning(f"⚠️ Failed to log GPU memory: {e}")

async def get_all_session_chunks(session_id: str) -> List[Dict[str, Any]]:
    """Get all chunks from the session for GPU initialization"""
    try:
        all_chunks = await app.state.vector_store.get_all_session_chunks(session_id)

        if not all_chunks:
            logger.warning(f"⚠️ No chunks found for session {session_id}")
            return []

        estimated_memory = len(all_chunks) * (384 * 4 + 2000) / 1024 / 1024
        logger.info(f"✅ Retrieved {len(all_chunks)} total chunks for CPU archive initialization")
        logger.info(f"   Estimated archive size: ~{estimated_memory:.2f} MB")
        
        return all_chunks
    
    except Exception as e:
        logger.error(f"❌ Failed to get all chunks: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services at startup"""

    logger.info("Starting S-GAS Manager API initialization...")
    
    try:
        # Initialize vector store
        vector_store = ChromaVectorStore(
            persist_directory=settings['database']['chroma_persist_dir'],
            collection_name=settings['database'].get('collection_name', 'documents'),
        )
        logger.info("✅ ChromaVectorStore initialized")
        
        # Initialize chunker with adaptive tracking
        chunker = SemanticChunker(
            max_chunk_size=settings['chunking']['max_chunk_size'],
            overlap_size=settings['chunking']['overlap_size'],
        )
        logger.info("✅ SemanticChunker initialized")
        
        # Initialize document processor
        document_processor = DocumentProcessor(
            vector_store=vector_store,
            chunker=chunker
        )
        logger.info("✅ DocumentProcessor initialized")
        
        # Initialize Graph Builder
        graph_builder = KnowledgeGraphBuilder(
            priority_model_for_en="en_core_web_sm",
            priority_model_for_ru="natasha",
            priority_kw_extractor_for_ru="yake",
            priority_kw_extractor_for_en="yake",
            use_gpu=False # True if GPU is requiered
        )

        
        logger.info("✅ KnowledgeGraphBuilder initialized")
        
        # Initialize Hybrid Scorer
        hybrid_scorer = HybridScorer(
            alpha=settings['graph']['alpha'],
            beta=settings['graph']['beta']
        )
        logger.info("✅ HybridScorer initialized")
        
        # Initialize Swap Manager
        swap_manager = SwapManager(
            threshold=settings['swap']['threshold'],
            prefetch_count=settings['swap']['prefetch_count'],
            memory_check_interval_ms=settings['swap']['memory_check_interval'],
            max_gpu_memory_tokens=settings['vllm']['max_model_len'],
            debug_mode=settings['swap'].get('debug_mode', False),
            force_offload_on_iteration=settings['swap'].get('force_offload_on_iteration', -1)
        )
        logger.info("✅ SwapManager initialized")

        # Initialize KV Cahce Monitor
        kv_monitor = KVCacheMonitor()
        logger.info("✅ KVCacheMonitor initialized")
        
        # Store all components in app.state
        app.state.vector_store = vector_store
        app.state.chunker = chunker
        app.state.document_processor = document_processor
        app.state.graph_builder = graph_builder
        app.state.hybrid_scorer = hybrid_scorer
        app.state.swap_manager = swap_manager
        app.state.kv_monitor = kv_monitor
        
        # Initialize iteration tracking
        app.state.sessions = {}
        app.state._swap_initialized = False
        
        logger.info("✅ All S-GAS components initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("Cleaning up resources...")
    app.state.vector_store = None
    app.state.chunker = None
    app.state.document_processor = None
    app.state.graph_builder = None
    app.state.hybrid_scorer = None
    app.state.swap_manager = None
    app.state.kv_monitor = None


# Create FastAPI app with lifespan
app = FastAPI(
    title="S-GAS Manager API",
    description="Semantic-Graph Adaptive Swapping (S-GAS) for Small Language Models",
    version=VERSION,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        f"http://{settings['api']['host']}:{settings['api']['port']}",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ============================================================================
# Data Models
# ============================================================================

class SessionCreateRequest(BaseModel):
    user_id: Optional[str] = None


class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class SearchResponse(BaseModel):
    query: str
    session_id: str
    results: List[Dict[str, Any]]


class ChatRequest(BaseModel):
    """Chat request with S-GAS parameters"""

    message: str
    use_rag: bool = True
    n_chunks: int = 5
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    """Chat response with full S-GAS statistics"""

    response: str
    metadata: Optional[Dict[str, Any]] = None


# ============================================================================
# Helper Functions
# ============================================================================


def build_prompt(
    context_chunks: List[Dict[str, Any]],
    user_message: str,
    max_context_tokens: Optional[int] = None,
    enable_context_limit: Optional[bool] = None
    ) -> str:
    """Build prompt with context"""

    # Getting defaults from config if not provided
    if enable_context_limit is None:
        enable_context_limit = settings['prompt'].get('enable_context_limit', True)
    if max_context_tokens is None:
        max_context_tokens = settings['prompt'].get('max_context_tokens', 5000)

    if context_chunks:
        context_text = ""
        token_count = 0

        for chunk in context_chunks:
            chunk_text = chunk.get("text") or chunk.get("document") or ""
            chunk_tokens = len(chunk_text) // 4

            # Applying limit only if enabled
            if enable_context_limit:
                if token_count + chunk_tokens <= max_context_tokens:
                    context_text += chunk_text + "\n\n"
                    token_count += chunk_tokens
                else:
                    break
            else:
                # No limit: addding all chunks
                context_text += chunk_text + "\n\n"

        return (
            f"Context from the knowledge base:\n{context_text}\n\n"
            f"User's request: {user_message}\n\n"
            f"Instructions: Analyze the context and give a brief, precise, full and specific answer based on the fragments provided. "
            f"If the context lacks information, state so and respond briefly using general knowledge. Use the same language as the user uses.\n\n"
            f"Response:"
        )
    else:
        return (
            f"User's request: {user_message}\n\n"
            f"Give a short and precise answer. Use the same language as the user uses.\n\n"
            f"Response:"
        )


async def call_vllm_chat(prompt: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Call vLLM API"""

    api_base = settings['vllm']['api_base']
    model_name = settings['vllm']['model_name']
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": overrides.get("temperature", settings['vllm']['temperature']),
        "max_tokens": overrides.get("max_tokens", settings['vllm']['max_tokens']),
        "top_p": overrides.get("top_p", settings['vllm']['top_p']),
        "stop": ["\n\nRequest:", "\n\nUser:", "\n\nAssistant:"],
        "stream": False,
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{api_base}/chat/completions", json=payload)
            
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"❌ vLLM error {resp.status_code}: {resp.text}"
                )
            
            data = resp.json()
            choices = data.get("choices")
            
            if not isinstance(choices, list) or not choices or "message" not in choices[0]:
                raise HTTPException(
                    status_code=502,
                    detail=f"❌ vLLM malformed response: {data}"
                )
            
            if "content" not in choices[0]["message"]:
                raise HTTPException(
                    status_code=502,
                    detail=f"❌ vLLM response without content: {data}"
                )
            
            return data
    
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"❌ vLLM unavailable: {str(e)}"
        )

# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/")
async def serve_web_client():
    """Serve web client or show API info"""

    static_index = STATIC_DIR / "index.html"
    if static_index.exists():
        return FileResponse(str(static_index))
    
    return JSONResponse(
        {
            "message": "S-GAS Manager API",
            "version": VERSION,
            "endpoints": [
                "/api/session/new",
                "/api/session/{session_id}/upload-document",
                "/api/session/{session_id}/search",
                "/api/session/{session_id}/chat",
                "/api/session/{session_id}/clear",
                "/health",
            ],
            "web_client": "No files in src/web/static/",
        },
        status_code=200,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings['vllm']['api_base']}/models")
            vllm_status = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        vllm_status = "unavailable"

    vector_store_status = "ready" if app.state.vector_store is not None else "not_initialized"
    chunker_status = "ready" if app.state.chunker is not None else "not_initialized"

    return {
        "status": "healthy",
        "vllm_status": vllm_status,
        "vector_store_status": vector_store_status,
        "chunker_status": chunker_status,
        "model": settings['vllm']['model_name'],
        "active_sessions": len(app.state.sessions),
        "time": datetime.now(timezone.utc).isoformat(),
        "api_version": VERSION,
    }


@app.post("/api/session/new", response_model=SessionCreateResponse)
async def create_new_session(request: SessionCreateRequest):
    """Create a new session"""

    session_id = f"session_{uuid.uuid4().hex[:12]}"

    app.state.sessions[session_id] = {
        'iteration': 0,
        'excluded_chunks': [],
        'created_at': datetime.now(timezone.utc).isoformat(),
        'user_id': request.user_id if hasattr(request, 'user_id') else None,
        'swap_initialized': False,
    }
    
    logger.info(f"✅ Created new session: {session_id}")

    return SessionCreateResponse(
        session_id=session_id,
        created_at=datetime.now(timezone.utc).isoformat()
    )


@app.post("/api/session/{session_id}/upload-document")
async def upload_document(
    session_id: str,
    file: UploadFile,
    document_type: str = "general"
):
    """Upload a document to a session"""

    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="❌ Session not found")

    # Save temporary file
    temp_path = Path(f"/tmp/{file.filename}")
    
    try:
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(await file.read())
    
        result = app.state.document_processor.process_document(
            temp_path,
            session_id=session_id,
            metadata={"document_type": document_type}
        )

        if result.status == 'success':
            logger.info(f"✅ Document uploaded to session {session_id}: {file.filename}")
            logger.info(f"   UUID: {result.document_uuid}")
            logger.info(f"   Chunks: {result.chunks_created}")
            
            if result.previous_uuid:
                logger.info(f"   Updated from version (prev_uuid={result.previous_uuid[:8]}...)")
        
        else:
            logger.error(f"❌ Document upload FAILED for {file.filename}")
            logger.error(f"   Error: {result.error}")
            if result.error_details:
                logger.error(f"   Details: {result.error_details}")
        

        return result.to_dict()
    
    except Exception as e:
        logger.error(f"❌ Unexpected error uploading document: {e}")
        return DocumentProcessingResult(
            status='error',
            document_uuid='',
            document_name=file.filename,
            session_id=session_id,
            chunks_created=0,
            error=str(e),
            error_details={'exception_type': type(e).__name__}
        ).to_dict()
    
    finally:
        # Delete temporary file
        if temp_path.exists():
            temp_path.unlink()
            

@app.post("/api/session/{session_id}/search", response_model=SearchResponse)
async def search_documents(session_id: str, request: SearchRequest):
    """Search documents in session context"""

    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="❌ Session not found")

    logger.info(f"Searching in session {session_id}: {request.query[:50]}...")

    # Embed query
    query_embeddings = get_embeddings([request.query])
    query_embedding = query_embeddings[0]

    # Search with session isolation
    results = await app.state.vector_store.search(
        query_embedding,
        session_id=session_id,
        top_k=request.top_k
    )

    logger.info(f"✅ Found {len(results)} chunks")

    return SearchResponse(
        query=request.query,
        session_id=session_id,
        results=results
    )


@app.delete("/api/session/{session_id}/clear")
async def clear_session(session_id: str):
    """Clear a session (delete all chunks and reset state)"""

    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="❌ Session not found")
    
    logger.info(f"Clearing session {session_id}...")

    # Delete chunks from vector store
    result = app.state.vector_store.delete_session_chunks(session_id)

    # Reset chunker tracking
    app.state.chunker.reset_session_tracking(session_id)

    # Clear the session data in processor
    app.state.document_processor.clear_session(session_id)

    # Remove session from tracking
    app.state.sessions.pop(session_id)

    logger.info(f"✅ Session {session_id} cleared: {result['deleted']} chunks deleted")

    return {
        "status": "success",
        "cleared_chunks": result['deleted'],
        "session_id": session_id,
        "message": "Session completely cleared"
    }


@app.post("/api/session/{session_id}/chat")
async def chat_endpoint(session_id: str, request: ChatRequest):
    """
    Chat with S-GAS adaptive chunking
    
    1. Get excluded chunks from previous iterations (if not the first iteration)
    2. Search for NEW chunks excluding the old ones
    3. Graph analysis and Hybrid reranking
    4. Swap manager initialization (only if first iteration)
    5. Memory swapping management
    6. Mark used chunks for the next iteration
    7. LLM Inference
    """
    
    # ════════════════════════════════════════════════════════════════════════════
    # STEP 1: Initialize iteration and get excluded chunks
    # ════════════════════════════════════════════════════════════════════════════
    
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="❌ Session not found")
    
    session_data = app.state.sessions[session_id]
    
    # Increment iteration for this session
    session_data['iteration'] += 1
    iteration = session_data['iteration']
    
    logger.info(f"Session {session_id} | Iteration {iteration}: Query: {request.message[:50]}...")
    
    # Get chunks excluded in previous iterations using session state
    excluded_chunks = session_data.get('excluded_chunks', [])
    logger.info(f"Excluding {len(excluded_chunks)} chunks from previous iterations")
    
    # ════════════════════════════════════════════════════════════════════════════
    # STEP 2: Search for chunks with exclusion
    # ════════════════════════════════════════════════════════════════════════════
    
    context_chunks = []
    query_embedding_vec = None

    if request.use_rag:
        try:
            # Embed query once
            query_embeddings = get_embeddings([request.message])
            query_embedding_vec = query_embeddings[0]
            
            initial_top_k = max(20, request.n_chunks * 3)
            search_k = (initial_top_k * 2) if excluded_chunks else initial_top_k
        
            if excluded_chunks:
                excluded_set = set(excluded_chunks)

                # Searching for more chunks
                candidates = await app.state.vector_store.search(
                    query_embedding=query_embedding_vec,
                    session_id=session_id,
                    top_k=search_k
                )

                # Filtering out excluded chunks
                context_chunks = [
                    c for c in candidates
                    if c.get('id') not in excluded_set
                ][:initial_top_k]

                logger.info(f"✅ Retrieved {len(candidates)} candidates, filtered to {len(context_chunks)} new chunks")
            
            else:
                # First iteration is just searching
                context_chunks = await app.state.vector_store.search(
                    query_embedding=query_embedding_vec,
                    session_id=session_id,
                    top_k=initial_top_k
                )

                logger.info(f"✅ Iteration 1: Got {len(context_chunks)} chunks")
        
        except Exception as e:
            logger.warning(f"❌ RAG search failed: {e}")
            context_chunks = []
        
    # ════════════════════════════════════════════════════════════════════════════
    # STEP 3: Knowledge graph building and analysis
    # ════════════════════════════════════════════════════════════════════════════
    
    graph_distances = {}
    chunk_embeddings = []
    final_context = context_chunks  # Default if no graph

    if request.use_rag and len(context_chunks) > 0:
        try:
            logger.info("Stage 1: Building Knowledge Graph from retrieved chunks...")

            # Getting chunk texts and embeddings for graph
            
            # Sanitize and validate chunk texts
            chunk_texts = [
                (c.get('text') or '').strip()[:5000]  # Max 5000 chars per chunk
                for c in context_chunks
            ]

            non_empty_texts = [t for t in chunk_texts if t]

            if not non_empty_texts:
                logger.warning("⚠️ All chunk texts are empty!")
                chunk_embeddings = np.zeros((len(context_chunks), 384))
                logger.warning("   Using zero embeddings as fallback")
            else:
                try:
                    chunk_embeddings = get_embeddings(chunk_texts)

                    if isinstance(chunk_embeddings, np.ndarray):
                        nan_count = np.isnan(chunk_embeddings).sum()

                    if nan_count > 0:
                        logger.warning(f"⚠️ Found {nan_count} NaN values in embeddings!")
                        chunk_embeddings = np.nan_to_num(chunk_embeddings, nan=0.0)
                except Exception as e:
                    logger.error(f"❌ Embedding failed: {e}")
                    chunk_embeddings = np.zeros((len(context_chunks), 384))
            
            graph_builder = app.state.graph_builder

            if graph_builder is None:
                raise ValueError("❌ Graph builder not initialized")
            
            # Building graph: extract entities, relations from chunks
            graph = graph_builder.build_graph(
                chunks=context_chunks,
                embeddings=chunk_embeddings
            )
            
            logger.info(f"✅ Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

            # Graph analysis: computing relevance via graph structure
            # Logging chunk_ids before passing to compute_graph_distances
            chunk_ids = [c.get('id', f'chunk_{i}') for i, c in enumerate(context_chunks)]
            logger.debug(f"Chunk IDs for graph analysis: {chunk_ids[:5]}...")

            # Computing how each chunk relates to query through graph structure
            try:
                graph_distances = graph_builder.compute_graph_distances(
                    query_text=request.message,
                    chunk_ids=chunk_ids
                )
                logger.info(f"✅ Graph distances computed for {len(graph_distances)} chunks")
            except Exception as e_compute:
                logger.error(f"❌ Error inside compute_graph_distances: {e_compute}")
                logger.exception("Detailed error inside compute_graph_distances:")
                raise e_compute

            # Log graph structure info
            graph_stats = graph_builder.get_graph_statistics()
            logger.info(f"Graph Statistics:")
            logger.info(f"   - Nodes (entities): {graph_stats.get('node_count', 0)}")
            logger.info(f"   - Edges (relations): {graph_stats.get('edge_count', 0)}")
            logger.info(f"   - Density: {graph_stats.get('density', 0):.3f}")
    
        except Exception as e:
            logger.error(f"❌ Graph building/analysis failed: {e}")
            logger.exception("Detailed error in graph building/analysis:")
            logger.warning("⚠️ Continuing without graph analysis (fallback to semantic only)")
            
            graph_distances = {
                c.get('id', f'chunk_{i}'): 0.5  # Uniform distance (neutral)
                for i, c in enumerate(context_chunks)
            }
    
            if chunk_embeddings is None or len(chunk_embeddings) == 0:
                logger.info("Getting embeddings for fallback reranking...")
                
                # Sanitize and validate chunk texts
                chunk_texts = [
                    (c.get('text') or '').strip()[:5000]  # Max 5000 chars per chunk
                    for c in context_chunks
                ]

                non_empty_texts = [t for t in chunk_texts if t]

                if not non_empty_texts:
                    logger.warning("⚠️ All chunk texts are empty!")
                    chunk_embeddings = np.zeros((len(context_chunks), 384))
                    logger.warning("   Using zero embeddings as fallback")
                else:
                    try:
                        chunk_embeddings = get_embeddings(chunk_texts)

                        if isinstance(chunk_embeddings, np.ndarray):
                            nan_count = np.isnan(chunk_embeddings).sum()

                        if nan_count > 0:
                            logger.warning(f"⚠️ Found {nan_count} NaN values in embeddings!")
                            chunk_embeddings = np.nan_to_num(chunk_embeddings, nan=0.0)
                    except Exception as e:
                        logger.error(f"❌ Embedding failed: {e}")
                        chunk_embeddings = np.zeros((len(context_chunks), 384))
    
            logger.info(f"✅ Fallback: using semantic-only reranking (without graph)")
    
    log_gpu_memory_detailed("AFTER_RETRIEVAL", iteration)

    # ════════════════════════════════════════════════════════════════════════════
    # STEP 4: Hybrid Reranking (semantic + graph-based)
    # ════════════════════════════════════════════════════════════════════════════

    if request.use_rag and len(context_chunks) > 0 and query_embedding_vec is not None:
        try:
            logger.info("Stage 2: Hybrid Reranking (semantic + graph-based)...")
            
            hybrid_scorer = app.state.hybrid_scorer
            if hybrid_scorer is None:
                raise ValueError("❌ Hybrid scorer not initialized")

            # Ensure embeddings are available
            chunk_embeddings_is_empty = (
                chunk_embeddings is None or 
                (isinstance(chunk_embeddings, list) and len(chunk_embeddings) == 0) or
                (isinstance(chunk_embeddings, np.ndarray) and chunk_embeddings.size == 0)
            )

            if chunk_embeddings_is_empty:
                logger.info("⚠️ chunk_embeddings is empty, getting embeddings...")
                
                # Sanitize and validate chunk texts
                chunk_texts = [
                    (c.get('text') or '').strip()[:5000]  # Max 5000 chars per chunk
                    for c in context_chunks
                ]

                non_empty_texts = [t for t in chunk_texts if t]

                if not non_empty_texts:
                    logger.warning("⚠️ All chunk texts are empty!")
                    chunk_embeddings = np.zeros((len(context_chunks), 384))
                    logger.warning("   Using zero embeddings as fallback")
                else:
                    try:
                        chunk_embeddings = get_embeddings(chunk_texts)

                        if isinstance(chunk_embeddings, np.ndarray):
                            nan_count = np.isnan(chunk_embeddings).sum()

                        if nan_count > 0:
                            logger.warning(f"⚠️ Found {nan_count} NaN values in embeddings!")
                            chunk_embeddings = np.nan_to_num(chunk_embeddings, nan=0.0)
                    except Exception as e:
                        logger.error(f"❌ Embedding failed: {e}")
                        chunk_embeddings = np.zeros((len(context_chunks), 384))

                logger.info(f"✅ Got embeddings")
            else:
                logger.info("chunk_embeddings is not empty")

            # Check type of chunk_embeddings
            logger.info(f"Type of chunk_embeddings: {type(chunk_embeddings)}")
            logger.info(f"Type of query_embedding_vec: {type(query_embedding_vec)}")

            logger.info("Checking is chunk_embeddings an instance...")
            if isinstance(chunk_embeddings, list):
                chunk_embeddings = np.array(chunk_embeddings)
        
            logger.info("Checking is query_embedding an instance...")
            if isinstance(query_embedding_vec, list):
                query_embedding_vec = np.array(query_embedding_vec)
            
            # Handle empty graph_distances (use fallback)
            if not graph_distances:
                logger.warning("⚠️ No graph distances, using fallback (all chunks equally distant)")
                graph_distances = {
                    c.get('id', f'chunk_{i}'): 0.5
                    for i, c in enumerate(context_chunks)
                }

            logger.info("Stage 2: Calling hybrid_scorer.rerank_chunks()...")

            # Reranking
            final_context = hybrid_scorer.rerank_chunks(
                query_embedding=query_embedding_vec,
                chunks=context_chunks,
                chunk_embeddings=chunk_embeddings,
                graph_distances=graph_distances,
                top_k=request.n_chunks
            )

            logger.info(f"✅ Reranked to top {request.n_chunks} chunks")

            # Show top reranked chunks
            if final_context:
                for i, chunk in enumerate(final_context[:3]):
                    chunk_id = chunk.get('id', 'unknown')
                    hybrid_score = chunk.get('hybrid_score', 'N/A')
                    semantic_score = chunk.get('semantic_score', 'N/A')
                    graph_score = chunk.get('graph_score', 'N/A')
        
                    # Formating scores
                    hybrid_str = f"{hybrid_score:.4f}" if isinstance(hybrid_score, float) else "N/A"
                    semantic_str = f"{semantic_score:.4f}" if isinstance(semantic_score, float) else "N/A"
                    graph_str = f"{graph_score:.4f}" if isinstance(graph_score, float) else "N/A"
        
                    logger.info(
                        f"   [{i+1}] {chunk_id}: "
                        f"hybrid={hybrid_str} "
                        f"(semantic={semantic_str}, graph={graph_str})"
                    )

        except Exception as e:
            logger.error(f"❌ Hybrid reranking failed: {e}")
            import traceback
            logger.error(f"Full Traceback:\n{traceback.format_exc()}")
            logger.warning("⚠️ Fallback: using original chunk order")
            final_context = context_chunks[:request.n_chunks]

        log_gpu_memory_detailed("AFTER_RANKING", iteration)

    else:
        # No RAG or no chunks
        logger.info("⚠️ Stage 2: Skipped (no RAG or empty context)")
        final_context = context_chunks[:request.n_chunks] if context_chunks else []

    # ════════════════════════════════════════════════════════════════════════════
    # STEP 5.1: Mark used chunks for next iteration before LLM call
    # ════════════════════════════════════════════════════════════════════════════

    if final_context:
        used_chunk_ids = [c['id'] for c in final_context if 'id' in c]
        session_data['excluded_chunks'].extend(used_chunk_ids)
        
        # Marking used chunks in chunker for tracking
        app.state.chunker.mark_chunks_used(session_id, used_chunk_ids, iteration)
        
        logger.info(f"Marked {len(used_chunk_ids)} chunks as used for next iteration")
    else:
        logger.warning(f"⚠️ No marked used chunks for next iteration")
        used_chunk_ids = []

    # ════════════════════════════════════════════════════════════════════════════
    # STEP 5.2: Starting KV Cache Monitor
    # ════════════════════════════════════════════════════════════════════════════
    # Approximate count of prompt length
    context_text_for_prompt = "\n\n".join([c.get('text', '') for c in final_context])
    full_prompt_text = f"Context:\n{context_text_for_prompt}\n\nQuery: {request.message}\nAnswer:"
    prompt_length_tokens_approx = len(full_prompt_text.split())

    # TODO: To get acccess to tokenizer of model to recieve exact length of prompt
    # Meanwhile just using the appoximate value
    prompt_length_tokens = prompt_length_tokens_approx

    n_selected_chunks = len(final_context)
    n_archived_chunks_this_iter = len(used_chunk_ids)

    # Calling the logging of KV Monitor
    app.state.kv_monitor.log_iteration_start(
        iteration=iteration,
        prompt_length_tokens=prompt_length_tokens,
        n_selected_chunks=n_selected_chunks,
        n_archived_chunks_this_iter=n_archived_chunks_this_iter
    )
    
    # ════════════════════════════════════════════════════════════════════════════
    # STEP 6.1: Initialize memory management
    # ════════════════════════════════════════════════════════════════════════════

    logger.info("⚠️ Stage 3.1 (experimental): Memory Management...")
    log_gpu_memory_detailed("BEFORE_SWAP_INIT", iteration)

    if iteration == 1 and not session_data.get('swap_initialized', False):
        logger.info("First iteration - initializing swap manager...")

        try:
            all_chunks = await get_all_session_chunks(session_id)

            if all_chunks and len(all_chunks) > 0:
                logger.info(f"Building permanent CPU RAM archive for {len(all_chunks)} chunks...")

                swap_manager = app.state.swap_manager
                swap_manager.initialize_chunks(all_chunks)

                app.state._swap_initialized = True
                session_data['swap_initialized'] = True

                logger.info(f" ✅ Swap manager initialized successfully!")
                logger.info(f"    - CPU archive: {len(swap_manager.cpu_chunks)} chunks")
                logger.info(f"    - GPU preload: {len(swap_manager.gpu_chunks)} chunks (top-5)")
            
            else:
                logger.warning("⚠️ No chunks to initialize swap manager with")
                session_data['swap_initialized'] = False

        except Exception as e:
            logger.error(f"❌ Swap initialization failed: {e}")
            import traceback
            logger.error(f"Detailed error:\n{traceback.format_exc()}")
            # Continue anyway - swap won't work but system still functional
            pass
    
    elif iteration > 1 and app.state._swap_initialized:
        logger.info(f"ℹ️ Iteration {iteration}: Swap manager already initialized, using existing archives")

    log_gpu_memory_detailed("AFTER_SWAP_INIT", iteration)

    # ════════════════════════════════════════════════════════════════════════════
    # STEP 6.2: Making swapping decisions
    # ════════════════════════════════════════════════════════════════════════════

    swap_action = 'none'

    try:
        logger.info("⚠️ Stage 3.2 (experimental): Swap Decision and Execution...")

        swap_manager = app.state.swap_manager

        # Update prefetch buffer with current chunks
        swap_manager.update_prefetch_buffer(final_context)

        # Calculate context size
        current_context_tokens = sum(
            c.get('metadata', {}).get('chunk_size', 500)
            for c in final_context
        )

        # Decide swap action based on memory stats
        swap_decision = app.state.swap_manager.decide_swap_action(
            final_context, 
            current_context_tokens,
            iteration=iteration
        )

        # Execute swap
        swap_manager.execute_swap_decision(swap_decision)
        swap_action = swap_decision.get('action', 'none')

        logger.info(f" ✅ Swap action: {swap_action}")
        logger.info(f" Context tokens: {current_context_tokens}")

        log_gpu_memory_detailed("AFTER_SWAP_DECISION", iteration)

    except Exception as e:
        logger.warning(f"⚠️ Swap management skipped: {e}")
        swap_action = 'skipped'

    # ════════════════════════════════════════════════════════════════════════════
    # STEP 7: LLM Inference with prepared context
    # ════════════════════════════════════════════════════════════════════════════

    prompt = build_prompt(
        final_context, 
        request.message,
        max_context_tokens=settings['prompt'].get('max_context_tokens', 5000),
        enable_context_limit=settings['prompt'].get('enable_context_limit', True)
    )
    
    inference_start = time.time()

    try:
        logger.info("Stage 4: LLM Inference...")
        
        llm_response = await call_vllm_chat(
            prompt,
            {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
            }
        )
        
        inference_time = time.time() - inference_start
        
        # Record for swap manager stats
        if app.state.swap_manager:
            app.state.swap_manager.record_compute_time(inference_time)
        
        logger.info(f"   ✅ Inference complete: {inference_time:.3f}s")
        
        # Extract response
        response_text = llm_response['choices'][0]['message']['content']
        usage = llm_response.get('usage', {})
        
    except Exception as e:
        logger.error(f"❌ LLM inference failed: {e}")
        raise HTTPException(status_code=502, detail=f"LLM inference failed: {str(e)}")
    
    # Calling KV Monitor in the end of inference
    generation_time_ms = inference_time * 1000 # Converting s in ms

    app.state.kv_monitor.log_iteration_end(
        iteration=iteration,
        generation_time_ms=generation_time_ms,
        prompt_length_tokens=prompt_length_tokens,
        n_selected_chunks=n_selected_chunks,
        n_archived_chunks_this_iter=n_archived_chunks_this_iter
    )

    log_gpu_memory_detailed("AFTER_INFERENCE", iteration)

    # ════════════════════════════════════════════════════════════════════════════
    # STEP 8: Collect detailed statistics
    # ════════════════════════════════════════════════════════════════════════════

    chunking_stats = app.state.chunker.get_statistics(session_id)
    new_chunks_count = len([c for c in final_context if c.get('id') not in excluded_chunks]) if excluded_chunks else len(final_context)
    
    logger.info(f"Iteration {iteration} Complete:")
    logger.info(f"   - Chunks used: {len(final_context)}")
    logger.info(f"   - New in this iteration: {new_chunks_count}")
    logger.info(f"   - Total explored: {chunking_stats['used_chunks']}/{chunking_stats['total_chunks_in_pool']}")
    logger.info(f"   - Coverage: {chunking_stats['coverage_percent']:.1f}%")
    logger.info(f"   - Inference time: {inference_time:.3f}s")

    
    # ════════════════════════════════════════════════════════════════════════════
    # STEP 9: Log session metrics for analytics
    # ════════════════════════════════════════════════════════════════════════════

    try:
        session_metrics = {
            "session_id": session_id,
            "iteration": iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        
            # Chunk statistics
            "chunks_used": len(final_context),
            "new_chunks_in_iteration": new_chunks_count,
            "excluded_from_previous": len(excluded_chunks),
            "total_explored": chunking_stats['used_chunks'],
            "coverage_percent": chunking_stats['coverage_percent'],
        
            # Performance
            "inference_time_sec": round(inference_time, 3),
            "context_tokens": sum(c.get('metadata', {}).get('chunk_size', 500) for c in final_context),
            "swap_action": swap_action,
        
            # Memory
            "gpu_memory_mb": {
                "allocated": round(torch.cuda.memory_allocated() / 1024 / 1024, 2) if torch.cuda.is_available() else 0,
                "reserved": round(torch.cuda.memory_reserved() / 1024 / 1024, 2) if torch.cuda.is_available() else 0,
            } if torch.cuda.is_available() else {},
        
            # LLM output
            "tokens_generated": usage.get('completion_tokens', 0),
            "tokens_in_prompt": usage.get('prompt_tokens', 0),
        }
    
        # Line-delimited JSON
        metrics_file = Path("logs/session_metrics.jsonl")
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
        async with aiofiles.open(metrics_file, 'a') as f:
            await f.write(json.dumps(session_metrics) + "\n")
    
        logger.debug(f"✅ Metrics logged for iteration {iteration}")
    
    except Exception as e:
        logger.warning(f"⚠️ Failed to log metrics: {e}")

    # ════════════════════════════════════════════════════════════════════════════
    # RETURN DETAILED RESPONSE with GRAPH STATISTICS
    # ════════════════════════════════════════════════════════════════════════════

    # Collect S-GAS statistics
    sgas_statistics = {
        "graph_analysis": {},
        "hybrid_reranking": {},
        "swap_management": {},
        "chunking": chunking_stats
    }

    try:
        if app.state.graph_builder:
            sgas_statistics["graph_analysis"] = app.state.graph_builder.get_graph_statistics()
    except Exception as e:
        logger.warning(f"⚠️ Failed to get graph stats: {e}")
    
    try:
        if app.state.swap_manager:
            sgas_statistics["swap_management"] = app.state.swap_manager.get_statistics()
    except Exception as e:
        logger.warning(f"⚠️ Failed to get swap stats: {e}")
    
    # Hybrid reranking stats
    sgas_statistics["hybrid_reranking"] = {
        "chunks_before_reranking": len(context_chunks),
        "chunks_after_reranking": len(final_context),
        "graph_distances_used": len(graph_distances),
        "semantic_embeddings_used": len(chunk_embeddings),
    }

    return ChatResponse(
        response=response_text,
        metadata={
            # LLM Output
            "model_used": settings['vllm']['model_name'],
            "tokens_generated": usage.get('completion_tokens', 0),
            "tokens_in_prompt": usage.get('prompt_tokens', 0),
            
            # S-GAS Settings
            "use_rag": request.use_rag,
            "session_id": session_id,
            "iteration": iteration,
            
            # Chunk Statistics (S-GAS Adaptive)
            "context_chunks_used": len(final_context),
            "new_chunks_in_iteration": new_chunks_count,
            "chunks_excluded_from_previous": len(excluded_chunks),
            "total_chunks_explored": chunking_stats['used_chunks'],
            "total_chunks_available": chunking_stats['total_chunks_in_pool'],
            "coverage_percent": chunking_stats['coverage_percent'],
            
            # Performance
            "inference_time_sec": inference_time,
            "context_tokens": sum(c.get('metadata', {}).get('chunk_size', 500) for c in final_context),
            "swap_action": swap_action,
            
            # ✅ COMPLETE S-GAS ALGORITHM STATISTICS
            "swap_statistics": {
                "total_swap_operations": app.state.swap_manager.total_swaps if app.state.swap_manager else 0,
                "swap_to_ram_count": app.state.swap_manager.swap_to_ram_count if app.state.swap_manager else 0,
                "swap_to_gpu_count": app.state.swap_manager.swap_to_gpu_count if app.state.swap_manager else 0,
                "chunks_in_gpu": len(app.state.swap_manager.gpu_chunks) if app.state.swap_manager else 0,
                "chunks_in_archive": len(app.state.swap_manager.cpu_chunks) if app.state.swap_manager else 0,
                "chunks_in_ram": (len(app.state.swap_manager.cpu_chunks) - len(app.state.swap_manager.gpu_chunks)) if app.state.swap_manager else 0,
                "last_action": app.state.swap_manager.last_action if app.state.swap_manager else "none",
                "swap_triggered": app.state.swap_manager.total_swaps > 0 if app.state.swap_manager else False,
            },

            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.get("/api/session/{session_id}/info")
async def get_session_info(session_id: str):
    """Getting info about session and documents"""
    
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="❌ Session not found")
    
    # Getting state of documents
    session_state = app.state.document_processor.get_session_state(session_id)
    
    # Getting statistics of chunker
    chunker_stats = app.state.chunker.get_statistics(session_id)
    
    logger.info(f"Session {session_id} info requested")
    
    return {
        "session_id": session_id,
        "documents": session_state,
        "chunker_statistics": chunker_stats,
        "session_data": app.state.sessions.get(session_id, {}),
    }


@app.get("/api/session/{session_id}/documents")
async def list_session_documents(session_id: str):
    """List documents in session"""

    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Getting documents
        doc_processor = app.state.document_processor
        session_docs_metadata = doc_processor.get_session_documents(session_id)
        documents_data = session_docs_metadata.to_dict()

        # Getting active documents
        active_documents = documents_data.get('active_documents', [])

        # Converting
        result_documents = []
        for doc_info in active_documents:
            if isinstance(doc_info, dict):
                result_documents.append({
                    'filename': doc_info.get('document_name', 'unknown'),
                    'chunks': len(doc_info.get('chunks', [])),  # Count of chunks
                    'size': doc_info.get('file_size', 0),
                    'uuid': doc_info.get('document_uuid', ''),
                    'version': doc_info.get('version', 1)
                })
            else:
                logger.warning(f"⚠️ Skipping invalid document entry: {type(doc_info)} - {doc_info}")

        total_chunks = sum(d.get('chunks', 0) for d in result_documents)
        logger.info(f"✅ Listed {len(result_documents)} documents for session {session_id}")

        return {
            "session_id": session_id,
            "total_chunks": total_chunks,
            "documents": result_documents
        }

    except Exception as e:
        logger.error(f"❌ Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sgas-statistics")
async def get_sgas_statistics():
    """Get S-GAS algorithm global statistics"""
    try:
        graph_builder = app.state.graph_builder
        swap_manager = app.state.swap_manager
        hybrid_scorer = app.state.hybrid_scorer
        
        statistics = {
            "graph": graph_builder.get_graph_statistics() if graph_builder else {},
            "swap": swap_manager.get_statistics() if swap_manager else {},
            "scorer": {
                "alpha": hybrid_scorer.alpha if hybrid_scorer else 0.6,
                "beta": hybrid_scorer.beta if hybrid_scorer else 0.4
            }
        }
        
        return JSONResponse(content=statistics, status_code=200)
    except Exception as e:
        logger.error(f"❌ Error retrieving S-GAS statistics: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""

    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host=settings['api']['host'],
        port=settings['api']['port'],
        reload=True
    )
