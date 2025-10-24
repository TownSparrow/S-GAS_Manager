from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import logging
import shutil
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager

# Configs and modules
from src.config import settings
from src.modules.retrieval.vector_store import ChromaVectorStore
from src.modules.retrieval.chunking import SemanticChunker
from src.modules.retrieval.document_processor import DocumentProcessor
from src.modules.retrieval.embedder import get_embeddings

# -----------------------------------------------------------------------------
# Init of app and basic services
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOADS_DIR = Path("data/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Services init with settings['...']
    vector_store = ChromaVectorStore(
        persist_directory=settings['database']['chroma_persist_dir'],
        collection_name=settings['database'].get('collection_name', 'documents'),
    )
    chunker = SemanticChunker(
        max_chunk_size=settings['chunking']['max_chunk_size'],
        overlap_size=settings['chunking']['overlap_size'],
    )
    try:
        document_processor = DocumentProcessor(vector_store=vector_store, chunker=chunker)
        app.state.vector_store = vector_store
        app.state.chunker = chunker
        app.state.document_processor = document_processor
        logger.info("✅ All S-GAS components initialized with config parameters")
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
    yield
    app.state.vector_store = None
    app.state.chunker = None
    app.state.document_processor = None

app = FastAPI(
    title="S-GAS Manager API",
    version="0.1.2",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:8080", "http://127.0.0.1:8080",
        f"http://{settings['api']['host']}:{settings['api']['port']}",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Статика
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# -----------------------------------------------------------------------------
# Models of requests and responses
# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True
    n_chunks: int = 5
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    metadata: Dict[str, Any] = {}

# -----------------------------------------------------------------------------
# Additional functions
# -----------------------------------------------------------------------------
def build_prompt(context_chunks: List[Dict[str, Any]], user_message: str) -> str:
    if context_chunks:
        context_text = "\n\n".join([c.get("text") or c.get("document") or "" for c in context_chunks if c])
        return (
            f"Context from the knowledge base:\n{context_text}\n\n"
            f"User's request: {user_message}\n\n"
            f"Instructions: Analyze the context and give a brief, precise, and specific answer based on the fragments provided. If the context lacks information, state so and respond briefly using general knowledge.\n\n"
            f"Response:"
        )
    else:
        return (
            f"User's request: {user_message}\n\n"
            f"Give a short and precise answer.\n\n"
            f"Response:"
        )

async def call_vllm_chat(prompt: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
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
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{api_base}/chat/completions", json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"vLLM error {resp.status_code}: {resp.text}")
        data = resp.json()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or "message" not in choices[0]:
            raise HTTPException(status_code=502, detail=f"vLLM malformed response: {data}")
        if "content" not in choices[0]["message"]:
            raise HTTPException(status_code=502, detail=f"vLLM response without content: {data}")
        return data

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/")
async def serve_web_client():
    static_index = STATIC_DIR / "index.html"
    if static_index.exists():
        return FileResponse(str(static_index))
    return JSONResponse(
        {
            "message": "S-GAS Manager API",
            "version": "1.0",
            "endpoints": ["/api/chat", "/health", "/api/upload-document", "/api/documents"],
            "web_client": "No files in src/web/static/",
        },
        status_code=200,
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        logger.info(f"Request receieved: {request.message}")

        # 1) Embedding of request
        query_embedding = await get_embeddings([request.message])
        try:
            dim = getattr(query_embedding, "shape", None)
            logger.info(f"Embedding is recieved. Dimension is : {dim}")
        except Exception:
            logger.info("Embedding is recieved.")

        # 2) Search of context (optionally)
        context_chunks: List[Dict[str, Any]] = []
        if request.use_rag:
            try:
                vector_store = app.state.vector_store
                if vector_store is not None:
                    context_chunks = await vector_store.search(
                        query_embedding,
                        top_k=max(1, int(request.n_chunks)),
                    )
                    logger.info(f"Retrieved {len(context_chunks)} context chunks")
            except Exception as e:
                logger.warning(f"RAG search failed: {e}")

        # 3) Prompt is always defined
        enhanced_prompt = build_prompt(context_chunks, request.message)

        # 4) Calling vLLM
        overrides = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
        }
        data = await call_vllm_chat(enhanced_prompt, overrides)

        # 5) Response
        msg = data["choices"][0]["message"]
        answer = msg.get("content", "")

        usage = data.get("usage", {})
        return ChatResponse(
            response=answer,
            metadata={
                "usage": usage,
                "model_used": settings['vllm']['model_name'],
                "use_rag": request.use_rag,
                "context_chunks_used": len(context_chunks),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in /api/chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or '..' in file.filename or '/' in file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = UPLOADS_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File uploaded: {file.filename} ({file_path.stat().st_size} bytes)")
        result = await app.state.document_processor.process_document(file_path)
        return result
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/documents")
async def list_documents():
    documents = [
        {"filename": f.name, "size": f.stat().st_size, "modified": f.stat().st_mtime}
        for f in UPLOADS_DIR.iterdir()
        if f.is_file()
    ]
    return {"documents": documents}

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    file_path = UPLOADS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        file_path.unlink()
        logger.info(f"Deleted document: {filename}")
        return {"message": f"Document {filename} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.get("/health")
async def health_check():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings['vllm']['api_base']}/models")
        vllm_status = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        vllm_status = "unavailable"
    vector_store_status = "ready" if app.state.vector_store is not None else "not_initialized"
    return {
        "status": "healthy",
        "vllm_status": vllm_status,
        "vector_store_status": vector_store_status,
        "model": settings['vllm']['model_name'],
        "time": datetime.now(timezone.utc).isoformat(),
    }
