from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import logging
import shutil
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager

# Import of all needed custom modules
from src.config import settings
from src.modules.retrieval.vector_store import ChromaVectorStore
from src.modules.retrieval.chunking import SemanticChunker
from src.modules.retrieval.document_processor import DocumentProcessor
from src.modules.retrieval.embedder import get_embeddings

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and Paths
UPLOADS_DIR = Path("data/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setting the global services
    vector_store = ChromaVectorStore(
        persist_directory=settings['database']['chroma_persist_dir'],
        collection_name=settings['database']['collection_name']
    )
    chunker = SemanticChunker(
        max_chunk_size=settings['chunking']['max_chunk_size'],
        overlap_size=settings['chunking']['overlap_size']
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


# App Instance
app = FastAPI(
    title="S-GAS Manager API",
    version="0.1.1",
    lifespan=lifespan
    )


# CORS settings for access from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000",
                   "http://localhost:8080", "http://127.0.0.1:8080",
                   f"http://{settings['api']['host']}:{settings['api']['port']}"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True


class ChatResponse(BaseModel):
    response: str
    metadata: dict = {}


@app.get("/")
async def serve_web_client():
    """ Main page - web-client """
    static_index = STATIC_DIR / "index.html"
    if static_index.exists():
        return FileResponse(str(static_index))
    else:
        return {
            "message": "S-GAS Manager API",
            "version": "0.1.1",
            "endpoints": ["/api/chat", "/health", "/api/upload-document", "/api/documents"],
            "web_client": "No files in src/web/static/"
        }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """ Main endpoint for chat """
    try:
        logger.info(f"Request receieved: {request.message}")
        
        # 1. Receiving the embedding of request
        try:
            query_embedding = await get_embeddings([request.message])
            logger.info(f"Embedding is recieved. Dimension is : {query_embedding.shape}")
        except Exception as e:
            logger.error(f"Error of receving the embedding: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing request: {e}")
        
        # 2. Recieving the chunks (if RAG is available)
        context_chunks = []
        if request.use_rag:
            try:
                vector_store = app.state.vector_store
                if vector_store is not None:
                    context_chunks = await vector_store.search(query_embedding, top_k=settings['rag']['top_k'])
                    logger.info(f"Retrieved {len(context_chunks)} context chunks")
            except Exception as e:
                logger.warning(f"RAG search failed: {e}")
        
        # 3. Main logic (not ready yet)
        # 2.1 Searching in base of embeddings
        # TODO: Make the search in ChromaDB
        
        # 3.2 Graph analysis
        # TODO: Make the graph building
        
        # 3.3 Swapping
        # TODO: Make the adaptive swapping

        # 4. Forming the final prompt (including the chunks)
        if context_chunks:
            context_text = "\n\n".join([chunk['text'] for chunk in context_chunks])
            enhanced_prompt = (
                f"Context from knowledge base:\n{context_text}\n\n"
                f"User's request: {request.message}\n"
                f"Answer based on the context above. If the context doesn't contain relevant information,"
                f" answer based on your general knowledge."
            )
        else:
            enhanced_prompt = (
                f"Context: This is a test query as part of the development of the S-GAS algorithm.\n\n"
                f"User's request: {request.message}\n\nAnswer briefly and to the point."
            )
        
        # 5. Sending request to vLLM for response generation
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{settings['vllm']['api_base']}/chat/completions",
                    json={
                    "model": settings['vllm']['model_name'],
                    "messages": [{"role": "user", "content": enhanced_prompt}],
                    "temperature": settings['vllm']['temperature'],
                    "max_tokens": settings['vllm']['max_tokens']
                    }
                )
                response.raise_for_status()
                result = response.json()   
                answer = result["choices"][0]["message"]["content"]
                logger.info("Response received successfully from vLLM")
                
                return ChatResponse(
                    response=answer,
                    metadata={
                        "embedding_shape": query_embedding.shape,
                        "model_used": settings['vllm']['model_name'],
                        "use_rag": request.use_rag,
                        "context_chunks_used": len(context_chunks),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                
        except httpx.TimeoutException:
            logger.error("Timeout while accessing vLLM")
            raise HTTPException(status_code=504, detail="Timeout waiting for response from model")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from vLLM: {e.response.status_code}")
            raise HTTPException(status_code=502, detail="Language model server error")
        except Exception as e:
            logger.error(f"Unexpected error accessing vLLM: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Critical error in chat_endpoint: {e}")
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
    """ Checking the service health"""
    try:
        # Checking the access to vLLM
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings['vllm']['api_base']}/models")
            vllm_status = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        vllm_status = "unavailable"
    vector_store_status = "ready" if app.state.vector_store is not None else "not_initialized"
    return {
        "status": "healthy",
        "vllm_status": vllm_status,
        "vector_store_status": vector_store_status,
        "model": settings['vllm']['model_name']
    }
