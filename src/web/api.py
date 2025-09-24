from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.modules.retrieval.embedder import get_embeddings
from src.config import settings
import httpx
import logging
from pathlib import Path
from datetime import datetime, timezone


# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="S-GAS Manager API", version="0.1.0")


# CORS settings for access from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000",
                   "http://localhost:8080", "http://127.0.0.1:8080"], # or ["*"] (but not for the production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Connecting the static files for web-client
static_dir = Path(__file__).parent / "static"
if not static_dir.exists():
    static_dir.mkdir()

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True


class ChatResponse(BaseModel):
    response: str
    metadata: dict = {}


@app.get("/")
async def serve_web_client():
    """ Main page - web-client """
    static_index = static_dir / "index.html"
    if static_index.exists():
        return FileResponse(str(static_index))
    else:
        return {
            "message": "S-GAS Manager API",
            "version": "0.1.0",
            "endpoints": ["/api/chat", "/health"],
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
        
        # 2. Main logic (not ready yet)
        # 2.1 Searching in base of embeddings
        # TODO: Make the search in ChromaDB
        
        # 2.2 Graph analysis
        # TODO: Make the graph building
        
        # 2.3 Swapping
        # TODO: Make the adaptive swapping
        
        # 3. Forming the final prompt
        enhanced_prompt = f"""Context: This is a test query as part of the development of the S-GAS algorithm.

User's request: {request.message}

Answer briefly and to the point."""
        
        # 4. Sending request to vLLM for response generation
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{settings.VLLM_API_URL}/chat/completions",
                    json={
                        "model": settings.VLLM_MODEL_NAME,
                        "messages": [
                            {
                                "role": "user", 
                                "content": enhanced_prompt
                            }
                        ],
                        "temperature": 0.7,
                        "max_tokens": 256
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
                        "model_used": settings.VLLM_MODEL_NAME,
                        "use_rag": request.use_rag,
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


@app.get("/health")
async def health_check():
    """ Checking the service health"""
    try:
        # Checking the access to vLLM
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.VLLM_API_URL}/models")
            vllm_status = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        vllm_status = "unavailable"
    
    return {
        "status": "healthy",
        "vllm_status": vllm_status,
        "model": settings.VLLM_MODEL_NAME
    }


# @app.get("/")
# async def root():
#     """ Root endpoint """
#     return {
#         "message": "S-GAS Manager API",
#         "version": "0.1.0",
#         "endpoints": ["/api/chat", "/health"]
#     }