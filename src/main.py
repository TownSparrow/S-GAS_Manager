import asyncio
import logging
from pathlib import Path
import sys

# Adding the project root directory to the import path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Addding src to the import path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from modules.retrieval.embedder import get_embeddings
from config import settings

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_embeddings():
    """ Testing the embedding module """
    try:
        test_texts = [
            "Example of the first chunk for testing",
            "Example of the second chunk with another content for testing",
            "The third test chunk to check the algorithm's operation"
        ]
        
        logger.info("Starting the receiving of embeddings......")
        embeddings = await get_embeddings(test_texts)
        
        logger.info(f"✅ Embeddings were received!")
        logger.info(f"📊 Dimension: {embeddings.shape}")
        logger.info(f"📝 Texts processed: {len(test_texts)}")
        
        return embeddings
        
    except Exception as e:
        logger.error(f"❌ Error of getting embeddings: {e}")
        return None


async def test_configuration():
    """ Checking the correctness of the configuration """
    try:
        logger.info("🔧 Checking configuration...")
        logger.info(f"vLLM model: {settings.VLLM_MODEL_NAME}")
        logger.info(f"vLLM API URL: {settings.VLLM_API_URL}")
        logger.info(f"Embedding model: {settings.EMBEDDING_MODEL_NAME}")
        logger.info(f"GPU utilization: {settings.vllm_config.gpu_memory_utilization}")
        logger.info(f"Max length: {settings.vllm_config.max_model_len}")
        logger.info("✅ Configuration loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Error of configuration: {e}")
        return False

async def main():
    """ Basic function for testing components """
    logger.info("🚀 Launching S-GAS Manager...")
    
    # Checking the configuration
    if not await test_configuration():
        return
    
    # Testing the embeddings
    embeddings = await test_embeddings()
    if embeddings is None:
        logger.error("❌ Critical error: Failed to get embeddings")
        return
    
    # TODO: Add tests for other modules in the future
    # logger.info("🔄 Testing the graph module...")
    # graph = build_knowledge_graph(texts, embeddings)

    # logger.info("🔄 Testing the swap manager...")
    # swap_manager = SwapManager()
    
    logger.info("✅ All basic components are tested!")
    logger.info("🌐 To run the API use: uvicorn src.web.api:app --reload --host 0.0.0.0 --port 8080")

if __name__ == "__main__":
    asyncio.run(main())