from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from src.config import config
import logging

logger = logging.getLogger(__name__)


def get_embedder() -> Embeddings:
    """
    Factory function to return the correct LangChain Embeddings instance
    based on the configuration.
    """
    base_url = config.EMBEDDER_BASE_URL
    model_name = config.EMBEDDER_MODEL
    api_key = config.EMBEDDER_API_KEY

    # Simple heuristic to determine provider if not explicitly set
    if base_url and "api.openai.com" in base_url:
        logger.info(f"Using OpenAI Embeddings with model {model_name}")
        return OpenAIEmbeddings(
            openai_api_key=api_key, model=model_name, openai_api_base=base_url
        )
    elif api_key and api_key.startswith("AIza"):
        # Google Gemini heuristic
        logger.info(f"Using Google Generative AI Embeddings with model {model_name}")
        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
    elif (
        base_url
        and ("localhost" in base_url or "127.0.0.1" in base_url)
        and "ollama" not in base_url.lower()
        and not api_key
    ):
        # Assuming Ollama if local and no API key (can be customized)
        logger.info(f"Using Ollama Embeddings with model {model_name}")
        return OllamaEmbeddings(base_url=base_url, model=model_name)
    elif base_url:
        # Fallback for OpenAI compatible APIs (Mistral, vLLM, text-generation-webui, etc)
        logger.info(
            f"Using OpenAI Compatible Embeddings at {base_url} with model {model_name}"
        )
        return OpenAIEmbeddings(
            openai_api_key=api_key
            or "sk-dummy",  # Some local providers require a dummy key
            model=model_name,
            openai_api_base=base_url,
        )
    else:
        # Default to OpenAI if nothing else is specified but API key exists
        logger.info(f"Defaulting to OpenAI Embeddings with model {model_name}")
        return OpenAIEmbeddings(openai_api_key=api_key, model=model_name)
