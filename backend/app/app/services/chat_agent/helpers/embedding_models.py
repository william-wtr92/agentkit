# -*- coding: utf-8 -*-
# mypy: disable-error-code="call-arg"
# TODO: Change langchain param names to match the new langchain version

import logging
import inspect 
from typing import List, Optional, Union

from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.embeddings import Embeddings

from app.api.deps import get_redis_store
from app.core.config import settings

logger = logging.getLogger(__name__)

ollama_signature = inspect.signature(OllamaEmbeddings)
logger.info(f"Inspecting OllamaEmbeddings signature: {ollama_signature}")

class CacheBackedEmbeddingsExtended(CacheBackedEmbeddings):
    def embed_query(self, text: str) -> List[float]:
        """
        Embed query text.

        Extended to support caching

        Args:
            text: The text to embed.

        Returns:
            The embedding for the given text.
        """
        vectors: List[Union[List[float], None]] = self.document_embedding_store.mget([text])
        text_embeddings = vectors[0]

        if text_embeddings is None:
            text_embeddings = self.underlying_embeddings.embed_query(text)
            self.document_embedding_store.mset(list(zip([text], [text_embeddings])))

        return text_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.underlying_embeddings.embed_documents(texts)


def get_embedding_model(emb_model: Optional[str]) -> CacheBackedEmbeddingsExtended:
     if settings.OLLAMA_ENABLED:
        return get_ollama_embedding_model(emb_model)


def get_ollama_embedding_model(emb_model: Optional[str]) -> CacheBackedEmbeddingsExtended:
    """
    Gets the embedding model from the embedding model type.
    """
    if emb_model is None:
        emb_model = settings.OLLAMA_DEFAULT_MODEL

    logger.info(f"Getting Ollama embedding model: {emb_model}")

    valid_options = {"base_url", "model", "show_progress"}
    
    ollama_kwargs = {
        "base_url": settings.OLLAMA_URL,
        "model": emb_model,
        "show_progress": True
    }
    
    filtered_options = {k: v for k, v in ollama_kwargs.items() if k in valid_options}
    
    logger.info(f"Using OllamaEmbeddings with options: {filtered_options}")

    underlying_embeddings = OllamaEmbeddings(**filtered_options)
    store = get_redis_store()
    embedder = CacheBackedEmbeddingsExtended.from_bytes_store(
        underlying_embeddings, store, namespace=underlying_embeddings.model
    )
    return embedder
