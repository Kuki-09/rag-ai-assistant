from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Return a cached HuggingFaceEmbeddings instance.
    lru_cache ensures the model is loaded once and reused.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )