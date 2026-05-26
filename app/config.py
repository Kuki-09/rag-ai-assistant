import os
from dotenv import load_dotenv

load_dotenv()

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Reranker
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 4))

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

# Paths
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vectorstore")
DATA_PATH = os.getenv("DATA_PATH", "data")

# Ollama / LLM
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Retrieval
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 8))

# Whisper
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")