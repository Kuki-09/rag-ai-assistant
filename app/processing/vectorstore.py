import os
from langchain_community.vectorstores import FAISS
from app.processing.embeddings import get_embeddings
from app.config import VECTOR_DB_PATH


class VectorStoreManager:
    def __init__(self):
        self.base_path = VECTOR_DB_PATH
        os.makedirs(self.base_path, exist_ok=True)

    def _path(self, doc_id: str) -> str:
        return os.path.join(self.base_path, doc_id)

    def create_vectorstore(self, chunks: list, doc_id: str) -> None:
        """Embed chunks and save a FAISS index for this doc_id."""
        if not chunks:
            raise ValueError("No chunks to embed — document may be empty.")

        embeddings = get_embeddings()  # cached singleton
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(self._path(doc_id))

    def load(self, doc_id: str):
        """Load a FAISS index. Returns None if not found."""
        path = self._path(doc_id)
        if not os.path.exists(path):
            return None

        return FAISS.load_local(
            path,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )

    def exists(self, doc_id: str) -> bool:
        return os.path.exists(self._path(doc_id))

    def delete(self, doc_id: str) -> bool:
        """Remove a vectorstore index."""
        import shutil
        path = self._path(doc_id)
        if os.path.exists(path):
            shutil.rmtree(path)
            return True
        return False