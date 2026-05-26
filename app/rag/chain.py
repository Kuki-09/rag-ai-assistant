from langchain_ollama import OllamaLLM
from app.processing.vectorstore import VectorStoreManager
from app.processing.reranker import Reranker
from app.rag.memory import ChatMemory
from app.config import OLLAMA_MODEL, OLLAMA_URL, RETRIEVAL_TOP_K, RERANK_TOP_N


def _build_prompt(context: str, history_text: str, query: str) -> str:
    return f"""You are a strict document-based question answering assistant.

RULES:
1. Answer ONLY using the provided context below.
2. Do NOT use any outside knowledge or assumptions.
3. If the answer is not explicitly present in the context, say:
   "I don't know based on the provided document."
4. Do not guess, infer, or add extra information.
5. Keep answers clear, factual, and grounded strictly in the context.
6. If context is relevant, use only the necessary parts to answer the question.

CONTEXT:
{context}

CHAT HISTORY:
{history_text}

QUESTION: {query}

ANSWER (be detailed and specific):"""


class RAGChain:
    def __init__(self):
        self.vs = VectorStoreManager()
        self.llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_URL)  # ✅ FIX
        self.reranker = Reranker()
        self.memory = ChatMemory()

    def _get_context(self, query: str, doc_id: str):
        """Retrieve, rerank, and format context from vectorstore."""
        db = self.vs.load(doc_id)
        if db is None:
            return None, None

        docs = db.similarity_search(query, k=RETRIEVAL_TOP_K)
        docs = self.reranker.score(query, docs, top_n=RERANK_TOP_N)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        return context, docs

    def _format_history(self, doc_id: str) -> str:
        history = self.memory.get_history(doc_id, limit=5)
        if not history:
            return "None"
        return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history)

    def ask(self, query: str, doc_id: str = "default") -> str:
        """Non-streaming answer."""
        context, _ = self._get_context(query, doc_id)
        if context is None:
            return "No document found. Please upload a document first."

        history_text = self._format_history(doc_id)
        prompt = _build_prompt(context, history_text, query)

        result = self.llm.invoke(prompt)
        self.memory.save(doc_id, query, result)
        return result

    def stream_answer(self, query: str, doc_id: str = "default"):
        """Streaming answer — yields text chunks."""
        context, _ = self._get_context(query, doc_id)
        if context is None:
            yield "No document found. Please upload a document first."
            return

        history_text = self._format_history(doc_id)
        prompt = _build_prompt(context, history_text, query)

        full_answer = ""
        try:
            for chunk in self.llm.stream(prompt):
                if chunk:
                    full_answer += chunk
                    yield chunk
        except Exception as e:
            error_msg = f"\n[Stream error: {e}]"
            full_answer += error_msg
            yield error_msg
        finally:
            # ✅ FIX: always save, even on partial stream
            if full_answer:
                self.memory.save(doc_id, query, full_answer)