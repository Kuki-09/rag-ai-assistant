import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.config import RERANK_MODEL


class Reranker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
        self.model.eval()

    def score(self, query: str, docs: list, top_n: int = None) -> list:
        """
        Score and re-rank documents by relevance to the query.
        Returns top_n most relevant docs (or all if top_n is None).
        """
        if not docs:
            return docs

        queries = [query] * len(docs)
        passages = [doc.page_content for doc in docs]

        # ✅ FIX: Pass as two separate lists, not list-of-tuples
        inputs = self.tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        # ✅ FIX: flatten to 1-D regardless of batch size
        scores = logits.view(-1).tolist()

        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        ranked = [doc for doc, _ in scored_docs]

        return ranked[:top_n] if top_n else ranked