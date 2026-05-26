import os
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str):
    """Load a PDF and return a list of LangChain Document objects with metadata."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    filename = os.path.basename(file_path)
    for doc in docs:
        doc.metadata["source"] = filename

    return docs