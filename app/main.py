import os
import uuid
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from app.loaders.pdf_loader import load_pdf
from app.loaders.docx_loader import load_docx
from app.loaders.web_loader import load_web
from app.processing.chunking import chunk_text
from app.processing.vectorstore import VectorStoreManager
from app.rag.chain import RAGChain
from app.config import DATA_PATH
import tempfile
# -----------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------
rag: RAGChain = None
_whisper_model = None
vectorstore = VectorStoreManager()

os.makedirs(DATA_PATH, exist_ok=True)




# -----------------------------------------------------------------------
# Lifespan
# -----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = RAGChain()
    yield


# -----------------------------------------------------------------------
# App — must be defined BEFORE any @app.xxx decorators
# -----------------------------------------------------------------------
app = FastAPI(title="RAG AI Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str
    doc_id: str = "default"


class WebIngestRequest(BaseModel):
    url: str


# -----------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------
def _ingest_docs(docs, doc_id: str) -> dict:
    chunks = chunk_text(docs)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced no text chunks.")
    vectorstore.create_vectorstore(chunks, doc_id)
    return {"doc_id": doc_id, "chunks": len(chunks)}


# -----------------------------------------------------------------------
# Upload routes
# -----------------------------------------------------------------------
@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        docs = load_pdf(path)
        return _ingest_docs(docs, doc_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {e}")
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.post("/upload/docx")
async def upload_docx(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        docs = load_docx(path)
        return _ingest_docs(docs, doc_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DOCX processing failed: {e}")
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.post("/upload/web")
async def upload_web(req: WebIngestRequest):
    doc_id = str(uuid.uuid4())
    try:
        docs = load_web(req.url)
        return _ingest_docs(docs, doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Web load failed: {e}")


# -----------------------------------------------------------------------
# Chat routes
# -----------------------------------------------------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG not initialized.")
    answer = rag.ask(req.query, req.doc_id)
    return {"answer": answer}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG not initialized.")

    def generate():
        try:
            for chunk in rag.stream_answer(req.query, req.doc_id):
                if chunk:
                    yield chunk
        except Exception as e:
            yield f"\n[ERROR]: {e}\n{traceback.format_exc()}"

    return StreamingResponse(generate(), media_type="text/plain")



# -----------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "rag_ready": rag is not None}



@app.get("/")
def index():
    return FileResponse("index.html")
