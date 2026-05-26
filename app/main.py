import os
import uuid
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.loaders.pdf_loader import load_pdf
from app.loaders.docx_loader import load_docx
from app.loaders.web_loader import load_web
from app.processing.chunking import chunk_text
from app.processing.vectorstore import VectorStoreManager
from app.rag.chain import RAGChain
from app.config import DATA_PATH, WHISPER_MODEL_SIZE

# -----------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------
rag: RAGChain = None
_whisper_model = None
vectorstore = VectorStoreManager()

os.makedirs(DATA_PATH, exist_ok=True)


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


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
    path = os.path.join(DATA_PATH, f"{doc_id}.pdf")
    with open(path, "wb") as f:
        f.write(await file.read())
    try:
        docs = load_pdf(path)
        return _ingest_docs(docs, doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {e}")


@app.post("/upload/docx")
async def upload_docx(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    path = os.path.join(DATA_PATH, f"{doc_id}.docx")
    with open(path, "wb") as f:
        f.write(await file.read())
    try:
        docs = load_docx(path)
        return _ingest_docs(docs, doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DOCX processing failed: {e}")


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
# Voice route
# -----------------------------------------------------------------------
@app.post("/voice")
async def voice_transcribe(file: UploadFile = File(...)):
    path = os.path.join(DATA_PATH, f"voice_{uuid.uuid4()}.wav")
    with open(path, "wb") as f:
        f.write(await file.read())
    try:
        model = get_whisper()
        segments, _ = model.transcribe(path)
        text = " ".join(seg.text.strip() for seg in segments)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        if os.path.exists(path):
            os.remove(path)


# -----------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "rag_ready": rag is not None}


