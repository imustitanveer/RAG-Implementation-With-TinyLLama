# uvicorn main:app --reload

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import boto3
import uuid
import json
from tempfile import NamedTemporaryFile
from typing import List
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from get_embedding import get_embedding_function

"""
main.py – FastAPI application for:
  • Uploading PDFs → chunk → embed (Ollama Mistral) → store JSON docs in S3
  • Clearing all stored chunks from S3
  • RAG chat endpoint that streams an answer based on S3-stored chunks
"""

load_dotenv()

# ─── AWS CONFIG ───────────────────────────────────────────────────────────────
S3_BUCKET_NAME = "dam-chatbot-saas"
S3_PREFIX = "embeddings/"  # folder-like path in bucket
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name=AWS_REGION,
)

# ─── FASTAPI APP ──────────────────────────────────────────────────────────────
app = FastAPI(title="RAG-as-a-Service", version="0.1.0")

# ─── GLOBALS (IN-MEMORY CACHE) ────────────────────────────────────────────────
EMBEDDINGS: List[np.ndarray] = []  # shape: (N, d)
CHUNKS_META: List[dict] = []      # keeps id, content, etc.
EMBED_DIM = None                  # set after first load
EMBEDDER = get_embedding_function()
LLM = Ollama(model="mistral")
PROMPT = ChatPromptTemplate.from_template("""Answer the question using ONLY the context below. If the answer isn't contained, say you don't know.

{context}

---

Q: {question}
A:""")
PARSER = StrOutputParser()

# ─── helpers ──────────────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_s3_embeddings() -> None:
    """Load all JSON docs from S3 into memory as vectors & metadata."""
    global EMBEDDINGS, CHUNKS_META, EMBED_DIM
    EMBEDDINGS.clear()
    CHUNKS_META.clear()

    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)

    for page in pages:
        for obj in page.get("Contents", []):
            resp = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=obj["Key"])
            body = resp["Body"].read()
            doc = json.loads(body)
            EMBEDDINGS.append(np.array(doc["embedding"], dtype=np.float32))
            CHUNKS_META.append(doc)

    if EMBEDDINGS:
        EMBED_DIM = EMBEDDINGS[0].shape[0]
        EMBEDDINGS[:] = [v / np.linalg.norm(v) for v in EMBEDDINGS]  # normalise once
    print(f"📥 Loaded {len(EMBEDDINGS)} chunks from S3 into memory.")

# Call once on startup
after_start_event = app.on_event("startup")
@after_start_event
def startup_load():
    try:
        load_s3_embeddings()
    except Exception as e:
        print("⚠️  Failed to preload embeddings:", e)


def _split_pdf(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    last_page_id = None
    idx = 0
    for chunk in chunks:
        src, page = chunk.metadata.get("source"), chunk.metadata.get("page")
        current = f"{src}:{page}"
        idx = idx + 1 if current == last_page_id else 0
        chunk.metadata["id"] = f"{current}:{idx}"
        last_page_id = current
    return chunks


def _embed_and_upload(chunks):
    global EMBEDDINGS, CHUNKS_META
    for chunk in chunks:
        emb = EMBEDDER.embed_query(chunk.page_content)
        doc_json = {
            "id": chunk.metadata["id"],
            "source": chunk.metadata.get("source"),
            "page": chunk.metadata.get("page"),
            "content": chunk.page_content,
            "embedding": emb,
        }
        key = f"{S3_PREFIX}{uuid.uuid4()}.json"
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=json.dumps(doc_json),
            ContentType="application/json",
        )
        # Update in-memory cache
        vec = np.array(emb, dtype=np.float32)
        vec /= np.linalg.norm(vec)
        EMBEDDINGS.append(vec)
        CHUNKS_META.append(doc_json)

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.post("/upload", summary="Upload PDF & embed")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        chunks = _split_pdf(tmp_path)
        _embed_and_upload(chunks)
        os.remove(tmp_path)
        return {"message": f"Processed {len(chunks)} chunks and uploaded."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.delete("/clear", summary="Delete all embeddings from S3 & memory")
def clear_embeddings():
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)
        to_del = [{"Key": o["Key"]} for p in pages for o in p.get("Contents", [])]
        if to_del:
            s3_client.delete_objects(Bucket=S3_BUCKET_NAME, Delete={"Objects": to_del})
        # Clear in-memory
        EMBEDDINGS.clear()
        CHUNKS_META.clear()
        return {"message": f"Deleted {len(to_del)} S3 objects and cleared cache."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

class ChatRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/chat", summary="RAG chat endpoint")
async def chat(req: ChatRequest):
    if not EMBEDDINGS:
        return JSONResponse(status_code=400, content={"error": "No documents loaded. Upload first."})

    q_vec = np.array(EMBEDDER.embed_query(req.question), dtype=np.float32)
    q_vec /= np.linalg.norm(q_vec)

    # Compute cosine similarities
    sims = [float(np.dot(q_vec, v)) for v in EMBEDDINGS]
    top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[: req.top_k]

    context_parts, sources = [], []
    for i in top_indices:
        meta = CHUNKS_META[i]
        context_parts.append(meta["content"])
        sources.append(meta["id"])
    context = "\n\n---\n\n".join(context_parts)

    # Build prompt & stream response
    chain = PROMPT | LLM | PARSER
    async def streamer():
        for token in chain.stream({"context": context, "question": req.question}):
            yield token
    headers = {"Sources": ",".join(sources)}
    return StreamingResponse(streamer(), headers=headers, media_type="text/plain")

@app.get("/reload", summary="Reload embeddings from S3")
def reload_cache():
    try:
        load_s3_embeddings()
        return {"message": f"Cache reloaded with {len(EMBEDDINGS)} chunks."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def root():
    return {"message": "Welcome to the RAG API: /upload → /chat. Enjoy!"}