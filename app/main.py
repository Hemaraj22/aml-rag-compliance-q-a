import time

from fastapi import FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app import rag
from app.config import get_settings

app = FastAPI(title="AML/Compliance RAG API", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=20)


class Citation(BaseModel):
    id: int
    source: str | None
    section: str | None
    page: int | None
    edition: int | None = None
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    latency_ms: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return {"collection": get_settings().collection_name, "chunks": rag.get_vectorstore()._collection.count()}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    start = time.perf_counter()
    result = await run_in_threadpool(rag.answer, req.question, req.top_k)
    return AskResponse(
        answer=result["answer"],
        citations=result["citations"],
        latency_ms=int((time.perf_counter() - start) * 1000),
    )


@app.post("/ingest")
async def ingest(x_api_key: str | None = Header(None)):
    admin_key = get_settings().admin_api_key
    if admin_key and x_api_key != admin_key:
        raise HTTPException(status_code=401, detail="invalid API key")
    from app.ingest import build_index

    count = await run_in_threadpool(build_index)
    rag.get_vectorstore.cache_clear()
    rag.get_bm25.cache_clear()
    return {"indexed_chunks": count}
