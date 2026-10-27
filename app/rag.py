"""Retrieval + grounded generation with numbered citations.

Retrieval pipeline: vector search + BM25 keyword search -> reciprocal rank fusion
-> LLM reranker -> top_k passages -> grounded answer.
"""
import re
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable
from pydantic import BaseModel, Field

from app.config import get_settings

NO_ANSWER = "I could not find this in the FFIEC BSA/AML material I have indexed."

SYSTEM_PROMPT = """You are a BSA/AML compliance assistant. Answer ONLY using the numbered context \
passages from the FFIEC BSA/AML Examination Manual below.

Rules:
- Cite every factual claim with the passage number in square brackets, e.g. [1] or [2][3].
- Quote dollar thresholds, deadlines and record-retention periods exactly as written in the context.
- Each passage shows its manual edition year. If passages conflict, follow the most recent \
edition and do not repeat the superseded rule as current.
- Give a complete answer: include the conditions, exceptions and aggregation rules the context \
states for the requirement.
- If the context does not contain the answer, reply exactly: "{no_answer}"
- Do not give legal advice; describe what the manual says.

Context:
{context}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])


@lru_cache
def get_vectorstore() -> Chroma:
    s = get_settings()
    return Chroma(
        collection_name=s.collection_name,
        embedding_function=OpenAIEmbeddings(model=s.embedding_model),
        persist_directory=s.chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


@lru_cache
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=get_settings().llm_model, temperature=0)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9$]+", text.lower())


@lru_cache
def get_bm25() -> BM25Retriever:
    data = get_vectorstore()._collection.get(include=["documents", "metadatas"])
    docs = [Document(page_content=t, metadata=m) for t, m in zip(data["documents"], data["metadatas"])]
    return BM25Retriever.from_documents(docs, k=get_settings().fetch_k, preprocess_func=_tokenize)


def reciprocal_rank_fusion(rankings: list[list[Document]], k: int = 60) -> list[Document]:
    """Merge ranked lists: each doc scores sum(1 / (k + rank)) over the lists it appears in."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking, start=1):
            key = d.metadata.get("chunk_id") or d.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs[key] = d
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


class PassageScore(BaseModel):
    id: int
    score: int = Field(ge=0, le=10)


class RerankResult(BaseModel):
    scores: list[PassageScore]


RERANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You rank passages from the FFIEC BSA/AML Examination Manual for a question. "
               "Score every passage 0-10 for how directly it helps answer the question "
               "(10 = states the answer, 0 = irrelevant). Return a score for every id."),
    ("human", "Question: {question}\n\nPassages:\n{passages}"),
])


@traceable(run_type="chain", name="rerank")
def rerank(question: str, docs: list[Document], top_k: int) -> list[tuple[Document, float]]:
    s = get_settings()
    passages = "\n\n".join(f"[{i}] {d.page_content[:1500]}" for i, d in enumerate(docs))
    llm = ChatOpenAI(model=s.rerank_model, temperature=0).with_structured_output(RerankResult, method="function_calling")
    result = (RERANK_PROMPT | llm).invoke({"question": question, "passages": passages})
    scores = {p.id: p.score for p in result.scores if 0 <= p.id < len(docs)}
    # stable sort keeps the fused order as the tie-breaker
    order = sorted(range(len(docs)), key=lambda i: scores.get(i, 0), reverse=True)
    return [(docs[i], scores.get(i, 0) / 10) for i in order[:top_k] if scores.get(i, 0) >= s.min_rerank_score]


@traceable(run_type="retriever", name="retrieve")
def retrieve(question: str, k: int | None = None) -> list[tuple[Document, float]]:
    s = get_settings()
    top_k = k or s.top_k
    fetch_k = max(s.fetch_k, top_k) if (s.use_hybrid or s.use_rerank) else top_k

    vector_hits = [
        (d, score)
        for d, score in get_vectorstore().similarity_search_with_relevance_scores(question, k=fetch_k)
        if score >= s.min_relevance
    ]
    if not s.use_hybrid and not s.use_rerank:
        return vector_hits[:top_k]

    candidates = [d for d, _ in vector_hits]
    if s.use_hybrid:
        candidates = reciprocal_rank_fusion([candidates, get_bm25().invoke(question)])[:fetch_k]
    if not candidates:
        return []
    if s.use_rerank:
        return rerank(question, candidates, top_k)

    vector_score = {d.metadata.get("chunk_id"): score for d, score in vector_hits}
    return [(d, vector_score.get(d.metadata.get("chunk_id"), 0.0)) for d in candidates[:top_k]]


def format_context(hits: list[tuple[Document, float]]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {d.metadata.get('source')}, p. {d.metadata.get('page')}, "
        f"edition {d.metadata.get('edition') or 'unknown'})\n{d.page_content}"
        for i, (d, _) in enumerate(hits, start=1)
    )


@traceable(run_type="chain", name="aml_rag_answer")
def answer(question: str, k: int | None = None) -> dict:
    hits = retrieve(question, k)
    if not hits:
        return {"answer": NO_ANSWER, "citations": [], "contexts": []}

    chain = PROMPT | get_llm() | StrOutputParser()
    text = chain.invoke({"question": question, "context": format_context(hits), "no_answer": NO_ANSWER})

    citations = [
        {
            "id": i,
            "source": d.metadata.get("source"),
            "section": d.metadata.get("section"),
            "page": d.metadata.get("page"),
            "edition": d.metadata.get("edition") or None,
            "score": round(score, 4),
            "snippet": d.page_content[:300],
        }
        for i, (d, score) in enumerate(hits, start=1)
        if f"[{i}]" in text
    ]
    return {"answer": text, "citations": citations, "contexts": [d.page_content for d, _ in hits]}
