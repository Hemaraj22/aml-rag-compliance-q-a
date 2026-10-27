"""API tests with the RAG layer stubbed out (no OpenAI calls)."""
from fastapi.testclient import TestClient

from app import main, rag

client = TestClient(main.app)


def fake_answer(question, top_k=None):
    return {
        "answer": "CTRs are required for currency transactions over $10,000 [1].",
        "citations": [
            {"id": 1, "source": "ctr.pdf", "section": "ctr", "page": 3, "score": 0.82, "snippet": "..."}
        ],
        "contexts": ["..."],
    }


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_with_citations(monkeypatch):
    monkeypatch.setattr(rag, "answer", fake_answer)
    r = client.post("/ask", json={"question": "What is the CTR threshold?"})
    assert r.status_code == 200
    body = r.json()
    assert "[1]" in body["answer"]
    assert body["citations"][0]["page"] == 3
    assert body["latency_ms"] >= 0


def test_ask_validates_input():
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "top_k": 99}).status_code == 422


def test_format_context_numbers_passages():
    from langchain_core.documents import Document

    hits = [(Document("alpha", metadata={"source": "a.pdf", "page": 1, "edition": 2021}), 0.9),
            (Document("beta", metadata={"source": "b.pdf", "page": 7}), 0.8)]
    ctx = rag.format_context(hits)
    assert ctx.startswith("[1] (source: a.pdf, p. 1, edition 2021)")
    assert "[2] (source: b.pdf, p. 7, edition unknown)\nbeta" in ctx


def test_detect_edition_prefers_header_dates():
    from app.ingest import detect_edition

    assert detect_edition(["Title\nFFIEC BSA/AML Examination Manual 3 February 2021\nCites a 2010 rule"]) == 2021
    assert detect_edition(["No header. Amended in 2011 and again in 2023."]) == 2023
    assert detect_edition(["no dates here"]) is None


def test_answer_without_hits_refuses(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q, k=None: [])
    out = rag.answer("unrelated question")
    assert out["answer"] == rag.NO_ANSWER
    assert out["citations"] == []


def test_reciprocal_rank_fusion_rewards_docs_found_by_both():
    from langchain_core.documents import Document

    a, b, c = (Document(x, metadata={"chunk_id": x}) for x in "abc")
    fused = rag.reciprocal_rank_fusion([[a, b], [c, b]])
    assert fused[0] is b  # in both lists, so it outranks a and c
    assert {d.page_content for d in fused} == {"a", "b", "c"}


def test_rerank_orders_by_score_and_drops_weak_passages(monkeypatch):
    from langchain_core.documents import Document

    class FakeLLM:
        def with_structured_output(self, schema, **kwargs):
            return lambda _: rag.RerankResult(scores=[
                rag.PassageScore(id=0, score=2), rag.PassageScore(id=1, score=9), rag.PassageScore(id=2, score=6)])

    monkeypatch.setattr(rag, "ChatOpenAI", lambda **kw: FakeLLM())
    docs = [Document(x) for x in ("weak", "best", "ok")]
    out = rag.rerank("q", docs, top_k=5)
    assert [d.page_content for d, _ in out] == ["best", "ok"]  # "weak" (2 < 3) is dropped
    assert out[0][1] == 0.9
