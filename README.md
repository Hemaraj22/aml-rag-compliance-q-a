# AML/Compliance RAG

Retrieval-Augmented Generation over the **FFIEC BSA/AML Examination Manual**, with
citation-backed answers, Ragas evaluation, LangSmith tracing, a FastAPI service, and Docker.

```
PDFs ──► PyPDFLoader ──► SemanticChunker (OpenAI embeddings) ──► ChromaDB (cosine)
                                                                   │
question ──► FastAPI /ask ──► top-k retrieval + relevance filter ──┘
                                   │
                                   ▼
                     GPT (temp 0) with numbered context ──► answer + [n] citations
                                   │
                         LangSmith traces (latency, tokens, cost)

eval/benchmark.jsonl ──► run_eval.py ──► Ragas: faithfulness, context precision, answer relevancy
```

## Layout

| Path | Purpose |
|---|---|
| `app/ingest.py` | Load PDFs, semantic chunking with size cap, dedupe, index into Chroma |
| `app/rag.py` | Retrieval, grounded prompt, citation extraction, `@traceable` spans |
| `app/main.py` | FastAPI: `GET /health`, `GET /stats`, `POST /ask`, `POST /ingest` |
| `eval/benchmark.jsonl` | Hand-curated Q&A benchmark (question + reference answer) |
| `eval/run_eval.py` | Runs the pipeline over the benchmark and scores it with Ragas; CI gate via `--fail-under` |
| `tests/` | API and unit tests (no OpenAI calls) |

## 1. Get the source documents

Download sections of the manual as PDFs from <https://bsaaml.ffiec.gov/manual> and save them to
`data/raw/`. Good starting sections: *Introduction / BSA/AML Compliance Program*, *Customer
Identification Program*, *Customer Due Diligence / Beneficial Ownership*, *Suspicious Activity
Reporting*, *Currency Transaction Reporting*, *CTR Exemptions*, *Purchase and Sale of Monetary
Instruments*, *Funds Transfers Recordkeeping*. Use descriptive filenames (for example
`suspicious_activity_reporting.pdf`); the filename becomes the citation's section name.

## 2. Run locally

```bash
cp .env.example .env        # add OPENAI_API_KEY and LANGSMITH_API_KEY
make install
make ingest                 # builds ./chroma_db
make serve                  # http://localhost:8000/docs
make test
make eval
```

Example request:

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "What is the deadline for filing a SAR?"}' | python -m json.tool
```

## 3. Run with Docker

```bash
docker compose --profile jobs run --rm ingest   # index PDFs into the chroma volume
docker compose up --build api                   # serve on :8000
docker compose --profile jobs run --rm eval     # Ragas eval, fails if any metric < 0.80
```

## 4. Evaluation

`eval/benchmark.jsonl` ships with 14 starter questions (CTR, SAR, CIP, beneficial ownership,
recordkeeping, plus one out-of-scope question that should be refused). **Before you rely on
the scores, check every reference answer against the manual text and expand the set to 50+
questions.** A small benchmark gives noisy metrics.

Metrics (Ragas 0.2):
- **Faithfulness**: share of answer claims supported by the retrieved context
- **Context precision** (`LLMContextPrecisionWithReference`): whether relevant chunks are ranked above irrelevant ones
- **Answer relevancy** (`ResponseRelevancy`): how directly the answer addresses the question

Results go to `eval/results/<timestamp>.csv` (per question) and `-summary.json`.

### Results

13 in-scope questions plus 1 out-of-scope question, graded by Ragas with a gpt-4o judge.
Corpus: FFIEC BSA/AML Examination Manual (2014 edition, Feb 2015 revision) plus current sections
(Introduction 2026, SAR 2025, CIP and CTR 2021, CDD and Beneficial Ownership 2018, Iranian-linked FI
reporting 2023): 1,532 chunks from 499 pages. Each chunk carries its edition year, and the prompt tells
the model to follow the most recent edition when passages conflict.

| Retrieval | Faithfulness | Context precision | Answer relevancy | Refusal accuracy |
|---|---|---|---|---|
| Baseline: vector top-5 | 0.904 | 0.674 | 0.710 | 1.00 |
| Hybrid (vector + BM25, RRF) with LLM reranker | 1.000 | 0.908 | 0.807 | 1.00 |
| Same, with current manual sections and edition-aware prompt | **0.974** | **0.954** | **0.794** | 1.00 |

What changed: vector search alone missed exact-wording passages. For example, it ranked the
"retain CTRs for five years" chunk above the "file within 15 calendar days" chunk from the same
page. Fusing BM25 keyword results and re-ranking 20 candidates with gpt-4o-mini fixed this.
Adding the current sections fixed `bo-01` (beneficial ownership): the 2014 manual predates the
2016 CDD rule, so context precision on that question went from 0.00 to 0.83.

Answer relevancy is the weakest metric. Ragas computes it by generating questions back from the
answer, so complete answers that include exceptions (for example, the SAR 60-day extension) score
lower even when correct. With 13 questions, each one moves an average by about 7 points, so
differences of 1 to 2 points between runs are noise.

## 5. Improving the scores

Change one setting at a time and re-run `make eval`:
- `breakpoint_threshold_amount` in `ingest.py` (85–95) controls chunk granularity
- `TOP_K` and `MIN_RELEVANCE` in `.env` trade context precision against recall
- `EMBEDDING_MODEL=text-embedding-3-large` usually improves retrieval
- Add a reranker, such as a cross-encoder, between retrieval and generation
- In LangSmith, filter for slow or expensive runs, then compare the `retrieve` and LLM spans

## 6. LangSmith

With `LANGSMITH_TRACING=true`, every `/ask` call creates an `aml_rag_answer` trace containing a
`retrieve` span and the LLM call, with token counts and cost. Eval runs are traced too, so you
can compare runs in the LangSmith project `aml-rag`.
