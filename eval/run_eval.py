"""Run the RAG pipeline over the benchmark and score it with Ragas.

Usage:  python -m eval.run_eval [--limit N] [--fail-under 0.85]

Out-of-scope questions (reference == rag.NO_ANSWER) are not sent to Ragas, which scores a
correct refusal as 0; they are graded by refusal accuracy instead.

Writes per-question scores to eval/results/<timestamp>.csv and a summary JSON.
Exits non-zero if any metric falls below --fail-under (for CI gating).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, ResponseRelevancy

from app import rag
from app.config import get_settings

ROOT = Path(__file__).parent
METRICS = {
    "faithfulness": Faithfulness(),
    "llm_context_precision_with_reference": LLMContextPrecisionWithReference(),
    "answer_relevancy": ResponseRelevancy(),
}


def load_benchmark(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=ROOT / "benchmark.jsonl", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fail-under", type=float, default=0.0)
    ap.add_argument("--judge-model", default="gpt-4o")
    args = ap.parse_args()

    bench = load_benchmark(args.benchmark, args.limit)
    negatives = [r for r in bench if r["reference"] == rag.NO_ANSWER]
    bench = [r for r in bench if r["reference"] != rag.NO_ANSWER]

    refused = 0
    for row in negatives:
        out = rag.answer(row["question"])
        ok = out["answer"].strip() == rag.NO_ANSWER
        refused += ok
        print(f"[negative] {row['id']}: {'refused (correct)' if ok else 'ANSWERED (wrong): ' + out['answer'][:120]}")

    samples = []
    for i, row in enumerate(bench, 1):
        out = rag.answer(row["question"])
        print(f"[{i}/{len(bench)}] {row['id']}: {len(out['contexts'])} contexts")
        samples.append(
            {
                "user_input": row["question"],
                "response": out["answer"],
                # Ragas needs at least one context; out-of-scope questions retrieve none
                "retrieved_contexts": out["contexts"] or [""],
                "reference": row["reference"],
            }
        )

    judge = LangchainLLMWrapper(ChatOpenAI(model=args.judge_model, temperature=0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=get_settings().embedding_model))
    result = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=list(METRICS.values()),
        llm=judge,
        embeddings=emb,
        run_config=RunConfig(max_workers=4, timeout=120),
    )

    df = result.to_pandas()
    df.insert(0, "id", [r["id"] for r in bench])
    summary = {name: round(float(df[name].mean(skipna=True)), 4) for name in METRICS}
    if negatives:
        summary["refusal_accuracy"] = round(refused / len(negatives), 4)

    st = get_settings()
    config = {k: getattr(st, k) for k in
              ("llm_model", "embedding_model", "top_k", "use_hybrid", "fetch_k", "use_rerank", "rerank_model")}

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    df.to_csv(out_dir / f"{stamp}.csv", index=False)
    (out_dir / f"{stamp}-summary.json").write_text(
        json.dumps({"n": len(bench), "n_negative": len(negatives), "config": config, "judge": args.judge_model, **summary}, indent=2)
    )

    print(f"\n=== Ragas summary ({len(bench)} questions) | config: {config} ===")
    for k, v in summary.items():
        print(f"{k:40s} {v:.3f}")

    failed = [k for k, v in summary.items() if v < args.fail_under]
    if failed:
        print(f"\nFAIL: below {args.fail_under}: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
