"""Абляция поиска: как каждый компонент влияет на качество и скорость.

Для каждого режима (vector, fts, hybrid, vector_rerank, hybrid_rerank) считает
Hit@1, Recall@3, Recall@5, MRR@10 на уровне документов и задержку поиска.
LLM не нужна — только БД и модели поиска, поэтому прогон быстрый и детерминированный.

Запуск (после `python -m src.ingest data/kb`):
    python -m scripts.eval_retrieval
    python -m scripts.eval_retrieval --modes vector hybrid_rerank
"""

import argparse
import json
import time
from pathlib import Path

from src.metrics import hit_at_k, percentile, recall_at_k, reciprocal_rank, summarize
from src.models import warmup
from src.retriever import MODES, retrieve


def load_questions(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [q for q in rows if q["gold_doc_ids"]]


def evaluate_mode(mode: str, questions) -> dict:
    rows, latencies = [], []
    for q in questions:
        t0 = time.perf_counter()
        docs = retrieve(q["question"], top_n=10, mode=mode)
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked = [d["doc_id"] for d in docs]
        gold = q["gold_doc_ids"]
        rows.append(
            {
                "hit@1": hit_at_k(ranked, gold, 1),
                "recall@3": recall_at_k(ranked, gold, 3),
                "recall@5": recall_at_k(ranked, gold, 5),
                "mrr@10": reciprocal_rank(ranked, gold, 10),
            }
        )
    result = summarize(rows)
    result["p50_ms"] = percentile(latencies, 50)
    result["p95_ms"] = percentile(latencies, 95)
    return result


def to_markdown(results: dict, n: int) -> str:
    header = (
        "| Режим | Hit@1 | Recall@3 | Recall@5 | MRR@10 | p50, мс | p95, мс |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    lines = [
        f"| {mode} | {r['hit@1']:.3f} | {r['recall@3']:.3f} | {r['recall@5']:.3f} | {r['mrr@10']:.3f} "
        f"| {r['p50_ms']:.0f} | {r['p95_ms']:.0f} |"
        for mode, r in results.items()
    ]
    return f"Вопросов: {n}\n\n" + header + "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--out", type=Path, default=Path("results/retrieval_ablation.md"))
    args = parser.parse_args()

    questions = load_questions(args.questions)
    warmup()
    results = {mode: evaluate_mode(mode, questions) for mode in args.modes}

    table = to_markdown(results, len(questions))
    print(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table, encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
