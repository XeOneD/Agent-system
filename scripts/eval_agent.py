"""Оценка агента целиком (нужна LLM).

Считает:
- точность роутера (faq / action / escalate) по всем вопросам;
- для FAQ-вопросов: долю ответов, прошедших проверку с первой попытки, долю эскалаций,
  среднюю оценку судьи и задержку;
- с флагом --ragas: метрики RAGAS (faithfulness, context recall) по эталонным ответам.

Запуск:
    python -m scripts.eval_agent
    python -m scripts.eval_agent --ragas      # нужен requirements-eval.txt

Важно: судить ответы маленькой локальной моделью ненадёжно. Для --ragas и JUDGE_*
лучше указать более сильную модель (например, через OpenRouter), см. .env.example.
Вопросы action/escalate создают заявки в таблице tickets - запускайте на тестовой БД.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean

from src.agent import run
from src.metrics import percentile


def load(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_ragas(samples) -> dict:
    try:
        from langchain_openai import ChatOpenAI
        from ragas import EvaluationDataset, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, LLMContextRecall
    except ImportError as exc:
        raise SystemExit(
            f"Для --ragas установите зависимости: pip install -r requirements-eval.txt ({exc})"
        ) from exc

    from src.config import get_settings

    s = get_settings()
    judge = ChatOpenAI(
        model=s.judge_model or s.llm_model,
        base_url=s.judge_base_url or s.llm_base_url,
        api_key=s.judge_api_key or s.llm_api_key,
        temperature=0.0,
    )
    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), LLMContextRecall()],
        llm=LangchainLLMWrapper(judge),
    )
    means = result.to_pandas().mean(numeric_only=True)
    return {name: float(value) for name, value in means.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    parser.add_argument("--ragas", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/agent_eval.json"))
    args = parser.parse_args()

    questions = load(args.questions)
    router_hits, confusion = 0, Counter()
    faq_rows, ragas_samples, latencies = [], [], []

    for q in questions:
        t0 = time.perf_counter()
        result = run(q["question"], user_id="eval")
        latencies.append((time.perf_counter() - t0) * 1000)

        predicted = result.get("intent", "faq")
        router_hits += predicted == q["intent"]
        confusion[(q["intent"], predicted)] += 1

        if q["intent"] == "faq" and predicted == "faq":
            escalated = result.get("ticket_id") is not None
            faq_rows.append(
                {
                    "escalated": escalated,
                    "first_try": not escalated and result.get("attempts") == 1,
                    "score": result.get("evaluation_score") or 0.0,
                }
            )
            if not escalated:
                ragas_samples.append(
                    {
                        "user_input": q["question"],
                        "response": result["response"],
                        "retrieved_contexts": [d["content"] for d in result.get("context", [])],
                        "reference": q["reference"],
                    }
                )
        print(f"{q['id']}: {q['intent']} → {predicted}", flush=True)

    report = {
        "questions": len(questions),
        "router_accuracy": router_hits / len(questions),
        "confusion": {f"{gold}->{pred}": n for (gold, pred), n in sorted(confusion.items())},
        "faq_answered": len(faq_rows),
        "faq_passed_first_try": mean(r["first_try"] for r in faq_rows) if faq_rows else None,
        "faq_escalation_rate": mean(r["escalated"] for r in faq_rows) if faq_rows else None,
        "faq_mean_judge_score": mean(r["score"] for r in faq_rows) if faq_rows else None,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
    }
    if args.ragas and ragas_samples:
        report["ragas"] = run_ragas(ragas_samples)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
