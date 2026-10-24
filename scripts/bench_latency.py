"""Замер задержки API: полный проход графа vs попадание в семантический кэш.

Для каждого вопроса делается два запроса подряд: первый идёт через граф
(cache miss), второй должен попасть в кэш. Перед замером кэш очищается.
Сервер должен быть запущен (docker compose up).

Запуск:
    python -m scripts.bench_latency --url http://localhost:8000 --n 20

Результат честно включает HTTP, эмбеддинг запроса и поиск в pgvector.
Указывайте в README железо, на котором делали замер.
"""

import argparse
import json
import urllib.request
from pathlib import Path

from src import cache
from src.metrics import percentile


def ask(url: str, query: str) -> dict:
    body = json.dumps({"query": query, "user_id": "bench"}).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/api/v1/chat", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    args = parser.parse_args()

    rows = [
        json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    questions = [q["question"] for q in rows if q["intent"] == "faq"][: args.n]

    cache.clear()
    miss, hit, not_cached = [], [], 0
    for q in questions:
        first = ask(args.url, q)
        second = ask(args.url, q)
        miss.append(first["latency_ms"])
        if second["cached"]:
            hit.append(second["latency_ms"])
        else:
            not_cached += 1  # ответ не прошёл проверку или ушёл в эскалацию — в кэш не попал

    def fmt(values):
        return f"p50 {percentile(values, 50):.0f} мс, p95 {percentile(values, 95):.0f} мс (n={len(values)})"

    print(f"Полный проход графа: {fmt(miss)}")
    print(f"Попадание в кэш:     {fmt(hit)}")
    print(f"Не закэшировано:     {not_cached}")


if __name__ == "__main__":
    main()
