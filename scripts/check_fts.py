"""Smoke-тест схемы и полнотекстового поиска на настоящем Postgres (используется в CI).

Создаёт схему, загружает базу знаний с нулевыми эмбеддингами (модели не нужны)
и проверяет, что FTS находит нужные документы. Работает только на пустой БД,
чтобы случайно не испортить настоящие данные.
"""

from pathlib import Path

from scripts.eval_retrieval import load_questions
from src.config import get_settings
from src.db import get_conn, init_db
from src.ingest import read_documents
from src.metrics import recall_at_k
from src.retriever import fts_search

MIN_RECALL_AT_5 = 0.5


def main() -> None:
    init_db()
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM document_chunks;")
        if cur.fetchone()[0]:
            raise SystemExit("document_chunks не пуста - запускайте проверку только на пустой БД.")
        zero = [0.0] * get_settings().embedding_dim
        for doc_id, title, body in read_documents(Path("data/kb")):
            cur.execute(
                "INSERT INTO document_chunks (doc_id, title, chunk_index, content, embedding) "
                "VALUES (%s, %s, 0, %s, %s);",
                (doc_id, title, body, zero),
            )

    questions = load_questions(Path("data/eval/questions.jsonl"))
    recalls = []
    for q in questions:
        found = [d["doc_id"] for d in fts_search(q["question"], limit=20)]
        ranked = list(dict.fromkeys(found))
        recalls.append(recall_at_k(ranked, q["gold_doc_ids"], 5))
    recall = sum(recalls) / len(recalls)
    print(f"FTS Recall@5 = {recall:.3f} на {len(questions)} вопросах")
    if recall < MIN_RECALL_AT_5:
        raise SystemExit(f"FTS Recall@5 ниже {MIN_RECALL_AT_5}: поиск или схема сломаны")


if __name__ == "__main__":
    main()
