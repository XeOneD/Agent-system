"""Поиск контекста для ответа.

Режимы (нужны для абляции в scripts/eval_retrieval.py):
  vector         — только pgvector (косинусная близость e5-эмбеддингов)
  fts            — только полнотекстовый поиск PostgreSQL (russian, ts_rank_cd)
  hybrid         — vector + fts, объединение через Reciprocal Rank Fusion
  vector_rerank  — vector + переранжирование cross-encoder'ом
  hybrid_rerank  — hybrid + переранжирование cross-encoder'ом (по умолчанию)
"""

from typing import Any

from src.config import get_settings
from src.db import get_conn
from src.fusion import build_or_tsquery, dedupe_by_doc, reciprocal_rank_fusion
from src.models import embed_query, rerank_scores

MODES = ("vector", "fts", "hybrid", "vector_rerank", "hybrid_rerank")

_COLUMNS = ("id", "doc_id", "title", "content")


def vector_search(query: str, limit: int) -> list[dict[str, Any]]:
    vector = embed_query(query)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, doc_id, title, content, 1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (vector, vector, limit),
        )
        rows = cur.fetchall()
    return [{**dict(zip(_COLUMNS, row[:4], strict=False)), "vector_score": float(row[4])} for row in rows]


def fts_search(query: str, limit: int) -> list[dict[str, Any]]:
    tsquery = build_or_tsquery(query)
    if not tsquery:
        return []
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, doc_id, title, content, ts_rank_cd(tsv, q) AS score
            FROM document_chunks, to_tsquery('russian', %s) AS q
            WHERE tsv @@ q
            ORDER BY score DESC
            LIMIT %s;
            """,
            (tsquery, limit),
        )
        rows = cur.fetchall()
    return [{**dict(zip(_COLUMNS, row[:4], strict=False)), "fts_score": float(row[4])} for row in rows]


def rerank(query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    scores = rerank_scores(query, [f"{c['title']}. {c['content']}" for c in candidates])
    for candidate, score in zip(candidates, scores, strict=False):
        candidate["rerank_score"] = score
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)


def retrieve(query: str, top_n: int | None = None, mode: str | None = None) -> list[dict[str, Any]]:
    """Возвращает top_n чанков из разных документов, лучшие первыми."""
    s = get_settings()
    mode = mode or s.retrieval_mode
    top_n = top_n or s.context_top_n
    if mode not in MODES:
        raise ValueError(f"Неизвестный режим поиска {mode!r}, допустимые: {MODES}")

    limit = s.candidates_per_retriever
    if mode == "vector":
        candidates = vector_search(query, limit)
    elif mode == "fts":
        candidates = fts_search(query, limit)
    elif mode == "vector_rerank":
        candidates = rerank(query, vector_search(query, s.rerank_candidates))
    else:
        fused = reciprocal_rank_fusion([vector_search(query, limit), fts_search(query, limit)], k=s.rrf_k)
        if mode == "hybrid_rerank":
            fused = rerank(query, fused[: s.rerank_candidates])
        candidates = fused

    return dedupe_by_doc(candidates, top_n)
