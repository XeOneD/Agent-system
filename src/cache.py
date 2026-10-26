"""Семантический кэш ответов на FAQ-вопросы.
"""

import json
from typing import Any

from src.config import get_settings
from src.db import get_conn
from src.models import embed_query


def lookup(query: str) -> dict[str, Any] | None:
    s = get_settings()
    if not s.cache_enabled:
        return None
    vector = embed_query(query)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT response, sources, 1 - (embedding <=> %s::vector) AS similarity
            FROM semantic_cache
            WHERE created_at > now() - make_interval(hours => %s)
            ORDER BY embedding <=> %s::vector
            LIMIT 1;
            """,
            (vector, s.cache_ttl_hours, vector),
        )
        row = cur.fetchone()
    if row is None or float(row[2]) < s.cache_similarity_threshold:
        return None
    return {"response": row[0], "sources": row[1], "similarity": float(row[2])}


def store(query: str, response: str, sources: list[dict[str, Any]]) -> None:
    if not get_settings().cache_enabled:
        return
    vector = embed_query(query)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO semantic_cache (query, response, sources, embedding) VALUES (%s, %s, %s, %s);",
            (query, response, json.dumps(sources, ensure_ascii=False), vector),
        )


def clear() -> int:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM semantic_cache;")
        return cur.rowcount
