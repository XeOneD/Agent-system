"""Пул соединений и схема БД.

Раньше каждое обращение открывало новое соединение и выполняло
CREATE EXTENSION. Теперь расширение создаётся один раз в init_db(),
а соединения берутся из пула.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock

from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection as PgConnection
from psycopg2.pool import ThreadedConnectionPool

from src.config import get_settings

_pool: ThreadedConnectionPool | None = None
_pool_lock = Lock()
_registered: set[int] = set()


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                s = get_settings()
                _pool = ThreadedConnectionPool(s.db_pool_min, s.db_pool_max, dsn=s.database_url)
    return _pool


@contextmanager
def get_conn() -> Iterator[PgConnection]:
    """Соединение из пула: commit при успехе, rollback при ошибке."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        if id(conn) not in _registered:
            register_vector(conn)
            _registered.add(id(conn))
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
        _registered.clear()


def init_db() -> None:
    dim = get_settings().embedding_dim
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            # Расширение должно существовать до register_vector.
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    finally:
        pool.putconn(conn)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id          SERIAL PRIMARY KEY,
                doc_id      TEXT NOT NULL,
                title       TEXT NOT NULL,
                chunk_index INT  NOT NULL,
                content     TEXT NOT NULL,
                embedding   vector({dim}) NOT NULL,
                tsv         tsvector GENERATED ALWAYS AS
                            (to_tsvector('russian', title || ' ' || content)) STORED,
                UNIQUE (doc_id, chunk_index)
            );
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
                ON document_chunks USING hnsw (embedding vector_cosine_ops);
            CREATE INDEX IF NOT EXISTS document_chunks_tsv_idx
                ON document_chunks USING gin (tsv);

            CREATE TABLE IF NOT EXISTS semantic_cache (
                id         SERIAL PRIMARY KEY,
                query      TEXT NOT NULL,
                response   TEXT NOT NULL,
                sources    JSONB NOT NULL DEFAULT '[]',
                embedding  vector({dim}) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            CREATE INDEX IF NOT EXISTS semantic_cache_embedding_idx
                ON semantic_cache USING hnsw (embedding vector_cosine_ops);

            CREATE TABLE IF NOT EXISTS tickets (
                id         SERIAL PRIMARY KEY,
                category   TEXT NOT NULL,
                user_id    TEXT NOT NULL,
                text       TEXT NOT NULL,
                status     TEXT NOT NULL DEFAULT 'open',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )


if __name__ == "__main__":
    init_db()
    print("Схема БД создана.")
