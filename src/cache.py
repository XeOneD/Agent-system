from typing import Optional
from src.retriever import bi_encoder
from src.db import get_connection

SIMILARITY_THRESHOLD = 0.85


def init_cache_db():
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id SERIAL PRIMARY KEY,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                embedding vector(384),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS cache_embedding_idx 
            ON semantic_cache USING hnsw (embedding vector_cosine_ops);
        """)
    conn.commit()
    conn.close()


def get_cached_response(query: str) -> Optional[str]:
    query_vector = bi_encoder.encode(query).tolist()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT response, (1 - (embedding <=> %s::vector)) AS similarity
        FROM semantic_cache
        WHERE (1 - (embedding <=> %s::vector)) >= %s
        ORDER BY similarity DESC
        LIMIT 1;
    """, (query_vector, query_vector, SIMILARITY_THRESHOLD))

    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row:
        return row[0]
    return None


def set_cached_response(query: str, response: str):
    query_vector = bi_encoder.encode(query).tolist()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO semantic_cache (query, response, embedding)
        VALUES (%s, %s, %s);
    """, (query, response, query_vector))
    conn.commit()
    cursor.close()
    conn.close()