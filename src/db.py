import os
import psycopg2
from pgvector.psycopg2 import register_vector

DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:postgres_password@localhost:5432/enterprise_rag")


def get_connection():
    conn = psycopg2.connect(DB_URI)

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    register_vector(conn)
    return conn


def init_db():
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                doc_id VARCHAR(64) NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding vector(384)
            );
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx 
            ON document_chunks USING hnsw (embedding vector_cosine_ops);
        """)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("БД и HNSW-индекс успешно инициализированы.")