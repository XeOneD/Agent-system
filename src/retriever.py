from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.db import get_connection

bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def search_candidates(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    query_vector = bi_encoder.encode(query).tolist()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, doc_id, content, metadata, (embedding <=> %s::vector) AS distance
        FROM document_chunks
        ORDER BY distance ASC
        LIMIT %s;
    """, (query_vector, top_k))

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    candidates = []
    for row in rows:
        candidates.append({
            "id": row[0],
            "doc_id": row[1],
            "content": row[2],
            "metadata": row[3],
            "distance": float(row[4])
        })
    return candidates


def get_relevant_context(query: str, top_k: int = 10, top_n: int = 3) -> List[Dict[str, Any]]:
    candidates = search_candidates(query, top_k=top_k)

    if not candidates:
        return []

    pairs = [[query, c["content"]] for c in candidates]
    scores = reranker.predict(pairs)

    for idx, score in enumerate(scores):
        candidates[idx]["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]


if __name__ == "__main__":
    test_query = "Не списалась оплата за подписку, что делать?"
    context = get_relevant_context(test_query, top_k=5, top_n=2)

    print(f"--- Результаты для запроса: '{test_query}' ---")
    for item in context:
        print(f"Rerank Score: {item['rerank_score']:.4f} | Cosine Dist: {item['distance']:.4f}")
        print(f"Текст: {item['content']}\n")