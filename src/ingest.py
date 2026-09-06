import json
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from db import get_connection, init_db

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def process_and_store_documents(json_filepath: str):
    init_db()

    with open(json_filepath, "r", encoding="utf-8") as f:
        documents = json.load(f)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    conn = get_connection()
    cursor = conn.cursor()

    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        embeddings = embedding_model.encode(chunks, show_progress_bar=False)

        for chunk, embedding in zip(chunks, embeddings):
            cursor.execute(
                """
                INSERT INTO document_chunks (doc_id, content, metadata, embedding)
                VALUES (%s, %s, %s, %s);
                """,
                (doc["id"], chunk, json.dumps({"title": doc.get("title", "")}), embedding.tolist())
            )

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Документы из {json_filepath} успешно обработаны и внесены в pgvector.")


if __name__ == "__main__":
    process_and_store_documents("data/kb_documents.json")