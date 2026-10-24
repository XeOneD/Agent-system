"""Загрузка базы знаний в pgvector.

Формат: папка с .md-файлами, первая строка — заголовок `# ...`, имя файла — doc_id.
Загрузка идемпотентна: документ перезаписывается целиком, повторный запуск
не создаёт дублей. После загрузки семантический кэш очищается, чтобы в нём
не остались ответы по старой версии базы.

Запуск:  python -m src.ingest data/kb
"""

import argparse
from pathlib import Path

from src import cache
from src.db import get_conn, init_db
from src.models import embed_passages


def read_documents(folder: Path) -> list[tuple[str, str, str]]:
    docs = []
    for path in sorted(folder.glob("*.md")):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines or not lines[0].startswith("# "):
            raise ValueError(f"{path}: первая строка должна быть заголовком '# ...'")
        title = lines[0][2:].strip()
        body = "\n".join(lines[1:]).strip()
        docs.append((path.stem, title, body))
    if not docs:
        raise ValueError(f"В {folder} нет .md-файлов")
    return docs


def ingest(folder: Path, chunk_size: int, chunk_overlap: int) -> None:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    init_db()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = read_documents(folder)
    total_chunks = 0
    with get_conn() as conn, conn.cursor() as cur:
        doc_ids = [doc_id for doc_id, _, _ in docs]
        cur.execute("DELETE FROM document_chunks WHERE NOT (doc_id = ANY(%s));", (doc_ids,))
        for doc_id, title, body in docs:
            chunks = splitter.split_text(body)
            # Заголовок добавляем в текст для эмбеддинга: короткие чанки без него теряют смысл.
            vectors = embed_passages([f"{title}. {chunk}" for chunk in chunks])
            cur.execute("DELETE FROM document_chunks WHERE doc_id = %s;", (doc_id,))
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=False)):
                cur.execute(
                    """
                    INSERT INTO document_chunks (doc_id, title, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                    (doc_id, title, index, chunk, vector),
                )
            total_chunks += len(chunks)
    cleared = cache.clear()
    print(f"Загружено документов: {len(docs)}, чанков: {total_chunks}. Очищено записей кэша: {cleared}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Загрузка базы знаний в pgvector")
    parser.add_argument("folder", type=Path, nargs="?", default=Path("data/kb"))
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args()
    ingest(args.folder, args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()
