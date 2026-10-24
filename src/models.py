"""Ленивая загрузка моделей поиска.

Модели грузятся при первом обращении, а не при импорте: так тесты и CLI-утилиты,
которым модели не нужны, запускаются мгновенно.

E5 обучалась с префиксами: запросы кодируются как "query: ...", документы как
"passage: ...". Без префиксов качество поиска заметно падает.
"""

from collections.abc import Sequence
from functools import lru_cache

from src.config import get_settings


@lru_cache
def _bi_encoder():
    from sentence_transformers import SentenceTransformer

    s = get_settings()
    model = SentenceTransformer(s.embedding_model, device=s.device)
    dim = model.get_sentence_embedding_dimension()
    if dim != s.embedding_dim:
        raise ValueError(
            f"Размерность {s.embedding_model} = {dim}, а EMBEDDING_DIM = {s.embedding_dim}. "
            "Поменяйте EMBEDDING_DIM и пересоздайте таблицы."
        )
    return model


@lru_cache
def _reranker():
    from sentence_transformers import CrossEncoder

    s = get_settings()
    return CrossEncoder(s.reranker_model, device=s.device)


def _uses_e5_prefixes() -> bool:
    return "e5" in get_settings().embedding_model.lower()


def embed_query(text: str) -> list[float]:
    prefix = "query: " if _uses_e5_prefixes() else ""
    return _bi_encoder().encode(prefix + text, normalize_embeddings=True).tolist()


def embed_passages(texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
    prefix = "passage: " if _uses_e5_prefixes() else ""
    vectors = _bi_encoder().encode(
        [prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True
    )
    return [v.tolist() for v in vectors]


def rerank_scores(query: str, passages: Sequence[str]) -> list[float]:
    pairs: list[tuple[str, str]] = [(query, p) for p in passages]
    return [float(x) for x in _reranker().predict(pairs)]


def warmup() -> None:
    """Прогрев при старте сервера, чтобы первый запрос не ждал загрузки моделей."""
    embed_query("прогрев")
    rerank_scores("прогрев", ["прогрев"])
