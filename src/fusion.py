"""Чистые функции гибридного поиска: построение FTS-запроса и Reciprocal Rank Fusion.

Модуль не зависит от БД и моделей, поэтому покрыт unit-тестами.
"""

import re
from collections.abc import Iterable, Sequence
from typing import Any

_TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)


def build_or_tsquery(text: str, min_len: int = 2) -> str:
    """Превращает вопрос пользователя в tsquery вида `слово1 | слово2 | ...`.

    websearch_to_tsquery требует совпадения ВСЕХ слов, и на разговорных вопросах
    полнотекстовый поиск почти ничего не находит. Поэтому объединяем термы через OR,
    а ранжирование оставляем ts_rank_cd. Стемминг и стоп-слова делает Postgres
    (конфигурация 'russian') при вызове to_tsquery.

    В выход попадают только буквы и цифры, так что спецсимволы tsquery
    (&, |, !, :, скобки, кавычки) не могут сломать запрос.
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= min_len]
    unique = list(dict.fromkeys(tokens))
    return " | ".join(unique)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[dict[str, Any]]],
    k: int = 60,
    key: str = "id",
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion (Cormack et al., 2009): score = sum(1 / (k + rank)).

    Не требует калибровки скоров разных поисков между собой, только рангов.
    Возвращает объединённый список, отсортированный по rrf_score.
    """
    fused: dict[Any, dict[str, Any]] = {}
    for results in ranked_lists:
        for rank, item in enumerate(results, start=1):
            item_key = item[key]
            if item_key not in fused:
                fused[item_key] = {**item, "rrf_score": 0.0}
            fused[item_key]["rrf_score"] += 1.0 / (k + rank)
    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)


def dedupe_by_doc(chunks: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Оставляет лучший чанк каждого документа, сохраняя порядок. Нужно для метрик
    на уровне документов и чтобы в контекст LLM не попадали дубли одного документа."""
    seen = set()
    out: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk["doc_id"] in seen:
            continue
        seen.add(chunk["doc_id"])
        out.append(chunk)
        if len(out) >= limit:
            break
    return out
