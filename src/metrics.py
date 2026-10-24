"""Метрики качества поиска на уровне документов."""

from collections.abc import Iterable, Sequence
from statistics import mean


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(relevant & set(retrieved[:k])) / len(relevant)


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    return 1.0 if set(relevant) & set(retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    relevant = set(relevant)
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Перцентиль с линейной интерполяцией (как numpy.percentile по умолчанию)."""
    if not values:
        return float("nan")
    data = sorted(values)
    pos = (len(data) - 1) * q / 100.0
    lo, hi = int(pos), min(int(pos) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys() if rows else []
    return {key: mean(row[key] for row in rows) for key in keys}
