"""Разбор ответов LLM. Принцип: при любой неоднозначности — безопасное значение
(fail-closed), а не «всё хорошо»."""

import re

INTENTS = ("faq", "action", "escalate")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_SCORE_RE = re.compile(r'"score"\s*:\s*"?(\d{1,2}(?:[.,]\d+)?)')
_NUMBER_RE = re.compile(r"(?<![\d.])(\d{1,2}(?:[.,]\d+)?)(?![\d])")


def strip_reasoning(text: str) -> str:
    """Убирает блоки рассуждений <think>...</think>, которые выдают reasoning-модели."""
    return _THINK_RE.sub("", text or "").strip()


def parse_intent(raw: str, default: str = "faq") -> str:
    """Ищет первую из допустимых категорий в ответе модели.

    Модели часто отвечают «Категория: action.» вместо одного слова,
    поэтому точное сравнение строки ломается.
    """
    text = strip_reasoning(raw).lower()
    positions = {intent: text.find(intent) for intent in INTENTS if intent in text}
    if not positions:
        return default
    return min(positions, key=positions.get)


def parse_score(raw: str) -> float | None:
    """Достаёт оценку 0..10 и нормирует в 0..1. Возвращает None, если оценки нет.

    Вызывающий код трактует None как провал проверки: непонятный ответ судьи
    не должен превращаться в «ответ подтверждён» и попадать в кэш.
    """
    text = strip_reasoning(raw)
    match = _JSON_SCORE_RE.search(text) or _NUMBER_RE.search(text)
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    if not 0.0 <= value <= 10.0:
        return None
    return value / 10.0
