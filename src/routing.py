"""Решения о переходах в графе. Вынесены в чистые функции, чтобы их можно было
протестировать без LLM и БД."""

from collections.abc import Mapping
from typing import Any


def route_after_router(state: Mapping[str, Any]) -> str:
    intent = state.get("intent")
    if intent == "action":
        return "action"
    if intent == "escalate":
        return "escalation"
    return "retriever"


def route_after_retrieval(state: Mapping[str, Any]) -> str:
    """Если поиск ничего не нашёл, генерировать нечего: сразу к человеку."""
    return "generator" if state.get("context") else "escalation"


def route_after_evaluation(state: Mapping[str, Any], threshold: float, max_attempts: int) -> str:
    """Принимаем ответ, перегенерируем или эскалируем.

    attempts - сколько раз уже сгенерирован ответ. Если оценка ниже порога и
    попытки остались, возвращаемся в генератор (в строгом режиме). Если попытки
    кончились - эскалация. Ответ с низкой оценкой пользователю не отдаётся.
    """
    score = state.get("evaluation_score")
    if score is not None and score >= threshold:
        return "__end__"
    if state.get("attempts", 0) < max_attempts:
        return "generator"
    return "escalation"
