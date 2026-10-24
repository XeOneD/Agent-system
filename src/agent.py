"""Граф обработки обращения на LangGraph.

router ──faq──▶ retriever ──есть контекст──▶ generator ──▶ evaluator
  │                  └──пусто──▶ escalation        ▲            │
  ├──action──▶ action (заявка в tickets)           │  score < порога и попытки есть
  └──escalate──▶ escalation (заявка в tickets)      └────────────┤
                                                   score ≥ порога ──▶ END
                                                   попытки кончились ──▶ escalation
"""

import logging
import time
from functools import lru_cache
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src import prompts
from src.config import get_settings
from src.llm import get_judge, get_llm
from src.parsing import parse_intent, parse_score, strip_reasoning
from src.retriever import retrieve
from src.routing import route_after_evaluation, route_after_retrieval, route_after_router
from src.tools import create_ticket

log = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    query: str
    user_id: str
    intent: str
    context: list[dict[str, Any]]
    response: str
    evaluation_score: float | None
    attempts: int
    ticket_id: int | None
    timings_ms: dict[str, float]


def _timed(state: AgentState, name: str, started: float) -> dict[str, float]:
    timings = dict(state.get("timings_ms") or {})
    timings[name] = round(timings.get(name, 0.0) + (time.perf_counter() - started) * 1000, 1)
    return timings


def _format_context(context: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[{i}] {doc['title']}\n{doc['content']}" for i, doc in enumerate(context, start=1))


def router_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    raw = get_llm().invoke([HumanMessage(content=prompts.ROUTER.format(query=state["query"]))]).content
    return {"intent": parse_intent(raw), "timings_ms": _timed(state, "router", t0)}


def retriever_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    return {"context": retrieve(state["query"]), "timings_ms": _timed(state, "retriever", t0)}


def generator_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    attempts = state.get("attempts", 0)
    system = prompts.GENERATOR_SYSTEM
    if attempts > 0:
        system += prompts.GENERATOR_STRICT_SUFFIX
    user = prompts.GENERATOR_USER.format(context=_format_context(state["context"]), query=state["query"])
    raw = get_llm().invoke([SystemMessage(content=system), HumanMessage(content=user)]).content
    return {
        "response": strip_reasoning(raw),
        "attempts": attempts + 1,
        "timings_ms": _timed(state, "generator", t0),
    }


def evaluator_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    prompt = prompts.JUDGE.format(context=_format_context(state["context"]), answer=state["response"])
    raw = get_judge().invoke([HumanMessage(content=prompt)]).content
    score = parse_score(raw)
    if score is None:
        # Fail-closed: непонятный ответ судьи — это провал проверки, а не «всё хорошо».
        log.warning("Судья вернул неразборчивый ответ: %r", raw[:200])
        score = 0.0
    return {"evaluation_score": score, "timings_ms": _timed(state, "evaluator", t0)}


def action_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    ticket_id = create_ticket("action_request", state.get("user_id", "anonymous"), state["query"])
    return {
        "ticket_id": ticket_id,
        "response": (
            f"Создал заявку №{ticket_id}. Операции с аккаунтом и платежами выполняет специалист, "
            "статус заявки придёт на почту, привязанную к аккаунту."
        ),
        "timings_ms": _timed(state, "action", t0),
    }


def escalation_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    ticket_id = create_ticket("escalation", state.get("user_id", "anonymous"), state["query"])
    return {
        "ticket_id": ticket_id,
        "response": (
            f"Передал обращение специалисту поддержки, номер заявки №{ticket_id}. "
            "Ответ придёт на почту, привязанную к аккаунту."
        ),
        "timings_ms": _timed(state, "escalation", t0),
    }


def _after_evaluation(state: AgentState) -> str:
    s = get_settings()
    return route_after_evaluation(state, s.eval_threshold, s.max_generation_attempts)


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("retriever", retriever_node)
    g.add_node("generator", generator_node)
    g.add_node("evaluator", evaluator_node)
    g.add_node("action", action_node)
    g.add_node("escalation", escalation_node)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        route_after_router,
        {"retriever": "retriever", "action": "action", "escalation": "escalation"},
    )
    g.add_conditional_edges(
        "retriever",
        route_after_retrieval,
        {"generator": "generator", "escalation": "escalation"},
    )
    g.add_edge("generator", "evaluator")
    g.add_conditional_edges(
        "evaluator",
        _after_evaluation,
        {"__end__": END, "generator": "generator", "escalation": "escalation"},
    )
    g.add_edge("action", END)
    g.add_edge("escalation", END)
    return g.compile()


@lru_cache
def get_graph():
    return build_graph()


def run(query: str, user_id: str = "anonymous") -> AgentState:
    state: AgentState = {"query": query, "user_id": user_id, "attempts": 0, "timings_ms": {}}
    return get_graph().invoke(state)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    result = run(" ".join(sys.argv[1:]) or "Как вернуть деньги за случайное продление подписки?")
    print(f"Намерение: {result.get('intent')}")
    print(f"Оценка:    {result.get('evaluation_score')}")
    print(f"Источники: {[d['doc_id'] for d in result.get('context', [])]}")
    print(f"Время, мс: {result.get('timings_ms')}")
    print(f"\n{result.get('response')}")
