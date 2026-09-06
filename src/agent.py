import os
from typing import Dict, Any, List, TypedDict, Literal
import json

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from src.retriever import get_relevant_context

llm = ChatOpenAI(
    model="llama3.2:3b",
    api_key="ollama",
    base_url="http://host.docker.internal:11434/v1"
)


class AgentState(TypedDict):
    query: str
    intent: str  # 'faq', 'action', 'escalate'
    context: List[Dict[str, Any]]
    response: str
    evaluation_score: float
    retry_count: int
    is_action_executed: bool


def router_node(state: AgentState) -> Dict[str, Any]:
    prompt = f"""Классифицируй намерение клиента по запросу: '{state['query']}'.
Выбери строго одну категорию:
- faq: вопрос по справке, регламентам, правилам
- action: просьба выполнить действие (сброс сессии, возврат средств)
- escalate: жалоба, агрессия или нетипичная ситуация

Ответь ТОЛЬКО названием категории в нижнем регистре."""

    res = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    intent = res if res in ["faq", "action", "escalate"] else "faq"

    return {"intent": intent}


def retriever_node(state: AgentState) -> Dict[str, Any]:
    docs = get_relevant_context(state["query"], top_k=10, top_n=3)
    return {"context": docs}


def generator_node(state: AgentState) -> Dict[str, Any]:
    context_str = "\n\n".join(
        [f"Документ #{i + 1}:\n{doc['content']}" for i, doc in enumerate(state.get("context", []))])

    system_prompt = """Ты — вежливый и точный ассистент технической поддержки.
Отвечай строго на основе предоставленного контекста. Если в контексте нет ответа, прямо скажи об этом и не придумывай факты."""

    user_prompt = f"Контекст:\n{context_str}\n\nВопрос клиента: {state['query']}"

    res = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]).content
    return {"response": res}


def action_node(state: AgentState) -> Dict[str, Any]:
    print(f"[ACTION] Выполнение автоматической операции для запроса: {state['query']}")
    response_msg = "Ваша заявка на сброс сессии/проверку платежа принята и успешно отправлена в обработку!"
    return {"response": response_msg, "is_action_executed": True}


def evaluator_node(state: AgentState) -> Dict[str, Any]:
    if not state.get("context"):
        return {"evaluation_score": 1.0}

    context_str = "\n".join([d["content"] for d in state["context"]])

    eval_prompt = f"""Оцени от 0 до 10, насколько данный ответ подтвержден контекстом и не содержит галлюцинаций:
Контекст: {context_str}
Ответ: {state['response']}

Напиши ТОЛЬКО число от 0 до 10."""

    try:
        score_raw = llm.invoke([HumanMessage(content=eval_prompt)]).content.strip()
        score = float(score_raw) / 10.0
    except ValueError:
        score = 1.0

    return {
        "evaluation_score": score,
        "retry_count": state.get("retry_count", 0) + 1
    }

def escalation_node(state: AgentState) -> Dict[str, Any]:
    return {"response": "Ваше обращение передано старшему специалист поддержки. Мы свяжемся с вами в течение 15 минут."}


def build_support_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("action", action_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("escalation", escalation_node)

    workflow.set_entry_point("router")

    def route_decision(state: AgentState) -> Literal["retriever", "action", "escalation"]:
        if state["intent"] == "action":
            return "action"
        elif state["intent"] == "escalate":
            return "escalation"
        return "retriever"

    workflow.add_conditional_edges("router", route_decision)

    workflow.add_edge("retriever", "generator")

    workflow.add_edge("generator", "evaluator")

    def evaluation_decision(state: AgentState) -> Literal["__end__", "escalation"]:
        if state["evaluation_score"] >= 0.7:
            return END
        if state["retry_count"] >= 2:
            return "escalation"
        return END

    workflow.add_conditional_edges("evaluator", evaluation_decision)
    workflow.add_edge("action", END)
    workflow.add_edge("escalation", END)

    return workflow.compile()


app_graph = build_support_graph()

if __name__ == "__main__":
    initial_state: AgentState = {
        "query": "Как мне вернуть деньги, если подписка оформилась случайно?",
        "intent": "",
        "context": [],
        "response": "",
        "evaluation_score": 0.0,
        "retry_count": 0,
        "is_action_executed": False
    }

    result = app_graph.invoke(initial_state)
    print("--- ИТОГОВЫЙ РЕЗУЛЬТАТ ГРАФА ---")
    print(f"Намерение: {result['intent']}")
    print(f"Ответ: {result['response']}")
    print(f"Score оценки: {result.get('evaluation_score', 'N/A')}")