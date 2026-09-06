import time
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional

from src.agent import app_graph, AgentState
from src.cache import get_cached_response, set_cached_response, init_cache_db

app = FastAPI(
    title="Enterprise Agentic RAG System",
    version="1.0.0",
    description="Высокопроизводительный асинхронный API с агентами и семантическим кэшированием."
)


@app.on_event("startup")
def startup_event():
    init_cache_db()


class ChatRequest(BaseModel):
    query: str = Field(..., example="Как оформить возврат средств за подписку?")
    user_id: Optional[str] = Field("user_default", example="usr_1029")


class ChatResponse(BaseModel):
    query: str
    response: str
    intent: str
    cached: bool
    execution_time_ms: float
    evaluation_score: Optional[float] = None


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    start_time = time.perf_counter()
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Текст запроса не может быть пустым.")

    cached_response = get_cached_response(query)
    if cached_response:
        exec_time = (time.perf_counter() - start_time) * 1000
        return ChatResponse(
            query=query,
            response=cached_response,
            intent="semantic_cache_hit",
            cached=True,
            execution_time_ms=round(exec_time, 2),
            evaluation_score=1.0
        )

    initial_state: AgentState = {
        "query": query,
        "intent": "",
        "context": [],
        "response": "",
        "evaluation_score": 0.0,
        "retry_count": 0,
        "is_action_executed": False
    }

    try:
        result = await app_graph.ainvoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка выполнения графа: {str(e)}")

    response_text = result.get("response", "Не удалось сформировать ответ.")
    eval_score = result.get("evaluation_score", 0.0)

    if eval_score >= 0.7:
        background_tasks.add_task(set_cached_response, query, response_text)

    exec_time = (time.perf_counter() - start_time) * 1000

    return ChatResponse(
        query=query,
        response=response_text,
        intent=result.get("intent", "unknown"),
        cached=False,
        execution_time_ms=round(exec_time, 2),
        evaluation_score=eval_score
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "RAG-Agent-Backend"}