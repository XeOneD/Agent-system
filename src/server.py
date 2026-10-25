"""HTTP API.

Эндпоинты объявлены обычными `def`: внутри синхронные вызовы БД, моделей и LLM,
и FastAPI выполняет такие обработчики в пуле потоков, не блокируя event loop.
(В первой версии синхронный код вызывался внутри `async def` и блокировал сервер.)
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import cache
from src.agent import run
from src.db import close_pool, get_conn, init_db
from src.models import warmup

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    warmup()
    yield
    close_pool()


app = FastAPI(title="Support RAG Agent", version="2.0.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, examples=["Как вернуть деньги за подписку?"])
    user_id: str = Field("anonymous", max_length=64)


class Source(BaseModel):
    doc_id: str
    title: str


class ChatResponse(BaseModel):
    response: str
    intent: str
    cached: bool
    sources: list[Source]
    evaluation_score: float | None = None
    ticket_id: int | None = None
    latency_ms: float
    timings_ms: dict = {}


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    started = time.perf_counter()
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Пустой запрос.")

    hit = cache.lookup(query)
    if hit:
        return ChatResponse(
            response=hit["response"],
            intent="faq",
            cached=True,
            sources=hit["sources"],
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    try:
        result = run(query, user_id=request.user_id)
    except Exception as exc:  # noqa: BLE001 - отдаём клиенту понятную ошибку, детали в лог
        log.exception("Ошибка графа")
        raise HTTPException(status_code=502, detail="Сервис ответов временно недоступен.") from exc

    sources = [{"doc_id": d["doc_id"], "title": d["title"]} for d in result.get("context", [])]
    passed = result.get("ticket_id") is None and result.get("intent") == "faq"
    if passed:
        cache.store(query, result["response"], sources)

    return ChatResponse(
        response=result["response"],
        intent=result.get("intent", "faq"),
        cached=False,
        sources=sources if result.get("ticket_id") is None else [],
        evaluation_score=result.get("evaluation_score"),
        ticket_id=result.get("ticket_id"),
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        timings_ms=result.get("timings_ms", {}),
    )


@app.get("/health")
def health() -> dict:
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM document_chunks;")
            chunks = cur.fetchone()[0]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"БД недоступна: {exc}") from exc
    return {"status": "ok", "chunks": chunks}
