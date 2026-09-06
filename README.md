# Enterprise Agentic Support System (LangGraph + RAG + pgvector)

Production-ready асинхронный мультиагентный сервис для автономии службы поддержки клиентов с гибридным RAG, семантическим кэшированием и непрерывным MLOps-трейсингом.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-orange)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-blue)

---

## Архитектура системы

```text
[Client Request] ──> [FastAPI Server] ──(Semantic Cache Check)──> [Cache Hit: ~12ms]
                             │
                      (Cache Miss)
                             ▼
                [LangGraph Agentic Workflow]
               ┌─────────────┼─────────────┐
               ▼             ▼             ▼
        [Router Agent]  [Retriever]  [Action Agent]
               │             │             │
        [Intent Class]  [pgvector]   [Mock REST API]
               └─────────────┼─────────────┘
                             ▼
                  [Evaluator Agent (Self-Check)]
                             │
                ┌────────────┴────────────┐
       (Score >= 0.7)                (Score < 0.7)
              │                           │
      [Return Answer & Cache]     [Escalate to Human]