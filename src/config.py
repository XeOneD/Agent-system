"""Все настройки - из переменных окружения или .env. Значения по умолчанию
рассчитаны на локальный запуск через docker compose."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- База данных
    database_url: str = "postgresql://postgres:postgres@localhost:5432/rag"
    db_pool_min: int = 1
    db_pool_max: int = 10

    # --- LLM: любой OpenAI-совместимый эндпоинт (Ollama, OpenRouter, vLLM, OpenAI)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen2.5:3b"
    llm_temperature: float = 0.0
    llm_timeout_s: float = 60.0

    # Отдельный судья для оценки ответа. Пусто - используется основная модель.
    # Судить ответ той же маленькой моделью, что его написала, - слабая проверка.
    judge_base_url: str = ""
    judge_api_key: str = ""
    judge_model: str = ""

    # --- Модели поиска (мультиязычные: база знаний на русском)
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    device: str = "cpu"

    # --- Поиск
    retrieval_mode: str = "vector"  # vector | fts | hybrid | vector_rerank | hybrid_rerank
    candidates_per_retriever: int = 20
    rerank_candidates: int = 20
    context_top_n: int = 3
    rrf_k: int = 60

    # --- Агент
    eval_threshold: float = 0.7
    max_generation_attempts: int = 2

    # --- Семантический кэш
    cache_enabled: bool = True
    # У e5 косинусная близость даже несвязанных фраз высокая (0.7-0.85),
    # поэтому порог должен быть строгим. Подбирайте на своих данных.
    cache_similarity_threshold: float = 0.95
    cache_ttl_hours: int = 24


@lru_cache
def get_settings() -> Settings:
    return Settings()
