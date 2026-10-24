"""Клиенты LLM. Любой OpenAI-совместимый API: Ollama, OpenRouter, vLLM, OpenAI.
Провайдер выбирается переменными LLM_BASE_URL / LLM_API_KEY / LLM_MODEL."""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from src.config import get_settings


@lru_cache
def get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.llm_model,
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        temperature=s.llm_temperature,
        timeout=s.llm_timeout_s,
        max_retries=1,
    )


@lru_cache
def get_judge() -> ChatOpenAI:
    s = get_settings()
    if not s.judge_model:
        return get_llm()
    return ChatOpenAI(
        model=s.judge_model,
        base_url=s.judge_base_url or s.llm_base_url,
        api_key=s.judge_api_key or s.llm_api_key,
        temperature=0.0,
        timeout=s.llm_timeout_s,
        max_retries=1,
    )
