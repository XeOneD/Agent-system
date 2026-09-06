import os
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

# Инициализация клиента Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-1234567890"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-1234567890"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

# Callback для автоматического трейсинга LangChain / LangGraph
def get_langfuse_callback(session_id: str = "default_session"):
    return CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST"),
        session_id=session_id
    )

def log_custom_metric(trace_id: str, metric_name: str, value: float):
    """Отправка кастомных метрик качества (например, score от Evaluator)."""
    langfuse.score(
        trace_id=trace_id,
        name=metric_name,
        value=value
    )