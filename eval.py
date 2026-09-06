import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from src.agent import app_graph, AgentState

# Тестовый датасет бенчмарка (Ground Truth)
TEST_QUESTIONS = [
    {
        "question": "Какой срок возврата денежных средств при отмене подписки?",
        "ground_truth": "Возврат осуществляется в течение 14 рабочих дней на тот же счет."
    },
    {
        "question": "Что делать, если списались деньги, но подписка не активировалась?",
        "ground_truth": "Нужно проверить статус в личном кабинете или сбросить сессию."
    }
]


async def run_evaluation():
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in TEST_QUESTIONS:
        state: AgentState = {
            "query": item["question"],
            "intent": "",
            "context": [],
            "response": "",
            "evaluation_score": 0.0,
            "retry_count": 0,
            "is_action_executed": False
        }

        res = await app_graph.ainvoke(state)

        questions.append(item["question"])
        answers.append(res["response"])
        contexts.append([c["content"] for c in res.get("context", [])])
        ground_truths.append(item["ground_truth"])

    # Формируем HuggingFace Dataset для RAGAS
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    # Запуск расчёта метрик
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )

    print("\n================ RAGAS BENCHMARK RESULTS ================")
    print(results)
    print("=========================================================")


if __name__ == "__main__":
    asyncio.run(run_evaluation())