import pytest

from src.routing import route_after_evaluation, route_after_retrieval, route_after_router


@pytest.mark.parametrize(
    "intent, expected",
    [("faq", "retriever"), ("action", "action"), ("escalate", "escalation"), (None, "retriever")],
)
def test_route_after_router(intent, expected):
    assert route_after_router({"intent": intent}) == expected


def test_empty_context_goes_to_escalation():
    assert route_after_retrieval({"context": []}) == "escalation"
    assert route_after_retrieval({"context": [{"doc_id": "a"}]}) == "generator"


def test_good_answer_is_accepted():
    assert route_after_evaluation({"evaluation_score": 0.8, "attempts": 1}, 0.7, 2) == "__end__"


def test_bad_answer_is_regenerated_while_attempts_left():
    assert route_after_evaluation({"evaluation_score": 0.3, "attempts": 1}, 0.7, 2) == "generator"


def test_bad_answer_is_escalated_when_attempts_exhausted():
    # Без этой ветки плохой ответ уходил бы клиенту.
    assert route_after_evaluation({"evaluation_score": 0.3, "attempts": 2}, 0.7, 2) == "escalation"


def test_missing_score_is_not_accepted():
    assert route_after_evaluation({"evaluation_score": None, "attempts": 2}, 0.7, 2) == "escalation"
