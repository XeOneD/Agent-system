import pytest

from src.parsing import parse_intent, parse_score, strip_reasoning


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"intent": "action"}', "action"),
        ("Категория: escalate.", "escalate"),
        ("FAQ", "faq"),
        ("<think>может быть action</think> faq", "faq"),
        ("не знаю", "faq"),
    ],
)
def test_parse_intent(raw, expected):
    assert parse_intent(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"score": 8}', 0.8),
        ("Оценка: 7/10", 0.7),
        ("10", 1.0),
        ("0", 0.0),
        ("8,5", 0.85),
        ('В ответе 2 утверждения. {"score": 9}', 0.9),
        ("<think>сначала 3, потом</think> 6", 0.6),
    ],
)
def test_parse_score(raw, expected):
    assert parse_score(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "хороший ответ", "15", "100 из 100"])
def test_parse_score_returns_none_when_unclear(raw):
    # None означает «проверка не пройдена» — fail-closed.
    assert parse_score(raw) is None


def test_strip_reasoning():
    assert strip_reasoning("<think>\nдолго думаю\n</think>\nОтвет") == "Ответ"
