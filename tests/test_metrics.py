import pytest

from src.metrics import hit_at_k, percentile, recall_at_k, reciprocal_rank


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["c", "x"], k=3) == 0.5
    assert recall_at_k(["a", "b", "c"], ["c"], k=2) == 0.0


def test_hit_at_k():
    assert hit_at_k(["a", "b"], ["b"], 1) == 0.0
    assert hit_at_k(["a", "b"], ["b"], 2) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], ["z"]) == 0.0


def test_percentile_matches_linear_interpolation():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10], 95) == 10
