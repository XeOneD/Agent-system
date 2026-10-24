"""Проверка целостности базы знаний и eval-набора: битая ссылка на документ
тихо занизила бы метрики."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _questions():
    path = ROOT / "data" / "eval" / "questions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_every_document_has_title():
    for path in (ROOT / "data" / "kb").glob("*.md"):
        assert path.read_text(encoding="utf-8").startswith("# "), path.name


def test_gold_documents_exist():
    doc_ids = {p.stem for p in (ROOT / "data" / "kb").glob("*.md")}
    for q in _questions():
        for doc_id in q["gold_doc_ids"]:
            assert doc_id in doc_ids, f"{q['id']}: нет документа {doc_id}"


def test_question_schema():
    ids = set()
    for q in _questions():
        assert q["intent"] in {"faq", "action", "escalate"}
        assert (q["intent"] == "faq") == bool(q["gold_doc_ids"]), q["id"]
        assert q["id"] not in ids
        ids.add(q["id"])
