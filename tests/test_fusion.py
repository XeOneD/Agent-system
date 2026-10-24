from src.fusion import build_or_tsquery, dedupe_by_doc, reciprocal_rank_fusion


def test_tsquery_joins_terms_with_or():
    assert build_or_tsquery("Как вернуть деньги?") == "как | вернуть | деньги"


def test_tsquery_drops_special_characters():
    query = build_or_tsquery("возврат & (деньги) | !карта: 'x'")
    assert set("&()!:'\"") & set(query) == set()
    assert query == "возврат | деньги | карта"


def test_tsquery_removes_duplicates_and_short_tokens():
    assert build_or_tsquery("А я деньги ДЕНЬГИ деньги") == "деньги"


def test_tsquery_empty_input():
    assert build_or_tsquery("?? !!") == ""


def _item(chunk_id, doc_id):
    return {"id": chunk_id, "doc_id": doc_id}


def test_rrf_rewards_items_found_by_both_retrievers():
    vector = [_item(1, "a"), _item(2, "b"), _item(3, "c")]
    fts = [_item(3, "c"), _item(4, "d")]
    fused = reciprocal_rank_fusion([vector, fts], k=60)
    assert fused[0]["id"] == 3  # третий в векторном, но первый в FTS
    assert {x["id"] for x in fused} == {1, 2, 3, 4}


def test_rrf_score_formula():
    fused = reciprocal_rank_fusion([[_item(1, "a")], [_item(1, "a")]], k=60)
    assert abs(fused[0]["rrf_score"] - 2 / 61) < 1e-12


def test_dedupe_keeps_best_chunk_per_document():
    chunks = [_item(1, "a"), _item(2, "a"), _item(3, "b"), _item(4, "c")]
    assert [c["id"] for c in dedupe_by_doc(chunks, limit=2)] == [1, 3]
