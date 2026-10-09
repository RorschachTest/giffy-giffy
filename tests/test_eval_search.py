"""The ruler's arithmetic. No database."""
from . import conftest  # noqa: F401

from app import eval_search as ev


def test_rank_of_matches_share_or_database_id():
    rows = [{"id": 1, "share_id": "aaaaaaaaaa"}, {"id": 2, "share_id": "bbbbbbbbbb"}]
    assert ev.rank_of(rows, ["bbbbbbbbbb"]) == 2
    assert ev.rank_of(rows, [1]) == 1
    assert ev.rank_of(rows, ["zzzzzzzzzz"]) is None


def test_score_hits_and_mrr():
    s = ev.score([1, 3, None, 12])
    assert s["n"] == 4 and s["hit@1"] == 0.25 and s["hit@5"] == 0.5 and s["hit@10"] == 0.5
    assert s["mrr"] == round((1 + 1 / 3 + 1 / 12) / 4, 3)
    assert ev.score([])["mrr"] == 0.0
