"""Situation lines: what goes in, and that search runs with the lane on or off. No models."""
from . import conftest  # noqa: F401

from app import uses


def test_lines_are_cleaned_deduplicated_and_capped():
    desc = {"send_when": ["  when salary is late. ", "When salary is late", "ok", *[f"situation number {i}" for i in range(10)]],
            "poster_meant": "money talk"}
    got = uses.lines_for(desc)
    assert got[0] == ("money talk", "poster_meant")          # writer's intent first, kept under the cap
    assert got[1] == ("when salary is late", "send_when")
    assert sum(1 for line, _ in got if line.lower() == "when salary is late") == 1   # repeat dropped
    assert all(len(line) >= 8 for line, _ in got)                                   # "ok" dropped
    assert len(got) == uses.MAX_LINES


def test_poster_meant_is_a_line_too():
    assert ("using it to mock fake gyaan", "poster_meant") in uses.lines_for(
        {"send_when": [], "poster_meant": "using it to mock fake gyaan"})


def test_translated_versions_become_extra_rows(monkeypatch):
    from app import translate

    class Conn:
        def __init__(self): self.rows = []
        def execute(self, sql, params=None):
            if "INSERT INTO clip_uses" in sql: self.rows.append(params)
            class R:
                def fetchall(_):
                    if "FROM clip_descriptions" in sql:
                        return [{"clip_id": 7, "send_when": ["main nahi aa sakta"], "poster_meant": ""}]
                    return []
            return R()
    monkeypatch.setattr(translate, "line_versions",
                        lambda conn, lines: {"main nahi aa sakta": [("I can't come", "en"), ("मैं नहीं आ सकता", "hi")]})
    monkeypatch.setattr(uses, "get_embedder", lambda: type("E", (), {"embed": lambda self, t: [[0.0] * 3 for _ in t]})())
    conn = Conn()
    assert uses.build(conn, [7]) == 3
    assert [(r[1], r[2]) for r in conn.rows] == [("main nahi aa sakta", "send_when"),
                                                 ("I can't come", "send_when:en"), ("मैं नहीं आ सकता", "send_when:hi")]
