"""Search-box suggestions: cached, safe inside the page, and seeded by the schema."""
from . import conftest  # noqa: F401

from app import config, suggestions


def test_script_json_cannot_close_the_tag():
    out = suggestions.as_script_json(["a </script><b>", "मूड"])
    assert "</" not in out and "मूड" in out


def test_cache_serves_from_memory(monkeypatch):
    calls = []

    class Session:
        def __enter__(self): return "conn"
        def __exit__(self, *a): return False
    monkeypatch.setattr(suggestions.db, "session", lambda: Session())
    monkeypatch.setattr(suggestions, "load", lambda conn: calls.append(1) or ["x"])
    monkeypatch.setattr(config, "SUGGESTIONS_TTL_S", 60.0)
    monkeypatch.setattr(suggestions, "_cache", (0.0, []))
    assert suggestions.current() == ["x"] and suggestions.current() == ["x"]
    assert len(calls) == 1


def test_a_failed_read_keeps_the_last_list(monkeypatch):
    def boom(): raise RuntimeError("db down")
    monkeypatch.setattr(suggestions.db, "session", boom)
    monkeypatch.setattr(config, "SUGGESTIONS_TTL_S", 0.0)
    monkeypatch.setattr(suggestions, "_cache", (0.0, ["kept"]))
    assert suggestions.current() == ["kept"]
