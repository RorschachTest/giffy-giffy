"""Search insights: language guess and that the log stays anonymous."""
import logging

from . import conftest  # noqa: F401

from app import query_stats


def test_language_guess():
    assert query_stats.language("कौन हैं ये लोग") == "hindi"
    assert query_stats.language("kaun hai ye log") == "hinglish"
    assert query_stats.language("when my boss shouts at me") == "english"
    assert query_stats.language("salary") == "unclear"


def test_access_log_drops_search_text():
    from app import api

    rec = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                            ("1.2.3.4:5", "GET", "/search?q=kaun+hai+ye+log", "1.1", 200), None)
    api._NoQueryStrings().filter(rec)
    assert "kaun" not in rec.getMessage() and "/search" in rec.getMessage()
