"""The searches the empty search box suggests, from the search_suggestions table.

Read once and kept in memory for SUGGESTIONS_TTL_S, so a page load costs no query; the page
gets them written in (api._themed_page), so the browser needs no extra request either.
Change the table (add a row, set active = false, reorder by position) and the site shows it
within the TTL, no deploy.
"""
from __future__ import annotations

import json
import logging
import threading
import time

from . import config, db

log = logging.getLogger("suggestions")

_lock = threading.Lock()
_cache: tuple[float, list[str]] = (0.0, [])


def load(conn) -> list[str]:
    return [r["text"] for r in conn.execute(
        "SELECT text FROM search_suggestions WHERE active ORDER BY position, id").fetchall()]


def current() -> list[str]:
    """Active suggestions, from memory when fresh. A database error keeps the last list."""
    global _cache
    at, items = _cache
    if time.monotonic() - at < config.SUGGESTIONS_TTL_S:
        return items
    with _lock:
        at, items = _cache
        if time.monotonic() - at < config.SUGGESTIONS_TTL_S:
            return items
        try:
            with db.session() as conn:
                items = load(conn)
        except Exception as exc:   # the page must render even if this read fails
            log.warning("could not load search suggestions: %s", exc)
        _cache = (time.monotonic(), items)
        return items


def as_script_json(items: list[str]) -> str:
    """JSON safe inside <script type="application/json">: no '</' can close the tag."""
    return json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
