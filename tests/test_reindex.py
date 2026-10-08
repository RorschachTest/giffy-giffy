"""The backfills keep going when one clip fails (needs the test database)."""
import os

import pytest

from . import conftest  # noqa: F401

pytestmark = pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")

from app import db, laya, reindex  # noqa: E402


def test_backfill_laya_skips_a_failing_clip_and_continues(monkeypatch, capsys):
    db.init_db()
    with db.session() as conn:
        conn.execute("TRUNCATE query_log, clip_faces, face_refs, people, clips RESTART IDENTITY CASCADE")
        for n in (1, 2, 3):
            conn.execute("INSERT INTO clips (sha256, file, title) VALUES (%s, %s, %s)", (f"{n:064d}", f"{n}.mp4", f"clip {n}"))

    def flaky(meta, native, roman, people):
        if meta["title"] == "clip 2":
            raise TimeoutError("laya timed out")
        return {"topics": ["work"], "mood": {}, "scores": {}}
    monkeypatch.setattr(laya, "describe_clip", flaky)

    reindex.backfill_laya()
    with db.session() as conn:
        topics = {r["title"]: r["topics"] for r in conn.execute("SELECT title, topics FROM clips")}
    assert topics == {"clip 1": ["work"], "clip 2": [], "clip 3": ["work"]}   # 3 was still done after 2 failed
    assert "1 of 3 clip(s) skipped" in capsys.readouterr().err
