"""Rebuild the searchable columns for every clip.

    python -m app.reindex              after changing rules in text.py or indexing.py
    python -m app.reindex --no-embed   skip embeddings (much faster)
    python -m app.reindex --vibe       first fill empty mood/description/use-when (local models)
    python -m app.reindex --laya       first add Laya topics to every clip (keeps existing ones)
    python -m app.reindex --vibe --replace
                                       recompute mood on every clip, overwriting hand edits too

Does not re-run speech to text or faces; it only re-derives from stored facts.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from psycopg.types.json import Jsonb

from . import config, db, faces, indexing, laya, media, pipeline, vibe


def backfill_laya() -> None:
    """Ask Laya about every clip; its topics are added after any already there."""
    with db.session() as conn:
        rows = conn.execute("SELECT * FROM clips ORDER BY id").fetchall()
    for r in rows:
        found = laya.describe_clip(r, r["transcript_native"], r["transcript_roman"], r["people"])
        topics = pipeline.pipeline_topics(r["topics"], found["topics"])
        raw = {**(r["raw"] or {}), "laya": found["scores"]}
        with db.session() as conn:
            conn.execute("UPDATE clips SET topics = %s, raw = %s WHERE id = %s",
                         (topics, Jsonb(raw), r["id"]))
        print(f"laya: clip {r['id']} {r['title']!r}: {topics}")


def backfill_vibe(replace: bool = False) -> None:
    """Fill empty mood fields on clips indexed before the mood step existed.
    With replace=True every clip is redone and its mood fields overwritten."""
    where = "TRUE" if replace else "description = '' OR reactions = '{}' OR use_when = '{}'"
    with db.session() as conn:
        rows = conn.execute(
            f"""SELECT id, file, duration, title, caption, source_title, hashtags, people,
                       transcript_native, transcript_roman, description, reactions, use_when, raw
                  FROM clips WHERE {where} ORDER BY id"""
        ).fetchall()
    engine = faces.get_engine() if config.ENABLE_FACES else None
    for r in rows:
        with tempfile.TemporaryDirectory(prefix="memeclip_") as tmp:
            frames = media.extract_keyframes(config.MEDIA_DIR / r["file"], Path(tmp),
                                             r["duration"], config.KEYFRAMES)
            detected = [engine.detect(f) for f in frames] if engine else None
            text_moods = None
            if laya.available():
                text_moods = laya.describe_clip(r, r["transcript_native"], r["transcript_roman"],
                                                r["people"])["mood"]
            found = vibe.analyse(frames, detected, r["transcript_native"], text_moods)
        filled = found["fields"] if replace else vibe.fill_missing(r, found["fields"])
        raw = {**(r["raw"] or {}), "vibe": found["scores"]}
        with db.session() as conn:
            conn.execute("UPDATE clips SET description=%s, reactions=%s, use_when=%s, raw=%s WHERE id=%s",
                         (filled["description"], filled["reactions"], filled["use_when"],
                          Jsonb(raw), r["id"]))
        print(f"vibe: clip {r['id']} {r['title']!r}: {filled['reactions']}")


def main() -> None:
    embed = "--no-embed" not in sys.argv
    db.init_db()
    if "--laya" in sys.argv:
        if not laya.available():
            sys.exit("--laya needs LAYA_URL")
        backfill_laya()
    if "--vibe" in sys.argv:
        if not config.ENABLE_VIBE:
            sys.exit("--vibe needs ENABLE_VIBE=1")
        backfill_vibe(replace="--replace" in sys.argv)
    with db.session() as conn:
        ids = [r["id"] for r in conn.execute("SELECT id FROM clips ORDER BY id")]
    for i, clip_id in enumerate(ids, 1):
        with db.session() as conn:
            indexing.refresh(conn, clip_id, embed=embed)
        if i % 50 == 0 or i == len(ids):
            print(f"{i}/{len(ids)}")


if __name__ == "__main__":
    main()
