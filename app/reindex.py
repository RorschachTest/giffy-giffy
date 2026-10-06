"""Rebuild the searchable columns for every clip.

    python -m app.reindex              after changing rules in text.py or indexing.py
    python -m app.reindex --no-embed   skip embeddings (much faster)
    python -m app.reindex --vibe       first fill empty mood/description/use-when (local models)

Does not re-run speech to text or faces; it only re-derives from stored facts.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from psycopg.types.json import Jsonb

from . import config, db, faces, indexing, media, vibe


def backfill_vibe() -> None:
    """Fill empty mood fields on clips indexed before the mood step existed."""
    with db.session() as conn:
        rows = conn.execute(
            """SELECT id, file, duration, title, transcript_native, description, reactions, use_when, raw
                 FROM clips WHERE description = '' OR reactions = '{}' OR use_when = '{}' ORDER BY id"""
        ).fetchall()
    engine = faces.get_engine() if config.ENABLE_FACES else None
    for r in rows:
        with tempfile.TemporaryDirectory(prefix="memeclip_") as tmp:
            frames = media.extract_keyframes(config.MEDIA_DIR / r["file"], Path(tmp),
                                             r["duration"], config.KEYFRAMES)
            detected = [engine.detect(f) for f in frames] if engine else None
            found = vibe.analyse(frames, detected, r["transcript_native"])
        filled = vibe.fill_missing(r, found["fields"])
        raw = {**(r["raw"] or {}), "vibe": found["scores"]}
        with db.session() as conn:
            conn.execute("UPDATE clips SET description=%s, reactions=%s, use_when=%s, raw=%s WHERE id=%s",
                         (filled["description"], filled["reactions"], filled["use_when"],
                          Jsonb(raw), r["id"]))
        print(f"vibe: clip {r['id']} {r['title']!r}: {filled['reactions']}")


def main() -> None:
    embed = "--no-embed" not in sys.argv
    db.init_db()
    if "--vibe" in sys.argv:
        if not config.ENABLE_VIBE:
            sys.exit("--vibe needs ENABLE_VIBE=1")
        backfill_vibe()
    with db.session() as conn:
        ids = [r["id"] for r in conn.execute("SELECT id FROM clips ORDER BY id")]
    for i, clip_id in enumerate(ids, 1):
        with db.session() as conn:
            indexing.refresh(conn, clip_id, embed=embed)
        if i % 50 == 0 or i == len(ids):
            print(f"{i}/{len(ids)}")


if __name__ == "__main__":
    main()
