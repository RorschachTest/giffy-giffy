"""Rebuild the searchable columns for every clip.

    python -m app.reindex              after changing rules in text.py or indexing.py
    python -m app.reindex --no-embed   skip embeddings (much faster)

Does not re-run speech to text or faces; it only re-derives from stored facts.
"""
from __future__ import annotations

import sys

from . import db, indexing


def main() -> None:
    embed = "--no-embed" not in sys.argv
    db.init_db()
    with db.session() as conn:
        ids = [r["id"] for r in conn.execute("SELECT id FROM clips ORDER BY id")]
    for i, clip_id in enumerate(ids, 1):
        with db.session() as conn:
            indexing.refresh(conn, clip_id, embed=embed)
        if i % 50 == 0 or i == len(ids):
            print(f"{i}/{len(ids)}")


if __name__ == "__main__":
    main()
