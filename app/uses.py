"""The situation lane: when you would send a clip, one line at a time, ready for search.

    python -m app.uses            (re)build lines for every clip with a description
    python -m app.uses 11 228     just these clips

Lines come from clip_descriptions (app/describe.py): every "send it when" situation plus
what the poster meant. Each line gets its own meaning vector in clip_uses, and search
scores a clip by its BEST line. Not one vector for all of them: measured earlier, stuffing
more text into a clip's single vector blurred it ("who are these people" fell from rank 1
to 1,166). One clip fits many situations; a best-line match keeps each one sharp.

Lines Gemini wrote in Hinglish ("jab dil garden garden ho jaaye") get the same English
gloss as search queries before embedding (hinglish.gloss): the meaning model reads
English and Devanagari, not romanised Hindi, so both sides are treated the same way.
"""
from __future__ import annotations

import sys

from . import db, hinglish
from .db import vec
from .embed import get_embedder

MAX_LINES = 8


def lines_for(desc: dict) -> list[tuple[str, str]]:
    """(line, source) pairs for one description row, cleaned and without repeats."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line, source in [*((s, "send_when") for s in desc.get("send_when") or []),
                         (desc.get("poster_meant") or "", "poster_meant")]:
        line = " ".join(str(line).split()).strip(" .")
        key = line.lower()
        if len(line) >= 8 and key not in seen:
            seen.add(key)
            out.append((line, source))
    return out[:MAX_LINES]


def build(conn, clip_ids: list[int] | None = None) -> int:
    """Replace the lines of these clips (all described clips when None) from their newest
    description. Returns how many lines were written."""
    where = "WHERE d.clip_id = ANY(%(ids)s)" if clip_ids else ""
    descs = conn.execute(
        f"""SELECT DISTINCT ON (d.clip_id) d.clip_id, d.send_when, d.poster_meant
              FROM clip_descriptions d {where}
             ORDER BY d.clip_id, d.at DESC""", {"ids": clip_ids or []}).fetchall()
    pairs = [(d["clip_id"], line, source) for d in descs for line, source in lines_for(d)]
    vectors = get_embedder().embed([hinglish.gloss(line) for _, line, _ in pairs]) if pairs else []
    ids = [d["clip_id"] for d in descs]
    conn.execute("DELETE FROM clip_uses WHERE clip_id = ANY(%s) AND source <> 'edited'", (ids,))
    for (clip_id, line, source), v in zip(pairs, vectors):
        conn.execute("INSERT INTO clip_uses (clip_id, line, source, embedding) VALUES (%s, %s, %s, %s)",
                     (clip_id, line, source, vec(v)))
    return len(pairs)


def main(argv: list[str]) -> int:
    ids = [int(a) for a in argv if a.isdigit()] or None
    with db.session() as conn:
        n = build(conn, ids)
        clips = conn.execute("SELECT count(DISTINCT clip_id) AS n FROM clip_uses").fetchone()["n"]
    print(f"{n} lines written; {clips} clips now have situation lines")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
