"""The meme understanding engine: why a clip is funny and what it says in a chat.

    humor.py    why it is funny (humor theories as measurements, logistic funniness)
    intent.py   what it communicates as a reply (product of experts over intents)
    this file   loads the clip, applies what feedback has taught, stores new feedback

Learning (no LLM, no training run): every POST /understand/feedback stores the
chat, the reading people agreed on and whether it was funny, together with the
dialogue acts and humor features measured at that moment. From those rows:

  * each clip gets a Dirichlet posterior over how it is actually used
  * the dialogue-act -> intent transition matrix is re-estimated
  * the funniness weights are refitted (MAP logistic regression around the
    literature priors)

All three are recomputed from the table, so deleting rows undoes them.

    python -m app.understand 12                        the clip on its own
    python -m app.understand 12 "I got the job!"       sent after that message
    python -m app.understand 12 "I got the job" --caption "me rn"
"""
from __future__ import annotations

import json
import re
import sys

from psycopg.types.json import Jsonb

from . import humor, intent

SHARE_LINK = re.compile(r"/c/([0-9a-f]{10})\b")
SHARE_ID = re.compile(r"^([0-9a-f]{10})$")
MAX_CONTEXT = 6

_weights_cache: dict = {"key": None, "weights": None}


def find_clip(conn, clip_id: int | None = None, link: str | None = None) -> dict | None:
    """By id, or by a share link / share id the way people actually send memes."""
    if clip_id is not None:
        return conn.execute("SELECT * FROM clips WHERE id = %s", (clip_id,)).fetchone()
    link = (link or "").strip().lower()
    m = SHARE_LINK.search(link) or SHARE_ID.match(link)
    if not m:
        return None
    return conn.execute("SELECT * FROM clips WHERE left(sha256, 10) = %s", (m.group(1),)).fetchone()


def clip_counts(conn, clip_id: int) -> dict[str, float]:
    rows = conn.execute(
        "SELECT intent, count(*) AS n FROM meme_feedback WHERE clip_id = %s AND intent IS NOT NULL GROUP BY intent",
        (clip_id,)).fetchall()
    return {r["intent"]: float(r["n"]) for r in rows}


def act_counts(conn) -> dict[str, dict[str, float]]:
    """Soft counts: a feedback row whose message was 70% "brag" adds 0.7 to brag -> intent."""
    out: dict[str, dict[str, float]] = {}
    for r in conn.execute("SELECT acts, intent FROM meme_feedback WHERE intent IS NOT NULL AND acts <> '{}'"):
        for act, p in (r["acts"] or {}).items():
            if act in intent.CONTEXT_ACTS:
                row = out.setdefault(act, {})
                row[r["intent"]] = row.get(r["intent"], 0.0) + float(p)
    return out


def funny_weights(conn) -> dict[str, float]:
    """Funniness weights refitted from feedback; cached until new feedback arrives."""
    key = tuple(conn.execute(
        "SELECT count(*) AS n, coalesce(max(id), 0) AS last FROM meme_feedback WHERE funny IS NOT NULL"
    ).fetchone().values())
    if _weights_cache["key"] != key:
        rows = conn.execute("SELECT features, funny FROM meme_feedback WHERE funny IS NOT NULL").fetchall()
        examples = [(r["features"], 1.0 if r["funny"] else 0.0) for r in rows if r["features"]]
        _weights_cache.update(key=key, weights=humor.fit_funny(examples))
    return _weights_cache["weights"]


def _clean_context(context: list[str] | str | None) -> list[str]:
    if isinstance(context, str):
        context = [context]
    return [c.strip() for c in (context or []) if c and c.strip()][-MAX_CONTEXT:]


def understand(conn, clip: dict, context: list[str] | str | None = None, caption: str = "") -> dict:
    """The full reading of one clip, alone (no context) or as a reply in a chat."""
    context = _clean_context(context)
    caption = (caption or "").strip()
    facts = humor.facts_from_clip(clip)
    funny = humor.analyse(facts, context, caption, weights=funny_weights(conn))
    said = intent.read(facts, context, caption, mech=funny["mechanisms"],
                       clip_counts=clip_counts(conn, clip["id"]), act_counts=act_counts(conn))
    return {
        "clip": {"id": clip["id"], "share_id": clip["sha256"][:10], "title": clip["title"],
                 "said": clip["transcript_roman"], "moods": facts["reactions"], "topics": facts["topics"]},
        "context": context, "caption": caption,
        "communicates": {k: said[k] for k in ("intent", "reading", "aimed_at", "confidence", "intents", "evidence")},
        "funny": funny["funny"],
        "why_funny": funny["reasons"],
        "features": funny["features"],
        "details": {"experts": said["experts"], "acts": said["acts"], "mechanisms": funny["mechanisms"]},
    }


def record_feedback(conn, clip: dict, context: list[str] | str | None, caption: str,
                    label: str | None, funny: bool | None) -> dict:
    """Store what a person says the meme meant here (and whether it landed). The
    acts and features are measured now and stored, so learning never re-embeds."""
    if label is not None and label not in intent.INTENTS:
        raise ValueError(f"unknown intent {label!r}; one of {sorted(intent.INTENTS)}")
    if label is None and funny is None:
        raise ValueError("give an intent, a funny verdict, or both")
    reading = understand(conn, clip, context, caption)
    conn.execute(
        """INSERT INTO meme_feedback (clip_id, context, caption, intent, funny, acts, features, predicted)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
        (clip["id"], reading["context"], reading["caption"], label, funny,
         Jsonb(reading["details"]["acts"]), Jsonb(reading["features"]), reading["communicates"]["intent"]),
    )
    return {"ok": True, "predicted": reading["communicates"]["intent"], "label": label,
            "agreed": label == reading["communicates"]["intent"] if label else None}


def accuracy(conn) -> dict:
    """How often the engine's reading matched what people said (at the time)."""
    r = conn.execute(
        """SELECT count(*) AS n, count(*) FILTER (WHERE predicted = intent) AS hits
             FROM meme_feedback WHERE intent IS NOT NULL""").fetchone()
    return {"labelled": r["n"], "agreed": r["hits"], "accuracy": round(r["hits"] / r["n"], 3) if r["n"] else None}


def main(argv: list[str]) -> None:
    from . import db

    if not argv:
        sys.exit(__doc__)
    caption = ""
    if "--caption" in argv:
        i = argv.index("--caption")
        caption = argv[i + 1] if i + 1 < len(argv) else ""
        argv = argv[:i] + argv[i + 2:]
    target, context = argv[0], argv[1:]
    db.init_db()
    with db.session() as conn:
        clip = find_clip(conn, int(target)) if target.isdigit() else find_clip(conn, link=target)
        if clip is None:
            sys.exit(f"no clip {target!r}")
        out = understand(conn, clip, context, caption)
    out.pop("details")
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
