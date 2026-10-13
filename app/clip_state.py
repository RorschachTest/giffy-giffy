"""What a typed-decision model (Jev, Laya) would read about each clip, and how ready the
library is for it.

    python -m app.clip_state                 coverage report: how many clips carry each kind of
                                             evidence, and how many are too thin to judge
    python -m app.clip_state 161 630         the state for those clips, exactly as it would be sent
    GET /clips/{id}/state                    the same for one clip, from the app

These models only read text, so a clip's "state" is every piece of text evidence it has,
labelled, in the order a person would weigh it: the title someone wrote, the caption
burned into the picture, what is said (Devanagari, which models read better than
romanised Hindi, plus the Roman form), who is in it, what people call it, and the
visual mood as a weak hint. A clip with none of title, caption, on-screen text or
speech is "thin": no text model can say what it means, so a person (or a visual model)
has to. Keep the state honest: no field is invented to fill a gap.
"""
from __future__ import annotations

import json
import sys

from . import db

THIN_FIELDS = ("title", "caption", "on_screen_text", "said")   # at least one, or the clip is thin
MAX_FIELD = 500


def _clean_title(t: str | None) -> str:
    t = (t or "").strip()
    return "" if t.lower() in ("", "(untitled)", "untitled") or len(t) <= 3 else t


def build(clip: dict) -> dict:
    """The labelled text evidence for one clip row. Empty fields are left out."""
    stt = (clip.get("raw") or {}).get("stt") or {}
    said = clip.get("transcript_native") or clip.get("transcript_roman") or ""
    trusted = not (stt.get("avg_logprob") is not None and stt["avg_logprob"] < -0.6)
    state = {
        "title": _clean_title(clip.get("title")),
        "caption": (clip.get("caption") or "").strip(),
        "on_screen_text": (clip.get("screen_text") or "").strip(),
        "said": said.strip() if trusted else "",
        "said_roman": (clip.get("transcript_roman") or "").strip()
                      if trusted and clip.get("transcript_roman") != said else "",
        "people": ", ".join(clip.get("people") or []),
        "known_as": ", ".join(clip.get("folk_names") or []),
        "film_or_show": (clip.get("source_title") or "").strip(),
        "hashtags": " ".join(clip.get("hashtags") or []),
        "visual_mood_hint": ", ".join(clip.get("reactions") or []),
    }
    return {k: v[:MAX_FIELD] for k, v in state.items() if v}


def is_thin(state: dict) -> bool:
    return not any(state.get(f) for f in THIN_FIELDS)


def _rows(conn, ids: list[int] | None = None):
    where = "duplicate_of IS NULL" if _has(conn, "duplicate_of") else "TRUE"
    if ids:
        where += " AND id = ANY(%(ids)s)"
    return conn.execute(f"SELECT * FROM clips WHERE {where} ORDER BY id", {"ids": ids or []}).fetchall()


def _has(conn, column: str) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM information_schema.columns WHERE table_name = 'clips' AND column_name = %s",
        (column,)).fetchone())


def report(conn) -> dict:
    rows = _rows(conn)
    states = [build(r) for r in rows]
    n = len(states) or 1
    fields = ["title", "caption", "on_screen_text", "said", "people", "known_as", "visual_mood_hint"]
    cover = {f: round(sum(1 for s in states if s.get(f)) / n, 3) for f in fields}
    thin = [r["id"] for r, s in zip(rows, states) if is_thin(s)]
    words = sorted(len(" ".join(s.values()).split()) for s in states)
    return {"clips": len(states), "coverage": cover, "thin": len(thin), "thin_share": round(len(thin) / n, 3),
            "thin_examples": thin[:20], "median_words": words[len(words) // 2] if words else 0,
            "on_screen_text_column": _has(conn, "screen_text")}


def main(argv: list[str]) -> int:
    with db.session() as conn:
        if argv:
            for r in _rows(conn, [int(a) for a in argv if a.isdigit()]):
                print(json.dumps({"id": r["id"], "state": build(r)}, ensure_ascii=False, indent=2))
            return 0
        rep = report(conn)
    print(f"{rep['clips']} clips; median {rep['median_words']} words of evidence each")
    for f, share in rep["coverage"].items():
        print(f"  {f:18} {share:6.1%}")
    print(f"thin (no title, caption, on-screen text or trusted speech): {rep['thin']} ({rep['thin_share']:.1%})"
          f", e.g. {rep['thin_examples'][:10]}")
    if not rep["on_screen_text_column"]:
        print("note: on-screen text is not read yet on this database (python -m app.reindex --screen-text)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
