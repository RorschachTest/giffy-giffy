"""Builds the searchable columns from a clip's facts.

The clip row holds FACTS (transcript, people, title ...). The s_* columns and the
embedding are DERIVED from them here and nowhere else, so they can always be
rebuilt: after an edit, after a share teaches us a new query, or after the rules
in text.py change (python -m app.reindex).

Which fact feeds which column is the single most important tuning decision:

  s_names   what the meme is called        <- folk names + cleaned title
  s_said    what is said                   <- transcript
  s_people  who / where it is from         <- people + their aliases + film or show
  s_meta    everything else                <- caption, hashtags, description,
                                              reactions, use-when, topics, top comments
  s_learned real queries that led to a share
"""
from __future__ import annotations

from . import config, faces
from .db import vec
from .embed import embed_one
from .text import clean_caption, normalise

MAX_LEARNED = 30   # most-used queries kept in s_learned; the rest stay in learned_queries

MAX_COMMENTS_INDEXED = 10


def search_fields(clip: dict, aliases: list[str]) -> dict:
    title, title_tags = clean_caption(clip["title"])
    caption, caption_tags = clean_caption(clip["caption"])
    tags = list(dict.fromkeys([*clip["hashtags"], *title_tags, *caption_tags]))

    names = [*clip["folk_names"], title]
    # what the clip says: its speech, and the caption burned into the picture
    said = " ".join(p for p in (clip["transcript_roman"], clip.get("screen_text") or "") if p)
    people = [*clip["people"], *aliases, clip["source_title"]]
    meta = [caption, *tags, clip["description"], *clip["reactions"], *clip["use_when"],
            *clip["topics"],
            *clip["comments"][:MAX_COMMENTS_INDEXED]]

    f = {
        "s_names": normalise(" ".join(names)),
        "s_said": normalise(said),
        "s_people": normalise(" ".join(people)),
        "s_meta": normalise(" ".join(meta)),
        # learned queries were normalised when they were recorded
        "s_learned": "" if not config.LEARN_FROM_SHARES else " ".join(sorted(clip["learned_queries"], key=clip["learned_queries"].get, reverse=True)[:MAX_LEARNED]),
    }
    f["s_all"] = " ".join(v for v in f.values() if v)
    return f


def embedding_text(clip: dict) -> str:
    """Plain sentences for the meaning-based model: the clip's MESSAGE (title, names,
    on-screen text, what is said, who, from where) plus a short mood word. Not normalised:
    the model understands real spelling better than our squashed spelling.

    Machine-written sentences (description, use-when, topics) stay out: the small model
    averages everything, and that boilerplate drowned the real words. Measured on the
    library: "who are these people" ranked "Kon hain ye log" 1,166th with it, 1st without;
    "boys are simple" matched its on-screen caption at 0.79 instead of 0.52. They remain
    in the keyword field (s_meta)."""
    title, _ = clean_caption(clip["title"])
    caption, _ = clean_caption(clip["caption"])
    said = clip.get("transcript_native") or clip["transcript_roman"]   # Devanagari reads far better
    parts = [
        title,
        ", ".join(clip["folk_names"]),
        f"On screen: {clip['screen_text']}" if clip.get("screen_text") else "",
        f"Dialogue: {said}" if said else "",
        f"People: {', '.join(clip['people'])}" if clip["people"] else "",
        f"From: {clip['source_title']}" if clip["source_title"] else "",
        caption,
        f"Mood: {', '.join(clip['reactions'])}" if clip["reactions"] else "",
    ]
    return ". ".join(p for p in parts if p)


def refresh(conn, clip_id: int, embed: bool = True) -> None:
    """Recompute the derived columns for one clip."""
    clip = conn.execute("SELECT * FROM clips WHERE id = %s", (clip_id,)).fetchone()
    if clip is None:
        raise ValueError(f"no clip with id {clip_id}")
    f = search_fields(clip, faces.aliases_for(conn, clip["people"]))
    conn.execute(
        """UPDATE clips SET s_names=%(s_names)s, s_said=%(s_said)s, s_people=%(s_people)s,
                            s_meta=%(s_meta)s, s_learned=%(s_learned)s, s_all=%(s_all)s,
                            updated_at=now()
            WHERE id=%(id)s""",
        {**f, "id": clip_id},
    )
    if embed:
        conn.execute("UPDATE clips SET embedding = %s::vector WHERE id = %s",
                     (vec(embed_one(embedding_text(clip))), clip_id))
        from . import uses   # its situation lines use the same embedding model: keep them in step
        uses.build(conn, [clip_id])
