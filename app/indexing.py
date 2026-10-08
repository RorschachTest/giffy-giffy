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

from . import faces
from .db import vec
from .embed import embed_one
from .text import clean_caption, normalise

MAX_COMMENTS_INDEXED = 10


def search_fields(clip: dict, aliases: list[str]) -> dict:
    title, title_tags = clean_caption(clip["title"])
    caption, caption_tags = clean_caption(clip["caption"])
    tags = list(dict.fromkeys([*clip["hashtags"], *title_tags, *caption_tags]))

    names = [*clip["folk_names"], title]
    said = clip["transcript_roman"]
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
        "s_learned": " ".join(clip["learned_queries"].keys()),
    }
    f["s_all"] = " ".join(v for v in f.values() if v)
    return f


def embedding_text(clip: dict) -> str:
    """Plain sentences for the meaning-based model. Not normalised: the model
    understands real spelling better than our squashed spelling."""
    title, _ = clean_caption(clip["title"])
    caption, _ = clean_caption(clip["caption"])
    parts = [
        title,
        ", ".join(clip["folk_names"]),
        f"Dialogue: {clip['transcript_roman']}" if clip["transcript_roman"] else "",
        clip["description"],
        f"People: {', '.join(clip['people'])}" if clip["people"] else "",
        f"From: {clip['source_title']}" if clip["source_title"] else "",
        f"Mood: {', '.join(clip['reactions'])}" if clip["reactions"] else "",
        f"Use when: {'; '.join(clip['use_when'])}" if clip["use_when"] else "",
        f"About: {', '.join(clip['topics'])}" if clip["topics"] else "",
        caption,
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
