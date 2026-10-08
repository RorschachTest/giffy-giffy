"""The assembly line every clip goes through.

    file + optional sidecar JSON
        -> probe -> exact-duplicate check -> keyframes -> near-duplicate check
        -> web-ready copy + thumbnail
        -> speech to text -> faces -> names found in captions -> topics + text mood (Laya) -> mood (local models)
        -> review flags -> searchable columns + embedding
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

from psycopg.types.json import Jsonb

from . import config, faces, indexing, laya, media, vibe
from .text import has_devanagari, to_roman

log = logging.getLogger("pipeline")

LIST_FIELDS = ("hashtags", "comments", "folk_names", "people", "reactions", "use_when", "topics")
MAX_COMMENTS = 20


class Rejected(Exception):
    """The clip is not something we index (too long, unreadable ...)."""


# --------------------------------------------------------------------------- #
# Sidecar metadata
# --------------------------------------------------------------------------- #

def find_sidecar(video: Path) -> Path | None:
    """clip.mp4 -> clip.json, or clip.info.json as written by `yt-dlp --write-info-json`."""
    for candidate in (video.with_suffix(".json"), video.with_suffix(".info.json")):
        if candidate.exists():
            return candidate
    return None


def _as_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = value.split(",")
    return [str(v).strip() for v in value if str(v).strip()]


def load_meta(sidecar: Path | None) -> dict:
    """Read our own sidecar format, and also yt-dlp's, into one shape."""
    raw: dict = {}
    if sidecar:
        try:
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("ignoring unreadable sidecar %s: %s", sidecar.name, exc)
    comments = raw.get("comments") or []
    comments = [c.get("text", "") if isinstance(c, dict) else c for c in comments]
    meta = {
        "title": (raw.get("title") or "").strip(),
        # yt-dlp calls the caption "description"; ours is the visual description.
        "caption": (raw.get("caption") or (raw.get("description") if "webpage_url" in raw else "") or "").strip(),
        "description": ("" if "webpage_url" in raw else (raw.get("description") or "")).strip(),
        "hashtags": _as_list(raw.get("hashtags") or raw.get("tags")),
        "comments": _as_list(comments)[:MAX_COMMENTS],
        "source_url": (raw.get("source_url") or raw.get("webpage_url") or "").strip(),
        "language": (raw.get("language") or "").strip() or None,
        "folk_names": _as_list(raw.get("folk_names")),
        "people": _as_list(raw.get("people")),
        "source_title": (raw.get("source_title") or "").strip(),
        "reactions": _as_list(raw.get("reactions")),
        "use_when": _as_list(raw.get("use_when")),
        "topics": _as_list(raw.get("topics")),
        "transcript": (raw.get("transcript") or "").strip(),
    }
    return meta


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #

def find_duplicate(conn, sha: str, phash: list[int] | None, duration: float) -> int | None:
    row = conn.execute("SELECT id FROM clips WHERE sha256 = %s", (sha,)).fetchone()
    if row:
        return row["id"]
    if phash is None:
        return None
    # A plain scan is fine up to tens of thousands of clips.
    candidates = conn.execute(
        "SELECT id, phash FROM clips WHERE abs(duration - %s) <= %s",
        (duration, config.DEDUP_MAX_DURATION_DIFF),
    ).fetchall()
    best_id, best = None, float(config.DEDUP_MAX_HAMMING)
    for c in candidates:
        d = media.mean_hamming(phash, c["phash"])
        if d <= best:
            best_id, best = c["id"], d
    return best_id


def _union(a: list[str], b: list[str], limit: int | None = None) -> list[str]:
    seen = {x.lower() for x in a}
    out = list(a)
    for x in b:
        if x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out[:limit] if limit else out


def pipeline_topics(given: list[str], found: list[str]) -> list[str]:
    """Topics from the sidecar or a human come first; Laya's are added after them."""
    return _union(given, found)


def merge_duplicate(conn, clip_id: int, meta: dict) -> None:
    """Same clip seen again: keep one record, absorb the new context, count it.

    The count is the trend signal; the extra captions are free folk names.
    """
    clip = conn.execute("SELECT * FROM clips WHERE id = %s FOR UPDATE", (clip_id,)).fetchone()
    updates = {f: _union(clip[f], meta[f]) for f in LIST_FIELDS if f != "comments"}
    # Other uploaders' titles and captions are "what people call it": keep them.
    extra_words = [t for t in (meta["title"], meta["caption"])
                   if t and t not in (clip["title"], clip["caption"])]
    updates["comments"] = _union(clip["comments"], [*extra_words, *meta["comments"]], MAX_COMMENTS)
    updates["title"] = clip["title"] or meta["title"]
    updates["caption"] = clip["caption"] or meta["caption"]
    updates["source_title"] = clip["source_title"] or meta["source_title"]
    updates["description"] = clip["description"] or meta["description"]
    updates["source_urls"] = _union(clip["source_urls"], [meta["source_url"]] if meta["source_url"] else [])
    sets = ", ".join(f"{k} = %({k})s" for k in updates)
    conn.execute(
        f"UPDATE clips SET {sets}, duplicates_seen = duplicates_seen + 1 WHERE id = %(id)s",
        {**updates, "id": clip_id},
    )
    indexing.refresh(conn, clip_id)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def process_clip(conn, video: Path, sidecar: Path | None = None) -> dict:
    """Process one file. Returns {"status": "new" | "duplicate", "clip_id": ...}.

    Raises Rejected for clips we do not want, MediaError for broken files.
    The caller owns the transaction (commit on success, rollback on error).
    """
    meta = load_meta(sidecar or find_sidecar(video))
    info = media.probe(video)
    if info.duration > config.MAX_CLIP_SECONDS:
        raise Rejected(f"clip is {info.duration:.0f}s; the limit is {config.MAX_CLIP_SECONDS:.0f}s")

    sha = media.sha256(video)
    dup = find_duplicate(conn, sha, None, info.duration)
    if dup:
        merge_duplicate(conn, dup, meta)
        return {"status": "duplicate", "clip_id": dup, "match": "exact"}

    written: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="memeclip_") as tmp_name:
        tmp = Path(tmp_name)
        frames = media.extract_keyframes(video, tmp, info.duration, config.KEYFRAMES)
        phash = [media.dhash(f) for f in frames]

        dup = find_duplicate(conn, sha, phash, info.duration)
        if dup:
            merge_duplicate(conn, dup, meta)
            return {"status": "duplicate", "clip_id": dup, "match": "near"}

        try:
            # --- web-ready copy -------------------------------------------------
            stem = sha[:16]
            out_video = media.transcode(video, config.MEDIA_DIR / f"{stem}.mp4")
            written.append(out_video)
            out_thumb = media.make_thumb(frames[len(frames) // 2], config.MEDIA_DIR / f"{stem}.jpg")
            written.append(out_thumb)

            raw: dict = {"sidecar": meta}
            reasons: list[str] = []

            # --- what is said ---------------------------------------------------
            language = meta["language"] or config.DEFAULT_LANGUAGE
            native = meta["transcript"]
            if native:
                raw["stt"] = {"source": "sidecar"}
            elif info.has_audio and config.ENABLE_STT:
                from . import stt  # lazy: the model is only loaded when needed

                wav = media.extract_audio(video, tmp / "audio.wav")
                result = stt.transcribe(wav, language)
                raw["stt"] = result
                native = result["text"]
                language = language or result["language"]
                if native and ((result["avg_logprob"] or 0) < -1.0 or result["language_probability"] < 0.5):
                    reasons.append("low_stt_confidence")
            roman = to_roman(native) if has_devanagari(native) else native
            if not native:
                reasons.append("no_speech")

            # --- the record -----------------------------------------------------
            clip_id = conn.execute(
                """INSERT INTO clips (sha256, phash, file, thumb, duration, width, height, has_audio,
                                      language, transcript_native, transcript_roman,
                                      title, caption, hashtags, comments, source_urls,
                                      folk_names, people, source_title, description,
                                      reactions, use_when, topics)
                   VALUES (%(sha)s, %(phash)s, %(file)s, %(thumb)s, %(duration)s, %(width)s,
                           %(height)s, %(has_audio)s, %(language)s, %(native)s, %(roman)s,
                           %(title)s, %(caption)s, %(hashtags)s, %(comments)s, %(source_urls)s,
                           %(folk_names)s, %(people)s, %(source_title)s, %(description)s,
                           %(reactions)s, %(use_when)s, %(topics)s)
                   RETURNING id""",
                {
                    "sha": sha, "phash": phash, "file": out_video.name, "thumb": out_thumb.name,
                    "duration": info.duration, "width": info.width, "height": info.height,
                    "has_audio": info.has_audio, "language": language,
                    "native": native, "roman": roman,
                    "title": meta["title"], "caption": meta["caption"],
                    "hashtags": meta["hashtags"], "comments": meta["comments"],
                    "source_urls": [meta["source_url"]] if meta["source_url"] else [],
                    "folk_names": meta["folk_names"], "people": meta["people"],
                    "source_title": meta["source_title"], "description": meta["description"],
                    "reactions": meta["reactions"], "use_when": meta["use_when"],
                    "topics": meta["topics"],
                },
            ).fetchone()["id"]

            # --- who is in it ---------------------------------------------------
            people = {name: "sidecar" for name in meta["people"]}
            said_about_it = " ".join([meta["title"], meta["caption"], *meta["hashtags"], *meta["comments"]])
            from_text = faces.people_in_text(conn, said_about_it)
            for name in from_text:
                people.setdefault(name, "metadata")

            detected = None
            if config.ENABLE_FACES:
                engine = faces.get_engine()
                detected = [engine.detect(f) for f in frames]
                seen = faces.record_faces(conn, clip_id, detected)
                raw["faces"] = {"faces": seen["faces"], "unknown": seen["unknown"],
                                "matched": {k: round(v, 3) for k, v in seen["people"].items()}}
                for name, score in seen["people"].items():
                    confirmed = name in people
                    people.setdefault(name, "face")
                    if not confirmed and score < config.FACE_MATCH_THRESHOLD + 0.08:
                        reasons.append("weak_face_match")
                if seen["unknown"] and not seen["people"]:
                    reasons.append("unknown_faces")
            raw["people_evidence"] = people

            # --- topics + text mood (Laya): sidecar topics are kept, Laya adds ----
            text_moods = None
            if laya.available():
                try:
                    said = laya.describe_clip(meta, native, roman, list(people))
                    raw["laya"] = said["scores"]
                    text_moods = said["mood"]
                    topics = pipeline_topics(meta["topics"], said["topics"])
                    conn.execute("UPDATE clips SET topics = %s WHERE id = %s", (topics, clip_id))
                except Exception:
                    log.exception("laya failed for %s", video.name)
                    reasons.append("laya_failed")

            # --- mood: only fills what the sidecar left empty --------------------
            if config.ENABLE_VIBE:
                try:
                    found = vibe.analyse(frames, detected, native, text_moods)
                    raw["vibe"] = found["scores"]
                    filled = vibe.fill_missing({k: meta[k] for k in found["fields"]}, found["fields"])
                    conn.execute(
                        "UPDATE clips SET description = %s, reactions = %s, use_when = %s WHERE id = %s",
                        (filled["description"], filled["reactions"], filled["use_when"], clip_id),
                    )
                except Exception:
                    log.exception("mood extraction failed for %s", video.name)
                    reasons.append("vibe_failed")

            if not (native or meta["title"] or meta["caption"] or meta["folk_names"]):
                reasons.append("no_context")

            reasons = list(dict.fromkeys(reasons))
            conn.execute(
                """UPDATE clips SET people = %s, needs_review = %s, review_reasons = %s, raw = %s
                    WHERE id = %s""",
                (list(people), bool(reasons), reasons, Jsonb(raw), clip_id),
            )
            indexing.refresh(conn, clip_id)
        except Exception:
            for p in written:
                p.unlink(missing_ok=True)
            raise

    return {"status": "new", "clip_id": clip_id, "review": reasons}


def move_aside(video: Path, dest_dir: Path, note: str | None = None) -> None:
    """Move a finished inbox file (and its sidecar) out of the inbox."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in (video, video.with_suffix(".json"), video.with_suffix(".info.json")):
        if not src.exists():
            continue
        dest = dest_dir / src.name
        n = 1
        while dest.exists():
            dest = dest_dir / f"{src.stem}_{n}{src.suffix}"
            n += 1
        shutil.move(str(src), str(dest))
    if note:
        (dest_dir / f"{video.name}.error.txt").write_text(note, encoding="utf-8")
