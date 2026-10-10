"""Rebuild the searchable columns for every clip.

    python -m app.reindex              after changing rules in text.py or indexing.py
    python -m app.reindex --no-embed   skip embeddings (much faster)
    python -m app.reindex --vibe       first fill empty mood/description/use-when (local models)
    python -m app.reindex --laya       first add Laya topics to every clip (keeps existing ones)
    python -m app.reindex --vibe --replace
                                       recompute mood on every clip, overwriting hand edits too
    python -m app.reindex --roman      first rebuild the Roman transcript from the stored Devanagari
                                       (after changing text.to_roman or the loanwords in hinglish.py)
    python -m app.reindex --stt        first re-run speech to text on every clip with audio
                                       (not typed sidecar transcripts); then --laya is worth a rerun

Without --stt it does not re-run speech to text or faces; it only re-derives from stored facts.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from psycopg.types.json import Jsonb

from . import config, db, faces, indexing, laya, media, pipeline, storage, vibe


def _report_failed(step: str, failed: list[int], total: int) -> None:
    if failed:
        print(f"{step}: {len(failed)} of {total} clip(s) skipped (ids {failed}); fix the cause and run again",
              file=sys.stderr)


def backfill_roman() -> None:
    """Re-romanise every stored Devanagari transcript with the current rules. No models."""
    from .text import has_devanagari, to_roman

    with db.session() as conn:
        rows = conn.execute("SELECT id, transcript_native, transcript_roman FROM clips ORDER BY id").fetchall()
        changed = 0
        for r in rows:
            native = r["transcript_native"] or ""
            roman = to_roman(native) if has_devanagari(native) else native
            if roman != (r["transcript_roman"] or ""):
                conn.execute("UPDATE clips SET transcript_roman = %s WHERE id = %s", (roman, r["id"]))
                changed += 1
    print(f"roman: {changed} of {len(rows)} transcript(s) changed")


def backfill_stt() -> None:
    """Transcribe every clip with audio again, with the current model and settings.
    Transcripts that came typed in a sidecar are a person's and are left alone."""
    from . import stt
    from .text import has_devanagari, to_roman

    with db.session() as conn:
        rows = conn.execute("SELECT * FROM clips WHERE has_audio ORDER BY id").fetchall()
    failed = []
    for r in rows:
        raw = r["raw"] or {}
        if (raw.get("stt") or {}).get("source") == "sidecar":
            continue
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with storage.local_copy(r["file"]) as clip_file:
                    wav = media.extract_audio(clip_file, Path(tmp) / "audio.wav")
                language = (raw.get("sidecar") or {}).get("language") or config.DEFAULT_LANGUAGE
                result = stt.transcribe(wav, language, stt.hint(r["title"], r["folk_names"], r["source_title"]))
            native = result["text"]
            roman = to_roman(native) if has_devanagari(native) else native
            with db.session() as conn:
                conn.execute("""UPDATE clips SET transcript_native = %s, transcript_roman = %s, raw = %s
                                 WHERE id = %s""", (native, roman, Jsonb({**raw, "stt": result}), r["id"]))
            print(f"stt: clip {r['id']} {r['title']!r}: {roman[:100]!r}")
        except Exception as exc:   # one bad clip must not stop the rest
            failed.append(r["id"])
            print(f"stt: clip {r['id']} {r['title']!r} SKIPPED: {exc}", file=sys.stderr)
    _report_failed("stt", failed, len(rows))


def backfill_laya() -> None:
    """Ask Laya about every clip; its topics are added after any already there."""
    with db.session() as conn:
        rows = conn.execute("SELECT * FROM clips ORDER BY id").fetchall()
    failed = []
    for r in rows:
        try:
            found = laya.describe_clip(r, r["transcript_native"], r["transcript_roman"], r["people"])
            topics = pipeline.pipeline_topics(r["topics"], found["topics"])
            raw = {**(r["raw"] or {}), "laya": found["scores"]}
            with db.session() as conn:
                conn.execute("UPDATE clips SET topics = %s, raw = %s WHERE id = %s",
                             (topics, Jsonb(raw), r["id"]))
            print(f"laya: clip {r['id']} {r['title']!r}: {topics}")
        except Exception as exc:   # one bad clip must not stop the rest
            failed.append(r["id"])
            print(f"laya: clip {r['id']} {r['title']!r} SKIPPED: {exc}", file=sys.stderr)
    _report_failed("laya", failed, len(rows))


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
    failed = []
    for r in rows:
        try:
            with tempfile.TemporaryDirectory(prefix="memeclip_") as tmp, storage.local_copy(r["file"]) as clip_file:
                frames = media.extract_keyframes(clip_file, Path(tmp), r["duration"], config.KEYFRAMES)
                detected = [engine.detect(f) for f in frames] if engine else None
                text_moods = None
                if laya.available():
                    try:   # Laya down: fall back to the plain text signal, as at ingest
                        text_moods = laya.describe_clip(r, r["transcript_native"], r["transcript_roman"],
                                                        r["people"])["mood"]
                    except Exception as exc:
                        print(f"vibe: clip {r['id']} without Laya: {exc}", file=sys.stderr)
                found = vibe.analyse(frames, detected, r["transcript_native"], text_moods)
            filled = found["fields"] if replace else vibe.fill_missing(r, found["fields"])
            raw = {**(r["raw"] or {}), "vibe": found["scores"]}
            with db.session() as conn:
                conn.execute("UPDATE clips SET description=%s, reactions=%s, use_when=%s, raw=%s WHERE id=%s",
                             (filled["description"], filled["reactions"], filled["use_when"],
                              Jsonb(raw), r["id"]))
            print(f"vibe: clip {r['id']} {r['title']!r}: {filled['reactions']}")
        except Exception as exc:   # one bad clip must not stop the rest
            failed.append(r["id"])
            print(f"vibe: clip {r['id']} {r['title']!r} SKIPPED: {exc}", file=sys.stderr)
    _report_failed("vibe", failed, len(rows))


def main() -> None:
    embed = "--no-embed" not in sys.argv
    db.init_db()
    if "--roman" in sys.argv:
        backfill_roman()
    if "--stt" in sys.argv:
        if not config.ENABLE_STT:
            sys.exit("--stt needs ENABLE_STT=1")
        backfill_stt()
    if "--laya" in sys.argv:
        if not laya.available():
            sys.exit("--laya needs ENABLE_LAYA=1 and LAYA_URL (and the laya service: docker compose --profile laya up -d laya)")
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
