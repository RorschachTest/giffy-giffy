"""Report on what the pipeline made of the clips, and run a few test searches.

    python -m app.smoke

Run inside the api container (scripts/smoke_test.sh does that). Exits non-zero
when a clip failed to process or a test search missed.

Test searches come from DATA_DIR/smoke_queries.txt, one per line:

    what you would type => words from the title of the clip you expect on top

Lines starting with # are ignored. Without that file only the report is printed.
"""
from __future__ import annotations

import sys

from . import config, db, faces, laya, search
from .text import normalise

QUERIES_FILE = config.DATA_DIR / "smoke_queries.txt"


def _label(clip: dict) -> str:
    return (clip["folk_names"] or [clip["title"] or f"clip {clip['id']}"])[0]


def report_clips(conn) -> int:
    clips = conn.execute("SELECT * FROM clips ORDER BY id").fetchall()
    print(f"clips indexed: {len(clips)}")
    for c in clips:
        stt = c["raw"].get("stt") or {}
        seen = c["raw"].get("faces") or {}
        print(f"\n[{c['id']}] {_label(c)}")
        print(f"    file        {c['file']}  {c['duration']:.1f}s  {c['width']}x{c['height']}"
              f"  audio={'yes' if c['has_audio'] else 'no'}  seen {c['duplicates_seen']}x")
        print(f"    language    {c['language']}  (confidence {stt.get('language_probability')},"
              f" avg_logprob {stt.get('avg_logprob')}, model {stt.get('model') or stt.get('source')})")
        print(f"    heard       {c['transcript_native'] or '(nothing)'}")
        if c["transcript_roman"] != c["transcript_native"]:
            print(f"    romanised   {c['transcript_roman']}")
        print(f"    people      {c['people'] or '(none)'}   evidence {c['raw'].get('people_evidence')}")
        if seen:
            print(f"    faces       {seen.get('faces')} seen, {seen.get('unknown')} unknown,"
                  f" matched {seen.get('matched')}")
        print(f"    mood        {c['reactions'] or '(none)'}   use when {c['use_when'] or '(none)'}")
        print(f"    topics      {c['topics'] or '(none)'}")
        print(f"    review      {c['review_reasons'] or 'ok'}")
        print(f"    s_names     {c['s_names']}")
        print(f"    s_said      {c['s_said']}")
        print(f"    s_people    {c['s_people']}")
        print(f"    embedding   {'yes' if c['embedding'] else 'MISSING'}")
    return len(clips)


def report_failed() -> int:
    notes = sorted(config.FAILED_DIR.glob("*.error.txt")) if config.FAILED_DIR.exists() else []
    print(f"\nclips that failed: {len(notes)}")
    for note in notes:
        lines = note.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        print(f"    {note.name[:-len('.error.txt')]}: {lines[-1] if lines else '(no message)'}")
    return len(notes)


def report_people(conn) -> None:
    rows = conn.execute(
        """SELECT p.name, p.aliases, count(r.id) AS photos
             FROM people p LEFT JOIN face_refs r ON r.person_id = p.id
            GROUP BY p.id ORDER BY p.name"""
    ).fetchall()
    print(f"\npeople known: {len(rows)}")
    for r in rows:
        print(f"    {r['name']}  aliases={r['aliases']}  reference faces={r['photos']}")
    clusters = faces.unknown_clusters(conn)
    print(f"unknown face groups: {len(clusters)}")
    for c in clusters[:10]:
        print(f"    face_id {c['face_id']}: {c['faces']} face(s) in clip(s) {c['clips']}")


def run_queries(conn) -> tuple[int, int]:
    if not QUERIES_FILE.exists():
        print(f"\nno {QUERIES_FILE.name}; skipping test searches")
        return 0, 0
    passed = failed = 0
    print("\ntest searches")
    for line in QUERIES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=>" not in line:
            continue
        query, expected = (part.strip() for part in line.split("=>", 1))
        rows = search.search(conn, query, limit=3, log_query=False)
        top = rows[0] if rows else None
        ok = bool(top) and normalise(expected) in normalise(f"{top['title']} {' '.join(top['folk_names'])}")
        passed, failed = passed + ok, failed + (not ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {query!r}  (want: {expected})"
              + (f"  read as {laya.query_intent(query)}" if laya.available() else ""))
        for r in rows:
            print(f"          {r['score']:.3f}  kw {r['kw']:.2f}  sem {r['sem']:.2f}"
                  f"  tm {r['tm']:.2f}  {_label(r)}")
        if not rows:
            print("          no results")
    print(f"\nsearches: {passed} passed, {failed} failed")
    return passed, failed


def main() -> int:
    print(f"settings: whisper={config.WHISPER_MODEL if config.ENABLE_STT else 'off'}"
          f"  faces={'on' if config.ENABLE_FACES else 'off'}"
          f"  embeddings={config.EMBED_BACKEND}:{config.EMBED_MODEL}"
          f"  mood={'on' if config.ENABLE_VIBE else 'off'}"
          f"  laya={config.LAYA_URL if laya.available() else 'off'}")
    with db.session() as conn:
        indexed = report_clips(conn)
        broken = report_failed()
        report_people(conn)
        _, missed = run_queries(conn)
    if not indexed:
        print("\nnothing was indexed: put clips in data/inbox and run again")
    return 1 if (broken or missed or not indexed) else 0


if __name__ == "__main__":
    sys.exit(main())
