"""Find and merge duplicate clips already in the library. Nothing is deleted.

    python -m app.dedupe              list duplicate groups (changes nothing)
    python -m app.dedupe --rehash     first recompute the picture hashes from the stored web
                                      copies with black bars trimmed (ffmpeg only, no models)
    python -m app.dedupe --apply      merge each group: the best copy stays; the others point
                                      at it (clips.duplicate_of), give it their titles, links,
                                      share and seen counts, and drop out of search
    python -m app.dedupe --undo       un-hide every merged copy (and take back their counts)

Candidates are clips of about the same length that look alike or say the same thing;
media.same_clip decides (the same rule new uploads use). In short: same words need a
similar picture too (a reused sound over other footage is another meme), a talking
clip never matches a silent one, and silent clips must look the same strictly.
Blank or near-uniform frames and Whisper's filler lines ("thanks for watching") are
ignored: they are alike whatever the clip.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from psycopg.types.json import Jsonb

from . import config, db, indexing, media, storage
from .pipeline import duration_tolerance

_informative = media.informative_hash


def candidate_pairs(conn) -> dict[tuple[int, int], dict]:
    """Pairs of about the same length that look alike or say the same thing, with both
    measures filled in, ready for media.same_clip to decide."""
    rows = conn.execute(
        """SELECT id, phash, duration, transcript_roman FROM clips
            WHERE duplicate_of IS NULL ORDER BY duration"""
    ).fetchall()
    info = {r["id"]: r for r in rows}
    hashes = {r["id"]: [h for h in (r["phash"] or []) if _informative(h)] for r in rows}
    pairs: dict[tuple[int, int], dict] = {}
    for i, a in enumerate(rows):                       # look-alikes, by duration window
        for b in rows[i + 1:]:
            if b["duration"] - a["duration"] > duration_tolerance(a["duration"]):
                break
            ha, hb = hashes[a["id"]], hashes[b["id"]]
            if ha and hb:
                d = media.frame_distance(ha, hb)
                if d <= media.SIMILAR_PICTURE:
                    pairs[(min(a["id"], b["id"]), max(a["id"], b["id"]))] = {"frames": d}
    for r in conn.execute(                              # same words
            """SELECT a.id AS a, b.id AS b FROM clips a JOIN clips b ON a.id < b.id
                WHERE a.duplicate_of IS NULL AND b.duplicate_of IS NULL
                  AND length(a.transcript_roman) >= %(min)s AND length(b.transcript_roman) >= %(min)s
                  AND abs(a.duration - b.duration) <= greatest(%(diff)s, %(share)s * least(a.duration, b.duration))
                  AND similarity(a.transcript_roman, b.transcript_roman) >= %(sim)s""",
            {"min": config.DEDUP_MIN_SPEECH_CHARS, "diff": config.DEDUP_MAX_DURATION_DIFF,
             "share": config.DEDUP_MAX_DURATION_SHARE, "sim": media.SAME_SPEECH}).fetchall():
        key = (r["a"], r["b"])
        if key not in pairs:
            ha, hb = hashes[r["a"]], hashes[r["b"]]
            pairs[key] = {"frames": media.frame_distance(ha, hb) if ha and hb else 64.0}
    if pairs:                                           # speech similarity for every candidate
        keys = list(pairs)
        for r in conn.execute(
                """SELECT p.a, p.b, similarity(coalesce(ca.transcript_roman, ''), coalesce(cb.transcript_roman, '')) AS s
                     FROM unnest(%s::bigint[], %s::bigint[]) AS p(a, b)
                     JOIN clips ca ON ca.id = p.a JOIN clips cb ON cb.id = p.b""",
                ([a for a, _ in keys], [b for _, b in keys])).fetchall():
            pairs[(r["a"], r["b"])]["speech"] = float(r["s"])
    for (a, b), m in pairs.items():
        m["same"] = media.same_clip(m["frames"], m.get("speech", 0.0),
                                    info[a]["transcript_roman"], info[b]["transcript_roman"])
    return pairs


def groups(conn) -> list[list[int]]:
    """Connected groups of duplicates (union-find over the pairs media.same_clip accepts)."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), m in candidate_pairs(conn).items():
        if m["same"]:
            parent[find(a)] = find(b)
    out: dict[int, list[int]] = {}
    for x in parent:
        out.setdefault(find(x), []).append(x)
    return sorted((sorted(g) for g in out.values() if len(g) > 1), key=lambda g: -len(g))


def _keeper(conn, ids: list[int]) -> int:
    """The copy to keep: biggest picture, then most shared/seen, then the oldest."""
    return conn.execute(
        """SELECT id FROM clips WHERE id = ANY(%s)
            ORDER BY coalesce(width, 0) * coalesce(height, 0) DESC, shares + duplicates_seen DESC, id
            LIMIT 1""", (ids,)).fetchone()["id"]


def _union(*lists) -> list:
    out: list = []
    for xs in lists:
        for x in xs or []:
            if x and x not in out:
                out.append(x)
    return out


def merge_group(conn, ids: list[int]) -> int:
    """Keep the best copy; the others give it their names, links and counts and hide."""
    keep = _keeper(conn, ids)
    others = [i for i in ids if i != keep]
    k = conn.execute("SELECT * FROM clips WHERE id = %s FOR UPDATE", (keep,)).fetchone()
    dups = conn.execute("SELECT * FROM clips WHERE id = ANY(%s)", (others,)).fetchall()
    # other uploaders' titles and captions are "what people call it": keep them as comments
    names = [t for d in dups for t in (d["title"], d["caption"]) if t and t not in (k["title"], k["caption"])]
    conn.execute(
        """UPDATE clips SET folk_names = %s, people = %s, hashtags = %s, source_urls = %s, comments = %s,
                            shares = shares + %s, duplicates_seen = duplicates_seen + %s,
                            raw = jsonb_set(coalesce(raw, '{}'), '{merged}', %s)
            WHERE id = %s""",
        (_union(k["folk_names"], *[d["folk_names"] for d in dups]),
         _union(k["people"], *[d["people"] for d in dups]),
         _union(k["hashtags"], *[d["hashtags"] for d in dups]),
         _union(k["source_urls"], *[d["source_urls"] for d in dups]),
         _union(k["comments"], names),
         sum(d["shares"] for d in dups), sum(d["duplicates_seen"] for d in dups),
         Jsonb(_union((k["raw"] or {}).get("merged"), others)), keep),
    )
    conn.execute(
        """UPDATE clips SET duplicate_of = %s, raw = jsonb_set(coalesce(raw, '{}'), '{merged_into}', to_jsonb(%s::bigint))
            WHERE id = ANY(%s)""", (keep, keep, others))
    indexing.refresh(conn, keep, embed=False)
    return keep


def undo(conn) -> int:
    rows = conn.execute("SELECT id, duplicate_of, shares, duplicates_seen FROM clips WHERE duplicate_of IS NOT NULL").fetchall()
    for r in rows:
        conn.execute("""UPDATE clips SET shares = greatest(0, shares - %s),
                               duplicates_seen = greatest(1, duplicates_seen - %s), raw = raw - 'merged'
                         WHERE id = %s""", (r["shares"], r["duplicates_seen"], r["duplicate_of"]))
        conn.execute("UPDATE clips SET duplicate_of = NULL, raw = raw - 'merged_into' WHERE id = %s", (r["id"],))
    return len(rows)


def rehash() -> None:
    """Picture hashes again, from the stored web copies, black bars trimmed."""
    with db.session() as conn:
        rows = conn.execute("SELECT id, file, duration FROM clips WHERE duplicate_of IS NULL ORDER BY id").fetchall()
    failed = []
    for n, r in enumerate(rows, 1):
        try:
            with tempfile.TemporaryDirectory(prefix="memeclip_") as tmp, storage.local_copy(r["file"]) as f:
                frames = media.extract_keyframes(f, Path(tmp), r["duration"], config.KEYFRAMES)
                hashes = [media.dhash(fr) for fr in frames]
            with db.session() as conn:
                conn.execute("UPDATE clips SET phash = %s WHERE id = %s", (hashes, r["id"]))
        except Exception as exc:   # one bad clip must not stop the rest
            failed.append(r["id"])
            print(f"rehash: clip {r['id']} SKIPPED: {exc}", file=sys.stderr)
        if n % 100 == 0 or n == len(rows):
            print(f"rehash: {n}/{len(rows)}", flush=True)
    if failed:
        print(f"rehash: {len(failed)} skipped: {failed[:20]}", file=sys.stderr)


def main() -> int:
    db.init_db()
    if "--undo" in sys.argv:
        with db.session() as conn:
            print(f"un-hid {undo(conn)} merged copies")
        return 0
    if "--rehash" in sys.argv:
        rehash()
    with db.session() as conn:
        found = groups(conn)
        titles = {r["id"]: (r["title"] or "(untitled)", r["duration"]) for r in conn.execute(
            "SELECT id, title, duration FROM clips WHERE id = ANY(%s)", ([i for g in found for i in g],)).fetchall()}
    extra = sum(len(g) - 1 for g in found)
    print(f"{len(found)} duplicate group(s), {extra} extra cop{'y' if extra == 1 else 'ies'}")
    for g in found[:40]:
        print("  " + " | ".join(f"{i}: {titles[i][0][:28]} ({titles[i][1]:.1f}s)" for i in g))
    if "--apply" in sys.argv:
        with db.session() as conn:
            kept = [merge_group(conn, g) for g in found]
        print(f"merged: kept {len(kept)} clips, hid {extra} copies (python -m app.dedupe --undo brings them back)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
