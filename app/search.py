"""Hybrid search: spelling-tolerant keyword match + meaning match, then one score.

keyword (kw)   best of:
                 the whole query as a phrase against each column, weighted by
                 how much that column should count
                 OR the share of query words found anywhere in the clip, which
                 handles mixed queries like "nana patekar haath jodiye"
semantic (sem) cosine similarity between the query and the clip's description
topic (tm)     Laya reads what the query is about ("work", "late", "shocked" ...);
               tm is how strongly, summed over the clip's topics and reactions
               that match (capped at 1). 0 when Laya is off or unsure.
score          W_KEYWORD * kw + W_SEMANTIC * sem + W_TOPIC * tm * (1 - kw)
               + small popularity bonus. The topic boost only fills the gap the
               words leave: "kya karu main" names one clip, and a different clip
               with the same mood must not overtake it.
shown if       kw >= MIN_KEYWORD or tm >= MIN_TOPIC, or the meaning match stands out from the
               library (z >= MIN_SEMANTIC_Z; below SEMANTIC_Z_MIN_CLIPS clips: sem >= MIN_SEMANTIC)

The weights are guesses until they are tuned against real queries. /search
returns kw and sem for every hit so you can see why something ranked.
"""
from __future__ import annotations

from . import config, hinglish, laya
from .db import vec
from .embed import embed_one
from .text import normalise

COLUMNS = """id, left(sha256, 10) AS share_id, file, thumb, duration, title, transcript_roman, transcript_native, folk_names,
             people, source_title, description, reactions, use_when, topics, hashtags,
             shares, duplicates_seen, needs_review, review_reasons, first_seen"""

# Note: `%%` is a literal % for the driver. `a <% b` means "a fuzzily appears in b".
SEARCH_SQL = f"""
WITH q AS (
    SELECT %(qn)s::text AS qn, %(qv)s::vector AS qv, %(tokens)s::text[] AS tokens,
           %(qt)s::text[] AS qt, %(qw)s::real[] AS qw
),
-- how the whole library scores against this query: a meaning match must STAND OUT from
-- it, since raw similarities are not comparable across queries (nonsense like "xyzzy
-- qwfp" scored 0.49, above the real "salary day already feeling gareeb" at 0.40)
lib AS (
    SELECT count(*) AS n, avg(1 - (c.embedding <=> q.qv)) AS m, stddev_pop(1 - (c.embedding <=> q.qv)) AS sd
      FROM clips c, q WHERE c.embedding IS NOT NULL AND c.duplicate_of IS NULL
),
-- situation lane: each clip's best "send it when" line (clip_uses), and how the library
-- scores, so a clip only counts when its best line stands out (as the meaning lane does)
-- The library's spread comes from a fixed quarter of the clips (clip_id %% 4 = 0): the same
-- mean and spread within noise, a quarter of the work. Candidates come from the vector index.
use_sample AS (
    SELECT max(1 - (u.embedding <=> q.qv)) AS us
      FROM clip_uses u, q WHERE %(use_on)s AND u.clip_id %% 4 = 0 GROUP BY u.clip_id
),
ulib AS (SELECT count(*) * 4 AS n, avg(us) AS m, stddev_pop(us) AS sd FROM use_sample),
use_top AS (
    SELECT DISTINCT u.clip_id FROM (
        SELECT u.clip_id FROM clip_uses u, q WHERE %(use_on)s
         ORDER BY u.embedding <=> q.qv LIMIT 400) u
),
candidates AS (
    (SELECT c.id FROM clips c, q
      WHERE q.qn <%% c.s_names OR q.qn <%% c.s_said OR q.qn <%% c.s_people
         OR q.qn <%% c.s_learned OR q.qn <%% c.s_all
         OR EXISTS (SELECT 1 FROM unnest(q.tokens) t WHERE t <%% c.s_all)
      ORDER BY word_similarity(q.qn, c.s_all) DESC
      LIMIT 500)
    UNION
    (SELECT c.id FROM clips c, q
      WHERE c.embedding IS NOT NULL
      ORDER BY c.embedding <=> q.qv
      LIMIT 50)
    UNION
    (SELECT c.id FROM clips c, q
      WHERE cardinality(q.qt) > 0 AND (c.topics && q.qt OR c.reactions && q.qt)
      LIMIT 200)
    UNION
    (SELECT clip_id FROM use_top)
),
-- each candidate's best line (only candidates: not every clip in the library)
uses AS (
    SELECT u.clip_id, max(1 - (u.embedding <=> q.qv)) AS us
      FROM clip_uses u JOIN candidates c ON c.id = u.clip_id, q WHERE %(use_on)s GROUP BY u.clip_id
),
scored AS (
    SELECT c.*,
           GREATEST(
               1.00 * word_similarity(q.qn, c.s_names),
               1.00 * word_similarity(q.qn, c.s_learned),
               0.95 * word_similarity(q.qn, c.s_said),
               0.90 * word_similarity(q.qn, c.s_people),
               0.60 * word_similarity(q.qn, c.s_meta),
               -- a word counts only if it is a clear match; weak overlaps are noise
               0.85 * COALESCE((SELECT avg(CASE WHEN word_similarity(t, c.s_all) >= 0.5
                                                THEN word_similarity(t, c.s_all) ELSE 0 END)
                                  FROM unnest(q.tokens) t), 0)
           ) AS kw,
           CASE WHEN c.embedding IS NULL THEN 0
                ELSE GREATEST(0, 1 - (c.embedding <=> q.qv)) END AS sem,
           CASE WHEN c.embedding IS NULL OR coalesce(lib.sd, 0) = 0 THEN 0
                ELSE ((1 - (c.embedding <=> q.qv)) - lib.m) / lib.sd END AS sem_z,
           lib.n AS lib_n,
           COALESCE(uses.us, 0) AS use_sim,
           CASE WHEN uses.us IS NULL OR coalesce(ulib.sd, 0) = 0 THEN 0
                ELSE (uses.us - ulib.m) / ulib.sd END AS use_z,
           ulib.n AS use_n,
           LEAST(1, COALESCE((SELECT sum(x.w) FROM unnest(q.qt, q.qw) AS x(t, w)
                               WHERE x.t = ANY(c.topics) OR x.t = ANY(c.reactions)), 0)) AS tm
      FROM clips c JOIN candidates USING (id) LEFT JOIN uses ON uses.clip_id = c.id, q, lib, ulib
)
SELECT {COLUMNS}, kw, sem, tm,
       %(wk)s * kw + %(ws)s * sem + %(wt)s * tm * (1 - LEAST(kw, 1))
         -- only a line that stands out counts: below the bar every clip would get a little,
         -- and that noise buried exact quote and name matches (measured)
         + CASE WHEN use_n >= %(z_min_clips)s AND use_z >= %(min_use_z)s THEN %(wu)s * use_sim ELSE 0 END
         + %(wp)s * LEAST(ln(%(learn)s * shares + duplicates_seen), 3) AS score
  FROM scored
 WHERE duplicate_of IS NULL
   AND (kw >= %(min_kw)s OR tm >= %(min_tm)s
        -- big libraries: the clip must stand out; tiny ones (tests, a fresh install) cannot
        -- produce large z-scores, so they keep the plain similarity floor
        OR (lib_n >= %(z_min_clips)s AND sem_z >= %(min_sem_z)s)
        OR (lib_n < %(z_min_clips)s AND sem >= %(min_sem)s)
        -- situation lane: same rule, the best line must stand out among clips that have lines
        OR (use_n >= %(z_min_clips)s AND use_z >= %(min_use_z)s))
 ORDER BY score DESC, id
 LIMIT %(limit)s OFFSET %(offset)s
"""

# Browsing order: most shared first. Every key DESC, so "the next page" is simply
# "rows below the last one seen" as one row comparison, which the clips_browse index
# answers directly (keyset paging). OFFSET would re-read every skipped row.
BROWSE_ORDER = "(shares + duplicates_seen) DESC, first_seen DESC, id DESC"

TRENDING_SQL = f"""
SELECT {COLUMNS}, NULL::real AS kw, NULL::real AS sem, NULL::real AS tm, NULL::real AS score
  FROM clips
 WHERE duplicate_of IS NULL
 ORDER BY {BROWSE_ORDER}
 LIMIT %(limit)s OFFSET %(offset)s
"""

BROWSE_AFTER_SQL = f"""
SELECT {COLUMNS}, NULL::real AS kw, NULL::real AS sem, NULL::real AS tm, NULL::real AS score
  FROM clips
 WHERE ((shares + duplicates_seen), first_seen, id) < (%(pop)s, %(seen)s::timestamptz, %(id)s)
   AND duplicate_of IS NULL
 ORDER BY {BROWSE_ORDER}
 LIMIT %(limit)s
"""


def browse_cursor(row: dict) -> str:
    """Where a browse page ended, as an opaque token for the next request."""
    return f"{row['shares'] + row['duplicates_seen']}~{row['first_seen'].isoformat()}~{row['id']}"


def parse_cursor(token: str) -> dict:
    """The token back into its three keys. ValueError when it is not one of ours."""
    pop, seen, clip_id = token.split("~")
    from datetime import datetime
    return {"pop": int(pop), "seen": datetime.fromisoformat(seen), "id": int(clip_id)}


def search(conn, query: str, limit: int = 24, log_query: bool = True, intent: dict | None = None,
           offset: int = 0, after: str | None = None) -> list[dict]:
    query = (query or "").strip()
    if not query:
        if after:   # keyset: the rows after the last one the client has
            return conn.execute(BROWSE_AFTER_SQL, {**parse_cursor(after), "limit": limit}).fetchall()
        return conn.execute(TRENDING_SQL, {"limit": limit, "offset": offset}).fetchall()

    qn = normalise(query)
    if not qn:  # emoji-only or punctuation-only query
        return []
    if intent is None:                  # callers that need it too pass it in: one Laya call per search
        intent = laya.query_intent(query)   # {} when Laya is off, slow or unsure
    rows = conn.execute(SEARCH_SQL, {
        "qn": qn,
        "qv": vec(embed_one(hinglish.gloss(query))),   # as typed, plus an English gloss of its Hindi words
        "tokens": [t for t in qn.split() if len(t) >= 3] or qn.split(),
        "qt": list(intent), "qw": list(intent.values()),
        "wk": config.W_KEYWORD, "ws": config.W_SEMANTIC, "wt": config.W_TOPIC,
        "wp": config.W_POPULARITY, "learn": 1 if config.LEARN_FROM_SHARES else 0,
        "min_kw": config.MIN_KEYWORD, "min_sem": config.MIN_SEMANTIC,
        "min_sem_z": config.MIN_SEMANTIC_Z, "z_min_clips": config.SEMANTIC_Z_MIN_CLIPS,
        "min_tm": config.MIN_TOPIC, "limit": limit, "offset": offset,
        "use_on": config.ENABLE_USES, "wu": config.W_USE if config.ENABLE_USES else 0.0,
        "min_use_z": config.MIN_USE_Z,
    }).fetchall()
    if log_query:
        conn.execute(
            # anonymous on purpose: no account, session or IP, and only the hour, so a search
            # cannot be matched to who was signed in at that minute
            "INSERT INTO query_log (at, query, query_norm, results, top_score) "
            "VALUES (date_trunc('hour', now()), %s, %s, %s, %s)",
            (query, qn, len(rows), rows[0]["score"] if rows else None),
        )
    return rows


def record_share(conn, clip_id: int, query: str | None) -> bool:
    """Someone picked this clip. If they got here by searching, the query they
    typed becomes a way to find the clip: real behaviour, not a model's guess."""
    from . import indexing

    qn = normalise(query or "")
    if qn:
        updated = conn.execute(
            """UPDATE clips
                  SET shares = shares + 1,
                      learned_queries = jsonb_set(
                          learned_queries, ARRAY[%(qn)s]::text[],
                          to_jsonb(COALESCE((learned_queries ->> %(qn)s)::int, 0) + 1))
                WHERE id = %(id)s""",
            {"qn": qn, "id": clip_id},
        ).rowcount
    else:
        updated = conn.execute("UPDATE clips SET shares = shares + 1 WHERE id = %s",
                               (clip_id,)).rowcount
    if updated and qn and config.LEARN_FROM_SHARES:
        indexing.refresh(conn, clip_id, embed=False)
    return bool(updated)
