"""Hybrid search: spelling-tolerant keyword match + meaning match, then one score.

keyword (kw)   best of:
                 the whole query as a phrase against each column, weighted by
                 how much that column should count
                 OR the share of query words found anywhere in the clip, which
                 handles mixed queries like "nana patekar haath jodiye"
semantic (sem) cosine similarity between the query and the clip's description
score          W_KEYWORD * kw + W_SEMANTIC * sem + small popularity bonus
shown if       kw >= MIN_KEYWORD or sem >= MIN_SEMANTIC (anything less is noise)

The weights are guesses until they are tuned against real queries. /search
returns kw and sem for every hit so you can see why something ranked.
"""
from __future__ import annotations

from . import config
from .db import vec
from .embed import embed_one
from .text import normalise

COLUMNS = """id, file, thumb, duration, title, transcript_roman, transcript_native, folk_names,
             people, source_title, description, reactions, use_when, hashtags,
             shares, duplicates_seen, needs_review, review_reasons, first_seen"""

# Note: `%%` is a literal % for the driver. `a <% b` means "a fuzzily appears in b".
SEARCH_SQL = f"""
WITH q AS (
    SELECT %(qn)s::text AS qn, %(qv)s::vector AS qv, %(tokens)s::text[] AS tokens
),
candidates AS (
    (SELECT c.id FROM clips c, q
      WHERE q.qn <%% c.s_names OR q.qn <%% c.s_said OR q.qn <%% c.s_people
         OR q.qn <%% c.s_learned OR q.qn <%% c.s_all
         OR EXISTS (SELECT 1 FROM unnest(q.tokens) t WHERE t <%% c.s_all)
      LIMIT 500)
    UNION
    (SELECT c.id FROM clips c, q
      WHERE c.embedding IS NOT NULL
      ORDER BY c.embedding <=> q.qv
      LIMIT 50)
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
                ELSE GREATEST(0, 1 - (c.embedding <=> q.qv)) END AS sem
      FROM clips c JOIN candidates USING (id), q
)
SELECT {COLUMNS}, kw, sem,
       %(wk)s * kw + %(ws)s * sem
         + %(wp)s * LEAST(ln(shares + duplicates_seen), 3) AS score
  FROM scored
 WHERE kw >= %(min_kw)s OR sem >= %(min_sem)s
 ORDER BY score DESC, id
 LIMIT %(limit)s
"""

TRENDING_SQL = f"""
SELECT {COLUMNS}, NULL::real AS kw, NULL::real AS sem, NULL::real AS score
  FROM clips
 ORDER BY shares + duplicates_seen DESC, first_seen DESC
 LIMIT %(limit)s
"""


def search(conn, query: str, limit: int = 24, log_query: bool = True) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return conn.execute(TRENDING_SQL, {"limit": limit}).fetchall()

    qn = normalise(query)
    if not qn:  # emoji-only or punctuation-only query
        return []
    rows = conn.execute(SEARCH_SQL, {
        "qn": qn,
        "qv": vec(embed_one(query)),          # the model gets the query as typed
        "tokens": [t for t in qn.split() if len(t) >= 3] or qn.split(),
        "wk": config.W_KEYWORD, "ws": config.W_SEMANTIC, "wp": config.W_POPULARITY,
        "min_kw": config.MIN_KEYWORD, "min_sem": config.MIN_SEMANTIC, "limit": limit,
    }).fetchall()
    if log_query:
        conn.execute(
            "INSERT INTO query_log (query, query_norm, results, top_score) VALUES (%s, %s, %s, %s)",
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
    if updated and qn:
        indexing.refresh(conn, clip_id, embed=False)
    return bool(updated)
