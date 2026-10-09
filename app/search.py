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
shown if       kw >= MIN_KEYWORD or sem >= MIN_SEMANTIC or tm >= MIN_TOPIC

The weights are guesses until they are tuned against real queries. /search
returns kw and sem for every hit so you can see why something ranked.
"""
from __future__ import annotations

from . import config, laya
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
           LEAST(1, COALESCE((SELECT sum(x.w) FROM unnest(q.qt, q.qw) AS x(t, w)
                               WHERE x.t = ANY(c.topics) OR x.t = ANY(c.reactions)), 0)) AS tm
      FROM clips c JOIN candidates USING (id), q
)
SELECT {COLUMNS}, kw, sem, tm,
       %(wk)s * kw + %(ws)s * sem + %(wt)s * tm * (1 - LEAST(kw, 1))
         + %(wp)s * LEAST(ln(shares + duplicates_seen), 3) AS score
  FROM scored
 WHERE kw >= %(min_kw)s OR sem >= %(min_sem)s OR tm >= %(min_tm)s
 ORDER BY score DESC, id
 LIMIT %(limit)s
"""

TRENDING_SQL = f"""
SELECT {COLUMNS}, NULL::real AS kw, NULL::real AS sem, NULL::real AS tm, NULL::real AS score
  FROM clips
 ORDER BY shares + duplicates_seen DESC, first_seen DESC
 LIMIT %(limit)s
"""


def search(conn, query: str, limit: int = 24, log_query: bool = True, intent: dict | None = None) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return conn.execute(TRENDING_SQL, {"limit": limit}).fetchall()

    qn = normalise(query)
    if not qn:  # emoji-only or punctuation-only query
        return []
    if intent is None:                  # callers that need it too pass it in: one Laya call per search
        intent = laya.query_intent(query)   # {} when Laya is off, slow or unsure
    rows = conn.execute(SEARCH_SQL, {
        "qn": qn,
        "qv": vec(embed_one(query)),          # the model gets the query as typed
        "tokens": [t for t in qn.split() if len(t) >= 3] or qn.split(),
        "qt": list(intent), "qw": list(intent.values()),
        "wk": config.W_KEYWORD, "ws": config.W_SEMANTIC, "wt": config.W_TOPIC,
        "wp": config.W_POPULARITY,
        "min_kw": config.MIN_KEYWORD, "min_sem": config.MIN_SEMANTIC,
        "min_tm": config.MIN_TOPIC, "limit": limit,
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
