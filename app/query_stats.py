"""What people search for, in aggregate (Curate > Searches). Never who.

query_log keeps only the search text, the hour it was made, how many clips came back and
the best score: no account, session, IP or cookie. This module reads it in aggregate:
the most common searches, how many find nothing, which language they are typed in, and a
count per day.

Language is a cheap guess from the words themselves:
    hindi      written in Devanagari
    hinglish   Latin letters with common Hindi words (hai, nahi, kya, bhai, yaar, ...)
    english    Latin letters with common English words and no Hindi ones
    unclear    too short to tell ("salary", "puneet")
"""
from __future__ import annotations

import re
from collections import Counter

HINDI_WORDS = frozenset("""
hai hain nahi nhi nahin naheen kya kyu kyun kyon ka ki ke ko se mein mai tu tum aap ap yaar yar
bhai tha thi hoon hu raha rahi rahe kar karo karna ho ye yeh wo woh aur bhi toh sab kuch kaise
kaun kon abhi jab phir fir mera meri mere tera teri tere apna apni log bas accha acha achha chal
chalo ji haan matlab bahut bohot bhot hum mujhe tujhe diya liya gaya gayi wala wali wale dekh
dekho bol bolo kaha kahan yahan wahan gareeb garib paisa paise dost dosti naukri pyar pyaar
""".split())
ENGLISH_WORDS = frozenset("""
the is are was were you your my when what this that and of it in on with for be have has not
does did i we they he she him her them who why how where can will just like so but if or all
about from at by an
""".split())
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def language(query: str) -> str:
    if _DEVANAGARI.search(query or ""):
        return "hindi"
    words = re.findall(r"[a-z]+", (query or "").lower())
    hindi = sum(w in HINDI_WORDS for w in words)
    english = sum(w in ENGLISH_WORDS for w in words)
    if hindi:
        return "hinglish"
    if english:
        return "english"
    return "unclear"


def summary(conn, days: int = 30, limit: int = 100) -> dict:
    window = {"days": max(1, min(days, 365))}
    rows = conn.execute(
        """SELECT query_norm, count(*) AS times, max(at) AS last_seen,
                  round(avg(results))::int AS avg_results,
                  sum((results = 0)::int) AS found_nothing
             FROM query_log WHERE at > now() - make_interval(days => %(days)s) AND query_norm <> ''
            GROUP BY query_norm ORDER BY times DESC, last_seen DESC""", window).fetchall()
    per_day = conn.execute(
        """SELECT date_trunc('day', at)::date AS day, count(*) AS searches
             FROM query_log WHERE at > now() - make_interval(days => %(days)s) AND query_norm <> ''
            GROUP BY 1 ORDER BY 1""", window).fetchall()
    by_language: Counter[str] = Counter()
    for r in rows:
        r["language"] = language(r["query_norm"])
        by_language[r["language"]] += r["times"]
    total = sum(r["times"] for r in rows)
    nothing = sum(r["found_nothing"] for r in rows)
    return {
        "days": window["days"], "searches": total, "distinct": len(rows),
        "found_nothing_share": round(nothing / total, 3) if total else 0.0,
        "by_language": dict(by_language.most_common()),
        "per_day": per_day,
        "top": rows[:max(1, min(limit, 500))],
    }
