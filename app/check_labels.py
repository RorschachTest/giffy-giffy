"""Is the label list any good? A rough first check, no Laya or Jev needed.

    python -m app.check_labels

1. Shape: unique labels per facet, descriptions at most 6 words, no empty group.
2. Variety: every clip and every test query is matched to its nearest labels by
   embedding similarity, then
      labels nearest to > 30% of texts   -> too broad, split them
      labels nearest to nothing          -> drop or merge
      texts with no label close enough   -> a label may be missing
   The embedding model is English-first and Hindi transcripts are noisy, so this is
   only a smell test. The real check is the pilot on hand-labelled clips.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

from . import config, db, taxonomy
from .embed import embed_one

MAX_DESC_WORDS = 6
TOO_BROAD = 0.30
MIN_SIM = 0.30   # guess: below this the nearest label is not really near


def shape_problems() -> list[str]:
    problems = []
    for facet, groups in taxonomy.FACETS.items():
        seen: dict[str, str] = {}
        for group, labels in groups.items():
            if not labels:
                problems.append(f"{facet}: group {group!r} is empty")
            for label, desc in labels.items():
                if label in seen:
                    problems.append(f"{facet}: {label!r} is in both {seen[label]!r} and {group!r}")
                seen[label] = group
                if label == taxonomy.NONE:
                    problems.append(f"{facet}: {label!r} is reserved")
                if len(desc.split()) > MAX_DESC_WORDS:
                    problems.append(f"{facet}.{label}: description has {len(desc.split())} words (max {MAX_DESC_WORDS})")
    return problems


def nearest(text_vecs: dict[str, list[float]], label_vecs: dict[str, list[float]]) -> dict[str, tuple[str, float]]:
    """For each text, its closest label and the similarity (vectors are unit length)."""
    out = {}
    for text, v in text_vecs.items():
        best = max(label_vecs, key=lambda lab: sum(a * b for a, b in zip(v, label_vecs[lab])))
        out[text] = (best, sum(a * b for a, b in zip(v, label_vecs[best])))
    return out


def variety(facet: str, matches: dict[str, tuple[str, float]]) -> dict:
    labels = taxonomy.labels(facet)
    used = Counter(lab for lab, sim in matches.values() if sim >= MIN_SIM)
    n = len(matches) or 1
    return {
        "broad": {lab: round(c / n, 2) for lab, c in used.items() if c / n > TOO_BROAD},
        "unused": [lab for lab in labels if lab not in used],
        "unmatched": [t for t, (_, sim) in matches.items() if sim < MIN_SIM],
        "used": dict(used.most_common()),
    }


def _texts() -> list[str]:
    texts = []
    with db.session() as conn:
        for r in conn.execute("SELECT title, transcript_roman, gist, why_funny FROM clips ORDER BY id").fetchall():
            parts = [r["title"], r.get("gist"), r.get("why_funny"), (r["transcript_roman"] or "")[:200]]
            texts.append(" | ".join(p for p in parts if p))
    queries = config.DATA_DIR / "search_queries.json"
    if queries.exists():
        texts += [q["q"] for q in json.loads(queries.read_text(encoding="utf-8")) if (q.get("q") or "").strip()]
    return [t for t in dict.fromkeys(texts) if t.strip()]


def main() -> int:
    problems = shape_problems()
    for p in problems:
        print("SHAPE", p)
    sizes = ", ".join(f"{f} {len(taxonomy.labels(f))}" for f in taxonomy.FACETS)
    print(f"labels: {sizes}" + ("" if problems else "   shape ok"))

    texts = _texts()
    if not texts:
        print("no clips or queries to check variety against")
        return 1 if problems else 0
    text_vecs = {t: embed_one(t) for t in texts}
    print(f"texts: {len(texts)} (clips + test queries)\n")
    for facet in taxonomy.FACETS:
        label_vecs = {lab: embed_one(f"{lab.replace('_', ' ')}: {desc}") for lab, desc in taxonomy.labels(facet).items()}
        v = variety(facet, nearest(text_vecs, label_vecs))
        print(f"[{facet}] used {len(v['used'])}/{len(label_vecs)} labels")
        if v["broad"]:
            print(f"  too broad (> {TOO_BROAD:.0%} of texts): {v['broad']}")
        print(f"  never nearest: {', '.join(v['unused']) or '-'}")
        for t in v["unmatched"][:5]:
            print(f"  no close label: {t[:90]!r}")
        print()
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
