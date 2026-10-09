"""The ruler: how well does search find the clip a person meant?

    python -m app.eval_search                    run data/search_queries.json, print the score
    python -m app.eval_search --template         write data/search_queries.json to fill in
    python -m app.eval_search --save baseline    keep this run as data/eval/baseline.json
    python -m app.eval_search --compare baseline show the change against a saved run

data/search_queries.json is a list of
    {"q": "when someone talks big", "expect": ["d28f9c7a36"], "kind": "situation"}
expect holds share ids (the part after /c/ in a clip link) or database ids; any one
of them counts as a hit. kind groups the score: quote, name, situation, message, mood, unknown.
Queries are not written to query_log, so measuring never changes what search learns.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from . import config, db
from . import search as search_mod

QUERIES = config.DATA_DIR / "search_queries.json"
EVAL_DIR = config.DATA_DIR / "eval"
CUTOFFS = (1, 5, 10)


def rank_of(rows: list[dict], expect: list[str]) -> int | None:
    """1-based position of the first expected clip in the results, or None."""
    want = {str(e) for e in expect}
    for i, r in enumerate(rows, 1):
        if str(r["id"]) in want or r["share_id"] in want:
            return i
    return None


def score(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {"n": n, "mrr": round(sum(1 / r for r in ranks if r) / n, 3) if n else 0.0}
    for k in CUTOFFS:
        out[f"hit@{k}"] = round(sum(1 for r in ranks if r and r <= k) / n, 3) if n else 0.0
    return out


def run(queries: list[dict]) -> dict:
    ranks_by_kind: dict[str, list] = defaultdict(list)
    misses = []
    with db.session() as conn:
        for item in queries:
            rows = search_mod.search(conn, item["q"], limit=max(CUTOFFS), log_query=False)
            rank = rank_of(rows, item["expect"])
            ranks_by_kind[item.get("kind") or "other"].append(rank)
            if rank is None or rank > 5:
                misses.append({"q": item["q"], "kind": item.get("kind"), "rank": rank,
                               "top": [r["share_id"] for r in rows[:3]]})
    every = [r for ranks in ranks_by_kind.values() for r in ranks]
    return {"all": score(every), "by_kind": {k: score(v) for k, v in sorted(ranks_by_kind.items())},
            "misses": misses}


def write_template() -> None:
    with db.session() as conn:
        rows = conn.execute("""SELECT left(sha256, 10) AS share_id, title, transcript_roman
                                 FROM clips ORDER BY id""").fetchall()
    items = [{"q": "", "expect": [r["share_id"]], "kind": "", "_clip": f"{r['title']} | {(r['transcript_roman'] or '')[:80]}"}
             for r in rows]
    QUERIES.parent.mkdir(parents=True, exist_ok=True)
    QUERIES.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {QUERIES}: {len(items)} clips; fill in q (and kind) for each query, add more rows per clip, delete rows you skip")


def show(result: dict, before: dict | None) -> None:
    def line(name: str, s: dict, b: dict | None) -> str:
        cols = [f"{k} {s[k]:.3f}" + (f" ({s[k] - b[k]:+.3f})" if b else "") for k in ("hit@1", "hit@5", "hit@10", "mrr")]
        return f"  {name:<10} n={s['n']:<4} " + "   ".join(cols)
    print(line("ALL", result["all"], before["all"] if before else None))
    for kind, s in result["by_kind"].items():
        print(line(kind, s, (before or {}).get("by_kind", {}).get(kind)))
    if result["misses"]:
        print("\nnot in the top 5:")
        for m in result["misses"]:
            print(f"  [{m['kind'] or '-'}] {m['q']!r}  rank={m['rank']}  top3={m['top']}")


def _arg(name: str) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv and sys.argv.index(name) + 1 < len(sys.argv) else None


def main() -> int:
    if "--template" in sys.argv:
        write_template()
        return 0
    if not QUERIES.exists():
        print(f"no queries yet: run with --template, fill in {QUERIES}", file=sys.stderr)
        return 1
    queries = [q for q in json.loads(QUERIES.read_text(encoding="utf-8")) if (q.get("q") or "").strip()]
    if not queries:
        print("every q is empty: write some queries first", file=sys.stderr)
        return 1
    before = None
    if _arg("--compare"):
        before = json.loads((EVAL_DIR / f"{_arg('--compare')}.json").read_text(encoding="utf-8"))
    result = run(queries)
    show(result, before)
    if _arg("--save"):
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        (EVAL_DIR / f"{_arg('--save')}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsaved {EVAL_DIR / (_arg('--save') + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
