"""English and Hindi-script versions of the situation lines, for the meaning model.

The meaning model reads English and Devanagari Hindi, not romanised Hindi: measured,
"main nahi araha" scored 0.27 against "I am not coming" but 0.81 against an unrelated
Hinglish line, while "मैं नहीं आ रहा" scored 0.90. So both sides get readable versions:

  lines     every "send it when" line is stored with natural English and Hindi (Devanagari)
            versions (line_translations); uses.build adds both as extra vectors, so a clip
            still scores by its best line, in whichever language matches.

    python -m app.translate lines      translate lines that have no translation yet
"""
from __future__ import annotations

import json
import logging
import sys
import time
import urllib.request

from . import config, db

log = logging.getLogger("translate")

MODEL = "google/gemini-2.5-flash-lite"
BATCH = 80
RETRY_WAITS_S = (10, 20, 40, 80, 160)

LINES_PROMPT = (
    'Each item is a short situation in which someone would send an Indian meme clip (English, Hindi or Hinglish). '
    'For each, give "en": the same situation in natural English, and "hi": the same situation in natural Hindi '
    'written in Devanagari. Keep meaning and tone; do not add anything. Return JSON {"out": [{"en": "...", "hi": "..."}, ...]}, '
    'one per item, same order.\n\n')


def _chat(prompt: str, max_tokens: int, waits=RETRY_WAITS_S) -> tuple[dict, float]:
    """One JSON reply from the gateway and its cost; waits out rate limits and server errors."""
    body = {"model": MODEL, "max_tokens": max_tokens, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(config.GATEWAY_CHAT_URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {config.GATEWAY_API_KEY}",
                                          "content-type": "application/json"})
    for wait in (*waits, None):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                reply = json.loads(resp.read())
            return json.loads(reply["choices"][0]["message"]["content"]), float((reply.get("usage") or {}).get("cost") or 0)
        except Exception:
            if wait is None:
                raise
            time.sleep(wait)
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------- #
# Lines
# --------------------------------------------------------------------------- #

def save_lines(conn, pairs: dict[str, dict]) -> int:
    """{line: {"en": ..., "hi": ...}} -> line_translations (existing rows are kept)."""
    n = 0
    for line, t in pairs.items():
        en, hi = str(t.get("en") or "").strip(), str(t.get("hi") or "").strip()
        if en or hi:
            n += conn.execute(
                """INSERT INTO line_translations (line, en, hi, model) VALUES (%s, %s, %s, %s)
                   ON CONFLICT (line) DO NOTHING""", (line, en, hi, MODEL)).rowcount
    return n


def translate_lines(conn, lines: list[str]) -> float:
    """Translate and store the lines that have no translation yet. Returns the cost."""
    have = {r["line"] for r in conn.execute(
        "SELECT line FROM line_translations WHERE line = ANY(%s)", (lines,)).fetchall()}
    todo = [l for l in dict.fromkeys(lines) if l not in have]
    cost = 0.0
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        out, c = _chat(LINES_PROMPT + json.dumps(batch, ensure_ascii=False), 8000)
        out = out.get("out") or []
        if len(out) != len(batch):
            log.warning("line batch came back with %d of %d", len(out), len(batch))
            continue
        save_lines(conn, dict(zip(batch, out)))
        conn.commit()
        cost += c
    return cost


def line_versions(conn, lines: list[str]) -> dict[str, list[tuple[str, str]]]:
    """line -> [(text, "en"), (text, "hi")], only versions that differ from the line itself."""
    out: dict[str, list[tuple[str, str]]] = {}
    for r in conn.execute("SELECT line, en, hi FROM line_translations WHERE line = ANY(%s)", (lines,)).fetchall():
        seen = {r["line"].lower()}
        for text, lang in ((r["en"], "en"), (r["hi"], "hi")):
            if text and text.lower() not in seen:
                seen.add(text.lower())
                out.setdefault(r["line"], []).append((text, lang))
    return out


def main(argv: list[str]) -> int:
    if argv[:1] != ["lines"]:
        print(__doc__)
        return 1
    with db.session() as conn:
        lines = [r["line"] for r in conn.execute("SELECT DISTINCT line FROM clip_uses WHERE source NOT LIKE '%:%'").fetchall()]
        cost = translate_lines(conn, lines)
        n = conn.execute("SELECT count(*) AS n FROM line_translations").fetchone()["n"]
    print(json.dumps({"lines": len(lines), "translated_total": n, "cost_usd": round(cost, 4)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
