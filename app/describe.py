"""What happens in a clip and when you would send it, written by a vision model.

    python -m app.describe                      every clip not yet described with this prompt
    python -m app.describe --limit 20           the first 20 of those
    python -m app.describe --ids 11 102         just these clips (described again if --redo)
    python -m app.describe --show 11            print what is stored for a clip

The model sees the whole clip as 8 frames in time order (one image, a 4x2 strip:
start top-left, end bottom-right) so it can follow setup -> punchline, plus every piece
of text we have (clip_state.build: title, caption, on-screen text, speech, people). It
returns what happens, what the poster meant, and 3-6 "send it when" situations written
the way people search. People search for situations ("salary khatam before month end"),
not labels, so these lines are what search will match.

Everything is kept in clip_descriptions, one row per clip and PROMPT_VERSION, with the
full reply, the text it was given, the strip it saw and the cost: nothing is asked
twice, and a new prompt version writes new rows beside the old ones to compare.
Measured on 10 clips (2026-10-20): Gemini 2.5 Flash with thinking off reads sarcasm
and the poster's intent, ~$0.0008 a clip; Flash-Lite took clips literally and made
situations up. The gateway's free credit is rate limited: 429s are waited out.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import clip_state, config, db, storage

PROMPT_VERSION = "v1"
FRAMES, COLS, ROWS, FRAME_WIDTH = 8, 4, 2, 240
PAUSE_S = 1.0                      # between clips: stays under the gateway's rate limit
RETRY_WAITS_S = (5, 10, 20, 40, 80)

PROMPT = """You are describing a short Indian meme video clip (Hindi, Hinglish or English) so people can find it by searching for the situation they are in.
You get 8 frames in time order (left to right, top to bottom) and the text we have about the clip. The poster's title and on-screen caption usually say what the clip is meant for: weigh them most.
Return JSON:
{"happens": "1-2 plain sentences: what happens in the clip, start to end",
 "poster_meant": "one short line: what the poster is using it to say, or empty if unclear",
 "send_when": ["3 to 6 short situations when someone would send this clip, written the way an Indian person would type a search: casual English or Hinglish, e.g. 'salary khatam before month end', 'when a friend gives gyaan he never follows'"],
 "confidence": "high | medium | low (low if the clip is unclear)"}
Do not invent facts that are not in the frames or text. Name a person only if the name is in the text we give you; otherwise describe them ("a man in a suit")."""


def make_strip(video: Path, duration: float | None, out: Path) -> Path:
    """8 frames spread over the whole clip, tiled 4x2 in time order, as one JPEG."""
    fps = FRAMES / max(duration or FRAMES, 1.0)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf",
                    f"fps={fps:.4f},scale={FRAME_WIDTH}:-2,tile={COLS}x{ROWS}", "-frames:v", "1", str(out)],
                   check=True)
    return out


def ask(strip_jpeg: bytes, state: dict, model: str) -> tuple[dict, dict]:
    """One gateway call. Returns (parsed answer, usage). Waits out rate limits."""
    body = {
        "model": model, "response_format": {"type": "json_object"}, "max_tokens": 1500,
        "reasoning_effort": "none",   # Flash "thinks" by default: 3x the cost, no better here
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT + "\n\nWhat we know:\n" + json.dumps(state, ensure_ascii=False)},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(strip_jpeg).decode()}},
        ]}],
    }
    req = urllib.request.Request(config.GATEWAY_CHAT_URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {config.GATEWAY_API_KEY}",
                                          "content-type": "application/json"})
    for wait in (*RETRY_WAITS_S, None):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                reply = json.loads(resp.read())
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or wait is None:
                raise
            time.sleep(wait)
    return json.loads(reply["choices"][0]["message"]["content"]), reply.get("usage") or {}


def clean(answer: dict) -> dict:
    """The reply's fields, trimmed and typed; anything missing becomes empty."""
    send = [str(s).strip() for s in answer.get("send_when") or [] if str(s).strip()][:8]
    conf = str(answer.get("confidence") or "").strip().lower()
    return {"happens": str(answer.get("happens") or "").strip(),
            "poster_meant": str(answer.get("poster_meant") or "").strip(),
            "send_when": send,
            "confidence": conf if conf in ("high", "medium", "low") else None}


def describe(conn, clip: dict, model: str) -> dict:
    state = clip_state.build(clip)
    with tempfile.TemporaryDirectory() as tmp:
        jpg = config.MEDIA_DIR / f"strip-{clip['id']}.jpg"
        with storage.local_copy(clip["file"]) as video:
            make_strip(video, clip.get("duration"), Path(tmp) / "strip.jpg")
        data = (Path(tmp) / "strip.jpg").read_bytes()
        jpg.write_bytes(data)
        strip_key = storage.save(jpg)          # kept: a later model or the eval page can reuse it
    answer, usage = ask(data, state, model)
    row = clean(answer)
    conn.execute(
        """INSERT INTO clip_descriptions (clip_id, prompt_version, model, happens, poster_meant, send_when,
                                          confidence, answer, state, strip, usage, cost)
           VALUES (%(id)s, %(v)s, %(m)s, %(happens)s, %(poster_meant)s, %(send_when)s, %(confidence)s,
                   %(answer)s, %(state)s, %(strip)s, %(usage)s, %(cost)s)
           ON CONFLICT (clip_id, prompt_version) DO UPDATE SET
             model = EXCLUDED.model, happens = EXCLUDED.happens, poster_meant = EXCLUDED.poster_meant,
             send_when = EXCLUDED.send_when, confidence = EXCLUDED.confidence, answer = EXCLUDED.answer,
             state = EXCLUDED.state, strip = EXCLUDED.strip, usage = EXCLUDED.usage, cost = EXCLUDED.cost, at = now()""",
        {**row, "id": clip["id"], "v": PROMPT_VERSION, "m": model, "answer": json.dumps(answer, ensure_ascii=False),
         "state": json.dumps(state, ensure_ascii=False), "strip": strip_key,
         "usage": json.dumps(usage), "cost": usage.get("cost")})
    return row


def todo(conn, ids: list[int] | None, redo: bool, limit: int | None) -> list[dict]:
    where = ["c.duplicate_of IS NULL"]
    if ids:
        where.append("c.id = ANY(%(ids)s)")
    if not redo:
        where.append("NOT EXISTS (SELECT 1 FROM clip_descriptions d WHERE d.clip_id = c.id AND d.prompt_version = %(v)s)")
    sql = f"SELECT c.* FROM clips c WHERE {' AND '.join(where)} ORDER BY c.id" + (" LIMIT %(n)s" if limit else "")
    return conn.execute(sql, {"ids": ids or [], "v": PROMPT_VERSION, "n": limit}).fetchall()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.describe")
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--model", default=config.DESCRIBE_MODEL)
    ap.add_argument("--show", type=int)
    args = ap.parse_args(argv)
    with db.session() as conn:
        if args.show:
            for r in conn.execute("SELECT * FROM clip_descriptions WHERE clip_id = %s ORDER BY at", (args.show,)).fetchall():
                print(json.dumps({k: r[k] for k in ("prompt_version", "model", "happens", "poster_meant",
                                                    "send_when", "confidence", "cost")}, ensure_ascii=False, indent=1))
            return 0
        if not config.GATEWAY_API_KEY:
            raise SystemExit("TYPESAFE_API_KEY (the AI Gateway key) is not set")
        clips = todo(conn, args.ids, args.redo, args.limit)
    print(f"{len(clips)} clips to describe with {args.model}, prompt {PROMPT_VERSION}", file=sys.stderr)
    done = failed = 0
    spent = 0.0
    for clip in clips:
        try:
            with db.session() as conn:          # one commit per clip: a stop loses nothing
                describe(conn, clip, args.model)
                spent += float(conn.execute(
                    "SELECT coalesce(cost, 0) AS c FROM clip_descriptions WHERE clip_id = %s AND prompt_version = %s",
                    (clip["id"], PROMPT_VERSION)).fetchone()["c"])
            done += 1
        except Exception as exc:                 # one bad clip must not stop the night's run
            failed += 1
            print(f"  clip {clip['id']}: {exc}", file=sys.stderr)
            if failed >= 5 and not done:
                raise SystemExit("first 5 clips failed: stopping")
        if (done + failed) % 25 == 0:
            print(f"  {done} described, {failed} failed, ${spent:.3f} spent", file=sys.stderr)
        time.sleep(PAUSE_S)
    print(json.dumps({"described": done, "failed": failed, "spent_usd": round(spent, 4)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
