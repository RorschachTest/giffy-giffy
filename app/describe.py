"""What happens in a clip and when you would send it, written by a vision model.

    python -m app.describe                      every clip not yet described with this prompt
    python -m app.describe --limit 20           the first 20 of those
    python -m app.describe --ids 11 102         just these clips (described again if --redo)
    python -m app.describe --show 11            print what is stored for a clip
    python -m app.describe --backfill-requests  rebuild the request of rows saved without one

The model sees the whole clip as 8 frames in time order (one image, a 4x2 strip:
start top-left, end bottom-right) so it can follow setup -> punchline, plus every piece
of text we have (clip_state.build: title, caption, on-screen text, speech, people). It
returns what happens, what the poster meant, and 3-6 "send it when" situations written
the way people search. People search for situations ("salary khatam before month end"),
not labels, so these lines are what search will match (app/uses.py).

Nothing paid for is thrown away:
  clip_descriptions          the current answer per clip and PROMPT_VERSION, with the exact
                             request (image -> its stored strip), the whole reply and the cost
  clip_descriptions_history  an answer replaced by --redo, moved here first
  clip_description_failures  a reply that could not be read (empty, cut off, not JSON)
Strips are stored under a name made from their own bytes, so a later prompt or a redo
never overwrites the image an older row points to.
Measured on 10 clips (2026-10-20): Gemini 2.5 Flash with thinking off reads sarcasm
and the poster's intent, ~$0.0008 a clip; Flash-Lite took clips literally and made
situations up. The gateway's free credit is rate limited: 429s are waited out.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import clip_state, config, db, media, storage

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


class UnreadableReply(Exception):
    """The gateway answered (and charged) but the content is not the JSON we asked for."""

    def __init__(self, reply: dict, reason: str):
        super().__init__(reason)
        self.reply = reply


def make_strip(video: Path, duration: float | None, tmp: Path) -> bytes:
    """8 frames spread over the whole clip (media.extract_keyframes, the same sampling OCR
    and faces use), each FRAME_WIDTH wide, tiled 4x2 in time order, as one JPEG."""
    from PIL import Image

    frames = media.extract_keyframes(video, tmp / "frames", duration or float(FRAMES), FRAMES)
    images = [Image.open(f).convert("RGB") for f in frames]
    w = FRAME_WIDTH
    h = max(2, round(images[0].height * w / images[0].width))
    sheet = Image.new("RGB", (w * COLS, h * ROWS))
    for i, im in enumerate(images[:COLS * ROWS]):
        sheet.paste(im.resize((w, h)), ((i % COLS) * w, (i // COLS) * h))
    out = io.BytesIO()
    sheet.save(out, "JPEG", quality=85)
    return out.getvalue()


def request_body(strip_jpeg: bytes, state: dict, model: str) -> dict:
    """Exactly what is sent: the prompt, the clip's text and the strip as an inline image."""
    return {
        "model": model, "response_format": {"type": "json_object"}, "max_tokens": 1500,
        "reasoning_effort": "none",   # Flash "thinks" by default: 3x the cost, no better here
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT + "\n\nWhat we know:\n" + json.dumps(state, ensure_ascii=False)},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(strip_jpeg).decode()}},
        ]}],
    }


def as_stored(body: dict, strip_key: str) -> dict:
    """The request as kept in the table: identical, except the inline image (~60 KB of
    base64) is replaced by the storage key of the same JPEG."""
    out = json.loads(json.dumps(body))
    for part in out["messages"][0]["content"]:
        if part.get("type") == "image_url":
            part["image_url"] = {"stored_as": strip_key, "mime": "image/jpeg"}
    return out


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError))


def ask(body: dict) -> dict:
    """One gateway call; returns the gateway's whole reply. Waits out rate limits, server
    errors and timeouts (RETRY_WAITS_S), then gives up."""
    req = urllib.request.Request(config.GATEWAY_CHAT_URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {config.GATEWAY_API_KEY}",
                                          "content-type": "application/json"})
    for wait in (*RETRY_WAITS_S, None):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            if wait is None or not _retryable(exc):
                raise
            time.sleep(wait)
    raise AssertionError("unreachable")


def read_answer(reply: dict) -> dict:
    """The JSON object in the reply's message, or UnreadableReply (keeping the reply)."""
    choice = (reply.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")
    if not content or not str(content).strip():
        raise UnreadableReply(reply, f"empty content (finish_reason={choice.get('finish_reason')})")
    try:
        answer = json.loads(content)
    except json.JSONDecodeError as exc:
        raise UnreadableReply(reply, f"not JSON ({exc}; finish_reason={choice.get('finish_reason')})") from None
    if not isinstance(answer, dict):
        raise UnreadableReply(reply, "JSON is not an object")
    return answer


def clean(answer: dict) -> dict:
    """The reply's fields, trimmed and typed; anything missing becomes empty."""
    send = [str(s).strip() for s in answer.get("send_when") or [] if s is not None and str(s).strip()][:8]
    conf = str(answer.get("confidence") or "").strip().lower()
    return {"happens": str(answer.get("happens") or "").strip(),
            "poster_meant": str(answer.get("poster_meant") or "").strip(),
            "send_when": send,
            "confidence": conf if conf in ("high", "medium", "low") else None}


def store_strip(clip_id: int, data: bytes) -> str:
    """Save the strip under a name from its own bytes: the same image is stored once, and a
    different one (new prompt, redo) never replaces the file an older row points to."""
    jpg = config.MEDIA_DIR / f"strip-{clip_id}-{hashlib.sha256(data).hexdigest()[:12]}.jpg"
    jpg.write_bytes(data)
    return storage.save(jpg)


COLUMNS = ("clip_id, prompt_version, model, happens, poster_meant, send_when, confidence, answer, "
           "state, strip, usage, cost, request, reply, at")


def describe(conn, clip: dict, model: str) -> float:
    """Ask, store, refresh the clip's situation lines. Returns what it cost (USD)."""
    from . import uses

    state = clip_state.build(clip)
    with tempfile.TemporaryDirectory() as tmp:
        with storage.local_copy(clip["file"]) as video:
            data = make_strip(video, clip.get("duration"), Path(tmp))
    strip_key = store_strip(clip["id"], data)
    body = request_body(data, state, model)
    reply = ask(body)
    usage = reply.get("usage") or {}
    request = json.dumps(as_stored(body, strip_key), ensure_ascii=False)
    try:
        answer = read_answer(reply)
    except UnreadableReply as exc:
        conn.execute(
            """INSERT INTO clip_description_failures (clip_id, prompt_version, model, error, request, reply, cost)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (clip["id"], PROMPT_VERSION, model, str(exc), request, json.dumps(reply, ensure_ascii=False),
             usage.get("cost")))
        conn.commit()          # the failure is kept even though the caller counts this clip as failed
        raise
    # a redo keeps what it replaces
    conn.execute(f"""INSERT INTO clip_descriptions_history ({COLUMNS})
                     SELECT {COLUMNS} FROM clip_descriptions WHERE clip_id = %s AND prompt_version = %s""",
                 (clip["id"], PROMPT_VERSION))
    row = clean(answer)
    conn.execute(
        """INSERT INTO clip_descriptions (clip_id, prompt_version, model, happens, poster_meant, send_when,
                                          confidence, answer, state, strip, usage, cost, request, reply)
           VALUES (%(id)s, %(v)s, %(m)s, %(happens)s, %(poster_meant)s, %(send_when)s, %(confidence)s,
                   %(answer)s, %(state)s, %(strip)s, %(usage)s, %(cost)s, %(request)s, %(reply)s)
           ON CONFLICT (clip_id, prompt_version) DO UPDATE SET
             model = EXCLUDED.model, happens = EXCLUDED.happens, poster_meant = EXCLUDED.poster_meant,
             send_when = EXCLUDED.send_when, confidence = EXCLUDED.confidence, answer = EXCLUDED.answer,
             state = EXCLUDED.state, strip = EXCLUDED.strip, usage = EXCLUDED.usage, cost = EXCLUDED.cost,
             request = EXCLUDED.request, reply = EXCLUDED.reply, at = now()""",
        {**row, "id": clip["id"], "v": PROMPT_VERSION, "m": model, "answer": json.dumps(answer, ensure_ascii=False),
         "state": json.dumps(state, ensure_ascii=False), "strip": strip_key,
         "usage": json.dumps(usage), "cost": usage.get("cost"),
         "request": request, "reply": json.dumps(reply, ensure_ascii=False)})
    uses.build(conn, [clip["id"]])     # search sees the new lines at once
    return float(usage.get("cost") or 0.0)


def backfill_requests(conn) -> int:
    """Rows described before the request was kept: rebuild it from what was stored (the
    prompt of their version, their state, their strip). No model call. Marked "rebuilt"
    because Postgres stores JSON keys in its own order, so the clip text may be listed in
    a different order than it was sent; the words are the same. Only for the current
    PROMPT_VERSION: an older prompt's text is not in this file any more."""
    rows = conn.execute(
        "SELECT clip_id, model, state, strip FROM clip_descriptions WHERE request IS NULL AND prompt_version = %s",
        (PROMPT_VERSION,)).fetchall()
    for r in rows:
        req = as_stored(request_body(b"", r["state"], r["model"]), r["strip"])
        req["rebuilt"] = True
        conn.execute("UPDATE clip_descriptions SET request = %s WHERE clip_id = %s AND prompt_version = %s",
                     (json.dumps(req, ensure_ascii=False), r["clip_id"], PROMPT_VERSION))
    return len(rows)


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
    ap.add_argument("--backfill-requests", action="store_true", help="rebuild the request of rows saved without one")
    args = ap.parse_args(argv)
    with db.session() as conn:
        if args.backfill_requests:
            print(json.dumps({"requests_rebuilt": backfill_requests(conn)}))
            return 0
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
                spent += describe(conn, clip, args.model)
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
