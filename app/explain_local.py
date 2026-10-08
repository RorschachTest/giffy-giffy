"""Explain backend: an open multimodal model running on this machine through Ollama.

    EXPLAIN_BACKEND=local
    EXPLAIN_LOCAL_URL=http://host.docker.internal:11434     (Ollama on the Mac, not in Docker)
    EXPLAIN_LOCAL_MODEL=qwen2.5vl:7b

Why Ollama on the host: Docker on macOS cannot use the Mac's GPU, Ollama on the
host does (Metal). Setup once:

    brew install ollama
    ollama serve            (or let the app start it)
    ollama pull qwen2.5vl:7b

Free and private (nothing leaves the machine); slower, and weaker than Claude on
Hindi humour. The comparison (python -m app.compare_explain) shows by how much.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

from . import config, explain

log = logging.getLogger("explain.local")

MAX_SIDE = 640
JPEG_QUALITY = 70


def _jpeg_b64(path: Path) -> str:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _call(path: str, body: dict | None, timeout: float) -> dict:
    req = urllib.request.Request(
        config.EXPLAIN_LOCAL_URL.rstrip("/") + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def generate(system: str, user: str, frames: list[Path]) -> dict:
    reply = _call("/api/chat", {
        "model": config.EXPLAIN_LOCAL_MODEL, "stream": False, "format": "json",
        "options": {"temperature": 0.3, "num_predict": 500},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user, "images": [_jpeg_b64(f) for f in frames]},
        ],
    }, timeout=config.EXPLAIN_TIMEOUT)
    return {"text": (reply.get("message") or {}).get("content", ""),
            "usage": {"input_tokens": reply.get("prompt_eval_count"), "output_tokens": reply.get("eval_count"),
                      "model": config.EXPLAIN_LOCAL_MODEL}}


def _ready() -> tuple[bool, str]:
    try:
        tags = _call("/api/tags", None, timeout=3)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return False, (f"Ollama does not answer at {config.EXPLAIN_LOCAL_URL} ({exc}). "
                       "Install and start it on the Mac: brew install ollama && ollama serve")
    names = {m.get("name", "") for m in tags.get("models", [])}
    want = config.EXPLAIN_LOCAL_MODEL
    if want not in names and f"{want}:latest" not in names:
        return False, f"Ollama has no model {want!r}; run: ollama pull {want}"
    return True, ""


generate.ready = _ready
explain.register("local", generate)
