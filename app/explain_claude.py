"""Explain backend: a cheap Claude model over the Anthropic API.

    EXPLAIN_BACKEND=claude
    ANTHROPIC_API_KEY=...                      in the gitignored .env
    EXPLAIN_CLAUDE_MODEL=claude-haiku-4-5-20251001   (default: the cheapest Claude)

It runs once per clip at ingest and in `reindex --explain`, never at search.
Privacy: the clip's frames (downscaled) and its transcript are sent to Anthropic.
Without a key the step is skipped with one warning.
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

from . import config, explain

log = logging.getLogger("explain.claude")

MAX_SIDE = 768        # px: enough to see a face and a scene, small enough to be cheap
JPEG_QUALITY = 72


def _image_block(path: Path) -> dict:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(buf.getvalue()).decode()}}


def generate(system: str, user: str, frames: list[Path]) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, timeout=config.EXPLAIN_TIMEOUT, max_retries=2)
    content = [_image_block(f) for f in frames] + [{"type": "text", "text": user}]
    reply = client.messages.create(
        model=config.EXPLAIN_CLAUDE_MODEL, max_tokens=500, system=system,
        messages=[{"role": "user", "content": content}],
    )
    return {"text": "".join(b.text for b in reply.content if b.type == "text"),
            "usage": {"input_tokens": reply.usage.input_tokens, "output_tokens": reply.usage.output_tokens,
                      "model": config.EXPLAIN_CLAUDE_MODEL}}


def _ready() -> tuple[bool, str]:
    if not config.ANTHROPIC_API_KEY:
        return False, "ANTHROPIC_API_KEY is not set (put it in .env)"
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, "the anthropic package is not installed (rebuild the image)"
    return True, ""


generate.ready = _ready
explain.register("claude", generate)
