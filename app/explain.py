"""Why is this clip funny, and when would you send it?

Fills three fields that no vision or speech model can fill by itself:

  gist       what is happening, one sentence
  why_funny  the joke mechanism: exaggeration, irony, broken English, mismatch ...
  send_when  situations to send it in ("when someone promises big and delivers nothing")

Nothing off the shelf explains humour without a generative model, so this module
only holds what every route shares: ONE prompt, ONE parser, ONE place that decides
what may be overwritten. The routes differ in who writes the text:

  local    an open model on this machine (app/explain_local.py)
  claude   a cheap Claude model over the API (app/explain_claude.py)
  manual   a person, through the Edit form: no model at all

A backend is a module that calls `register(name, generate)` where

    generate(system: str, user: str, frames: list[Path]) -> {"text": str, "usage": {...}}

so every route sees the identical prompt and returns text that is parsed the same
way, which is what makes them comparable (python -m app.compare_explain).

Human-written values always win: a clip whose explain_source is "manual" or
"sidecar" is never overwritten, and a field that already has text is kept.
"""
from __future__ import annotations

import importlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Callable

from . import config

log = logging.getLogger("explain")

BACKEND_MODULES = {"local": "app.explain_local", "claude": "app.explain_claude"}
HUMAN_SOURCES = {"manual", "sidecar"}
_REGISTRY: dict[str, Callable] = {}

SYSTEM = """You write short search notes for a library of Indian meme clips (Hindi, Hinglish and English).
You get a few frames from one clip, its title and a speech-to-text transcript.

Rules:
- The transcript comes from speech recognition and is often WRONG, especially for Hindi and for Hindi mixed with English. If it conflicts with the title or what you can see, trust the title and the frames, and do not quote garbled words.
- Say only what you can see, hear or know from the title. Do not guess names of people from their faces. Use names given in the title.
- Explain the joke the way a friend would: what is happening, then WHY it is funny or why people share it. Name the mechanism when there is one: exaggeration, empty bluster, irony or sarcasm, deadpan, a gap between what is said and what happens, broken or mixed-up English, an over-the-top reaction, a relatable situation, a reference to a film.
- If you cannot tell why it is funny, say what it is and leave why_funny short and honest. Never invent a story.
- Write in simple English. Short sentences.

Reply with ONLY a JSON object:
{"gist": "one sentence, at most 25 words: what is happening",
 "why_funny": "one or two sentences, at most 45 words: why it is funny or why people send it",
 "send_when": ["when ...", "when ...", "when ..."]}
send_when has 2 to 4 short situations, each starting with the word "when"."""

LIMITS = {"gist": 220, "why_funny": 420, "send_when": 120}


def user_prompt(title: str, transcript_native: str, transcript_roman: str, language: str | None,
                known: str = "") -> str:
    lines = [f"Title: {title or '(none)'}"]
    if language:
        lines.append(f"Language hint: {language}")
    if transcript_native:
        lines.append(f"Transcript (may be wrong): {transcript_native[:600]}")
        if transcript_roman and transcript_roman != transcript_native:
            lines.append(f"Same in Roman letters: {transcript_roman[:600]}")
    else:
        lines.append("Transcript: (none: no speech was found)")
    if known:
        lines.append(f"Already known about this clip: {known}")
    lines.append("The frames are in time order.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Parsing and merging (pure)
# --------------------------------------------------------------------------- #

def _text(value, limit: int) -> str:
    return re.sub(r"\s+", " ", value).strip()[:limit] if isinstance(value, str) else ""


def parse(reply: str) -> dict:
    """Pull the JSON object out of a model reply; empty fields on any problem."""
    m = re.search(r"\{.*\}", reply or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    when = data.get("send_when")
    when = [when] if isinstance(when, str) else when
    send_when = [_text(w, LIMITS["send_when"]) for w in (when if isinstance(when, list) else [])]
    return {
        "gist": _text(data.get("gist"), LIMITS["gist"]),
        "why_funny": _text(data.get("why_funny"), LIMITS["why_funny"]),
        "send_when": [w for w in send_when if len(w.split()) >= 2][:4],   # a bare "when" is not a situation
    }


def is_empty(found: dict) -> bool:
    return not (found["gist"] or found["why_funny"] or found["send_when"])


def merge(current: dict, found: dict, source: str, replace: bool = False) -> tuple[dict, str]:
    """What to store. A clip written by a person ("manual" / "sidecar") is never
    touched; otherwise a field that already has text is kept unless replace is set.
    Returns (fields, explain_source)."""
    current_source = current.get("explain_source") or ""
    if current_source in HUMAN_SOURCES:
        return {k: current.get(k) or ([] if k == "send_when" else "") for k in LIMITS}, current_source
    out = {k: (found[k] if replace or not current.get(k) else current[k]) for k in LIMITS}
    return out, (source if any(out.values()) else current_source)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

def register(name: str, generate: Callable) -> None:
    _REGISTRY[name] = generate


def backend(name: str) -> Callable:
    """The generate() of a backend; imports its module on first use."""
    if name not in _REGISTRY and name in BACKEND_MODULES:
        try:
            importlib.import_module(BACKEND_MODULES[name])
        except ModuleNotFoundError as exc:
            if exc.name != BACKEND_MODULES[name]:
                raise                        # a real missing dependency inside the module
    if name not in _REGISTRY:
        raise LookupError(f"explain backend {name!r} is not available on this branch")
    return _REGISTRY[name]


def available_backends() -> list[str]:
    names = []
    for name in BACKEND_MODULES:
        try:
            backend(name)
            names.append(name)
        except Exception:
            pass
    return [*names, *(n for n in _REGISTRY if n not in names)]


def available() -> bool:
    """True when ingest should call a model. 'manual' and 'off' never do."""
    name = config.EXPLAIN_BACKEND
    if name in ("", "off", "manual"):
        return False
    try:
        backend(name)
        return True
    except Exception as exc:
        log.warning("explain backend %r unavailable: %s", name, exc)
        return False


def run(name: str, frames: list[Path], title: str, transcript_native: str, transcript_roman: str,
        language: str | None, known: str = "") -> dict:
    """One call to one backend. Returns the parsed fields plus how it went:
    {"gist", "why_funny", "send_when", "seconds", "usage", "raw_text", "backend"}."""
    generate = backend(name)
    t0 = time.time()
    reply = generate(SYSTEM, user_prompt(title, transcript_native, transcript_roman, language, known), frames)
    out = parse(reply.get("text", ""))
    out.update(seconds=round(time.time() - t0, 2), usage=reply.get("usage") or {},
               raw_text=reply.get("text", ""), backend=name)
    return out


def explain(frames: list[Path], title: str, transcript_native: str, transcript_roman: str,
            language: str | None, known: str = "") -> dict:
    """The configured backend (EXPLAIN_BACKEND) on one clip."""
    return run(config.EXPLAIN_BACKEND, frames, title, transcript_native, transcript_roman, language, known)
