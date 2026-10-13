"""Text burned into the video (captions, punchlines, names): read off the keyframes.

Measured on 40 random clips: 70% carry readable text, often the joke itself and often
on clips with no title ("Boys are simple", "When you help bro cheat and the teacher
takes only you to the principal's office"). RapidOCR (PP-OCR models on onnxruntime,
which the app already uses) reads it at ~0.3 s a frame on CPU.

Three clean-ups turn raw readings into clips.screen_text:
  watermarks   channel tags and logos repeated across clips ("VITA", "mememandir",
               "@memeverse143") are dropped: OCR_WATERMARKS plus any @handle
  repeats      the same caption read on several frames is kept once
  spaces       the default model drops spaces between Latin words ("Boysaresimple");
               run-together words are split with a word list of English plus the
               library's own Hinglish words (titles, transcripts), so "kyahaath" -> "kya haath"
The raw readings are kept in clips.raw["ocr"], so cleaning can be redone without reading
the frames again (python -m app.reindex --screen-text-clean).
"""
from __future__ import annotations

import gzip
import logging
import re
from functools import lru_cache
from pathlib import Path

from . import config

log = logging.getLogger("ocr")

MIN_CONFIDENCE = 0.7
MIN_CHARS = 3
MAX_CHARS = 600
WORDS_FILE = "ocr_words.txt.gz"   # in DATA_DIR, built by build_word_list()


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR   # lazy: loads the models
    return RapidOCR()


def read_frames(frames: list[Path]) -> list[str]:
    """Every confident line on every keyframe, in order, as read (no cleaning)."""
    lines: list[str] = []
    for frame in frames:
        try:
            found, _ = _engine()(str(frame))
        except Exception as exc:   # one unreadable frame must not lose the others
            log.warning("ocr failed on %s: %s", frame, exc)
            continue
        for _box, text, conf in found or []:
            text = text.strip()
            if float(conf) >= MIN_CONFIDENCE and len(text) >= MIN_CHARS:
                lines.append(text)
    return lines


def _watermarks() -> set[str]:
    return {w.strip().lower() for w in config.OCR_WATERMARKS.split(",") if w.strip()}


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9ऀ-ॿ]", "", text.lower())


@lru_cache(maxsize=1)
def _splitter():
    """wordninja with English + the library's words, when that list has been built."""
    import wordninja

    path = config.DATA_DIR / WORDS_FILE
    return wordninja.LanguageModel(str(path)) if path.exists() else wordninja.DEFAULT_LANGUAGE_MODEL


def _respace(line: str) -> str:
    """Split run-together Latin words; leave a token alone when splitting only yields
    crumbs (a name or a word the list does not know)."""
    out = []
    for token in line.split():
        letters = re.sub(r"[^A-Za-z]", "", token)
        if len(letters) >= 9 and letters == token.strip(".,!?:;'\"()"):
            parts = _splitter().split(token)
            if parts and sum(len(p) <= 2 for p in parts) <= len(parts) // 3:
                out.append(" ".join(parts))
                continue
        out.append(token)
    return " ".join(out)


def clean(lines: list[str]) -> str:
    """Raw readings -> one searchable text: no watermarks or handles, no repeats, spaced."""
    marks = _watermarks()
    kept: list[str] = []
    seen: set[str] = set()
    for line in lines:
        line = re.sub(r"[@G]\w*memeverse\w*|@\w+", "", line).strip(" .:-|")
        key = _squash(line)
        if len(key) < MIN_CHARS or key in seen or any(m and _squash(m) == key for m in marks):
            continue
        if any(key in s or s in key for s in seen if len(s) >= 6):   # a longer/shorter reading of a line kept
            continue
        seen.add(key)
        kept.append(_respace(line))
    return " / ".join(kept)[:MAX_CHARS]


def build_word_list(conn) -> Path:
    """English words (wordninja's list) followed by the library's own words, most frequent
    first, so Hinglish captions split into real words. Rebuilt by the backfill."""
    import wordninja
    from collections import Counter

    counts: Counter[str] = Counter()
    for r in conn.execute("SELECT title, transcript_roman, folk_names FROM clips").fetchall():
        text = " ".join([r["title"] or "", r["transcript_roman"] or "", " ".join(r["folk_names"] or [])])
        counts.update(w for w in re.findall(r"[a-z]{2,}", text.lower()))
    english_path = Path(wordninja.__file__).parent / "wordninja" / "wordninja_words.txt.gz"
    with gzip.open(english_path, "rt", encoding="utf-8") as f:
        english = [w.strip() for w in f if w.strip()]
    library = [w for w, n in counts.most_common() if n >= 2]
    # library words first: in this app "kya", "bhai", "nahi" are more common than most English
    words = list(dict.fromkeys([*library[:5000], *english]))
    path = config.DATA_DIR / WORDS_FILE
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\n".join(words))
    _splitter.cache_clear()
    return path
