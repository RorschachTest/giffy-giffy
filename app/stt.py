"""Speech to text with faster-whisper (runs on CPU, no GPU needed).

Tuned on the 13 sample clips (python -m app.compare_stt, 2026-10-09):
- large-v3 instead of small: most Hindi lines went from garbled to right
  ("tisam men onta" -> "are kya, kya karoon main iska ...").
- A hint prompt (title, folk names, film) keeps English words in Hinglish as English
  ("kantrol" -> "Control", "manee pholoj" -> "money follows") and fixes spellings.
- Auto-detect alone is unsafe on short meme clips: it said Urdu, Korean or Russian on
  4 of 13 and then wrote Urdu script or made text up. A language outside
  STT_LANGUAGES is re-run as STT_FALLBACK_LANGUAGE.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from . import config

MAX_HINT_CHARS = 200   # Whisper reads the prompt as the text before the clip; keep it short
RETRY_TEMPERATURES = [0.0, 0.2, 0.4]   # tried in order when a decode looks unreliable


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel  # imported lazily: it is a heavy import

    return WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE,
        download_root=str(config.MODELS_DIR / "whisper"),
    )


def hint(title: str = "", names: list[str] | None = None, film: str = "") -> str:
    """Words the clip probably contains, as a nudge for spelling. Never the transcript itself."""
    parts = [title.strip(), ", ".join(n for n in (names or []) if n), film.strip()]
    text = ". ".join(p for p in parts if p)
    return f"{text[:MAX_HINT_CHARS]}." if text else ""


def _run(wav: Path, language: str | None, prompt: str) -> dict:
    segments, info = _model().transcribe(
        str(wav),
        language=language,
        beam_size=5,
        vad_filter=True,                    # skip silence and music-only stretches
        condition_on_previous_text=False,   # reduces repeated-phrase hallucination
        initial_prompt=prompt or None,
        # A short retry ladder with one sample per step. The default (6 temperatures x 5
        # samples) took a 17 s clip to 171 s and 6.5 GB, enough to run Docker out of
        # memory next to Laya; this gives the same text in 52 s and 4.7 GB.
        temperature=RETRY_TEMPERATURES,
        best_of=1,
    )
    segs = list(segments)
    return {
        "text": " ".join(s.text.strip() for s in segs).strip(),
        "language": info.language,
        "language_probability": round(float(info.language_probability), 3),
        "avg_logprob": round(sum(s.avg_logprob for s in segs) / len(segs), 3) if segs else None,
        "no_speech_prob": round(max(s.no_speech_prob for s in segs), 3) if segs else None,
        "model": config.WHISPER_MODEL,
    }


def transcribe(wav: Path, language: str | None = None, prompt: str = "") -> dict:
    """Returns the text plus the confidence numbers we use to flag clips for review.

    Pass `language` ("hi", "en", ...) whenever the source tells you. Without it the
    language is detected, and a detection outside STT_LANGUAGES is re-run as the
    fallback language (detection on a 5 second clip with music under it is unreliable).
    """
    prompt = prompt if config.STT_HINT else ""
    result = _run(wav, language, prompt)
    if language is None and result["language"] not in config.STT_LANGUAGES:
        detected = result["language"]
        result = _run(wav, config.STT_FALLBACK_LANGUAGE, prompt)
        result["detected_language"] = detected
    result["hint"] = bool(prompt)
    return result
