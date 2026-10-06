"""Speech to text with faster-whisper (runs on CPU, no GPU needed)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from . import config


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel  # imported lazily: it is a heavy import

    return WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE,
        download_root=str(config.MODELS_DIR / "whisper"),
    )


def transcribe(wav: Path, language: str | None = None) -> dict:
    """Returns the text plus the confidence numbers we use to flag clips for review.

    Pass `language` ("hi", "en", "ta", ...) whenever the source tells you. Auto-detect
    is unreliable on a 5 second clip with music under it.
    """
    segments, info = _model().transcribe(
        str(wav),
        language=language,
        beam_size=5,
        vad_filter=True,                    # skip silence and music-only stretches
        condition_on_previous_text=False,   # reduces repeated-phrase hallucination
    )
    segs = list(segments)
    text = " ".join(s.text.strip() for s in segs).strip()
    return {
        "text": text,
        "language": info.language,
        "language_probability": round(float(info.language_probability), 3),
        "avg_logprob": round(sum(s.avg_logprob for s in segs) / len(segs), 3) if segs else None,
        "no_speech_prob": round(max(s.no_speech_prob for s in segs), 3) if segs else None,
        "model": config.WHISPER_MODEL,
    }
