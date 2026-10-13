"""Hinglish queries get an English gloss; the clip state a text model reads is honest."""
from . import conftest  # noqa: F401

from app import clip_state, hinglish


def test_gloss_adds_english_for_hindi_words():
    g = hinglish.gloss("salary day already feeling gareeb")
    assert g.startswith("salary day already feeling gareeb") and "poor" in g
    assert "who" in hinglish.gloss("kaun hai ye log") and "people" in hinglish.gloss("kaun hai ye log")
    assert hinglish.gloss("boss shouting at me") == "boss shouting at me"      # nothing to gloss


def test_state_prefers_devanagari_and_drops_untrusted_speech():
    clip = {"title": "Kon hain ye log", "transcript_native": "कौन है ये लोग?", "transcript_roman": "kaun hai ye log?",
            "people": ["Arshad Warsi"], "reactions": ["confused"], "raw": {"stt": {"avg_logprob": -0.2}}}
    s = clip_state.build(clip)
    assert s["said"] == "कौन है ये लोग?" and s["said_roman"] == "kaun hai ye log?" and s["people"] == "Arshad Warsi"
    noisy = {**clip, "raw": {"stt": {"avg_logprob": -1.2}}}
    assert "said" not in clip_state.build(noisy)                               # garbled speech is left out


def test_thin_clips_are_flagged_not_padded():
    s = clip_state.build({"title": "(untitled)", "transcript_roman": "", "reactions": ["shocked"]})
    assert clip_state.is_thin(s) and s == {"visual_mood_hint": "shocked"}
    assert not clip_state.is_thin(clip_state.build({"screen_text": "Boys are simple"}))
