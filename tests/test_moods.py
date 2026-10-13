"""Text-led moods: decisive scene scores, abstaining, Devanagari text. No video, no model for the rules."""
from . import conftest  # noqa: F401

from app import vibe


def test_undecided_scene_counts_for_little():
    flat = {"sad": 1.0, "helpless": 0.87, "confused": 0.86}      # the picture model cannot decide
    clear = {"preachy": 1.0, "shocked": 0.2}
    assert max(vibe.decisive(flat).values()) < 0.2
    assert vibe.decisive(clear)["preachy"] == 0.8


def test_no_mood_beats_a_wrong_one():
    assert vibe.pick_confident({"happy": 0.25, "sad": 0.2}) == []
    assert vibe.pick_confident({"helpless": 0.6, "sad": 0.55, "angry": 0.3}) == ["helpless", "sad"]
    assert len(vibe.pick_confident({m: 0.9 for m in vibe.MOODS})) == vibe.MOOD_LIMIT


def test_clip_text_prefers_devanagari_speech():
    t = vibe.clip_text({"title": "Kon hain ye log", "transcript_native": "कौन है ये लोग?",
                        "transcript_roman": "kaun hai ye log?", "caption": "(untitled)", "folk_names": []})
    assert "कौन है ये लोग?" in t and "kaun hai" not in t and "(untitled)" not in t


def test_every_mood_has_phrases():
    assert set(vibe.MOOD_PHRASES) == set(vibe.MOODS)
