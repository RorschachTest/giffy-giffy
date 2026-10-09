"""Language fallback and the hint prompt, with a stand-in for Whisper. No model download."""
from types import SimpleNamespace

from . import conftest  # noqa: F401

from app import config, stt


class FakeWhisper:
    """Answers with a fixed language when none is forced; records every call."""

    def __init__(self, detects: str):
        self.detects, self.calls = detects, []

    def transcribe(self, path, language=None, initial_prompt=None, **_):
        self.calls.append((language, initial_prompt))
        lang = language or self.detects
        seg = SimpleNamespace(text=f" said in {lang} ", avg_logprob=-0.1, no_speech_prob=0.01)
        return iter([seg]), SimpleNamespace(language=lang, language_probability=0.9)


def _use(monkeypatch, fake):
    monkeypatch.setattr(stt, "_model", lambda: fake)
    monkeypatch.setattr(config, "STT_HINT", True)


def test_unlikely_detection_is_rerun_as_hindi(monkeypatch):
    fake = FakeWhisper(detects="ko")     # what large-v3 said on a shocked-face clip
    _use(monkeypatch, fake)
    out = stt.transcribe("x.wav")
    assert [c[0] for c in fake.calls] == [None, "hi"]
    assert out["language"] == "hi" and out["detected_language"] == "ko" and out["text"] == "said in hi"


def test_hindi_or_english_detection_is_kept(monkeypatch):
    fake = FakeWhisper(detects="en")
    _use(monkeypatch, fake)
    assert stt.transcribe("x.wav")["language"] == "en" and len(fake.calls) == 1


def test_a_given_language_is_never_second_guessed(monkeypatch):
    fake = FakeWhisper(detects="ur")
    _use(monkeypatch, fake)
    stt.transcribe("x.wav", "hi")
    assert fake.calls == [("hi", None)]


def test_hint_is_passed_and_can_be_switched_off(monkeypatch):
    fake = FakeWhisper(detects="hi")
    _use(monkeypatch, fake)
    prompt = stt.hint("ravi kishan money follows", ["Ravi Kishan"], "")
    assert prompt == "ravi kishan money follows. Ravi Kishan."
    assert stt.transcribe("x.wav", "hi", prompt)["hint"] is True and fake.calls[-1][1] == prompt
    monkeypatch.setattr(config, "STT_HINT", False)
    assert stt.transcribe("x.wav", "hi", prompt)["hint"] is False and fake.calls[-1][1] is None


def test_empty_hint_and_long_hint():
    assert stt.hint("", [], "") == ""
    assert len(stt.hint("a" * 500)) == stt.MAX_HINT_CHARS + 1
