"""The Claude backend against a fake of the Anthropic client: request shape, usage, readiness."""
import sys
import types

import pytest

from . import conftest  # noqa: F401

from app import config, explain


class _Msg:
    def __init__(self, text):
        self.content = [types.SimpleNamespace(type="text", text=text)]
        self.usage = types.SimpleNamespace(input_tokens=1234, output_tokens=96)


@pytest.fixture
def fake_anthropic(monkeypatch, tmp_path):
    calls = {}

    class Client:
        def __init__(self, **kw):
            calls["client"] = kw
            self.messages = types.SimpleNamespace(create=self.create)

        def create(self, **kw):
            calls["request"] = kw
            return _Msg('{"gist": "Ravi Kishan boasts.", "why_funny": "Empty bluster.", "send_when": ["when he talks big"]}')

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=Client))
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")
    from PIL import Image
    frames = []
    for i in range(3):
        p = tmp_path / f"f{i}.jpg"
        Image.new("RGB", (1280, 720), (i * 40, 20, 90)).save(p)
        frames.append(p)
    return calls, frames


def test_request_shape_and_result(fake_anthropic):
    calls, frames = fake_anthropic
    out = explain.run("claude", frames, "ravi kishan", "नाम", "naam", "hi")
    assert out["gist"] == "Ravi Kishan boasts." and out["send_when"] == ["when he talks big"]
    assert out["usage"]["input_tokens"] == 1234 and out["usage"]["output_tokens"] == 96
    req = calls["request"]
    assert req["model"] == "claude-haiku-4-5-20251001" and req["system"] == explain.SYSTEM
    blocks = req["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["image", "image", "image", "text"]
    assert "Title: ravi kishan" in blocks[-1]["text"]
    import base64, io
    from PIL import Image
    im = Image.open(io.BytesIO(base64.b64decode(blocks[0]["source"]["data"])))
    assert max(im.size) <= 768                                   # downscaled before it is sent
    assert calls["client"]["api_key"] == "sk-test"


def test_without_a_key_the_step_is_skipped(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(config, "EXPLAIN_BACKEND", "claude")
    ok, why = explain.ready("claude")
    assert not ok and "ANTHROPIC_API_KEY" in why
    assert explain.available() is False
