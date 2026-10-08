"""The shared explain contract: prompt, parser, merge rules, dispatch. No model, no network."""
import pytest

from . import conftest  # noqa: F401

from app import config, explain


def test_parse_clean_json():
    out = explain.parse('{"gist": " A man  boasts. ", "why_funny": "He promises big.", "send_when": ["when a friend overpromises", "when "]}')
    assert out == {"gist": "A man boasts.", "why_funny": "He promises big.", "send_when": ["when a friend overpromises"]}


def test_parse_json_in_prose_fences_and_bad_shapes():
    out = explain.parse('Sure!\n```json\n{"gist": "x", "why_funny": 5, "send_when": "when late"}\n```')
    assert out == {"gist": "x", "why_funny": "", "send_when": ["when late"]}
    for reply in ("", "no json", "[1, 2]", '{"gist": '):
        assert explain.is_empty(explain.parse(reply))


def test_parse_caps_lengths():
    out = explain.parse('{"gist": "%s", "send_when": %s}' % ("a" * 900, str(["when " + "b" * 400] * 9).replace("'", '"')))
    assert len(out["gist"]) == explain.LIMITS["gist"] and len(out["send_when"]) == 4
    assert all(len(w) <= explain.LIMITS["send_when"] for w in out["send_when"])


def test_user_prompt_includes_what_the_backend_needs():
    p = explain.user_prompt("ravi kishan money", "नाम या दाम", "nam ya dam", "hi", "known: x")
    assert "Title: ravi kishan money" in p and "may be wrong" in p and "nam ya dam" in p and "Language hint: hi" in p
    assert "no speech was found" in explain.user_prompt("t", "", "", None)


FOUND = {"gist": "model gist", "why_funny": "model why", "send_when": ["when model"]}


def test_merge_fills_empty_fields_and_records_the_source():
    fields, source = explain.merge({"explain_source": "", "gist": "", "why_funny": "", "send_when": []}, FOUND, "claude")
    assert fields == FOUND and source == "claude"


def test_merge_keeps_what_is_already_written():
    cur = {"explain_source": "local", "gist": "mine", "why_funny": "", "send_when": []}
    fields, source = explain.merge(cur, FOUND, "claude")
    assert fields["gist"] == "mine" and fields["why_funny"] == "model why" and source == "claude"
    fields, _ = explain.merge(cur, FOUND, "claude", replace=True)
    assert fields["gist"] == "model gist"


@pytest.mark.parametrize("who", ["manual", "sidecar"])
def test_merge_never_touches_a_persons_words(who):
    cur = {"explain_source": who, "gist": "person wrote this", "why_funny": "", "send_when": []}
    for replace in (False, True):
        fields, source = explain.merge(cur, FOUND, "claude", replace=replace)
        assert fields == {"gist": "person wrote this", "why_funny": "", "send_when": []} and source == who


def test_dispatch_with_a_registered_backend(monkeypatch):
    seen = {}

    def fake(system, user, frames):
        seen.update(system=system, user=user, frames=frames)
        return {"text": '{"gist": "g", "why_funny": "w", "send_when": ["when x"]}', "usage": {"input_tokens": 12}}
    explain.register("fake", fake)
    monkeypatch.setattr(config, "EXPLAIN_BACKEND", "fake")
    assert explain.available()
    out = explain.explain(["a.jpg"], "Title T", "नाम", "naam", "hi")
    assert out["gist"] == "g" and out["backend"] == "fake" and out["usage"] == {"input_tokens": 12} and out["seconds"] >= 0
    assert seen["system"] == explain.SYSTEM and "Title T" in seen["user"] and seen["frames"] == ["a.jpg"]
    assert "fake" in explain.available_backends()


def test_off_manual_and_unknown_never_call_a_model(monkeypatch):
    for name in ("off", "", "manual", "no-such-backend", "claude"):   # claude/local modules live on their own branches
        monkeypatch.setattr(config, "EXPLAIN_BACKEND", name)
        if name == "claude":
            try:
                explain.backend("claude")
                continue          # this branch has the module
            except LookupError:
                pass
        assert explain.available() is False
    with pytest.raises(LookupError):
        explain.backend("no-such-backend")
