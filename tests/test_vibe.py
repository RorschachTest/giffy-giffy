"""Mood rules: combining signals, picking moods, filling fields. No models, no database."""
from . import conftest  # noqa: F401

from app import vibe


def test_softmax_sums_to_one():
    p = vibe.softmax([1.0, 2.0, 3.0])
    assert abs(sum(p) - 1) < 1e-9 and p[2] > p[1] > p[0]


def test_combine_drops_missing_signals():
    # no faces, no dialogue: the scene alone decides, at full strength
    out = vibe.combine({}, {"shocked": 1.0}, {})
    assert out["shocked"] == 1.0 and out["happy"] == 0.0


def test_combine_weights_faces_over_dialogue():
    out = vibe.combine({"angry": 1.0}, {}, {"happy": 1.0})
    assert out["angry"] > out["happy"]


def test_pick_threshold_order_and_limit():
    scores = {m: 0.0 for m in vibe.MOODS}
    scores.update({"shocked": 0.9, "sad": 0.5, "angry": 0.4, "happy": 0.35, "smug": 0.1})
    assert vibe.pick(scores) == ["shocked", "sad", "angry"]


def test_use_when_and_description():
    assert vibe.use_when_for(["in a hurry"]) == ["when you are late", "when you need to escape"]
    assert vibe.describe_text([], 2, []) == ""
    assert vibe.describe_text(["shocked"], 1, ["shocked"]) == "One person, looking shocked. Overall mood: shocked."


def test_every_fer_mood_is_a_known_mood():
    assert set(vibe.FER_TO_MOOD.values()) <= set(vibe.MOODS)


def test_existing_values_win():
    cur = {"description": "mine", "reactions": [], "use_when": ["when x"]}
    found = {"description": "theirs", "reactions": ["shocked"], "use_when": ["when y"]}
    assert vibe.fill_missing(cur, found) == {"description": "mine", "reactions": ["shocked"], "use_when": ["when x"]}


def test_fer_download_is_atomic(tmp_path, monkeypatch):
    """An interrupted or truncated download must not leave a file at the final path."""
    import urllib.request

    from app import vibe

    target = tmp_path / "ferplus" / "emotion-ferplus-8.onnx"

    def cut_off(url, dest):
        open(dest, "wb").write(b"x" * 1000)
        raise OSError("connection reset")
    monkeypatch.setattr(urllib.request, "urlretrieve", cut_off)
    try:
        vibe._download_fer(target)
        assert False, "expected the cut-off download to fail"
    except OSError:
        pass
    assert not target.exists() and not target.with_suffix(".part").exists()

    monkeypatch.setattr(urllib.request, "urlretrieve", lambda url, dest: open(dest, "wb").write(b"x" * 1000))
    try:
        vibe._download_fer(target)       # finishes, but far too small to be the model
        assert False, "expected the too-small file to be rejected"
    except OSError as exc:
        assert "expected about 35 MB" in str(exc)
    assert not target.exists()

    monkeypatch.setattr(urllib.request, "urlretrieve", lambda url, dest: open(dest, "wb").write(b"x" * vibe.FER_MIN_BYTES))
    vibe._download_fer(target)
    assert target.exists() and not target.with_suffix(".part").exists()
