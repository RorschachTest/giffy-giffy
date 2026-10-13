"""Cleaning on-screen text: watermarks, handles, repeats, run-together words. No OCR model."""
from . import conftest  # noqa: F401

from app import ocr


def test_watermarks_handles_and_repeats_are_dropped():
    lines = ["Boys are simple.", "@memeverse143", "Boysaresimple", "VITA", "Boys are simple", "mememandir"]
    assert ocr.clean(lines) == "Boys are simple"


def test_two_different_captions_are_both_kept():
    t = ocr.clean(["In Wedding Season", ":Photographer", "When Wedding Season Ends"])
    assert "In Wedding Season" in t and "When Wedding Season Ends" in t and "Photographer" in t


def test_run_together_words_split_with_the_library_words_first(tmp_path, monkeypatch):
    """English alone splits "brocheat" as "broc heat"; the library's words ("bro", "kya",
    "haath") come first in the list the backfill builds, which fixes that."""
    import gzip
    from pathlib import Path

    import wordninja

    english = gzip.open(Path(wordninja.__file__).parent / "wordninja" / "wordninja_words.txt.gz", "rt").read().split()
    words = tmp_path / "words.txt.gz"
    with gzip.open(words, "wt") as f:
        # first occurrence must win (wordninja keeps the LAST rank of a repeated word)
        f.write("\n".join(dict.fromkeys(["bro", "kya", "haath", "mein", "bhai", *english])))
    monkeypatch.setattr(ocr, "_splitter", lambda: wordninja.LanguageModel(str(words)))
    assert ocr._respace("Whenyouhelpbrocheat") == "When you help bro cheat"
    assert ocr._respace("Kyahaathmein") == "Kya haath mein"
    assert ocr._respace("Puneet") == "Puneet"                 # short words and names are left alone


def test_empty_and_noise():
    assert ocr.clean([]) == "" and ocr.clean(["a", "@x"]) == ""
