"""Duplicates: black bars, trimmed re-uploads, same speech; merging hides, never deletes."""
import os

import pytest
from PIL import Image, ImageDraw

from . import conftest  # noqa: F401

from app import media


def _picture(path, bars: int = 0, w: int = 320, h: int = 180):
    """A recognisable frame, optionally letterboxed inside black bars."""
    inner = Image.new("RGB", (w, h), (200, 180, 40))
    d = ImageDraw.Draw(inner)
    d.rectangle((20, 20, 140, 160), fill=(30, 60, 200))
    d.ellipse((180, 40, 300, 150), fill=(220, 30, 30))
    if bars:
        im = Image.new("RGB", (w, h + 2 * bars), (0, 0, 0))
        im.paste(inner, (0, bars))
    else:
        im = inner
    im.save(path)
    return path


def test_letterboxed_copy_hashes_like_the_original(tmp_path):
    a = media.dhash(_picture(tmp_path / "a.jpg"))
    b = media.dhash(_picture(tmp_path / "b.jpg", bars=90))       # the same picture inside black bars
    assert media.hamming(a, b) <= 6


def test_trimmed_copy_matches_in_any_order():
    a = [0x0F0F0F0F0F0F0F0F, 0x3333333333333333, 0x5555555555555555, 0x00FF00FF00FF00FF]
    shifted = a[1:] + a[:1]                       # a trim moved every keyframe one slot
    assert media.mean_hamming(a, shifted) > 6     # position by position: looks different
    assert media.frame_distance(a, shifted) == 0  # any order: the same frames


def test_blank_frames_are_not_evidence():
    from app import dedupe
    assert not dedupe._informative(0) and not dedupe._informative(-1)
    assert dedupe._informative(0x0F0F0F0F0F0F0F0F)


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")
def test_same_speech_found_merged_hidden_and_undone():
    from app import db, dedupe, indexing, search
    db.init_db()
    line = "kaun hai ye log kahan se aate hain ye log"
    with db.session() as conn:
        conn.execute("TRUNCATE clips, people RESTART IDENTITY CASCADE")
        ids = []
        look = [0x0F0F0F0F0F0F0F0F, 0x3333333333333333, 0x5555555555555555, 0x00FF00FF00FF00FF]
        rewatermarked = [h ^ 0x00000000000000FF for h in look]           # 8 bits off per frame
        other = [0x6A6A6A6A6A6A6A6A, 0x1C1C1C1C1C1C1C1C, 0x7E007E007E007E00, 0x2B2B2B2B2B2B2B2B]
        for i, (title, dur, w, said, ph) in enumerate([("Kon hain ye log", 4.5, 640, line, look),
                                                        ("kon ha ye log kaha say atay", 4.6, 1280, line, rewatermarked),
                                                        ("something else", 4.5, 640, "bilkul alag baat hai yahan par", other)]):
            ids.append(conn.execute(
                """INSERT INTO clips (sha256, file, title, duration, width, height, transcript_roman, phash)
                   VALUES (%s, %s, %s, %s, %s, 360, %s, %s) RETURNING id""",
                (f"{i:064x}", f"c{i}.mp4", title, dur, w, said, ph)).fetchone()["id"])
            indexing.refresh(conn, ids[-1], embed=False)
        found = dedupe.groups(conn)
        assert found == [[ids[0], ids[1]]]                         # the third says something else
        keep = dedupe.merge_group(conn, found[0])
        assert keep == ids[1]                                      # the bigger picture is kept
        k = conn.execute("SELECT comments, duplicates_seen FROM clips WHERE id = %s", (keep,)).fetchone()
        assert "Kon hain ye log" in k["comments"] and k["duplicates_seen"] == 2
        browse = [r["id"] for r in search.search(conn, "", 50, log_query=False)]
        assert ids[0] not in browse and keep in browse
        assert conn.execute("SELECT count(*) AS n FROM clips").fetchone()["n"] == 3   # nothing deleted
        assert dedupe.undo(conn) == 1
        assert conn.execute("SELECT duplicates_seen FROM clips WHERE id = %s", (keep,)).fetchone()["duplicates_seen"] == 1


def test_same_clip_rule_on_real_cases():
    from app.media import same_clip
    kaun = "kaun hai ye log? kahan se ate hain ye?"
    # letterboxed / watermarked re-upload: same words, picture moderately different
    assert same_clip(10.0, 1.0, kaun, kaun)
    # reused sound over other footage ("Apple right now" / "No Youtube"): different memes
    assert not same_clip(28.0, 0.93, "pagal to bana doonga desh ko guaranteed", "pagal to bana doonga desh ko guaranteed")
    # Whisper filler on two music clips is not "the same speech"
    assert not same_clip(28.0, 1.0, "Thanks for watching.", "Thanks for watching.")
    # silent look-alikes: only the same picture strictly ("Sheep laughing" / "Goat shout" at 4.9)
    assert not same_clip(4.9, 0.0, "", "") and same_clip(2.0, 0.0, "", "")
    # one talks, the other does not: a different soundtrack, a different meme
    assert not same_clip(4.6, 0.1, "ap kaha the jyanee baba ji kaha the", "hua tha")
    # same footage, the transcript only partly recognised (161 / 377): still the same clip
    assert same_clip(2.5, 0.45, kaun, "ho naheen ye log, kaha se ate hee.")
