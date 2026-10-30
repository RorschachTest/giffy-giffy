"""Meme insert editor: the cut snaps to a keyframe, and an insert re-encodes only the meme."""
import shutil
import subprocess

import pytest

from . import conftest  # noqa: F401

from app import edit, media


def test_snap_takes_the_keyframe_at_or_before():
    keys = [0.0, 0.5, 1.0, 1.5]
    assert edit.snap(keys, 1.2) == 1.0
    assert edit.snap(keys, 1.0) == 1.0
    assert edit.snap(keys, 0.1) == 0.0
    assert edit.snap(keys, 99) == 1.5


def _clip(path, seconds, size, audio):
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size={size}:rate=25"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=duration={seconds}"]
    subprocess.run(cmd + ["-pix_fmt", "yuv420p", str(path)], check=True)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_insert_joins_without_touching_the_original(tmp_path):
    _clip(tmp_path / "mine.mp4", 6, "640x360", audio=False)    # the user's video, silent
    _clip(tmp_path / "meme.mp4", 2, "480x480", audio=True)     # a square meme with sound

    info = edit._normalise(tmp_path / "mine.mp4", tmp_path / "v0.mp4")
    assert (info.width, info.height) == (640, 360) and info.has_audio      # silent track added
    keys = edit._keyframes(tmp_path / "v0.mp4")
    assert keys[:3] == [0.0, 0.5, 1.0]                                     # a keyframe every half second

    length = edit._meme(tmp_path / "meme.mp4", tmp_path / "m.mp4", info.width, info.height,
                        edit.Crop(x=.25, y=.25, w=.5, h=.5), "sepia")
    cut = edit.snap(keys, 2.3)
    assert cut == 2.0
    edit._join(tmp_path / "v0.mp4", tmp_path / "m.mp4", cut, tmp_path / "v1.mp4")

    out = media.probe(tmp_path / "v1.mp4")
    assert (out.width, out.height) == (640, 360)
    assert out.duration == pytest.approx(info.duration + length, abs=0.15)
    new_keys = edit._keyframes(tmp_path / "v1.mp4")
    assert cut in new_keys and any(abs(k - (cut + length)) < 0.05 for k in new_keys)   # meme starts and ends on keyframes

    # a second insert cuts the joined file (which starts a few ms late): no frames lost or doubled
    keys = edit._keyframes(tmp_path / "v1.mp4")
    edit._join(tmp_path / "v1.mp4", tmp_path / "m.mp4", edit.snap(keys, 1.0), tmp_path / "v2.mp4")
    assert media.probe(tmp_path / "v2.mp4").duration == pytest.approx(out.duration + length, abs=0.15)
