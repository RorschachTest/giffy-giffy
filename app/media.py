"""Everything that touches the video file itself. Uses ffmpeg and ffprobe."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


class MediaError(RuntimeError):
    pass


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MediaError(f"{cmd[0]} failed: {proc.stderr.strip()[-400:]}")
    return proc


@dataclass
class Probe:
    duration: float
    width: int
    height: int
    has_audio: bool


def probe(path: Path) -> Probe:
    out = _run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]).stdout
    info = json.loads(out)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError("no video stream found")
    audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    duration = float(info.get("format", {}).get("duration") or video.get("duration") or 0)
    if duration <= 0:
        raise MediaError("could not read the clip's duration")
    return Probe(duration, int(video["width"]), int(video["height"]), audio)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_keyframes(path: Path, out_dir: Path, duration: float, n: int) -> list[Path]:
    """Grab n frames spread evenly through the clip."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(n):
        t = duration * (i + 0.5) / n
        out = out_dir / f"frame_{i}.jpg"
        _run([
            "ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path),
            "-frames:v", "1", "-q:v", "3", str(out),
        ])
        if out.exists():
            frames.append(out)
    if not frames:
        raise MediaError("could not extract any frame")
    return frames


def extract_audio(path: Path, out: Path) -> Path:
    """16 kHz mono WAV, which is what speech models expect."""
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", "16000", str(out),
    ])
    return out


def transcode(path: Path, out: Path) -> Path:
    """One consistent, small, browser-playable format for every clip."""
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(path),
        "-vf", "scale=-2:'min(480,ih)'", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "26", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart", str(out),
    ])
    return out


def make_thumb(frame: Path, out: Path, height: int = 360) -> Path:
    with Image.open(frame) as im:
        im = im.convert("RGB")
        w = max(2, round(im.width * height / im.height))
        im.resize((w, height)).save(out, "JPEG", quality=80)
    return out


# --------------------------------------------------------------------------- #
# Picture hashing for duplicate detection
# --------------------------------------------------------------------------- #

BORDER_DARK = 28        # a row or column this dark (and this even) is a letterbox bar, not picture
BORDER_SPREAD = 18


def trim_borders(im: "Image.Image") -> "Image.Image":
    """Cut black bars (letterbox / pillarbox) off a frame, so a re-upload with bars
    hashes like the original. Keeps the frame as it is if the bars would leave
    less than a third of it."""
    g = im.convert("L")
    w, h = g.size
    px = g.load()

    def is_bar(values) -> bool:
        vals = list(values)
        return max(vals) - min(vals) <= BORDER_SPREAD and sum(vals) / len(vals) <= BORDER_DARK

    step_x, step_y = max(1, w // 64), max(1, h // 64)
    rows = [is_bar(px[x, y] for x in range(0, w, step_x)) for y in range(h)]
    cols = [is_bar(px[x, y] for y in range(0, h, step_y)) for x in range(w)]
    top = next((i for i, b in enumerate(rows) if not b), 0)
    bottom = h - next((i for i, b in enumerate(reversed(rows)) if not b), 0)
    left = next((i for i, b in enumerate(cols) if not b), 0)
    right = w - next((i for i, b in enumerate(reversed(cols)) if not b), 0)
    if (bottom - top) * (right - left) < w * h / 3:
        return im
    return im.crop((left, top, right, bottom))


def dhash(frame: Path) -> int:
    """64-bit difference hash of the picture inside any black bars. Similar pictures
    give hashes that differ in few bits.

    Returned as a signed 64-bit integer so it fits a Postgres BIGINT.
    """
    with Image.open(frame) as im:
        px = list(trim_borders(im).convert("L").resize((9, 8), Image.LANCZOS).getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits - (1 << 64) if bits >= (1 << 63) else bits


def hamming(a: int, b: int) -> int:
    return bin((a ^ b) & 0xFFFFFFFFFFFFFFFF).count("1")


def mean_hamming(a: list[int], b: list[int]) -> float:
    if not a or len(a) != len(b):
        return 64.0
    return sum(hamming(x, y) for x, y in zip(a, b)) / len(a)


def frame_distance(a: list[int], b: list[int]) -> float:
    """How different two clips look, tolerant of trimming: every keyframe is matched
    to its closest keyframe in the other clip (any position), both ways, averaged.
    A trimmed or padded re-upload samples slightly different moments, so position-
    by-position comparison (mean_hamming) misses it."""
    if not a or not b:
        return 64.0
    one = sum(min(hamming(x, y) for y in b) for x in a) / len(a)
    two = sum(min(hamming(y, x) for x in a) for y in b) / len(b)
    return (one + two) / 2


# --------------------------------------------------------------------------- #
# Is it the same clip? One rule for uploads (pipeline) and clean-ups (dedupe)
# --------------------------------------------------------------------------- #

# Lines Whisper writes over music or silence: equal in unrelated clips, so not evidence.
FILLER_SPEECH = ("thanks for watching", "thank you for watching", "thank you", "subscribe",
                 "subtitles by", "please subscribe", "like and subscribe", "music", "hua hai")
SAME_PICTURE = 4.0       # frame_distance: the same footage (strict: plain frames look alike)
SIMILAR_PICTURE = 12.0   # frame_distance: the same footage re-encoded, watermarked or cropped
SAME_SPEECH = 0.8        # pg_trgm similarity of the Roman transcripts
DIFFERENT_SPEECH = 0.3   # below this the words are clearly different


def informative_hash(h: int) -> bool:
    """A frame hash that says something: not all-dark / all-even (popcount near 0 or 64)."""
    n = bin(h & 0xFFFFFFFFFFFFFFFF).count("1")
    return 4 <= n <= 60


def real_speech(text: str | None, min_chars: int = 20) -> bool:
    """Speech that tells clips apart: long enough, and not a known filler line."""
    t = " ".join((text or "").lower().replace(".", " ").replace(",", " ").split())
    if len(t) < min_chars:
        return False
    return not any(t == f or (t.startswith(f) and len(t) < len(f) + 8) for f in FILLER_SPEECH)


def same_clip(frame_dist: float, speech_sim: float, speech_a: str | None, speech_b: str | None) -> bool:
    """The rule, from the duplicates actually found in the library:
    - both talk:      same words and similar picture (a re-upload with bars or a watermark),
                      or the same picture unless the words are clearly different
    - neither talks:  the same picture, strictly (sheep and goat clips hash alike at 5)
    - one talks:      a different soundtrack over the footage is a different meme
    Same words over clearly different footage is a reused sound, also a different meme."""
    talk_a, talk_b = real_speech(speech_a), real_speech(speech_b)
    if talk_a and talk_b:
        return (speech_sim >= SAME_SPEECH and frame_dist <= SIMILAR_PICTURE) or \
               (frame_dist <= SAME_PICTURE and speech_sim >= DIFFERENT_SPEECH)
    if not talk_a and not talk_b:
        return frame_dist <= SAME_PICTURE
    return False
