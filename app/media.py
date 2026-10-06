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

def dhash(frame: Path) -> int:
    """64-bit difference hash. Similar pictures give hashes that differ in few bits.

    Returned as a signed 64-bit integer so it fits a Postgres BIGINT.
    """
    with Image.open(frame) as im:
        px = list(im.convert("L").resize((9, 8), Image.LANCZOS).getdata())
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
