"""Meme insert editor: put library clips into your own video at any point on its timeline.

    GET  /edit                     the editor page
    POST /edits                    upload a video; returns the edit
    GET  /edits/{id}               the edit: current version, keyframes, inserts
    GET  /edits/{id}/v/{n}.mp4     version n of the video
    GET  /edits/{id}/v/{n}.jpg     filmstrip of version n (timeline background)
    POST /edits/{id}/insert        {clip_id, at, crop, filter}: insert a clip at `at` seconds
    POST /edits/{id}/undo          drop the last insert

How an insert stays cheap: the upload is converted ONCE to a fixed format (H.264 30 fps,
a keyframe every half second, AAC 48 kHz stereo, same encoder settings every time). An
insert then re-encodes only the meme into that exact format and joins the three parts
(video up to the cut, meme, rest of the video) with ffmpeg's concat demuxer and
`-c copy`: the user's video is never decoded again. Cuts snap to the keyframe at or
before the requested time (at most 0.5 s earlier).
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import config, db, media, storage

router = APIRouter()

EDITS = config.DATA_DIR / "edits"
MAX_BYTES = 95 * 1024 * 1024    # Cloudflare (free plan) refuses request bodies over 100 MB
MAX_SECONDS = 10 * 60
MAX_VERSIONS = 30
FPS = 30
KEY_EVERY = 0.5
STRIP_FRAMES = 16

# Identical encoder settings for the upload and every meme, so the parts join without re-encoding.
ENCODE = [
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-profile:v", "high", "-level", "4.0",
    "-pix_fmt", "yuv420p", "-r", str(FPS), "-force_key_frames", f"expr:gte(t,n_forced*{KEY_EVERY})",
    "-sc_threshold", "0", "-video_track_timescale", "90000",
    "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart",
]

# Filter presets: the ffmpeg look, matched by a CSS filter for the live preview in edit.html.
FILTERS = {
    "none": "",
    "mono": "hue=s=0",
    "sepia": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131",
    "vivid": "eq=saturation=1.6:contrast=1.1",
    "noir": "hue=s=0,eq=contrast=1.5",
    "fried": "eq=saturation=3:contrast=1.8,unsharp=5:5:2",
}


class Crop(BaseModel):
    """The kept part of the meme, as fractions of its width and height."""
    x: float = Field(0, ge=0, le=1)
    y: float = Field(0, ge=0, le=1)
    w: float = Field(1, gt=0.05, le=1)
    h: float = Field(1, gt=0.05, le=1)


class InsertBody(BaseModel):
    clip_id: int
    at: float = Field(ge=0)
    crop: Crop = Crop()
    filter: str = "none"


# --------------------------------------------------------------------------- #
# Storage: data/edits/<id>/{state.json, v0.mp4, v0.jpg, v1.mp4, ...}
# --------------------------------------------------------------------------- #

def _owner(request: Request) -> int | None:
    user = getattr(request.state, "user", None)
    return user["id"] if user else None          # None when sign-in is off


def _load(edit_id: str, request: Request) -> tuple[Path, dict]:
    if not edit_id.isalnum():
        raise HTTPException(404, "no such edit")
    folder = EDITS / edit_id
    try:
        state = json.loads((folder / "state.json").read_text())
    except FileNotFoundError:
        raise HTTPException(404, "no such edit") from None
    if state["owner"] != _owner(request):
        raise HTTPException(404, "no such edit")    # someone else's: don't confirm it exists
    return folder, state


def _save(folder: Path, state: dict) -> None:
    tmp = folder / "state.json.tmp"
    tmp.write_text(json.dumps(state))
    tmp.replace(folder / "state.json")


def _view(edit_id: str, state: dict) -> dict:
    v = state["versions"][-1]
    n = len(state["versions"]) - 1
    return {"id": edit_id, "version": n, "url": f"/edits/{edit_id}/v/{n}.mp4", "strip": f"/edits/{edit_id}/v/{n}.jpg",
            "duration": v["duration"], "width": state["width"], "height": state["height"],
            "keyframes": v["keyframes"], "inserts": v["inserts"], "can_undo": n > 0,
            "filters": list(FILTERS)}


# --------------------------------------------------------------------------- #
# Video work
# --------------------------------------------------------------------------- #

def _keyframe_times(path: Path) -> list[float]:
    """Keyframe timestamps as stored in the file (a joined file can start a few ms after 0)."""
    out = media._run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
                      "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(path)]).stdout
    return sorted({float(t) for t in out.replace(",", " ").split() if t != "N/A"})


def _keyframes(path: Path) -> list[float]:
    """Keyframes in seconds from the first frame: the times the player and the user see."""
    times = _keyframe_times(path)
    return [round(t - times[0], 3) for t in times]


def _strip(video: Path, out: Path, duration: float) -> None:
    """Evenly spaced frames side by side: the timeline's background."""
    frames = []
    for i in range(STRIP_FRAMES):
        f = out.with_name(f"{out.stem}_{i}.jpg")
        media._run(["ffmpeg", "-y", "-v", "error", "-ss", f"{duration * (i + .5) / STRIP_FRAMES:.3f}", "-i", str(video),
                    "-frames:v", "1", "-vf", "scale=-2:72", str(f)])
        frames.append(f)
    args = [a for f in frames for a in ("-i", str(f))]
    media._run(["ffmpeg", "-y", "-v", "error", *args, "-filter_complex",
                f"{''.join(f'[{i}:v]' for i in range(len(frames)))}hstack=inputs={len(frames)}", str(out)])
    for f in frames:
        f.unlink(missing_ok=True)


def _normalise(src: Path, out: Path) -> media.Probe:
    info = media.probe(src)
    if info.duration > MAX_SECONDS:
        raise HTTPException(400, f"videos up to {MAX_SECONDS // 60} minutes")
    # every version has an audio track, so a meme with sound can always join it
    audio = [] if info.has_audio else ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    media._run(["ffmpeg", "-y", "-v", "error", "-i", str(src), *audio,
                "-map", "0:v:0", "-map", "0:a:0" if info.has_audio else "1:a:0", "-shortest",
                "-vf", "scale=-2:'min(720,ih)',setsar=1", *ENCODE, str(out)])
    return media.probe(out)


def _meme(src: Path, out: Path, width: int, height: int, crop: Crop, look: str) -> float:
    info = media.probe(src)
    cw, ch = min(crop.w, 1 - crop.x), min(crop.h, 1 - crop.y)
    vf = (f"crop=trunc(iw*{cw}/2)*2:trunc(ih*{ch}/2)*2:iw*{crop.x}:ih*{crop.y},"
          + (FILTERS[look] + "," if FILTERS[look] else "")
          + f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
          f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1")
    audio = [] if info.has_audio else ["-f", "lavfi", "-t", f"{info.duration:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
    media._run(["ffmpeg", "-y", "-v", "error", "-i", str(src), *audio, "-map", "0:v:0",
                "-map", "0:a:0" if info.has_audio else "1:a:0", "-vf", vf, *ENCODE, str(out)])
    return media.probe(out).duration


def _join(current: Path, meme: Path, cut: float, out: Path) -> None:
    """current[0, cut) + meme + current[cut, end], stream copy. `cut` is a keyframe of current,
    in seconds from its first frame; concat's in/out points want the file's own timestamps."""
    cut += _keyframe_times(current)[0]
    parts = out.with_suffix(".txt")
    lines = []
    if cut > 0:
        lines += [f"file '{current}'", f"outpoint {cut:.6f}"]
    lines += [f"file '{meme}'", f"file '{current}'", f"inpoint {cut:.6f}"]
    parts.write_text("\n".join(lines) + "\n")
    try:
        media._run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(parts),
                    "-c", "copy", "-movflags", "+faststart", str(out)])
    finally:
        parts.unlink(missing_ok=True)


def snap(keyframes: list[float], at: float) -> float:
    """The last keyframe at or before `at` (the cut has to start a fresh picture)."""
    return max((k for k in keyframes if k <= at + 1e-3), default=0.0)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@router.post("/edits")
def create(request: Request, file: UploadFile = File(...)) -> dict:
    if Path(file.filename or "").suffix.lower() not in config.VIDEO_EXTENSIONS:
        raise HTTPException(400, "upload an mp4, mov, webm, mkv or m4v video")
    edit_id = uuid.uuid4().hex
    folder = EDITS / edit_id
    folder.mkdir(parents=True)
    try:
        src = folder / "upload"
        with src.open("wb") as f:
            while chunk := file.file.read(1 << 20):
                if f.tell() + len(chunk) > MAX_BYTES:
                    raise HTTPException(400, f"videos up to {MAX_BYTES >> 20} MB")
                f.write(chunk)
        try:
            info = _normalise(src, folder / "v0.mp4")
        except media.MediaError:
            raise HTTPException(400, "could not read that video") from None
        src.unlink()
        _strip(folder / "v0.mp4", folder / "v0.jpg", info.duration)
        state = {"owner": _owner(request), "width": info.width, "height": info.height,
                 "versions": [{"duration": info.duration, "keyframes": _keyframes(folder / "v0.mp4"), "inserts": []}]}
        _save(folder, state)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return _view(edit_id, state)


@router.get("/edits/{edit_id}")
def get(edit_id: str, request: Request) -> dict:
    return _view(edit_id, _load(edit_id, request)[1])


@router.api_route("/edits/{edit_id}/v/{name}", methods=["GET", "HEAD"])
def file(edit_id: str, name: str, request: Request):
    folder, state = _load(edit_id, request)
    stem, _, ext = name.partition(".")
    if not stem.isdigit() or int(stem) >= len(state["versions"]) or ext not in ("mp4", "jpg"):
        raise HTTPException(404, "no such file")
    # a version's file never changes, so browsers may keep it
    return FileResponse(folder / f"v{name}", media_type="video/mp4" if ext == "mp4" else "image/jpeg",
                        headers={"Cache-Control": "private, max-age=86400"})


@router.post("/edits/{edit_id}/insert")
def insert(edit_id: str, body: InsertBody, request: Request) -> dict:
    folder, state = _load(edit_id, request)
    if body.filter not in FILTERS:
        raise HTTPException(400, f"filter must be one of {', '.join(FILTERS)}")
    if len(state["versions"]) >= MAX_VERSIONS:
        raise HTTPException(400, "too many inserts in one video")
    v = state["versions"][-1]
    if body.at > v["duration"]:
        raise HTTPException(400, "that point is past the end of the video")
    with db.session() as conn:
        clip = conn.execute("SELECT id, file, title FROM clips WHERE id = %s AND duplicate_of IS NULL",
                            (body.clip_id,)).fetchone()
    if not clip:
        raise HTTPException(404, "no such clip")
    n = len(state["versions"])
    cur, out, meme = folder / f"v{n - 1}.mp4", folder / f"v{n}.mp4", folder / f"meme{n}.mp4"
    cut = snap(v["keyframes"], body.at)
    try:
        with storage.local_copy(clip["file"]) as src:
            length = _meme(src, meme, state["width"], state["height"], body.crop, body.filter)
        _join(cur, meme, cut, out)
        duration = media.probe(out).duration
        _strip(out, folder / f"v{n}.jpg", duration)
        keyframes = _keyframes(out)
    except media.MediaError:
        for f in (out, folder / f"v{n}.jpg"):
            f.unlink(missing_ok=True)
        raise HTTPException(500, "could not insert that clip") from None
    finally:
        meme.unlink(missing_ok=True)
    # earlier inserts after the cut move right by the meme's length
    inserts = [dict(i, at=round(i["at"] + length, 3)) if i["at"] >= cut else i for i in v["inserts"]]
    inserts.append({"at": round(cut, 3), "duration": round(length, 3), "clip_id": clip["id"],
                    "title": clip["title"] or "", "filter": body.filter})
    state["versions"].append({"duration": duration, "keyframes": keyframes,
                              "inserts": sorted(inserts, key=lambda i: i["at"])})
    _save(folder, state)
    return _view(edit_id, state)


@router.post("/edits/{edit_id}/undo")
def undo(edit_id: str, request: Request) -> dict:
    folder, state = _load(edit_id, request)
    if len(state["versions"]) > 1:
        n = len(state["versions"]) - 1
        state["versions"].pop()
        _save(folder, state)
        for ext in ("mp4", "jpg"):
            (folder / f"v{n}.{ext}").unlink(missing_ok=True)
    return _view(edit_id, state)
