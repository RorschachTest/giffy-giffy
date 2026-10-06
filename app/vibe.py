"""Emotional context for a clip from local models. No LLM, nothing leaves the machine.

Three signals, each a score per MOOD between 0 and 1:

  faces     FER+ (ONNX) reads the expression on every face insightface found
  scene     CLIP compares each keyframe with a short sentence per mood
  dialogue  the multilingual text embedder compares what is said with the same sentences

The mood scores are combined with fixed weights; moods that clear VIBE_MIN_SCORE
become `reactions`. `use_when` comes from a fixed table per mood and
`description` from a template. Sidecar and hand-edited values always win.

Every raw score is kept in clips.raw["vibe"], so thresholds can be tuned later.
"""
from __future__ import annotations

import logging
import math
import urllib.request
from functools import lru_cache
from pathlib import Path

from . import config

log = logging.getLogger("vibe")

# FER+ output order (fixed by the model)
FER_CLASSES = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
FER_TO_MOOD = {"happiness": "happy", "surprise": "shocked", "sadness": "sad", "anger": "angry",
               "disgust": "disgusted", "fear": "scared", "contempt": "smug"}

# mood -> (sentence for CLIP and the text model, situations to send it in)
MOODS: dict[str, tuple[str, list[str]]] = {
    "happy":     ("a person smiling happily", ["when you are pleased", "when things go your way"]),
    "laughing":  ("people laughing out loud", ["when something is hilarious"]),
    "shocked":   ("a person with a shocked, surprised face", ["when something unexpected happens", "when you can't believe it"]),
    "sad":       ("a sad, disappointed person", ["when you lose", "when plans fail"]),
    "angry":     ("an angry person shouting", ["when you are fed up", "when someone annoys you"]),
    "scared":    ("a scared, frightened person", ["when you are nervous", "when the deadline is close"]),
    "disgusted": ("a disgusted person pulling a face", ["when something is gross or cringe"]),
    "smug":      ("a smug, confident person smirking", ["when you were right all along", "when you show off"]),
    "confused":  ("a confused, puzzled person", ["when nothing makes sense", "when you don't understand"]),
    "helpless":  ("a helpless person giving up", ["when there is nothing you can do"]),
    "in a hurry": ("a person running fast", ["when you are late", "when you need to escape"]),
    "dancing":   ("people dancing and celebrating", ["when you celebrate", "when you get good news"]),
    "preachy":   ("a person giving a serious speech", ["when giving advice", "when someone lectures you"]),
}

W_FACES, W_SCENE, W_DIALOGUE = 0.45, 0.40, 0.15


# --------------------------------------------------------------------------- #
# Models (each loaded once, lazily)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _fer():
    import onnxruntime as ort

    path = config.MODELS_DIR / "ferplus" / "emotion-ferplus-8.onnx"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading FER+ to %s", path)
        urllib.request.urlretrieve(config.FER_URL, path)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def _clip():
    from fastembed import ImageEmbedding, TextEmbedding

    cache = str(config.MODELS_DIR / "fastembed")
    img = ImageEmbedding(model_name=config.VIBE_CLIP_IMAGE, cache_dir=cache)
    txt = TextEmbedding(model_name=config.VIBE_CLIP_TEXT, cache_dir=cache)
    prompts = [_unit(list(v)) for v in txt.embed([f"a photo of {s}" for s, _ in MOODS.values()])]
    return img, prompts


@lru_cache(maxsize=1)
def _mood_sentences() -> list[list[float]]:
    from .embed import get_embedder

    return get_embedder().embed([s for s, _ in MOODS.values()])


# --------------------------------------------------------------------------- #
# Pure helpers (tested without any model)
# --------------------------------------------------------------------------- #

def _unit(v) -> list[float]:
    v = [float(x) for x in v]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def softmax(xs: list[float], scale: float = 1.0) -> list[float]:
    m = max(xs)
    e = [math.exp((x - m) * scale) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def combine(face: dict[str, float], scene: dict[str, float], dialogue: dict[str, float]) -> dict[str, float]:
    """Weighted mood scores. A signal that is missing (no faces, no speech) is left
    out and the remaining weights are scaled up, so one clear signal can still win."""
    parts = [(w, s) for w, s in ((W_FACES, face), (W_SCENE, scene), (W_DIALOGUE, dialogue)) if s]
    total = sum(w for w, _ in parts) or 1.0
    return {m: round(sum(w * s.get(m, 0.0) for w, s in parts) / total, 3) for m in MOODS}


def pick(scores: dict[str, float], limit: int = 3) -> list[str]:
    ranked = sorted((s, m) for m, s in scores.items() if s >= config.VIBE_MIN_SCORE)
    return [m for _, m in reversed(ranked)][:limit]


def use_when_for(moods: list[str]) -> list[str]:
    out: list[str] = []
    for m in moods:
        for u in MOODS[m][1]:
            if u not in out:
                out.append(u)
    return out[:4]


def describe_text(moods: list[str], n_faces: int, face_moods: list[str]) -> str:
    if not moods:
        return ""
    who = {0: "No clear face", 1: "One person"}.get(n_faces, f"{n_faces} faces")
    look = f", looking {' and '.join(face_moods)}" if face_moods else ""
    return f"{who}{look}. Overall mood: {', '.join(moods)}."


def fill_missing(current: dict, found: dict) -> dict:
    """Keep anything already set (sidecar or manual edit); fill only the empty fields."""
    return {k: current.get(k) or v for k, v in found.items()}


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #

def face_scores(frames: list[Path], faces_per_frame: list[list[dict]]) -> tuple[dict[str, float], int]:
    """Mean FER+ probability per mood over every face crop. Neutral is dropped."""
    import cv2
    import numpy as np

    sess = _fer()
    totals = {m: 0.0 for m in FER_TO_MOOD.values()}
    n = 0
    for frame, faces in zip(frames, faces_per_frame):
        img = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        for f in faces:
            x1, y1, x2, y2 = (int(round(v)) for v in f["bbox"])
            crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                continue
            x = cv2.resize(crop, (64, 64)).astype(np.float32).reshape(1, 1, 64, 64)
            probs = softmax([float(v) for v in sess.run(None, {sess.get_inputs()[0].name: x})[0][0]])
            for cls, p in zip(FER_CLASSES, probs):
                if cls in FER_TO_MOOD:
                    totals[FER_TO_MOOD[cls]] += p
            n += 1
    return ({m: round(v / n, 3) for m, v in totals.items()} if n else {}), n


def scene_scores(frames: list[Path]) -> dict[str, float]:
    """CLIP zero-shot: per frame a softmax over the mood sentences, averaged."""
    if not frames:
        return {}
    img_model, prompts = _clip()
    names = list(MOODS)
    totals = [0.0] * len(names)
    for v in img_model.embed([str(f) for f in frames]):
        v = _unit(v)
        sims = [sum(a * b for a, b in zip(v, p)) for p in prompts]
        for i, p in enumerate(softmax(sims, scale=100.0)):   # CLIP's usual temperature
            totals[i] += p
    # Rescale so the best mood is 1.0: a softmax over 13 moods is never near 1 on its own.
    best = max(totals) or 1.0
    return {m: round(t / best, 3) for m, t in zip(names, totals)}


def dialogue_scores(said: str) -> dict[str, float]:
    if not said.strip():
        return {}
    from .embed import embed_one

    v = embed_one(said)
    sims = [sum(a * b for a, b in zip(v, s)) for s in _mood_sentences()]
    probs = softmax(sims, scale=20.0)
    best = max(probs) or 1.0
    return {m: round(p / best * max(0.0, max(sims)), 3) for m, p in zip(MOODS, probs)}


def analyse(frames: list[Path], faces_per_frame: list[list[dict]] | None, said: str) -> dict:
    """Returns the three clip fields plus the raw scores behind them."""
    face, n_faces = face_scores(frames, faces_per_frame) if faces_per_frame else ({}, 0)
    scene = scene_scores(frames)
    dialogue = dialogue_scores(said)
    combined = combine(face, scene, dialogue)
    moods = pick(combined)
    face_moods = [m for m, s in sorted(face.items(), key=lambda kv: -kv[1]) if s >= 0.35][:2]
    return {
        "fields": {"description": describe_text(moods, n_faces, face_moods),
                   "reactions": moods, "use_when": use_when_for(moods)},
        "scores": {"faces": face, "n_faces": n_faces, "scene": scene,
                   "dialogue": dialogue, "combined": combined},
    }
