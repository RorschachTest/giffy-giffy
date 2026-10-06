"""All settings come from environment variables (see .env.example)."""
from __future__ import annotations

import os
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://memeclip:memeclip@db:5432/memeclip")

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
INBOX_DIR = DATA_DIR / "inbox"          # drop clips here
MEDIA_DIR = DATA_DIR / "media"          # web-ready clips and thumbnails
PROCESSED_DIR = DATA_DIR / "processed"  # originals, after a successful run
FAILED_DIR = DATA_DIR / "failed"        # originals that could not be processed
GALLERY_DIR = DATA_DIR / "gallery"      # gallery/<Person Name>/*.jpg
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
MAX_CLIP_SECONDS = float(os.getenv("MAX_CLIP_SECONDS", "30"))
KEYFRAMES = int(os.getenv("KEYFRAMES", "4"))
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "3"))

# Speech to text
ENABLE_STT = _bool("ENABLE_STT", True)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")       # small | medium | large-v3
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")    # int8 on CPU, float16 on GPU
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "") or None  # e.g. "hi"; empty = auto-detect

# Text embeddings (meaning-based search)
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "fastembed")   # fastembed | hash (tests only)
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

# Faces
ENABLE_FACES = _bool("ENABLE_FACES", True)
FACE_MODEL = os.getenv("FACE_MODEL", "buffalo_l")
FACE_DIM = 512
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.42"))  # cosine similarity
FACE_MIN_DET_SCORE = float(os.getenv("FACE_MIN_DET_SCORE", "0.6"))
FACE_MIN_SIZE = int(os.getenv("FACE_MIN_SIZE", "40"))                     # pixels

# Duplicate detection
DEDUP_MAX_HAMMING = int(os.getenv("DEDUP_MAX_HAMMING", "6"))   # of 64 bits, averaged over frames
DEDUP_MAX_DURATION_DIFF = float(os.getenv("DEDUP_MAX_DURATION_DIFF", "0.7"))

# Ranking weights
W_KEYWORD = float(os.getenv("W_KEYWORD", "0.65"))
W_SEMANTIC = float(os.getenv("W_SEMANTIC", "0.35"))
W_POPULARITY = float(os.getenv("W_POPULARITY", "0.03"))
# A clip is shown only if it clears at least one of these. Below them a "match" is
# just letter-overlap noise. Starting values: tune them against your own test queries.
MIN_KEYWORD = float(os.getenv("MIN_KEYWORD", "0.35"))
MIN_SEMANTIC = float(os.getenv("MIN_SEMANTIC", "0.45"))


def ensure_dirs() -> None:
    for d in (INBOX_DIR, MEDIA_DIR, PROCESSED_DIR, FAILED_DIR, GALLERY_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
