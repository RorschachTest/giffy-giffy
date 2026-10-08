"""All settings come from environment variables (see .env.example)."""
from __future__ import annotations

import os
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


# The public https address people reach this server on (https://clips.example.com).
# Share links and their video previews use it; chat apps cannot unfurl localhost.
# Empty = the address each request came in on (fine for local testing).
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")

# The page design: the name of a file in app/static/themes/ (picker, sticker, viza, ...).
# Change it here, or set UI_THEME in .env. /?theme=<name> previews any theme.
UI_THEME = os.getenv("UI_THEME") or "viza"

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

# Emotional context (mood / use-when / description), all local models, no LLM:
# facial expressions (FER+), scene mood (CLIP zero-shot), dialogue tone (text embedder)
ENABLE_VIBE = _bool("ENABLE_VIBE", True)
VIBE_CLIP_IMAGE = os.getenv("VIBE_CLIP_IMAGE", "Qdrant/clip-ViT-B-32-vision")
VIBE_CLIP_TEXT = os.getenv("VIBE_CLIP_TEXT", "Qdrant/clip-ViT-B-32-text")
FER_URL = os.getenv("FER_URL", "https://github.com/onnx/models/raw/main/validated/vision/"
                    "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx")
VIBE_MIN_SCORE = float(os.getenv("VIBE_MIN_SCORE", "0.30"))   # combined score to keep a mood

# Laya (typed decisions over text, no text generation): topics + mood at ingest,
# query intent at search. Empty LAYA_URL switches it off.
LAYA_URL = os.getenv("LAYA_URL", "")
LAYA_API_KEY = os.getenv("LAYA_API_KEY", "")
LAYA_MODEL = os.getenv("LAYA_MODEL", "multilingual")       # Hindi + English; "" = let Laya route
LAYA_TIMEOUT = float(os.getenv("LAYA_TIMEOUT", "60"))      # ingest; the first call loads the model
LAYA_SEARCH_TIMEOUT = float(os.getenv("LAYA_SEARCH_TIMEOUT", "3"))
LAYA_TOPIC_MIN = float(os.getenv("LAYA_TOPIC_MIN", "0.45"))  # probability to keep a topic on a clip
LAYA_MAX_TOPICS = int(os.getenv("LAYA_MAX_TOPICS", "5"))
LAYA_QUERY_MIN = float(os.getenv("LAYA_QUERY_MIN", "0.3"))  # probability to treat a query as about X

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
# Topic / mood match between the query (Laya) and the clip's topics + reactions.
W_TOPIC = float(os.getenv("W_TOPIC", "0.25"))
MIN_TOPIC = float(os.getenv("MIN_TOPIC", "0.5"))


def ensure_dirs() -> None:
    for d in (INBOX_DIR, MEDIA_DIR, PROCESSED_DIR, FAILED_DIR, GALLERY_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
