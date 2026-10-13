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
UI_THEME = os.getenv("UI_THEME") or "sticker"

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://memeclip:memeclip@db:5432/memeclip")

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
INBOX_DIR = DATA_DIR / "inbox"          # drop clips here
MEDIA_DIR = DATA_DIR / "media"          # web-ready clips and thumbnails (STORAGE=local)
PROCESSED_DIR = DATA_DIR / "processed"  # originals, after a successful run
FAILED_DIR = DATA_DIR / "failed"        # originals that could not be processed
GALLERY_DIR = DATA_DIR / "gallery"      # gallery/<Person Name>/*.jpg
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))

# Where web-ready clips live (app/storage.py): "local" = MEDIA_DIR, "s3" = a bucket
# (Garage on this machine via scripts/setup_garage.sh, or R2 / S3 later).
STORAGE = os.getenv("STORAGE", "local").strip().lower()
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://garage:3900")
S3_REGION = os.getenv("S3_REGION", "garage")
S3_BUCKET = os.getenv("S3_BUCKET", "memeclip")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
MAX_CLIP_SECONDS = float(os.getenv("MAX_CLIP_SECONDS", "30"))
KEYFRAMES = int(os.getenv("KEYFRAMES", "4"))
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "3"))

# Speech to text
ENABLE_STT = _bool("ENABLE_STT", True)
# Read text burned into the video (captions, punchlines) off the keyframes: app/ocr.py
ENABLE_OCR = _bool("ENABLE_OCR", True)
# Lines dropped from on-screen text: channel tags and logos seen across many clips.
OCR_WATERMARKS = os.getenv("OCR_WATERMARKS", "vita,mememandir,memeverse143,memeverse,meme mandir,pememandir")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")    # small | medium | large-v3 (large-v3 needs ~4 GB RAM)
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")    # int8 on CPU, float16 on GPU
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "") or None  # e.g. "hi"; empty = auto-detect
# Auto-detect may only answer one of these; anything else is re-run as the fallback.
STT_LANGUAGES = {x.strip() for x in os.getenv("STT_LANGUAGES", "hi,en").split(",") if x.strip()}
STT_FALLBACK_LANGUAGE = os.getenv("STT_FALLBACK_LANGUAGE", "hi")
STT_HINT = _bool("STT_HINT", True)   # title / folk names / film as Whisper's prompt

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
# query intent at search. Off unless ENABLE_LAYA=1 (and LAYA_URL is set): the model
# holds ~2.7 GB even when idle, which large-v3 needs more. Search and ingest work
# without it; they only lose the topic boost and Laya's topics.
ENABLE_LAYA = _bool("ENABLE_LAYA", False)
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
# Cosine similarity for a face to be given a known person's name automatically (new
# clips, and spreading a name after a group is named). Was 0.42: after naming "puneet",
# 272 of 1331 faces had landed between 0.42 and 0.52, where meme frames (small, blurred,
# side-on) start confusing people. Weaker matches are left for a person to confirm.
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.5"))
# Below this a named face is shown as "unsure" in Curate > Faces, least sure first.
FACE_UNSURE_BELOW = float(os.getenv("FACE_UNSURE_BELOW", "0.55"))
# Curate > Faces groups unknown faces by similarity to a group's average face. Higher =
# purer groups but one person split into more of them (naming them alike merges them).
# Faces are linked only when each is among the other's FACE_GROUP_NEIGHBOURS closest and
# similar enough (mutual nearest neighbours). Measured on 7.7k unknown faces: 0.55 kept
# the top 30 groups one person each by title, with bigger groups than the old greedy
# method; 0.45-0.5 mixed people.
FACE_GROUP_THRESHOLD = float(os.getenv("FACE_GROUP_THRESHOLD", "0.55"))
FACE_GROUP_NEIGHBOURS = int(os.getenv("FACE_GROUP_NEIGHBOURS", "10"))
FACE_MIN_DET_SCORE = float(os.getenv("FACE_MIN_DET_SCORE", "0.6"))
FACE_MIN_SIZE = int(os.getenv("FACE_MIN_SIZE", "40"))                     # pixels

# Duplicate detection
# Re-uploads are often trimmed or padded: lengths may differ by this many seconds,
# or by DEDUP_MAX_DURATION_SHARE of the clip, whichever is more.
DEDUP_MAX_DURATION_DIFF = float(os.getenv("DEDUP_MAX_DURATION_DIFF", "1.5"))
DEDUP_MAX_DURATION_SHARE = float(os.getenv("DEDUP_MAX_DURATION_SHARE", "0.15"))
# Shorter transcripts are not compared as speech (too common). The same-clip rule itself
# (picture and speech thresholds) is media.same_clip.
DEDUP_MIN_SPEECH_CHARS = int(os.getenv("DEDUP_MIN_SPEECH_CHARS", "20"))   # shorter lines are too common

# Ranking weights
W_KEYWORD = float(os.getenv("W_KEYWORD", "0.65"))
W_SEMANTIC = float(os.getenv("W_SEMANTIC", "0.35"))
W_POPULARITY = float(os.getenv("W_POPULARITY", "0.03"))
# Shares teach search (the query becomes a way to find the clip, and shares lift the
# clip). Off until there is real traffic: early shares only echo what search already
# showed. Shares and their queries are still recorded either way, so turning this on
# later uses the history.
LEARN_FROM_SHARES = os.getenv("LEARN_FROM_SHARES", "0") == "1"
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
