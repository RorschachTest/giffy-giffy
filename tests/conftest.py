"""Test settings. Must be imported before anything from `app`.

The integration tests need a Postgres with the pgvector and pg_trgm extensions
and ffmpeg on the PATH. Point TEST_DATABASE_URL at a scratch database:

    docker compose up -d db
    TEST_DATABASE_URL=postgresql://memeclip:memeclip@localhost:5432/memeclip pytest

(expose the db port first, or run pytest inside the api container). Without
TEST_DATABASE_URL only the text tests run. The tests DROP the app's tables.

No models are downloaded: speech to text and faces are switched off and a
spelling-only stand-in replaces the embedding model.
"""
import os
import tempfile

os.environ["EMBED_BACKEND"] = "hash"
os.environ["ENABLE_STT"] = "0"
os.environ["ENABLE_FACES"] = "0"
os.environ["ENABLE_VIBE"] = "0"
os.environ["LAYA_URL"] = ""   # tests that need Laya start a fake one
os.environ["PUBLIC_BASE_URL"] = ""   # links use the test server, not the real domain in .env
os.environ["STORAGE"] = "local"    # tests never talk to the real bucket
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="memeclip_test_data_"))
os.environ.setdefault("MODELS_DIR", tempfile.mkdtemp(prefix="memeclip_test_models_"))
if os.getenv("TEST_DATABASE_URL"):
    os.environ["DATABASE_URL"] = os.environ["TEST_DATABASE_URL"]
