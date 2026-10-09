"""Where the web-ready clips and thumbnails live, and how they are served.

    STORAGE=local   files in MEDIA_DIR (the default; what ran before)
    STORAGE=s3      objects in an S3 bucket: Garage on this machine (scripts/setup_garage.sh),
                    or Cloudflare R2 / AWS S3 later by changing S3_ENDPOINT and the keys

Either way the app serves them at /media/<key>, so share links, the page and the
extension never change. The key is the file name (first 16 hex characters of the
SHA-256 + .mp4 / .jpg), so an object never changes once written and can be cached
forever. Video players ask for byte ranges to seek; both backends answer them.

Copying to the bucket never deletes the local files: python -m app.storage --upload-all
"""
from __future__ import annotations

import mimetypes
import re
import sys
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

from . import config

KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")   # a bare file name: no folders, no ".."
CHUNK = 256 * 1024
FOREVER = "public, max-age=31536000, immutable"            # keys are content hashes


def _check(key: str) -> str:
    if not KEY.match(key) or ".." in key:
        raise HTTPException(404, "no such file")
    return key


def _type(key: str) -> str:
    return mimetypes.guess_type(key)[0] or "application/octet-stream"


@lru_cache(maxsize=1)
def _s3():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3", endpoint_url=config.S3_ENDPOINT, region_name=config.S3_REGION,
        aws_access_key_id=config.S3_ACCESS_KEY, aws_secret_access_key=config.S3_SECRET_KEY,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                      retries={"max_attempts": 3}),
    )


def is_s3() -> bool:
    return config.STORAGE == "s3"


# --------------------------------------------------------------------------- #
# Writing (pipeline)
# --------------------------------------------------------------------------- #

def save(path: Path) -> str:
    """A file the pipeline just wrote into MEDIA_DIR becomes a stored object; returns its key.
    With s3 it is uploaded and the local copy removed (the bucket is the only copy)."""
    key = path.name
    if is_s3():
        _s3().upload_file(str(path), config.S3_BUCKET, key, ExtraArgs={"ContentType": _type(key)})
        path.unlink(missing_ok=True)
    return key


def delete(key: str) -> None:
    """Undo a save, for a clip that failed halfway."""
    if is_s3():
        _s3().delete_object(Bucket=config.S3_BUCKET, Key=key)
    else:
        (config.MEDIA_DIR / key).unlink(missing_ok=True)


@contextmanager
def local_copy(key: str) -> Iterator[Path]:
    """A path to read the file from (ffmpeg needs one): the file itself, or a temporary download."""
    if not is_s3():
        yield config.MEDIA_DIR / key
        return
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / key
        _s3().download_file(config.S3_BUCKET, key, str(dest))
        yield dest


# --------------------------------------------------------------------------- #
# Serving (api)
# --------------------------------------------------------------------------- #

def serve(key: str, request: Request) -> Response:
    key = _check(key)
    if not is_s3():
        path = config.MEDIA_DIR / key
        if not path.is_file():
            raise HTTPException(404, "no such file")
        return FileResponse(path, media_type=_type(key), headers={"Cache-Control": FOREVER})   # handles Range
    return _serve_s3(key, request)


def _serve_s3(key: str, request: Request) -> Response:
    from botocore.exceptions import ClientError

    wanted = request.headers.get("range")
    try:
        if request.method == "HEAD":
            meta = _s3().head_object(Bucket=config.S3_BUCKET, Key=key)
            return Response(headers={"Content-Length": str(meta["ContentLength"]), "Accept-Ranges": "bytes",
                                     "Content-Type": meta.get("ContentType") or _type(key),
                                     "Cache-Control": FOREVER})
        obj = _s3().get_object(Bucket=config.S3_BUCKET, Key=key, **({"Range": wanted} if wanted else {}))
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404", "NotFound"):
            raise HTTPException(404, "no such file") from None
        if code == "InvalidRange":
            raise HTTPException(416, "range not satisfiable") from None
        raise
    headers = {"Content-Length": str(obj["ContentLength"]), "Accept-Ranges": "bytes", "Cache-Control": FOREVER}
    if obj.get("ContentRange"):
        headers["Content-Range"] = obj["ContentRange"]
    return StreamingResponse(obj["Body"].iter_chunks(CHUNK), status_code=206 if wanted else 200,
                             media_type=obj.get("ContentType") or _type(key), headers=headers)


# --------------------------------------------------------------------------- #
# Copying what is already on disk into the bucket
# --------------------------------------------------------------------------- #

def upload_all() -> int:
    """Copy every file in MEDIA_DIR to the bucket (skipping ones already there). Keeps the files."""
    from botocore.exceptions import ClientError

    done = 0
    for path in sorted(config.MEDIA_DIR.iterdir()):
        if not path.is_file() or not KEY.match(path.name):
            continue
        try:
            _s3().head_object(Bucket=config.S3_BUCKET, Key=path.name)
            continue                                  # already there
        except ClientError:
            pass
        _s3().upload_file(str(path), config.S3_BUCKET, path.name, ExtraArgs={"ContentType": _type(path.name)})
        done += 1
    return done


if __name__ == "__main__":
    if "--upload-all" not in sys.argv:
        sys.exit("usage: python -m app.storage --upload-all")
    print(f"uploaded {upload_all()} file(s) from {config.MEDIA_DIR} to bucket {config.S3_BUCKET!r}; local files kept")
