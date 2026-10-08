"""HTTP API and the small test page.

    GET   /                     test page
    GET   /search?q=...         hybrid search (empty q = trending); `intent` is Laya's reading
    POST  /clips/{id}/share     count a share and learn the query that led to it
    GET   /clips/{id}           everything we know about one clip
    PATCH /clips/{id}           correct or add facts by hand
    GET   /review               clips the pipeline was unsure about
    GET   /queries/failed       searches that found nothing
    POST  /upload               drop a clip into the inbox from the browser
    GET   /media/...            the clip files
    GET   /c/{share_id}         share link: video page that chat apps unfurl (share.py)
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, db, indexing, laya, search as search_mod, share as share_mod
from .embed import get_embedder

STATIC = Path(__file__).with_name("static")

# Facts a human may correct. Anything else on the row is derived or measured.
EDITABLE_TEXT = {"title", "caption", "transcript_roman", "source_title", "description"}
EDITABLE_LISTS = {"folk_names", "people", "reactions", "use_when", "topics", "hashtags"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    db.init_db()
    get_embedder()  # load the model now, not on the first search
    yield


app = FastAPI(title="memeclip", lifespan=lifespan)
config.ensure_dirs()
app.mount("/media", StaticFiles(directory=str(config.MEDIA_DIR)), name="media")
app.include_router(share_mod.router)


def _public(row: dict, request: Request | None = None) -> dict:
    out = dict(row)
    if out.get("share_id"):
        out["share_url"] = share_mod.share_url(share_mod.public_base(request), out["share_id"])
    out["url"] = f"/media/{out.pop('file')}"
    out["thumb"] = f"/media/{out['thumb']}" if out.get("thumb") else None
    for key in ("kw", "sem", "tm", "score"):
        if out.get(key) is not None:
            out[key] = round(float(out[key]), 3)
    return out


@app.get("/")
def index() -> FileResponse:
    # no-cache: browsers re-check the page on every visit, so an update shows at once
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/search")
def search(request: Request, q: str = "", limit: int = 24) -> dict:
    limit = max(1, min(limit, 100))
    with db.session() as conn:
        rows = search_mod.search(conn, q, limit)
    return {"query": q, "intent": laya.query_intent(q) if q.strip() else {},  # cached: no second call
            "count": len(rows), "results": [_public(r, request) for r in rows]}


class ShareBody(BaseModel):
    q: str | None = None


@app.post("/clips/{clip_id}/share")
def share(clip_id: int, body: ShareBody) -> dict:
    with db.session() as conn:
        if not search_mod.record_share(conn, clip_id, body.q):
            raise HTTPException(404, "no such clip")
    return {"ok": True}


@app.get("/clips/{clip_id}")
def get_clip(clip_id: int, request: Request) -> dict:
    with db.session() as conn:
        row = conn.execute(
            """SELECT id, left(sha256, 10) AS share_id, file, thumb, duration, width, height, has_audio, language,
                      transcript_native, transcript_roman, title, caption, hashtags, comments,
                      source_urls, folk_names, people, source_title, description, reactions,
                      use_when, topics, learned_queries, s_names, s_said, s_people, s_meta, s_learned,
                      duplicates_seen, shares, first_seen, needs_review, review_reasons, raw
                 FROM clips WHERE id = %s""",
            (clip_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "no such clip")
    return _public(row, request)


class ClipPatch(BaseModel):
    title: str | None = None
    caption: str | None = None
    transcript_roman: str | None = None
    source_title: str | None = None
    description: str | None = None
    folk_names: list[str] | None = None
    people: list[str] | None = None
    reactions: list[str] | None = None
    use_when: list[str] | None = None
    topics: list[str] | None = None
    hashtags: list[str] | None = None
    reviewed: bool | None = None   # true clears the needs-review flag


@app.patch("/clips/{clip_id}")
def patch_clip(clip_id: int, patch: ClipPatch, request: Request) -> dict:
    changes = patch.model_dump(exclude_none=True)
    reviewed = changes.pop("reviewed", None)
    updates: dict = {}
    for key, value in changes.items():
        if key in EDITABLE_TEXT:
            updates[key] = value.strip()
        elif key in EDITABLE_LISTS:
            updates[key] = [v.strip() for v in value if v.strip()]
    if reviewed:
        updates["needs_review"] = False
        updates["review_reasons"] = []
    if not updates:
        raise HTTPException(400, "nothing to change")
    sets = ", ".join(f"{k} = %({k})s" for k in updates)
    with db.session() as conn:
        done = conn.execute(f"UPDATE clips SET {sets} WHERE id = %(id)s",
                            {**updates, "id": clip_id}).rowcount
        if not done:
            raise HTTPException(404, "no such clip")
        indexing.refresh(conn, clip_id)
    return get_clip(clip_id, request)


@app.get("/review")
def review(request: Request, limit: int = 50) -> dict:
    with db.session() as conn:
        rows = conn.execute(
            f"""SELECT {search_mod.COLUMNS}, NULL::real AS kw, NULL::real AS sem, NULL::real AS score
                  FROM clips WHERE needs_review ORDER BY first_seen DESC LIMIT %s""",
            (max(1, min(limit, 200)),),
        ).fetchall()
    return {"count": len(rows), "results": [_public(r, request) for r in rows]}


@app.get("/queries/failed")
def failed_queries(limit: int = 50) -> dict:
    """What people looked for and did not find: your to-do list for content and fields."""
    with db.session() as conn:
        rows = conn.execute(
            """SELECT query_norm AS query, count(*) AS times, max(at) AS last_seen
                 FROM query_log WHERE results = 0
                GROUP BY query_norm ORDER BY times DESC, last_seen DESC LIMIT %s""",
            (max(1, min(limit, 500)),),
        ).fetchall()
    return {"count": len(rows), "queries": rows}


@app.post("/upload")
def upload(
    file: UploadFile = File(...),
    title: str = Form(""),
    caption: str = Form(""),
    language: str = Form(""),
    folk_names: str = Form(""),
    people: str = Form(""),
    source_title: str = Form(""),
    source_url: str = Form(""),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in config.VIDEO_EXTENSIONS:
        raise HTTPException(400, f"unsupported file type {suffix or '(none)'}")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(file.filename or "clip").stem)[:60] or "clip"
    stem = f"{stem}_{uuid.uuid4().hex[:8]}"
    meta = {k: v for k, v in {
        "title": title, "caption": caption, "language": language, "folk_names": folk_names,
        "people": people, "source_title": source_title, "source_url": source_url,
    }.items() if v.strip()}
    # Sidecar first, video last: the worker only looks for videos, so it can
    # never pick up a clip whose metadata has not landed yet.
    if meta:
        (config.INBOX_DIR / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    partial = config.INBOX_DIR / f"{stem}{suffix}.part"
    with partial.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    partial.rename(config.INBOX_DIR / f"{stem}{suffix}")
    return {"ok": True, "queued_as": f"{stem}{suffix}"}
