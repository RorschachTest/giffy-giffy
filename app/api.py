"""HTTP API and the small test page.

    GET   /                     the page, in the theme set by UI_THEME (?theme=<name> to preview)
    GET   /themes               the available themes
    GET   /legal                where the clips and data come from, removal requests
    GET   /search?q=...         hybrid search (empty q = the whole library, most shared first;
                                &offset= pages through it); `intent` is Laya's reading
    POST  /clips/{id}/share     count a share and learn the query that led to it
    GET   /clips/{id}           everything we know about one clip
    PATCH /clips/{id}           correct or add facts by hand
    GET   /review               clips the pipeline was unsure about
    GET   /queries/failed       searches that found nothing
    POST  /upload               drop a clip into the inbox from the browser
    GET   /media/...            the clip files
    GET   /c/{share_id}         share link: video page that chat apps unfurl (share.py)
    GET   /clips/{id}/humor     why the clip is funny on its own (understand.py)
    POST  /understand           what a meme says sent after some chat messages, and why it is funny
    POST  /understand/feedback  what it really meant / whether it landed: the engine learns from it
    GET   /understand/stats     how often the engine's reading matched people's
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, db, faces, indexing, intent, laya, search as search_mod, share as share_mod, storage, understand as und
from .embed import get_embedder

STATIC = Path(__file__).with_name("static")
THEMES = STATIC / "themes"


def theme_names() -> list[str]:
    """Every app/static/themes/<name>.css is a theme: drop a file in, it is available."""
    return sorted(p.stem for p in THEMES.glob("*.css"))

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
@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> RedirectResponse:   # browsers ask for this on pages that name no icon
    return RedirectResponse("/static/icon.svg", status_code=301)


@app.api_route("/media/{key}", methods=["GET", "HEAD"], include_in_schema=False)
def media_file(key: str, request: Request):
    """A clip or thumbnail, from MEDIA_DIR or the bucket (STORAGE); answers byte ranges for seeking."""
    return storage.serve(key, request)


app.include_router(share_mod.router)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


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


def _themed_page(name: str, theme: str | None) -> HTMLResponse:
    names = theme_names()
    chosen = next((t for t in (theme, config.UI_THEME, "sticker") if t in names), names[0])
    css = [STATIC / "base.css", THEMES / f"{chosen}.css", STATIC / name]
    version = str(int(max(f.stat().st_mtime for f in css)))   # new file -> new URL, no stale CSS
    page = (STATIC / name).read_text(encoding="utf-8")
    page = page.replace("__THEME__", chosen).replace("__VERSION__", version)
    if "__SUGGESTIONS__" in page:
        from . import suggestions
        page = page.replace("__SUGGESTIONS__", suggestions.as_script_json(suggestions.current()))
    # no-cache: browsers re-check the page on every visit, so an update shows at once
    return HTMLResponse(page, headers={"Cache-Control": "no-cache"})


@app.get("/", response_class=HTMLResponse)
def index(theme: str | None = None) -> HTMLResponse:
    return _themed_page("index.html", theme)


@app.get("/legal", response_class=HTMLResponse)
def legal(theme: str | None = None) -> HTMLResponse:
    """Where the clips and data come from, and how to ask for removal."""
    return _themed_page("legal.html", theme)


@app.get("/themes")
def themes() -> dict:
    return {"active": config.UI_THEME, "themes": theme_names()}


@app.get("/search")
def search(request: Request, q: str = "", limit: int = 24, offset: int = 0, after: str = "") -> dict:
    limit, offset = max(1, min(limit, 100)), max(0, offset)
    browsing = not q.strip()
    intent = {} if browsing else laya.query_intent(q)
    with db.session() as conn:
        try:
            rows = search_mod.search(conn, q, limit, intent=intent, offset=offset, after=after or None,
                                     log_query=offset == 0 and not after)
        except ValueError:
            raise HTTPException(400, "bad 'after' cursor") from None
        # Browsing pages through the whole library: say how big it is, once, on the first page.
        first = browsing and offset == 0 and not after
        total = conn.execute("SELECT count(*) AS n FROM clips WHERE duplicate_of IS NULL").fetchone()["n"] if first else None
    # `next` asks for the page after this one (keyset); None when this page is the last
    nxt = search_mod.browse_cursor(rows[-1]) if browsing and len(rows) == limit else None
    return {"query": q, "intent": intent, "offset": offset, "total": total, "next": nxt,
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


@app.get("/clips/{clip_id}/state")
def clip_state(clip_id: int) -> dict:
    """What a text-only decision model (Jev, Laya) would read about this clip."""
    from . import clip_state as cs

    with db.session() as conn:
        row = conn.execute("SELECT * FROM clips WHERE id = %s", (clip_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "no such clip")
    state = cs.build(row)
    return {"id": clip_id, "state": state, "thin": cs.is_thin(state)}


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


# --------------------------------------------------------------------------- #
# Faces (Curate > Faces): who appears in which clips, and naming unknown faces
# --------------------------------------------------------------------------- #

def _crop_url(key: str | None) -> str | None:
    return f"/media/{key}" if key else None


@app.get("/faces")
def face_overview(limit: int = 120) -> dict:
    """Named people, then groups of unknown faces that probably are one person each
    (most clips first). A group's face_ids are what /faces/clips and /faces/name take."""
    with db.session() as conn:
        people = faces.known_people(conn)
        every = faces.face_groups(conn)
        groups = every[:max(1, min(limit, 500))]
        # Faces stored before face pictures existed have none: show their clips' thumbnails
        # instead (no reprocessing needed; reindex --face-crops can add close-ups later).
        want = {c for g in groups for c in g["clips"][:4]}
        thumbs = {r["id"]: r["thumb"] for r in conn.execute(
            "SELECT id, thumb FROM clips WHERE id = ANY(%s)", (list(want),)).fetchall()}
    for g in groups:
        g["pictures"] = g["crops"] or [thumbs.get(c) for c in g["clips"][:4] if thumbs.get(c)]
    return {
        "people": [{"id": p["id"], "name": p["name"], "aliases": p["aliases"], "clips": p["clips"],
                    "picture": _crop_url(p["crop"] or p["thumb"])} for p in people],
        "groups": [{"id": g["id"], "face_ids": g["face_ids"], "clips": len(g["clips"]),
                    "faces": len(g["face_ids"]), "close_up": bool(g["crops"]),
                    "pictures": [_crop_url(k) for k in g["pictures"]]}
                   for g in groups],
        "unknown_groups": len(every),
    }


@app.get("/faces/clips")
def face_clips(request: Request, face_ids: str = "", person_id: int | None = None) -> dict:
    """The clips a group of faces (comma-separated ids) or a named person appears in."""
    ids = [int(x) for x in face_ids.split(",") if x.strip().isdigit()][:2000]
    if not ids and person_id is None:
        raise HTTPException(400, "give face_ids or person_id")
    with db.session() as conn:
        if person_id is not None:
            # a person's clips, least sure first: the ones worth checking come up top
            rows = conn.execute(
                f"""SELECT {search_mod.COLUMNS}, NULL::real AS kw, NULL::real AS sem, NULL::real AS score,
                           m.face_score
                      FROM clips JOIN (SELECT clip_id, max(score) AS face_score FROM clip_faces
                                        WHERE person_id = %(pid)s GROUP BY clip_id) m ON m.clip_id = clips.id
                     WHERE duplicate_of IS NULL
                     ORDER BY m.face_score, {search_mod.BROWSE_ORDER}""",
                {"pid": person_id},
            ).fetchall()
        else:
            rows = conn.execute(
                f"""SELECT {search_mod.COLUMNS}, NULL::real AS kw, NULL::real AS sem, NULL::real AS score,
                           NULL::real AS face_score
                      FROM clips WHERE duplicate_of IS NULL
                       AND id IN (SELECT clip_id FROM clip_faces WHERE id = ANY(%(ids)s))
                     ORDER BY {search_mod.BROWSE_ORDER}""",
                {"ids": ids},
            ).fetchall()
    out = []
    for r in rows:
        item = _public(r, request)
        item["face_score"] = round(float(r["face_score"]), 3) if r["face_score"] is not None else None
        item["unsure"] = r["face_score"] is not None and r["face_score"] < config.FACE_UNSURE_BELOW
        out.append(item)
    return {"count": len(out), "unsure": sum(1 for o in out if o["unsure"]), "results": out}


class NameFacesBody(BaseModel):
    face_ids: list[int]
    name: str
    aliases: list[str] = []       # other names for the same person: "modi", "PM"


@app.post("/faces/name")
def name_face_group(body: NameFacesBody) -> dict:
    """Say who a group of faces is. The same name on two groups merges them into one person."""
    with db.session() as conn:
        try:
            person_id, changed = faces.name_faces(conn, body.face_ids, body.name, body.aliases)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        for clip_id in changed:
            indexing.refresh(conn, clip_id, embed=False)
    return {"person_id": person_id, "name": body.name.strip(), "clips_updated": len(changed)}


class RemovePersonBody(BaseModel):
    clip_id: int


@app.post("/people/{person_id}/remove")
def remove_person_from_clip(person_id: int, body: RemovePersonBody) -> dict:
    """"Not them": the person's name leaves this clip and is never matched back to its faces."""
    with db.session() as conn:
        try:
            name = faces.remove_person(conn, person_id, body.clip_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from None
        indexing.refresh(conn, body.clip_id, embed=False)
    return {"person_id": person_id, "name": name, "clip_id": body.clip_id}


class AliasesBody(BaseModel):
    aliases: list[str]


@app.post("/people/{person_id}/aliases")
def add_person_aliases(person_id: int, body: AliasesBody) -> dict:
    """More names for a known person; every clip of theirs becomes findable by them."""
    with db.session() as conn:
        try:
            aliases = faces.add_aliases(conn, person_id, body.aliases)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from None
        name = conn.execute("SELECT name FROM people WHERE id = %s", (person_id,)).fetchone()["name"]
        changed = faces.clips_of(conn, name)
        for clip_id in changed:
            indexing.refresh(conn, clip_id, embed=False)
    return {"person_id": person_id, "aliases": aliases, "clips_updated": len(changed)}


@app.get("/review")
def review(request: Request, limit: int = 50) -> dict:
    with db.session() as conn:
        rows = conn.execute(
            f"""SELECT {search_mod.COLUMNS}, NULL::real AS kw, NULL::real AS sem, NULL::real AS score
                  FROM clips WHERE needs_review AND duplicate_of IS NULL ORDER BY first_seen DESC LIMIT %s""",
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


class UnderstandBody(BaseModel):
    clip_id: int | None = None
    link: str | None = None            # a share link (/c/<id>) or bare share id, as sent in a chat
    context: list[str] = []            # the messages before the meme, oldest first
    caption: str = ""                  # what the sender typed with it


class FeedbackBody(UnderstandBody):
    intent: str | None = None          # what it really meant: one of GET /understand/intents
    funny: bool | None = None          # did it land


def _clip_for(conn, body: UnderstandBody) -> dict:
    if body.clip_id is None and not body.link:
        raise HTTPException(400, "give clip_id or link")
    clip = und.find_clip(conn, body.clip_id, body.link)
    if clip is None:
        raise HTTPException(404, "no such clip")
    return clip


@app.get("/clips/{clip_id}/humor")
def clip_humor(clip_id: int) -> dict:
    with db.session() as conn:
        clip = und.find_clip(conn, clip_id)
        if clip is None:
            raise HTTPException(404, "no such clip")
        return und.understand(conn, clip)


@app.post("/understand")
def understand(body: UnderstandBody) -> dict:
    with db.session() as conn:
        return und.understand(conn, _clip_for(conn, body), body.context, body.caption)


@app.get("/understand/intents")
def intents() -> dict:
    return {"intents": {k: v["gloss"] for k, v in intent.INTENTS.items()}}


@app.post("/understand/feedback")
def understand_feedback(body: FeedbackBody) -> dict:
    with db.session() as conn:
        clip = _clip_for(conn, body)
        try:
            return und.record_feedback(conn, clip, body.context, body.caption, body.intent, body.funny)
        except ValueError as exc:
            raise HTTPException(400, str(exc))


@app.get("/understand/stats")
def understand_stats() -> dict:
    with db.session() as conn:
        return und.accuracy(conn)
