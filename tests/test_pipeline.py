"""End to end: synthetic clips -> pipeline -> search -> share -> API.

Runs in order, top to bottom; later tests build on the clips earlier ones add.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

from . import conftest  # noqa: F401  (sets the environment before app is imported)

pytestmark = pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")

from app import config, db, faces, indexing, pipeline, search  # noqa: E402

WORK = Path(config.DATA_DIR) / "_fixtures"
TABLES = "query_log, clip_faces, face_refs, people, clips"


def make_clip(name: str, source: str, seconds: float = 3, extra: list[str] | None = None) -> Path:
    """Generate a small test video with ffmpeg's built-in pattern generators."""
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / name
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"{source}=size=320x240:rate=15",
         "-f", "lavfi", "-i", "sine=frequency=440",
         "-t", str(seconds), *(extra or []), "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )
    return out


def sidecar(video: Path, **meta) -> None:
    video.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def process(video: Path) -> dict:
    with db.session() as conn:
        return pipeline.process_clip(conn, video)


def clip(clip_id: int) -> dict:
    with db.session() as conn:
        return conn.execute("SELECT * FROM clips WHERE id = %s", (clip_id,)).fetchone()


def top(query: str) -> list[int]:
    with db.session() as conn:
        return [r["id"] for r in search.search(conn, query)]


STATE: dict = {}


def test_00_fresh_database():
    config.ensure_dirs()
    with db.session() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {TABLES} CASCADE")
    db.init_db()
    db.init_db()  # must be safe to run twice
    with db.session() as conn:
        faces.upsert_person(conn, "Nana Patekar", ["nana", "patekar", "vishwanath patekar"])
        faces.upsert_person(conn, "Rishi Kapoor", ["chintu"])


def test_01_new_clip_with_devanagari_transcript():
    video = make_clip("rasode.mp4", "testsrc")
    sidecar(video, title="Rasode mein kaun tha | Kokilaben meme template #viral #kokilaben",
            transcript="रसोड़े में कौन था", language="hi", source_title="Saath Nibhaana Saathiya",
            folk_names=["kokilaben meme", "cooker meme"],
            description="Stern older woman questions two nervous younger women",
            use_when=["finding out who did it"])
    result = process(video)
    assert result["status"] == "new"
    row = clip(result["clip_id"])
    STATE["rasode"] = row["id"]
    assert row["transcript_roman"] == "rasode men kaun tha"
    assert row["s_said"] == "rasode men kon ta"
    assert "kokilaben" in row["s_names"] and "viral" not in row["s_names"]
    assert row["embedding"] is not None
    assert len(row["phash"]) == config.KEYFRAMES
    assert (config.MEDIA_DIR / row["file"]).exists() and (config.MEDIA_DIR / row["thumb"]).exists()
    assert not row["needs_review"]


def test_02_exact_duplicate_is_merged():
    video = WORK / "rasode.mp4"
    sidecar(video, title="who was in the kitchen meme", hashtags=["saathiya"])
    result = process(video)
    assert result == {"status": "duplicate", "clip_id": STATE["rasode"], "match": "exact"}
    row = clip(STATE["rasode"])
    assert row["duplicates_seen"] == 2
    assert "saathiya" in row["hashtags"]
    assert "who was in the kitchen meme" in row["comments"]  # another uploader's words, kept
    assert "kitchen" in row["s_meta"]


def test_03_reencoded_copy_is_a_near_duplicate():
    # Same pictures, different bitrate and size: a different file, the same clip.
    video = make_clip("rasode_reupload.mp4", "testsrc", extra=["-vf", "scale=240:180", "-b:v", "120k"])
    result = process(video)
    assert result["status"] == "duplicate" and result["match"] == "near"
    assert clip(STATE["rasode"])["duplicates_seen"] == 3


def test_04_different_clips_stay_separate():
    video = make_clip("haath.mp4", "smptebars", seconds=4)
    sidecar(video, title="Haath jodiye please boliye - Nana Patekar Rishi Kapoor",
            transcript="हाथ जोड़िए, प्लीज बोलिए", language="hi")
    result = process(video)
    assert result["status"] == "new"
    STATE["haath"] = result["clip_id"]
    row = clip(result["clip_id"])
    # names in the title were matched against the people table
    assert sorted(row["people"]) == ["Nana Patekar", "Rishi Kapoor"]
    assert "chintu" in row["s_people"]  # aliases are searchable

    silent = make_clip("silent.mp4", "mandelbrot", seconds=2)
    result = process(silent)
    assert result["status"] == "new"
    STATE["silent"] = result["clip_id"]
    row = clip(result["clip_id"])
    assert row["needs_review"] and set(row["review_reasons"]) == {"no_speech", "no_context"}


def test_05_too_long_is_rejected():
    video = make_clip("long.mp4", "testsrc2", seconds=config.MAX_CLIP_SECONDS + 2)
    with pytest.raises(pipeline.Rejected):
        process(video)


def test_06_search_by_what_is_said():
    for typed in ["rasode me kon tha", "Rasode mein kaun tha?", "rasodey main kon thaa", "kon tha rasode"]:
        assert top(typed)[0] == STATE["rasode"], typed
    for typed in ["haath jodiye", "hath jodiye please", "please boliye"]:
        assert top(typed)[0] == STATE["haath"], typed


def test_07_search_by_name_person_and_mixed():
    assert top("kokilaben")[0] == STATE["rasode"]
    assert top("cooker meme")[0] == STATE["rasode"]
    assert top("saath nibhana saathiya")[0] == STATE["rasode"]
    assert top("nana patekar")[0] == STATE["haath"]
    assert top("rishi kapoor meme")[0] == STATE["haath"]
    assert top("chintu")[0] == STATE["haath"]                        # alias
    assert top("nana patekar haath jodiye")[0] == STATE["haath"]     # person + dialogue
    assert top("xyzzy qwfp zzkk") == []
    assert top("🔥") == []


def test_08_trending_when_query_is_empty():
    assert top("")[0] == STATE["rasode"]  # seen 3 times


def test_09_share_teaches_a_new_query():
    query = "mummy ne pakad liya"
    assert STATE["silent"] not in top(query)
    with db.session() as conn:
        assert search.record_share(conn, STATE["silent"], query)
        assert search.record_share(conn, STATE["silent"], "Mummy ne pakad liyaaa!!")  # same after normalising
        assert not search.record_share(conn, 999999, query)
    row = clip(STATE["silent"])
    assert row["shares"] == 2
    assert list(row["learned_queries"].values()) == [2]
    assert top(query)[0] == STATE["silent"]
    assert top("mumy ne pakad lia")[0] == STATE["silent"]


def test_10_failed_queries_are_logged():
    with db.session() as conn:
        rows = conn.execute("SELECT query_norm FROM query_log WHERE results = 0").fetchall()
    assert "xyjy kvfp jk" in {r["query_norm"] for r in rows}


def _face(axis: int, wobble: float = 0.0) -> list[float]:
    """A made-up unit-length face vector pointing mostly along one axis."""
    v = [0.0] * config.FACE_DIM
    v[axis] = 1.0
    v[axis + 1] = wobble
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def test_11_faces_match_cluster_and_name():
    with db.session() as conn:
        nana = faces.upsert_person(conn, "Nana Patekar")
        conn.execute("INSERT INTO face_refs (person_id, source, embedding) VALUES (%s, %s, %s::vector)",
                     (nana, "Nana Patekar/1.jpg", db.vec(_face(0))))
        # upsert without aliases must not wipe the aliases set earlier
        assert conn.execute("SELECT aliases FROM people WHERE id = %s", (nana,)).fetchone()["aliases"]

        assert faces.match_embedding(conn, _face(0, 0.3))["name"] == "Nana Patekar"
        assert faces.match_embedding(conn, _face(10)) is None

        # the silent clip shows Nana plus a stranger (twice)
        seen = faces.record_faces(conn, STATE["silent"], [[{"embedding": _face(0, 0.2)}, {"embedding": _face(20)}],
                                                          [{"embedding": _face(20, 0.2)}]])
        assert list(seen["people"]) == ["Nana Patekar"] and seen["faces"] == 3 and seen["unknown"] == 2
        # the same stranger also shows up in the rasode clip
        faces.record_faces(conn, STATE["rasode"], [[{"embedding": _face(20, 0.1)}]])

        clusters = faces.unknown_clusters(conn)
        assert len(clusters) == 1
        assert clusters[0]["faces"] == 3
        assert clusters[0]["clips"] == sorted([STATE["silent"], STATE["rasode"]])

        faces.promote(conn, clusters[0]["face_id"], "Rupal Patel")
        changed = faces.rematch_unknown(conn)
        assert changed == sorted([STATE["silent"], STATE["rasode"]])
        for clip_id in changed:
            indexing.refresh(conn, clip_id)
        assert faces.unknown_clusters(conn) == []
    assert "Rupal Patel" in clip(STATE["rasode"])["people"]
    assert top("rupal patel")[0] in (STATE["rasode"], STATE["silent"])
    assert set(top("rupal patel")[:2]) == {STATE["rasode"], STATE["silent"]}


def test_12_gallery_sync_names_only(tmp_path=None):
    root = Path(config.GALLERY_DIR)
    (root / "Paresh Rawal").mkdir(parents=True, exist_ok=True)
    (root / "Paresh Rawal" / "aliases.txt").write_text("Babu Bhaiya\nbaburao\n", encoding="utf-8")
    with db.session() as conn:
        stats = faces.sync_gallery(conn, engine=None)
        assert stats["people"] == 1 and stats["photos_added"] == 0
        assert faces.people_in_text(conn, "Babu bhaiya OP 😂 #heraPheri") == ["Paresh Rawal"]
        # "nana" alone is ambiguous (also means grandfather), so it must not match
        assert faces.people_in_text(conn, "mere nana ji") == []


def test_13_api():
    from fastapi.testclient import TestClient

    from app.api import app

    with TestClient(app) as client:
        assert client.get("/").status_code == 200

        data = client.get("/search", params={"q": "rasode me kon tha"}).json()
        hit = data["results"][0]
        assert hit["id"] == STATE["rasode"] and hit["url"].startswith("/media/") and hit["score"] > 0.5
        assert client.get(hit["url"]).status_code == 200
        assert client.get(hit["thumb"]).status_code == 200

        # a human adds what the pipeline could not know
        r = client.patch(f"/clips/{STATE['silent']}", json={
            "folk_names": ["zoom spiral"], "description": "A fractal zooms in forever",
            "use_when": ["overthinking at 3am"], "reviewed": True})
        assert r.status_code == 200 and r.json()["needs_review"] is False
        assert client.get("/search", params={"q": "overthinking"}).json()["results"][0]["id"] == STATE["silent"]
        assert client.get("/search", params={"q": "zoom spiral"}).json()["results"][0]["id"] == STATE["silent"]

        assert client.patch(f"/clips/{STATE['silent']}", json={}).status_code == 400
        assert client.patch("/clips/999999", json={"title": "x"}).status_code == 404
        assert client.get("/clips/999999").status_code == 404
        assert client.post("/clips/999999/share", json={"q": "x"}).status_code == 404
        assert client.post(f"/clips/{STATE['rasode']}/share", json={}).json() == {"ok": True}

        assert client.get("/review").json()["count"] == 0
        assert client.get("/queries/failed").json()["count"] >= 1

        video = make_clip("upload.mp4", "rgbtestsrc", seconds=2)
        with video.open("rb") as f:
            r = client.post("/upload", files={"file": ("my clip!.mp4", f, "video/mp4")},
                            data={"title": "RGB bars", "people": "Paresh Rawal"})
        assert r.status_code == 200
        queued = Path(config.INBOX_DIR) / r.json()["queued_as"]
        assert queued.exists() and queued.with_suffix(".json").exists()
        with video.open("rb") as f:
            assert client.post("/upload", files={"file": ("notes.txt", f, "text/plain")}).status_code == 400


def test_14_worker_processes_the_inbox():
    from app import worker

    inbox = Path(config.INBOX_DIR)
    (inbox / "broken.mp4").write_bytes(b"this is not a video")
    for p in inbox.iterdir():   # pretend the files finished copying a while ago
        os.utime(p, (p.stat().st_atime - 10, p.stat().st_mtime - 10))
    assert worker.run_once({}) == 2
    assert not [p for p in inbox.iterdir() if p.name != ".gitkeep"]
    assert (Path(config.FAILED_DIR) / "broken.mp4.error.txt").exists()
    assert top("rgb bars")
    row = clip(top("rgb bars")[0])
    assert row["people"] == ["Paresh Rawal"] and "babu bai" in row["s_people"]


def test_15_topics_from_laya_rank_search(monkeypatch):
    """Laya says the query is about 'work'; the clip tagged work comes first even
    though no word of the query is in it."""
    from app import laya

    with db.session() as conn:
        conn.execute("UPDATE clips SET topics = '{work}' WHERE id = %s", (STATE["silent"],))
        indexing.refresh(conn, STATE["silent"])
    query = "zzq vvx boss daanta"
    monkeypatch.setattr(laya, "query_intent", lambda q: {})
    assert STATE["silent"] not in top(query)
    monkeypatch.setattr(laya, "query_intent", lambda q: {"work": 0.8})
    with db.session() as conn:
        rows = search.search(conn, query, log_query=False)
    assert rows[0]["id"] == STATE["silent"] and rows[0]["tm"] == pytest.approx(0.8)


def test_16_sidecar_topics_are_kept_and_merged():
    assert pipeline.pipeline_topics(["Work", "late"], ["work", "money"]) == ["Work", "late", "money"]


def test_17_share_links_unfurl_as_video(monkeypatch):
    """The link a person pastes is /c/<id>; its page carries what chat apps read
    to draw the clip as a video."""
    from fastapi.testclient import TestClient

    from app import api

    with TestClient(api.app) as client:
        hit = client.get("/search", params={"q": "rasode me kon tha"}).json()["results"][0]
        share_id = hit["share_id"]
        assert len(share_id) == 10 and hit["share_url"] == f"http://testserver/c/{share_id}"

        page = client.get(f"/c/{share_id}")
        assert page.status_code == 200
        body = page.text
        assert f'<meta property="og:video" content="http://testserver/media/{hit["url"].split("/")[-1]}">' in body
        assert '<meta property="og:video:type" content="video/mp4">' in body
        assert '<meta name="twitter:card" content="player">' in body
        assert f'content="http://testserver/c/{share_id}/embed"' in body
        assert "application/json+oembed" in body

        assert client.get(f"/c/{share_id}/embed").status_code == 200
        o = client.get("/oembed", params={"url": f"http://testserver/c/{share_id}"}).json()
        assert o["type"] == "video" and f"/c/{share_id}/embed" in o["html"]

        assert client.get("/c/zzzzzzzzzz").status_code == 404
        assert client.get("/c/abc").status_code == 404
        assert client.get("/oembed", params={"url": "https://example.com/x"}).status_code == 404

        monkeypatch.setattr(config, "PUBLIC_BASE_URL", "https://clips.example.com/")
        assert client.get(f"/clips/{hit['id']}").json()["share_url"] == f"https://clips.example.com/c/{share_id}"
        assert 'content="https://clips.example.com/media/' in client.get(f"/c/{share_id}").text
