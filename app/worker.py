"""Watches the inbox folder and processes whatever lands in it.

    python -m app.worker           run forever
    python -m app.worker --once    process what is there, then exit
"""
from __future__ import annotations

import logging
import sys
import time
import traceback

from . import config, db, faces, indexing, media, pipeline

log = logging.getLogger("worker")


def sync_gallery_if_changed(state: dict) -> None:
    signature = faces.gallery_signature()
    if signature == state.get("gallery"):
        return
    engine = faces.get_engine() if config.ENABLE_FACES else None
    with db.session() as conn:
        stats = faces.sync_gallery(conn, engine)
        changed = faces.rematch_unknown(conn) if stats["photos_added"] else []
        for clip_id in changed:
            indexing.refresh(conn, clip_id)
    state["gallery"] = signature
    log.info("gallery: %d people, %d new photos, %d old clips gained a name",
             stats["people"], stats["photos_added"], len(changed))
    for photo in stats["photos_without_face"]:
        log.warning("gallery: no face found in %s", photo)


def pending() -> list:
    now = time.time()
    files = [
        p for p in config.INBOX_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in config.VIDEO_EXTENSIONS
        and now - p.stat().st_mtime > 2     # still being copied? leave it for the next pass
    ]
    return sorted(files, key=lambda p: p.stat().st_mtime)


def run_once(state: dict) -> int:
    sync_gallery_if_changed(state)
    done = 0
    for video in pending():
        try:
            with db.session() as conn:
                result = pipeline.process_clip(conn, video)
            pipeline.move_aside(video, config.PROCESSED_DIR)
            log.info("%s -> %s", video.name, result)
        except (pipeline.Rejected, media.MediaError) as exc:
            pipeline.move_aside(video, config.FAILED_DIR, str(exc))
            log.warning("%s rejected: %s", video.name, exc)
        except Exception:
            pipeline.move_aside(video, config.FAILED_DIR, traceback.format_exc())
            log.exception("%s failed", video.name)
        done += 1
    return done


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config.ensure_dirs()
    db.init_db()
    state: dict = {}
    if "--once" in sys.argv:
        print(f"processed {run_once(state)} file(s)")
        return
    log.info("watching %s", config.INBOX_DIR)
    while True:
        try:
            run_once(state)
        except Exception:
            log.exception("pass failed; will retry")
        time.sleep(config.POLL_SECONDS)


if __name__ == "__main__":
    main()
