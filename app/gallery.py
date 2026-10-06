"""Command line for the people gallery.

    python -m app.gallery sync                 read data/gallery into the database
    python -m app.gallery unknown              list groups of unrecognised faces
    python -m app.gallery name <face_id> "Full Name"
                                               name one unknown face; every clip
                                               showing that face inherits the name
"""
from __future__ import annotations

import sys

from . import config, db, faces, indexing


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    command = argv[0]
    db.init_db()

    if command == "sync":
        engine = faces.get_engine() if config.ENABLE_FACES else None
        with db.session() as conn:
            stats = faces.sync_gallery(conn, engine)
            changed = faces.rematch_unknown(conn)
            for clip_id in changed:
                indexing.refresh(conn, clip_id)
        print(f"{stats['people']} people, {stats['photos_added']} new photos, "
              f"{len(changed)} clips gained a name")
        for photo in stats["photos_without_face"]:
            print(f"  no face found in: {photo}")
        return 0

    if command == "unknown":
        with db.session() as conn:
            clusters = faces.unknown_clusters(conn)
        if not clusters:
            print("no unknown faces")
        for c in clusters[:50]:
            print(f"face_id {c['face_id']:>6}  seen in {len(c['clips']):>3} clip(s): "
                  f"{', '.join(map(str, c['clips'][:8]))}")
        return 0

    if command == "name" and len(argv) == 3:
        with db.session() as conn:
            faces.promote(conn, int(argv[1]), argv[2])
            changed = faces.rematch_unknown(conn)
            for clip_id in changed:
                indexing.refresh(conn, clip_id)
        print(f"named. {len(changed)} clip(s) updated: {changed}")
        return 0

    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
