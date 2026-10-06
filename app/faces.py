"""Recognising famous people, the same way a photo library groups faces.

No training is involved. A pretrained model turns any face into 512 numbers; two
photos of the same person give nearly the same numbers. So "adding a celebrity"
means saving the numbers for a few of their photos, and "recognising" means
finding the nearest saved face.

Only the FaceEngine class needs the model. Everything else is plain database work.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from . import config
from .db import vec
from .text import normalise

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class FaceEngine:
    def __init__(self) -> None:
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name=config.FACE_MODEL,
            root=str(config.MODELS_DIR / "insightface"),
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def detect(self, image: Path) -> list[dict]:
        """Every usable face in the picture, biggest first."""
        import cv2

        img = cv2.imread(str(image))
        if img is None:
            return []
        found = []
        for f in self.app.get(img):
            x1, y1, x2, y2 = (float(v) for v in f.bbox)
            size = min(x2 - x1, y2 - y1)
            if f.det_score < config.FACE_MIN_DET_SCORE or size < config.FACE_MIN_SIZE:
                continue
            found.append({
                "embedding": [float(x) for x in f.normed_embedding],
                "det_score": float(f.det_score),
                "size": size,
                "bbox": [x1, y1, x2, y2],
            })
        return sorted(found, key=lambda d: -d["size"])


@lru_cache(maxsize=1)
def get_engine() -> FaceEngine:
    return FaceEngine()


# --------------------------------------------------------------------------- #
# Gallery: data/gallery/<Person Name>/<any photos>  (+ optional aliases.txt)
# --------------------------------------------------------------------------- #

def gallery_signature(root: Path | None = None) -> tuple:
    """Changes whenever a file is added, removed or edited under the gallery."""
    root = root or config.GALLERY_DIR
    if not root.exists():
        return ()
    return tuple(sorted((str(p.relative_to(root)), p.stat().st_mtime_ns)
                        for p in root.rglob("*") if p.is_file()))


def upsert_person(conn, name: str, aliases: list[str] | None = None) -> int:
    row = conn.execute(
        """INSERT INTO people (name, aliases) VALUES (%s, %s)
           ON CONFLICT (name) DO UPDATE
             SET aliases = CASE WHEN %s THEN EXCLUDED.aliases ELSE people.aliases END
           RETURNING id""",
        (name, aliases or [], aliases is not None),
    ).fetchone()
    return row["id"]


def sync_gallery(conn, engine: FaceEngine | None, root: Path | None = None) -> dict:
    """Read the gallery folder into the database. Safe to run again and again.

    With engine=None only names and aliases are synced (enough for matching
    names in captions); photos are skipped.
    """
    root = root or config.GALLERY_DIR
    stats = {"people": 0, "photos_added": 0, "photos_without_face": []}
    if not root.exists():
        return stats
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = person_dir.name.strip()
        alias_file = person_dir / "aliases.txt"
        aliases = ([a.strip() for a in alias_file.read_text(encoding="utf-8").splitlines() if a.strip()]
                   if alias_file.exists() else [])
        person_id = upsert_person(conn, name, aliases)
        stats["people"] += 1
        if engine is None:
            continue
        known = {r["source"] for r in conn.execute(
            "SELECT source FROM face_refs WHERE person_id = %s", (person_id,))}
        for photo in sorted(person_dir.iterdir()):
            source = f"{name}/{photo.name}"
            if photo.suffix.lower() not in IMAGE_EXTENSIONS or source in known:
                continue
            faces = engine.detect(photo)
            if not faces:
                stats["photos_without_face"].append(source)
                continue
            # Reference photos should show one person; if not, take the biggest face.
            conn.execute(
                "INSERT INTO face_refs (person_id, source, embedding) VALUES (%s, %s, %s::vector)",
                (person_id, source, vec(faces[0]["embedding"])),
            )
            stats["photos_added"] += 1
    return stats


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #

def match_embedding(conn, embedding: list[float]) -> dict | None:
    """Nearest known person for one face, or None if nobody is close enough."""
    row = conn.execute(
        """SELECT p.id AS person_id, p.name, 1 - (r.embedding <=> %(e)s::vector) AS score
             FROM face_refs r JOIN people p ON p.id = r.person_id
            ORDER BY r.embedding <=> %(e)s::vector
            LIMIT 1""",
        {"e": vec(embedding)},
    ).fetchone()
    if row and row["score"] >= config.FACE_MATCH_THRESHOLD:
        return row
    return None


def record_faces(conn, clip_id: int, faces_per_frame: list[list[dict]]) -> dict:
    """Store every face seen in the clip and return who was recognised.

    Returns {"people": {name: best_score}, "faces": n, "unknown": n}
    """
    people: dict[str, float] = {}
    total = unknown = 0
    for frame_no, faces in enumerate(faces_per_frame):
        for face in faces:
            total += 1
            hit = match_embedding(conn, face["embedding"])
            if hit:
                people[hit["name"]] = max(people.get(hit["name"], 0.0), float(hit["score"]))
            else:
                unknown += 1
            conn.execute(
                """INSERT INTO clip_faces (clip_id, frame, person_id, score, embedding)
                   VALUES (%s, %s, %s, %s, %s::vector)""",
                (clip_id, frame_no, hit["person_id"] if hit else None,
                 hit["score"] if hit else None, vec(face["embedding"])),
            )
    return {"people": people, "faces": total, "unknown": unknown}


def people_in_text(conn, text: str) -> list[str]:
    """Known people whose name or alias appears in a caption, title or comment."""
    haystack = f" {normalise(text)} "
    found = []
    for row in conn.execute("SELECT name, aliases FROM people"):
        for label in [row["name"], *row["aliases"]]:
            needle = normalise(label)
            # A lone short word is too ambiguous ("nana" also means grandfather).
            if " " not in needle and len(needle) < 5:
                continue
            if f" {needle} " in haystack:
                found.append(row["name"])
                break
    return found


def aliases_for(conn, names: list[str]) -> list[str]:
    if not names:
        return []
    rows = conn.execute("SELECT aliases FROM people WHERE name = ANY(%s)", (names,)).fetchall()
    return [a for r in rows for a in r["aliases"]]


# --------------------------------------------------------------------------- #
# Unknown faces: group them, name a group once, every clip inherits the name
# --------------------------------------------------------------------------- #

def _parse_vec(text: str) -> list[float]:
    return [float(x) for x in text.strip("[]").split(",")]


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # embeddings are unit length


def unknown_clusters(conn, threshold: float = 0.5) -> list[dict]:
    """Greedy grouping of unrecognised faces. Biggest groups first: those are
    the people worth naming."""
    rows = conn.execute(
        "SELECT id, clip_id, embedding::text AS e FROM clip_faces WHERE person_id IS NULL ORDER BY id"
    ).fetchall()
    clusters: list[dict] = []
    for r in rows:
        e = _parse_vec(r["e"])
        best, best_sim = None, threshold
        for c in clusters:
            sim = _cos(e, c["centre"])
            if sim >= best_sim:
                best, best_sim = c, sim
        if best is None:
            clusters.append({"face_id": r["id"], "centre": e, "faces": 1, "clips": {r["clip_id"]}})
        else:
            best["faces"] += 1
            best["clips"].add(r["clip_id"])
    return [
        {"face_id": c["face_id"], "faces": c["faces"], "clips": sorted(c["clips"])}
        for c in sorted(clusters, key=lambda c: -len(c["clips"]))
    ]


def promote(conn, clip_face_id: int, name: str) -> int:
    """Name an unknown face. It becomes a reference photo for that person."""
    row = conn.execute(
        "SELECT embedding::text AS e FROM clip_faces WHERE id = %s", (clip_face_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"no face with id {clip_face_id}")
    person_id = upsert_person(conn, name)
    conn.execute(
        """INSERT INTO face_refs (person_id, source, embedding) VALUES (%s, %s, %s::vector)
           ON CONFLICT DO NOTHING""",
        (person_id, f"clip_face:{clip_face_id}", row["e"]),
    )
    return person_id


def rematch_unknown(conn) -> list[int]:
    """Try every unknown face against the gallery again. Returns the clips that changed.

    Run after adding people or photos, so old clips pick up the new names.
    """
    rows = conn.execute(
        "SELECT id, clip_id, embedding::text AS e FROM clip_faces WHERE person_id IS NULL"
    ).fetchall()
    gained: dict[int, set[str]] = defaultdict(set)
    for r in rows:
        hit = match_embedding(conn, _parse_vec(r["e"]))
        if not hit:
            continue
        conn.execute("UPDATE clip_faces SET person_id = %s, score = %s WHERE id = %s",
                     (hit["person_id"], hit["score"], r["id"]))
        gained[r["clip_id"]].add(hit["name"])
    for clip_id, names in gained.items():
        conn.execute(
            """UPDATE clips
                  SET people = ARRAY(SELECT DISTINCT unnest(people || %s::text[])),
                      review_reasons = array_remove(review_reasons, 'unknown_faces'),
                      needs_review = cardinality(array_remove(review_reasons, 'unknown_faces')) > 0
                WHERE id = %s""",
            (sorted(names), clip_id),
        )
    return sorted(gained)
