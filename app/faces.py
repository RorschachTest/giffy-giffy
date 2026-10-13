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


def record_faces(conn, clip_id: int, faces_per_frame: list[list[dict]],
                 frames: list[Path] | None = None) -> dict:
    """Store every face seen in the clip and return who was recognised. With the
    keyframes given, a small picture of each face is saved too (for the Faces view).

    Returns {"people": {name: best_score}, "faces": n, "unknown": n, "crops": [storage keys]}
    """
    people: dict[str, float] = {}
    total = unknown = 0
    crops: list[str] = []
    for frame_no, faces in enumerate(faces_per_frame):
        for face in faces:
            total += 1
            hit = match_embedding(conn, face["embedding"])
            if hit:
                people[hit["name"]] = max(people.get(hit["name"], 0.0), float(hit["score"]))
            else:
                unknown += 1
            face_id = conn.execute(
                """INSERT INTO clip_faces (clip_id, frame, person_id, score, embedding)
                   VALUES (%s, %s, %s, %s, %s::vector) RETURNING id""",
                (clip_id, frame_no, hit["person_id"] if hit else None,
                 hit["score"] if hit else None, vec(face["embedding"])),
            ).fetchone()["id"]
            if frames and face.get("bbox"):
                key = save_crop(conn, face_id, frames[frame_no], face["bbox"])
                if key:
                    crops.append(key)
    return {"people": people, "faces": total, "unknown": unknown, "crops": crops}


CROP_SIZE = 160      # px, square
CROP_MARGIN = 0.35   # extra room around the detected box, so hair and chin show


def save_crop(conn, face_id: int, frame: Path, bbox: list[float]) -> str | None:
    """Cut the face out of its keyframe, store it as face-<id>.jpg, remember the key.
    A failure only means the face has no picture; it never fails the clip."""
    from PIL import Image

    from . import storage
    try:
        with Image.open(frame) as img:
            x1, y1, x2, y2 = bbox
            side = max(x2 - x1, y2 - y1) * (1 + CROP_MARGIN)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            box = (max(0, cx - side / 2), max(0, cy - side / 2),
                   min(img.width, cx + side / 2), min(img.height, cy + side / 2))
            face = img.convert("RGB").crop(tuple(int(v) for v in box)).resize((CROP_SIZE, CROP_SIZE))
            path = config.MEDIA_DIR / f"face-{face_id}.jpg"
            face.save(path, "JPEG", quality=85)
        key = storage.save(path)
        conn.execute("UPDATE clip_faces SET crop = %s WHERE id = %s", (key, face_id))
        return key
    except Exception as exc:   # noqa: BLE001
        import logging
        logging.getLogger("faces").warning("no face picture for face %s: %s", face_id, exc)
        return None


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


def add_aliases(conn, person_id: int, aliases: list[str]) -> list[str]:
    """Other names for the same person ("modi", "PM"), added to the ones already there.
    Search finds every clip of the person by any of them. Returns the full list."""
    clean = [a.strip() for a in aliases if a and a.strip()]
    row = conn.execute(
        """UPDATE people SET aliases = ARRAY(SELECT DISTINCT unnest(aliases || %s::text[]) ORDER BY 1)
            WHERE id = %s RETURNING name, aliases""", (clean, person_id)).fetchone()
    if row is None:
        raise ValueError(f"no person with id {person_id}")
    return [a for a in row["aliases"] if a.lower() != row["name"].lower()]


def clips_of(conn, name: str) -> list[int]:
    return [r["id"] for r in conn.execute("SELECT id FROM clips WHERE %s = ANY(people)", (name,)).fetchall()]


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


def face_groups(conn, threshold: float | None = None) -> list[dict]:
    """Unknown faces grouped into probable people, like a photo app's "People".

    Greedy and quick: each face joins the closest group whose AVERAGE face is within
    `threshold`, and that average moves as faces join (so a group follows its
    person instead of its first photo). The same person can still end up as two
    groups; naming both with one name merges them (name_faces).
    """
    import numpy as np

    threshold = config.FACE_GROUP_THRESHOLD if threshold is None else threshold
    rows = conn.execute(
        """SELECT id, clip_id, crop, embedding::text AS e FROM clip_faces
            WHERE person_id IS NULL ORDER BY id"""
    ).fetchall()
    if not rows:
        return []
    vecs = np.array([_parse_vec(r["e"]) for r in rows], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    sums: list[np.ndarray] = []      # running sum per group; its direction is the average face
    members: list[list[int]] = []
    centres = np.zeros((0, vecs.shape[1]), dtype=np.float32)
    for i, v in enumerate(vecs):
        if len(sums):
            sims = centres @ v
            best = int(sims.argmax())
            if sims[best] >= threshold:
                members[best].append(i)
                sums[best] += v
                centres[best] = sums[best] / (np.linalg.norm(sums[best]) + 1e-9)
                continue
        sums.append(v.copy())
        members.append([i])
        centres = np.vstack([centres, v])
    groups = []
    for idx in members:
        faces_ = [rows[i] for i in idx]
        clips = sorted({f["clip_id"] for f in faces_})
        groups.append({
            "id": faces_[0]["id"],
            "face_ids": [f["id"] for f in faces_],
            "clips": clips,
            "crops": [f["crop"] for f in faces_ if f["crop"]][:4],
        })
    return sorted(groups, key=lambda g: (-len(g["clips"]), -len(g["face_ids"])))


def known_people(conn) -> list[dict]:
    """People with at least one recognised face: name, how many clips, one picture."""
    return conn.execute(
        """SELECT p.id, p.name, p.aliases, count(DISTINCT f.clip_id) AS clips,
                  (array_agg(f.crop ORDER BY f.score DESC NULLS LAST) FILTER (WHERE f.crop IS NOT NULL))[1] AS crop,
                  (array_agg(c.thumb ORDER BY f.score DESC NULLS LAST))[1] AS thumb
             FROM people p JOIN clip_faces f ON f.person_id = p.id JOIN clips c ON c.id = f.clip_id
            GROUP BY p.id ORDER BY clips DESC, p.name"""
    ).fetchall()


MAX_REFS_PER_NAMING = 5   # faces of a named group kept as references (spread across the group)


def name_faces(conn, face_ids: list[int], name: str, aliases: list[str] | None = None) -> tuple[int, list[int]]:
    """A person looked at these faces and said who it is. All of them become that
    person (even the ones a model would not have matched: a person checked them),
    a few become reference photos, and every other unknown face is checked again.
    Naming a second group with the same name merges it into the same person.
    Returns (person_id, ids of clips that changed)."""
    name = name.strip()
    if not name or not face_ids:
        raise ValueError("a name and at least one face are needed")
    person_id = upsert_person(conn, name)
    if aliases:
        add_aliases(conn, person_id, aliases)
    rows = conn.execute("SELECT id, clip_id FROM clip_faces WHERE id = ANY(%s) ORDER BY id",
                        (face_ids,)).fetchall()
    if not rows:
        raise ValueError("none of those faces exist")
    conn.execute("UPDATE clip_faces SET person_id = %s, score = 1 WHERE id = ANY(%s)",
                 (person_id, [r["id"] for r in rows]))
    step = max(1, len(rows) // MAX_REFS_PER_NAMING)
    for r in rows[::step][:MAX_REFS_PER_NAMING]:
        promote(conn, r["id"], name)
    named_clips = {r["clip_id"] for r in rows}
    for clip_id in named_clips:
        _add_person_to_clip(conn, clip_id, [name])
    changed = named_clips | set(rematch_unknown(conn))
    if aliases:                       # earlier clips of this person must learn the new names too
        changed |= set(clips_of(conn, name))
    return person_id, sorted(changed)


def _add_person_to_clip(conn, clip_id: int, names: list[str]) -> None:
    conn.execute(
        """UPDATE clips
              SET people = ARRAY(SELECT DISTINCT unnest(people || %s::text[])),
                  review_reasons = array_remove(review_reasons, 'unknown_faces'),
                  needs_review = cardinality(array_remove(review_reasons, 'unknown_faces')) > 0
            WHERE id = %s""",
        (sorted(names), clip_id),
    )


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
