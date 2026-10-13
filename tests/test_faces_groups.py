"""Faces view: grouping stored faces, naming (which merges groups), aliases in search.
Uses the stored face vectors only: no model, no reprocessing."""
import os
import random

import pytest

from . import conftest  # noqa: F401

pytestmark = pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")

from app import config, db, faces, indexing  # noqa: E402


def _person(seed: int) -> list[float]:
    rnd = random.Random(seed)
    return [rnd.gauss(0, 1) for _ in range(config.FACE_DIM)]


def _near(base: list[float], seed: int, noise: float = 0.25) -> list[float]:
    rnd = random.Random(seed)
    return [b + rnd.gauss(0, noise) for b in base]


@pytest.fixture
def two_strangers():
    """Three clips: A appears in clips 1+2, B in clip 3. Nobody is named yet."""
    db.init_db()
    a, b = _person(1), _person(2)
    with db.session() as conn:
        conn.execute("TRUNCATE clips, people RESTART IDENTITY CASCADE")
        ids = [conn.execute("INSERT INTO clips (sha256, file, title) VALUES (%s, %s, %s) RETURNING id",
                            (f"{i:064x}", f"c{i}.mp4", f"clip {i}")).fetchone()["id"] for i in (1, 2, 3)]
        for clip_id, emb in ((ids[0], _near(a, 10)), (ids[1], _near(a, 11)), (ids[1], _near(a, 12)),
                             (ids[2], _near(b, 20))):
            conn.execute("INSERT INTO clip_faces (clip_id, frame, embedding) VALUES (%s, 0, %s::vector)",
                         (clip_id, str(emb)))
        for i in ids:
            indexing.refresh(conn, i, embed=False)
    return ids


def test_stored_faces_group_by_person(two_strangers):
    with db.session() as conn:
        groups = faces.face_groups(conn)
    assert [len(g["clips"]) for g in groups] == [2, 1]          # A (2 clips) first, then B
    assert len(groups[0]["face_ids"]) == 3


def test_naming_two_groups_the_same_merges_them(two_strangers):
    with db.session() as conn:
        g1, g2 = faces.face_groups(conn)
        faces.name_faces(conn, g1["face_ids"], "Narendra Modi", ["modi", "PM"])
        _, changed = faces.name_faces(conn, g2["face_ids"], "Narendra Modi")   # same name: same person
        assert faces.face_groups(conn) == []                                    # nobody unknown left
        people = faces.known_people(conn)
        assert len(people) == 1 and people[0]["clips"] == 3
        assert sorted(people[0]["aliases"]) == ["PM", "modi"]
        assert set(changed) >= {two_strangers[2]}


def test_aliases_reach_search_text(two_strangers):
    with db.session() as conn:
        g1 = faces.face_groups(conn)[0]
        person_id, changed = faces.name_faces(conn, g1["face_ids"], "Narendra Modi")
        for c in changed:
            indexing.refresh(conn, c, embed=False)
        faces.add_aliases(conn, person_id, ["PM", "modi ji"])
        for c in faces.clips_of(conn, "Narendra Modi"):
            indexing.refresh(conn, c, embed=False)
        s = conn.execute("SELECT s_people FROM clips WHERE id = %s", (two_strangers[0],)).fetchone()["s_people"]
    assert "pm" in s and "modi ji" in s


def test_not_this_person_sticks(two_strangers):
    with db.session() as conn:
        g1 = faces.face_groups(conn)[0]
        person_id, _ = faces.name_faces(conn, g1["face_ids"], "Narendra Modi")
        clip = two_strangers[0]
        assert faces.remove_person(conn, person_id, clip) == "Narendra Modi"
        assert "Narendra Modi" not in conn.execute("SELECT people FROM clips WHERE id = %s", (clip,)).fetchone()["people"]
        faces.rematch_unknown(conn)                     # the "no" holds: not matched back
        back = conn.execute("SELECT count(*) AS n FROM clip_faces WHERE clip_id = %s AND person_id = %s",
                            (clip, person_id)).fetchone()["n"]
        assert back == 0


def test_grouping_does_not_depend_on_order(two_strangers):
    with db.session() as conn:
        first = faces.face_groups(conn)
        again = faces.face_groups(conn)
    assert [sorted(g["face_ids"]) for g in first] == [sorted(g["face_ids"]) for g in again]
