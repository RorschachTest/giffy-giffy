"""The label list keeps its shape, and the variety check counts correctly. No database."""
from . import conftest  # noqa: F401

from app import check_labels, taxonomy


def test_label_list_has_no_shape_problems():
    assert check_labels.shape_problems() == []


def test_every_facet_is_around_fifty_or_less():
    for facet in taxonomy.FACETS:
        assert 20 <= len(taxonomy.labels(facet)) <= 55, facet


def test_the_two_example_clips_have_their_labels():
    assert taxonomy.group_of("reply", "big_talk") == "about_me"     # /c/d28f9c7a36
    assert taxonomy.group_of("funny", "broken_language") == "language"   # /c/0e66616b4c
    assert set(taxonomy.FACETS) == set(taxonomy.QUESTIONS)


def test_variety_flags_broad_unused_and_unmatched(monkeypatch):
    labels = list(taxonomy.labels("reply"))
    matches = {"a": (labels[0], 0.9), "b": (labels[0], 0.8), "c": (labels[1], 0.1)}
    v = check_labels.variety("reply", matches)
    assert v["broad"] == {labels[0]: 0.67}
    assert labels[1] in v["unused"] and v["unmatched"] == ["c"]
