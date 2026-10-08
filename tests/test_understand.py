"""Meme understanding engine: the maths, the cues, and end-to-end readings.

No model is downloaded: WordSpace stands in for the sentence embedder. It is a
bag of words (one fixed random vector per word), so two sentences are close
exactly when they share words. That is enough to check the engine's logic:
the experts, the pooling, the learning. It says nothing about real model quality.
"""
import hashlib
import os
import re

import numpy as np
import pytest

from . import conftest  # noqa: F401

from app import humor, intent, semspace
from app.semspace import Space

STOP = {"a", "an", "the", "i", "is", "it", "to", "of", "and", "my", "me", "you", "so", "this", "that",
        "at", "in", "on", "for", "be", "we", "are", "was", "with", "all", "just"}


def word_embed(texts):
    out = []
    for t in texts:
        v = np.zeros(128)
        for w in re.findall(r"[a-z]+", t.lower()):
            if w in STOP:
                continue
            seed = int.from_bytes(hashlib.md5(w[:5].encode()).digest()[:4], "big")
            v += np.random.default_rng(seed).standard_normal(128)
        out.append(list(v if v.any() else np.random.default_rng(0).standard_normal(128) * 1e-3))
    return out


@pytest.fixture(scope="module")
def space():
    return Space(word_embed)


def clip(**over):
    base = {"id": 1, "title": "", "caption": "", "transcript_roman": "", "description": "", "reactions": [],
            "use_when": [], "topics": [], "folk_names": [], "people": [], "source_title": "", "comments": [],
            "shares": 0, "duplicates_seen": 1, "raw": {}}
    base.update(over)
    return humor.facts_from_clip(base)


# --------------------------------------------------------------------------- #
# Maths
# --------------------------------------------------------------------------- #

def test_product_of_experts_prefers_what_everyone_accepts():
    keys = ["a", "b", "c"]
    e1 = {"a": 0.6, "b": 0.4, "c": 0.0}
    e2 = {"a": 0.0, "b": 0.4, "c": 0.6}
    pooled = semspace.product_of_experts([(1, e1), (1, e2)], keys)
    assert max(pooled, key=pooled.get) == "b"            # the only option both find plausible
    mixed = semspace.mix([(1, e1), (1, e2)], keys)
    assert pooled["b"] > 0.9 > 0.5 > mixed["b"]               # pooling insists; a mixture only leans


def test_js_divergence_bounds():
    assert semspace.js_divergence({"x": 1}, {"x": 1}) == 0.0
    assert abs(semspace.js_divergence({"x": 1}, {"y": 1}) - 1.0) < 1e-9
    assert semspace.js_divergence({}, {"y": 1}) == 0.0


def test_entropy_confidence():
    assert semspace.entropy_confidence({"a": 1.0, "b": 0.0}) == 1.0
    assert semspace.entropy_confidence({"a": 0.5, "b": 0.5}) < 1e-9


def test_z_scores_separate_related_from_unrelated(space):
    a, b, c = space.vecs(["the boss is shouting at the meeting", "boss shouting meeting again",
                          "purple elephants juggling bananas"])
    assert space.z(a, b) > 3 and abs(space.z(a, c)) < 2.5


def test_clip_posterior_learns_from_use():
    prior = semspace.normalise_dist({"celebrate": 1}, intent.INTENTS)
    assert intent.clip_posterior(prior, None) == prior
    after = intent.clip_posterior(prior, {"call_out": 20})
    assert max(after, key=after.get) == "call_out"
    nudged = intent.clip_posterior(prior, {"call_out": 1})
    assert max(nudged, key=nudged.get) == "celebrate" and nudged["call_out"] > prior["call_out"]
    assert abs(sum(after.values()) - 1) < 1e-9


def test_transition_matrix_rows_are_distributions_and_learn():
    m = intent.transition_matrix()
    assert set(m) == set(intent.CONTEXT_ACTS)
    for row in m.values():
        assert abs(sum(row.values()) - 1) < 1e-9
    assert max(m["good_news"], key=m["good_news"].get) == "celebrate"
    learned = intent.transition_matrix({"good_news": {"sarcasm": 30}})
    assert max(learned["good_news"], key=learned["good_news"].get) == "sarcasm"


def test_fit_funny_stays_at_prior_without_data_and_moves_with_it():
    assert humor.fit_funny([]) == humor.PRIOR_WEIGHTS
    feats = {m: 0.0 for m in humor.MECHANISMS}
    # people keep finding relatable clips unfunny
    data = [({**feats, "relatability": 1.0}, 0.0)] * 60 + [({**feats, "laughter": 1.0}, 1.0)] * 60
    w = humor.fit_funny(data)
    assert w["relatability"] < humor.PRIOR_WEIGHTS["relatability"]
    assert humor.funniness({**feats, "laughter": 1.0}, w) > humor.funniness({**feats, "relatability": 1.0}, w)


# --------------------------------------------------------------------------- #
# Cues
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,rx", [
    ("hahaha 😂", humor.LAUGH), ("lmaooo", humor.LAUGH), ("I'm dead 💀", humor.LAUGH),
    ("me when the exam starts", humor.RELATABLE_FORMAT), ("POV: monday", humor.RELATABLE_FORMAT),
    ("yeah right 🙄", humor.SARCASM_MARK), ("nooooo", humor.EXAGGERATION), ("literally dying", humor.EXAGGERATION),
])
def test_cues_fire(text, rx):
    assert humor.count(rx, text) > 0


def test_cues_do_not_fire_on_plain_text():
    plain = "the train leaves at seven"
    for rx in (humor.LAUGH, humor.RELATABLE_FORMAT, humor.SARCASM_MARK, humor.EXAGGERATION):
        assert humor.count(rx, plain) == 0


def test_valence():
    assert humor.text_valence("I passed, congrats to me 🎉") > 0
    assert humor.text_valence("I failed and I'm so tired") < 0
    assert humor.mood_valence({"happy": 1.0}) == 1.0 and humor.mood_valence({}) == 0.0


# --------------------------------------------------------------------------- #
# Why it is funny
# --------------------------------------------------------------------------- #

def test_incongruity_needs_both_surprise_and_a_bridge(space):
    # Bag-of-words "meaning": a bridge relates to a sentence when they share words.
    bridges = {"kitchen thief caught": ["kitchen thief caught red handed"],
               "office deadline": ["office deadline tonight boss"],
               "beach holiday": ["beach holiday sunshine swimming"]}
    setup, punch, nonsense = space.vecs(["mom kitchen cooker empty", "thief caught red handed",
                                         "purple elephants juggling bananas"])
    bridged = humor.incongruity(space, setup, punch, bridges)
    unbridged = humor.incongruity(space, setup, nonsense, bridges)
    same = humor.incongruity(space, setup, setup, bridges)
    assert bridged["bridge"] == "kitchen thief caught"
    assert bridged["resolution"] > unbridged["resolution"]
    assert unbridged["absurdity"] > bridged["absurdity"]
    assert same["surprise"] < bridged["surprise"]
    assert bridged["score"] > max(unbridged["score"], same["score"])


def test_tonal_mismatch_from_stored_mood_scores():
    deadpan = clip(raw={"vibe": {"faces": {"angry": 0.9, "happy": 0.05},
                                 "dialogue": {"happy": 0.9, "angry": 0.05}}})
    same = clip(raw={"vibe": {"faces": {"angry": 0.9}, "dialogue": {"angry": 0.8}}})
    assert humor.tonal_mismatch(deadpan)["score"] > 0.5 > humor.tonal_mismatch(same)["score"]


def test_bare_template_is_flagged_and_known_memes_score_recognition(space):
    f = clip(title="Rasode mein kaun tha", transcript_roman="rasode men kaun tha",
             folk_names=["kokilaben meme"], reactions=["shocked"], duplicates_seen=8)
    out = humor.analyse(f, space=space)
    assert out["features"]["recognition"] >= 0.6
    assert any(r["mechanism"] == "template" for r in out["reasons"])
    assert 0.0 < out["funny"] < 1.0


def test_laughing_comments_raise_funniness(space):
    base = dict(title="boss dancing at the office party", reactions=["dancing"])
    quiet = humor.analyse(clip(**base, comments=["ok", "nice video"]), space=space)
    loud = humor.analyse(clip(**base, comments=["😂😂", "lmao", "hahaha dead 💀"]), space=space)
    assert loud["funny"] > quiet["funny"] and loud["features"]["laughter"] > 0.5


# --------------------------------------------------------------------------- #
# What it says in a chat
# --------------------------------------------------------------------------- #

KOKILABEN = dict(title="Rasode mein kaun tha", transcript_roman="rasode men kaun tha",
                 description="Stern woman questions two nervous women, who did this",
                 reactions=["shocked", "angry"], use_when=["finding out who did it"], folk_names=["kokilaben meme"])
DANCE = dict(title="uncle dancing at the wedding", reactions=["dancing", "happy"], topics=["celebration"],
             use_when=["when you celebrate", "when you get good news"])
SAD = dict(title="crying in the rain", reactions=["sad", "helpless"], topics=["fail"],
           use_when=["when plans fail"])
SMUG = dict(title="smirking guy laughing", reactions=["smug", "laughing"], topics=["roast"])


def top_intent(space, c, context, caption=""):
    facts = clip(**c)
    mech = humor.analyse(facts, context, caption, space=space)["mechanisms"]
    return intent.read(facts, context, caption, mech=mech, space=space)


def test_the_same_chat_reads_differently_with_different_memes(space):
    news = ["I got the job!"]
    assert top_intent(space, DANCE, news)["intent"] == "celebrate"
    assert top_intent(space, SMUG, ["I'm the best at this, easy"])["intent"] in {"mock", "sarcasm", "told_you_so"}


def test_confession_answered_by_kokilaben_is_a_call_out(space):
    r = top_intent(space, KOKILABEN, ["I ate the last piece"])
    assert r["intent"] in {"call_out", "disbelief"}
    assert r["aimed_at"] in {"recipient", "message"}


def test_bad_news_answered_by_sad_clip_is_sympathy(space):
    r = top_intent(space, SAD, ["I failed the exam"])
    assert r["intent"] in {"sympathise", "self_deprecate"}


def test_sender_caption_turns_it_on_themselves(space):
    r = top_intent(space, SAD, ["exam results are out"], caption="me rn")
    assert r["intent"] == "self_deprecate" and r["aimed_at"] == "self"


def test_feedback_counts_override_the_content_reading(space):
    facts = clip(**DANCE)
    plain = intent.read(facts, ["I got the job!"], space=space)
    taught = intent.read(facts, ["I got the job!"], space=space, clip_counts={"sarcasm": 40})
    assert plain["intent"] == "celebrate" and taught["intent"] == "sarcasm"


def test_reading_shape(space):
    r = top_intent(space, KOKILABEN, ["who finished the milk?"])
    assert set(r) >= {"intent", "reading", "aimed_at", "confidence", "intents", "evidence", "experts", "acts"}
    assert abs(sum(r["acts"].values()) - 1) < 1e-3 and 0 <= r["confidence"] <= 1
    assert r["intents"][0]["label"] == r["intent"]


# --------------------------------------------------------------------------- #
# Database + HTTP (needs TEST_DATABASE_URL)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")
def test_api_understand_feedback_and_learning():
    from fastapi.testclient import TestClient

    from app import api, db

    db.init_db()
    sha = hashlib.sha256(b"understand-test-clip").hexdigest()
    with db.session() as conn:
        conn.execute("DELETE FROM clips WHERE sha256 = %s", (sha,))
        clip_id = conn.execute(
            """INSERT INTO clips (sha256, file, title, transcript_roman, reactions, use_when, folk_names)
               VALUES (%s, 'x.mp4', 'Rasode mein kaun tha', 'rasode men kaun tha', '{shocked}',
                       '{finding out who did it}', '{kokilaben meme}') RETURNING id""", (sha,)).fetchone()["id"]
    try:
        with TestClient(api.app) as client:
            alone = client.get(f"/clips/{clip_id}/humor").json()
            assert alone["context"] == [] and 0 < alone["funny"] < 1 and alone["why_funny"]

            body = {"link": f"https://memes.example/c/{sha[:10]}", "context": ["I ate the last samosa"]}
            r = client.post("/understand", json=body).json()
            assert r["clip"]["id"] == clip_id and r["communicates"]["intent"] in intent.INTENTS

            assert client.post("/understand", json={"context": ["x"]}).status_code == 400
            assert client.post("/understand", json={"link": "nothing here"}).status_code == 404
            bad = client.post("/understand/feedback", json={**body, "intent": "nonsense"})
            assert bad.status_code == 400

            for _ in range(25):   # people keep saying this clip is a call-out here
                ok = client.post("/understand/feedback", json={**body, "intent": "call_out", "funny": True})
                assert ok.status_code == 200
            after = client.post("/understand", json=body).json()
            assert after["communicates"]["intent"] == "call_out"
            stats = client.get("/understand/stats").json()
            assert stats["labelled"] >= 25
            assert "call_out" in client.get("/understand/intents").json()["intents"]
    finally:
        with db.session() as conn:
            conn.execute("DELETE FROM clips WHERE id = %s", (clip_id,))
