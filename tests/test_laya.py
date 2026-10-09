"""Laya client: question building, answer reading, and the HTTP round trip
against a fake laya-serve. No model, no database."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from . import conftest  # noqa: F401

from app import config, laya


def test_clip_questions_cover_every_topic_and_mood():
    qs = laya.clip_questions()
    assert set(qs["topic"]["criteria"]) == {*laya.TOPICS, laya.NONE}
    assert qs["mood"]["type"] == "choice" and laya.NONE in qs["mood"]["criteria"]
    assert set(laya.clip_questions(with_mood=False)) == {"topic"}
    assert set(laya.query_questions()["topic"]["criteria"]) == {*laya.TOPICS, laya.NONE}


def test_clip_state_drops_empty_and_duplicate_fields():
    meta = {"title": "boss shouting", "caption": "", "source_title": "", "hashtags": []}
    assert laya.clip_state(meta, "kya hai", "kya hai", []) == {"title": "boss shouting", "dialogue": "kya hai"}
    assert laya.clip_state(meta, "kya hai", "kya hai", [], with_dialogue=False) == {"title": "boss shouting"}


def test_read_topics_takes_the_best_of_both_readings(monkeypatch):
    monkeypatch.setattr(config, "LAYA_MAX_TOPICS", 5)
    with_dialogue = {"probabilities": {"love": 0.3, "work": 0.2, "none": 0.5}}
    title_only = {"probabilities": {"money": 0.9, "love": 0.05, "none": 0.05}}
    kept, scores = laya.read_topics([with_dialogue, title_only], threshold=0.45)
    assert kept == ["money"] and scores["love"] == 0.3 and "none" not in scores
    assert laya.read_topics([None], threshold=0.45) == ([], {})


def test_read_topics_cap(monkeypatch):
    monkeypatch.setattr(config, "LAYA_MAX_TOPICS", 1)
    a = {"probabilities": {"work": 0.5, "late": 0.48}}
    assert laya.read_topics([a], threshold=0.45)[0] == ["work"]


def test_read_mood_keeps_real_probabilities_and_respects_none():
    ans = {"choice": "shocked", "probabilities": {"shocked": 0.6, "sad": 0.3, "none": 0.1}}
    assert laya.read_mood(ans) == {"shocked": 0.6, "sad": 0.3}
    assert laya.read_mood({"choice": "none", "probabilities": {"none": 0.9, "sad": 0.1}}) == {}
    assert laya.read_mood(None) == {}


def test_read_query_keeps_confident_options_only(monkeypatch):
    monkeypatch.setattr(config, "LAYA_QUERY_MIN", 0.3)
    answers = {"topic": {"probabilities": {"work": 0.7, "late": 0.2, "none": 0.1}},
               "mood": {"probabilities": {"angry": 0.5, "shocked": 0.35, "sad": 0.15}}}
    assert laya.read_query(answers) == {"work": 0.7, "angry": 0.5, "shocked": 0.35}
    assert laya.read_query({"topic": {"probabilities": {"none": 0.95, "work": 0.05}}}) == {}


class _FakeLaya(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).requests.append({"path": self.path, "body": body,
                                    "auth": self.headers.get("authorization")})
        answers = {}
        for qid, q in body["questions"].items():   # every question is a choice
            opts = list(q["criteria"])
            pick = "work" if "work" in opts else "angry"
            answers[qid] = {"type": "choice", "choice": pick,
                            "probabilities": {o: (0.8 if o == pick else 0.2 / (len(opts) - 1)) for o in opts}}
        out = json.dumps({"model": "laya-rl-agent", "answers": answers}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_laya(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _FakeLaya)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _FakeLaya.requests = []
    monkeypatch.setattr(config, "LAYA_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(config, "LAYA_API_KEY", "secret")
    laya._query_intent_cached.cache_clear()
    yield _FakeLaya.requests
    server.shutdown()
    laya._query_intent_cached.cache_clear()


def test_describe_clip_over_http(fake_laya):
    out = laya.describe_clip({"title": "boss shouting at me"}, "daant diya", "daant diya", ["Nana Patekar"])
    assert out["topics"] == ["work"]
    assert out["mood"]["angry"] == 0.8
    full, bare = fake_laya
    assert full["path"] == "/v1/systemone" and full["auth"] == "Bearer secret"
    assert full["body"]["model"] == config.LAYA_MODEL
    assert full["body"]["state"] == {"title": "boss shouting at me", "people": "Nana Patekar",
                                     "dialogue": "daant diya"}
    assert set(full["body"]["questions"]) == {"topic", "mood"}
    assert bare["body"]["state"] == {"title": "boss shouting at me", "people": "Nana Patekar"}
    assert set(bare["body"]["questions"]) == {"topic"}


def test_describe_clip_asks_once_without_dialogue(fake_laya):
    laya.describe_clip({"title": "boss shouting at me"}, "", "", [])
    assert len(fake_laya) == 1


def test_query_intent_is_cached(fake_laya):
    assert laya.query_intent("When the BOSS shouts ")["work"] == 0.8
    assert laya.query_intent("when the boss shouts")["work"] == 0.8
    assert len(fake_laya) == 1


def test_query_intent_never_fails(monkeypatch):
    monkeypatch.setattr(config, "LAYA_URL", "http://127.0.0.1:9")   # nothing listens there
    monkeypatch.setattr(laya, "_down_until", 0.0)
    laya._query_intent_cached.cache_clear()
    assert laya.query_intent("anything") == {}
    monkeypatch.setattr(config, "LAYA_URL", "")
    assert laya.query_intent("anything") == {}


def test_a_failed_query_intent_backs_off(monkeypatch):
    """lru_cache does not keep failures, so one hung Laya must not cost every search its timeout."""
    from app import laya
    calls = []

    def boom(q):
        calls.append(q)
        raise OSError("hung")
    monkeypatch.setattr(config, "LAYA_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(laya, "_query_intent_cached", boom)
    monkeypatch.setattr(laya, "_down_until", 0.0)
    assert laya.query_intent("a") == {} and laya.query_intent("b") == {}
    assert calls == ["a"]          # the second search did not call Laya at all


def test_learned_queries_in_the_index_are_bounded():
    from app import indexing
    learned = {f"query {i}": i for i in range(100)}
    clip = {"learned_queries": learned, "hashtags": [], "folk_names": [], "title": "", "transcript_roman": "",
            "transcript_native": "", "people": [], "reactions": [], "use_when": [], "topics": [],
            "description": "", "caption": "", "source_title": "", "comments": []}
    fields = indexing.search_fields(clip, [])
    kept = fields["s_learned"].split()
    assert "99" in kept and len(fields["s_learned"].split(" query ")) <= indexing.MAX_LEARNED + 1
