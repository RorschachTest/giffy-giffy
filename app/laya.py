"""Typed decisions from Laya (https://github.com/NandhaKishorM/laya), served by laya-serve.

Laya is not a chat model: it gets a text "state" plus typed questions (choice,
score, yes/no) and returns a probability for every option in one forward pass,
without generating any text. We use it twice:

  ingest   topics  a choice over TOPICS, asked twice: once on the metadata
                   alone (title, caption, people) and once with the dialogue
                   too; a topic is kept if either reading is confident  -> clips.topics
           mood    a choice over the moods in vibe.MOODS     -> the text signal for vibe
  search   what the query is about (topic + mood choice)     -> boosts clips tagged that way

Why choices and not one yes/no per topic: measured on the sample clips, separate
yes/no answers are not comparable with each other (every topic ~1.0 on some
clips, every topic < 0.25 on others), while one choice gives a clean
distribution. Why two readings: speech-to-text on film dialogue is often wrong,
and a noisy transcript pulls the answer towards generic topics; the title alone
is sharp when it names the topic.

This is the only file that talks to Laya. If LAYA_URL is empty or the server is
down, ingest flags the clip `laya_failed` and search works exactly as before.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from functools import lru_cache

from . import config
from .vibe import MOODS

log = logging.getLogger("laya")

# topic -> what it means. Kept short: every option costs tokens on every question.
TOPICS: dict[str, str] = {
    "politics":    "politicians, parliament, elections, government",
    "bollywood":   "a scene from a Hindi film or a film star",
    "interview":   "someone answering questions on camera, news or press",
    "speech":      "a lecture, sermon or advice given to others",
    "motivation":  "inspiring, never give up, keep going",
    "sarcasm":     "mocking or ironic, saying the opposite of what is meant",
    "roast":       "insulting or making fun of someone",
    "fail":        "something goes wrong, losing, a mistake",
    "celebration": "winning, a party, dancing, good news",
    "money":       "money, salary, being rich or poor, paying",
    "work":        "office, boss, job, deadlines",
    "study":       "exams, school, students, teachers",
    "love":        "romance, a crush, relationships, breakups",
    "family":      "parents, marriage, relatives",
    "late":        "being late, in a hurry, rushing",
    "confusion":   "not understanding, nothing makes sense",
    "cringe":      "awkward or embarrassing",
    "wholesome":   "kind, cute or warm",
    "sports":      "cricket, fitness, running, yoga",
}
NONE = "none"   # the "no clear topic / mood" option, so a name or quote is not forced into one


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

def available() -> bool:
    return bool(config.LAYA_URL)


def predict(state, questions: dict, timeout: float | None = None) -> dict:
    """POST /v1/systemone. Returns the `answers` object. Raises on any failure."""
    body = {"state": state, "questions": questions}
    if config.LAYA_MODEL:
        body["model"] = config.LAYA_MODEL
    req = urllib.request.Request(
        config.LAYA_URL.rstrip("/") + "/v1/systemone",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"content-type": "application/json",
                 **({"authorization": f"Bearer {config.LAYA_API_KEY}"} if config.LAYA_API_KEY else {})},
    )
    with urllib.request.urlopen(req, timeout=timeout or config.LAYA_TIMEOUT) as resp:
        return json.loads(resp.read())["answers"]


# --------------------------------------------------------------------------- #
# Questions and how their answers are read (pure, tested without a server)
# --------------------------------------------------------------------------- #

def _mood_question(instructions: str) -> dict:
    return {"type": "choice", "instructions": instructions,
            "criteria": {**{m: s for m, (s, _) in MOODS.items()}, NONE: "no particular mood"}}


def _topic_question(instructions: str, none: str) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": {**TOPICS, NONE: none}}


def clip_questions(with_mood: bool = True) -> dict:
    qs = {"topic": _topic_question("What is this meme clip about?", "nothing in particular")}
    if with_mood:
        qs["mood"] = _mood_question("Which mood does this meme clip show?")
    return qs


def query_questions() -> dict:
    return {
        "topic": _topic_question("What is the person searching for a meme about?",
                                 "a name, title or quote with no clear topic"),
        "mood": _mood_question("Which mood should the meme they want show?"),
    }


def clip_state(meta: dict, native: str, roman: str, people: list[str],
               with_dialogue: bool = True) -> dict:
    """The facts Laya decides on. Only text: it cannot see the video."""
    state = {"title": meta.get("title", ""), "caption": meta.get("caption", ""),
             "people": ", ".join(people), "film_or_show": meta.get("source_title", ""),
             "hashtags": " ".join(meta.get("hashtags") or [])}
    if with_dialogue:
        state.update(dialogue=native, dialogue_roman=roman if roman != native else "")
    return {k: v for k, v in state.items() if v}


def read_mood(answer: dict | None) -> dict[str, float]:
    """Choice probabilities over moods, as they are. Not rescaled: Laya is right
    when it is sure (0.8-0.95 on the sample clips) and noise when it is not
    (0.2-0.3), so an unsure reading must count for little. Empty if 'none' wins."""
    probs = {m: round(float(p), 3) for m, p in ((answer or {}).get("probabilities") or {}).items()
             if m in MOODS}
    if not probs or (answer or {}).get("choice") == NONE:
        return {}
    return probs


def read_topics(readings: list[dict | None], threshold: float | None = None) -> tuple[list[str], dict[str, float]]:
    """Topics any reading gives at least `threshold`, most likely first, plus the
    best probability each topic got. 'none' is never a topic."""
    threshold = config.LAYA_TOPIC_MIN if threshold is None else threshold
    best: dict[str, float] = {}
    for answer in readings:
        for t, p in ((answer or {}).get("probabilities") or {}).items():
            if t in TOPICS:
                best[t] = max(best.get(t, 0.0), round(float(p), 3))
    kept = sorted((t for t, p in best.items() if p >= threshold), key=lambda t: -best[t])
    return kept[:config.LAYA_MAX_TOPICS], best


def read_query(answers: dict) -> dict[str, float]:
    """Query intent as {topic or mood: weight}: the top options of each choice that
    clear LAYA_QUERY_MIN, 'none' dropped. Weights are the probabilities themselves."""
    out: dict[str, float] = {}
    for qid in ("topic", "mood"):
        probs = (answers.get(qid) or {}).get("probabilities") or {}
        ranked = sorted(((float(p), k) for k, p in probs.items() if k != NONE), reverse=True)
        for p, k in ranked[:2]:
            if p >= config.LAYA_QUERY_MIN:
                out[k] = max(out.get(k, 0.0), round(p, 3))
    return out


# --------------------------------------------------------------------------- #
# What the pipeline and search call
# --------------------------------------------------------------------------- #

def describe_clip(meta: dict, native: str, roman: str, people: list[str]) -> dict:
    """{"topics": [...], "mood": {mood: probability}, "scores": {...}} for one clip.
    One call with everything (topic + mood), and one on the metadata alone (topic)."""
    full = clip_state(meta, native, roman, people)
    answers = predict(full, clip_questions())
    readings = [answers.get("topic")]
    bare = clip_state(meta, native, roman, people, with_dialogue=False)
    if bare and bare != full:
        readings.append(predict(bare, clip_questions(with_mood=False)).get("topic"))
    topics, scores = read_topics(readings)
    mood_answer = answers.get("mood")
    return {"topics": topics, "mood": read_mood(mood_answer),
            "scores": {"topics": scores,
                       "mood": (mood_answer or {}).get("probabilities", {}),
                       "mood_choice": (mood_answer or {}).get("choice")}}


DOWN_BACKOFF_S = 30
_down_until = 0.0   # monotonic time before which query_intent does not call Laya


@lru_cache(maxsize=2048)
def _query_intent_cached(query: str) -> tuple[tuple[str, float], ...]:
    answers = predict({"search": query}, query_questions(), timeout=config.LAYA_SEARCH_TIMEOUT)
    return tuple(read_query(answers).items())


def query_intent(query: str) -> dict[str, float]:
    """What a search is about. Empty (never an error) when Laya is off, slow or down."""
    global _down_until
    if not available() or not query.strip() or time.monotonic() < _down_until:
        return {}
    try:
        return dict(_query_intent_cached(query.strip().lower()))
    except Exception as exc:  # search must never fail because of Laya
        _down_until = time.monotonic() + DOWN_BACKOFF_S   # a hung Laya must not cost every search its timeout
        log.warning("laya query intent skipped (retry in %ss): %s", DOWN_BACKOFF_S, exc)
        return {}
