"""Why a meme is funny: humor theories turned into measurements. No LLM.

Each mechanism below is a well-studied account of humor, measured as a number
between 0 and 1 from the clip's facts (and, when there is one, the chat it was
sent into). The funniness estimate is a logistic model over those numbers whose
starting weights come from the literature and which refits from feedback
(fit_funny).

  incongruity       Incongruity-resolution (Suls 1972; SICKNet, Zeng 2023): setup and
                    punchline are far apart in meaning (surprise) yet one situation
                    explains both (resolution). Measured as the semantic distance
                    between the two plus the best "bridge" situation they both fit.
  absurdity         surprise with no resolution: the nonsense / surreal kind.
  script_opposition Raskin's Semantic Script Theory: the text fits two opposed
                    scripts (expected vs real, high vs low status ...). The setup
                    leans to one pole, the punchline to the other.
  tonal_mismatch    what the face shows vs what the words say (deadpan, irony):
                    Jensen-Shannon divergence between the face/scene mood and the
                    dialogue mood, plus the CLIP picture-vs-words gap when known.
  benign_violation  McGraw & Warren 2010: something is wrong (a norm is broken) and
                    harmless at the same time. Geometric mean of the two.
  superiority       Hobbes / Gruner: laughing at someone (roast, fail, mockery).
  relief            Spencer / Freud: tension built up, then released.
  exaggeration      hyperbole: intensifiers, stretched letters, extreme moods.
  relatability      the "me when / POV / nobody:" observational formats.
  self_deprecation  the joke is on the speaker.
  recognition       a known template, person or film: the shared-reference laugh.
  laughter          people already laughed: laughing comments, the laughing mood.

The same code reads a clip on its own (setup = caption, punchline = what is said)
and a clip sent in a chat (setup = the messages before it, punchline = the clip):
that second reading is where reaction templates become jokes.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

from . import laya, vibe
from .semspace import Space, get_space, sigmoid, js_divergence
from .text import clean_caption

# --------------------------------------------------------------------------- #
# Lexical cues (English + Hinglish chat). Pure regex, tested without models.
# --------------------------------------------------------------------------- #

LAUGH = re.compile(r"(\b(lo+l+|lmf?ao+|rofl|(ha){2,}h?|(he){2,}|xd+|hahah\w*)\b|😂|🤣|💀|😹|😆|😭)", re.I)
SARCASM_MARK = re.compile(
    r"(\b(yeah right|sure jan|oh great|oh wow|wow+|totally|obviously|clearly|nice one|great job|"
    r"genius|so smart|waah|wah wah|kya baat hai|bahut badhiya|very good ji)\b|/s\b|🙄|😒|🙃)", re.I)
RELATABLE_FORMAT = re.compile(
    r"(^\s*(me when|me after|me before|me trying|when you|when ur|when u|when your|when my|"
    r"that moment when|pov\b|nobody\s*:|no one\s*:|me\s*:|mfw|tfw|jab\b|be like\b)|"
    r"\b(so me|literally me|me irl|relatable|every time|har baar)\b)", re.I)
EXAGGERATION = re.compile(
    r"(\b(literally|always|never|every single|forever|million|billion|1000|100%|dying|dead|"
    r"hamesha|kabhi nahi|bilkul|ekdum|sabse)\b|(?P<ch>[a-z])(?P=ch){2,}|!{2,}|\?{2,})", re.I)
FIRST_PERSON = re.compile(r"\b(i|i'm|im|me|my|mine|myself|main|mai|mera|meri|mere|mujhe|apun)\b", re.I)
SECOND_PERSON = re.compile(r"\b(you|your|you're|ur|u|tu|tum|tumhara|tumhari|tera|teri|aap|aapka|aapki)\b", re.I)
POSITIVE_WORDS = re.compile(
    r"\b(great|awesome|amazing|yay|congrats|congratulations|won|win|passed|got the|love|happy|best|"
    r"finally|promotion|selected|party|mubarak|badhai|mast|zabardast|jeet)\b|🎉|🥳|❤️|😍|🔥|💯", re.I)
NEGATIVE_WORDS = re.compile(
    r"\b(fail|failed|failing|lost|lose|sad|tired|hate|worst|cancelled|canceled|broke|broken|late|"
    r"fired|sick|angry|annoying|terrible|stuck|ruined|dead ?line|exam|boss|bekar|pareshan|"
    r"bura|gaya kaam se|nahi hua)\b|😞|😢|😡|🤬|😩|😤", re.I)


def match(z: float, at: float = 2.0) -> float:
    """A z-score read as "does this text match that prototype": ~0 for everyday
    text (z near 0), 0.5 at z = `at`, ~1 well above it."""
    return sigmoid(1.5 * (z - at))


def count(rx: re.Pattern, text: str) -> int:
    return len(rx.findall(text or ""))


def saturate(n: float, rate: float = 0.7) -> float:
    """0 -> 0, 1 -> ~0.5, 3+ -> ~0.9: more evidence counts, with diminishing returns."""
    return 1.0 - math.exp(-rate * max(0.0, n))


# --------------------------------------------------------------------------- #
# Prototype sentences (the only "knowledge" the engine has; edit freely)
# --------------------------------------------------------------------------- #

# Raskin's script oppositions: (name, (pole, sentences), (pole, sentences)).
SCRIPT_OPPOSITIONS: list[tuple[str, tuple[str, list[str]], tuple[str, list[str]]]] = [
    ("expectation vs reality",
     ("expectation", ["what I planned to do", "how it was supposed to go", "the perfect plan"]),
     ("reality", ["what actually happened", "it all went completely wrong", "the sad reality"])),
    ("serious vs silly",
     ("serious", ["a serious formal speech", "an important official matter", "a grave solemn moment"]),
     ("silly", ["a silly childish joke", "goofing around for no reason", "a ridiculous dance"])),
    ("high status vs low status",
     ("high status", ["the boss and the government", "a respected powerful leader", "rich famous people"]),
     ("low status", ["a clueless intern", "a broke student with no money", "the common man"])),
    ("normal vs bizarre",
     ("normal", ["an ordinary day", "the everyday routine", "a normal conversation"]),
     ("bizarre", ["something bizarre and weird", "a completely strange situation", "this makes no sense"])),
    ("competent vs incompetent",
     ("competent", ["doing it perfectly like an expert", "a confident professional", "a smart plan"]),
     ("incompetent", ["failing badly at a simple task", "a total mess and a disaster", "an idiot mistake"])),
    ("good vs bad",
     ("good", ["good news, things are great", "a happy success", "we won"]),
     ("bad", ["terrible news, everything is ruined", "a painful loss", "we lost everything"])),
    ("calm vs panic",
     ("calm", ["staying calm and relaxed", "no stress at all", "everything is under control"]),
     ("panic", ["panicking in total chaos", "screaming and running away", "everything is on fire"])),
    ("innocent vs guilty",
     ("innocent", ["acting innocent, I did nothing", "pretending to know nothing", "who me?"]),
     ("guilty", ["caught red-handed", "the one who did it", "the truth comes out"])),
    ("polite vs rude",
     ("polite", ["polite and respectful words", "speaking very sweetly", "with folded hands, please"]),
     ("rude", ["a rude insult", "shouting abuse at someone", "get lost"])),
    ("adult vs child",
     ("adult", ["responsible adult life, bills and taxes", "a mature serious grown-up"]),
     ("child", ["a kid's tantrum, crying for candy", "acting like a baby"])),
]

VIOLATION = {"violation": [
    "breaking the rules", "something rude or offensive", "gross and disgusting",
    "an embarrassing mistake in public", "a threat or an insult", "doing something forbidden",
    "a painful accident", "a shocking betrayal"]}
BENIGN = {"benign": [
    "it's just a joke", "harmless fun between friends", "nobody really gets hurt",
    "playful teasing", "not serious at all", "a silly harmless moment"]}
SUPERIORITY = {"superiority": [
    "laughing at someone's stupid mistake", "making fun of someone", "roasting a friend",
    "he looks like an idiot", "what a fool", "look at this loser"]}
TENSION = {"tension": [
    "stress before the exam", "the deadline is tonight", "scared of getting caught",
    "waiting nervously for the results", "an awkward tense silence", "we are in big trouble"]}
RELEASE = {"release": [
    "finally relieved", "phew, it's over", "we survived", "laughing it off", "never mind, all good"]}
RELATABLE = {"relatable": [
    "everyone has been there", "this is so me", "that feeling when", "every single time this happens to me",
    "we all do this"]}
SELF_DEPRECATION = {"self_deprecation": [
    "I am such a failure", "my life is a joke", "me being useless again", "I always mess things up"]}
VALENCE = {"positive": ["this is wonderful news", "I am so happy", "we did it, amazing", "love it"],
           "negative": ["this is terrible", "I am so sad and tired", "everything went wrong", "I hate this"]}


def _bridges() -> dict[str, list[str]]:
    """Situations that can explain a setup and a punchline at once: the "use when"
    lines of every mood and the description of every topic, plus a few classics."""
    out: dict[str, list[str]] = {}
    for mood, (sentence, uses) in vibe.MOODS.items():
        for u in uses:
            out[u] = [u, f"{u}, {sentence}"]
    for topic, desc in laya.TOPICS.items():
        out[f"about {topic}"] = [desc]
    for s in ("finding out who did it", "getting caught doing something wrong", "plans going wrong",
              "pretending everything is fine", "someone showing off", "the truth coming out",
              "asking for money", "being ignored", "getting scolded", "a sudden betrayal"):
        out[s] = [s]
    return out


BRIDGES = _bridges()

MOOD_VALENCE = {"happy": 1.0, "laughing": 1.0, "dancing": 1.0, "smug": 0.5, "shocked": -0.1,
                "confused": -0.3, "preachy": 0.0, "in a hurry": -0.3, "sad": -1.0, "angry": -1.0,
                "scared": -0.8, "disgusted": -0.8, "helpless": -0.9}

# Starting weights of the funniness model: literature-informed guesses (incongruity,
# laughter evidence and relatability are the strongest predictors in meme-humor
# studies; image features alone add little). fit_funny() moves them with feedback.
PRIOR_WEIGHTS = {
    "bias": -3.0, "incongruity": 2.4, "absurdity": 0.6, "script_opposition": 1.4,
    "tonal_mismatch": 1.1, "benign_violation": 1.4, "superiority": 1.0, "relief": 0.8,
    "exaggeration": 0.7, "relatability": 1.2, "self_deprecation": 0.8, "recognition": 0.9,
    "laughter": 2.0,
}
MECHANISMS = [k for k in PRIOR_WEIGHTS if k != "bias"]


# --------------------------------------------------------------------------- #
# From a clip row to the facts the engine reads
# --------------------------------------------------------------------------- #

def facts_from_clip(clip: dict) -> dict:
    """The parts of a clip row the engine needs, in one plain shape. Hand-entered
    reactions count as certain moods; the models' raw mood scores fill the rest."""
    raw = clip.get("raw") or {}
    vibe_raw = raw.get("vibe") or {}
    title, _ = clean_caption(clip.get("title") or "")
    caption, _ = clean_caption(clip.get("caption") or "")
    dialogue = (clip.get("transcript_roman") or "").strip()
    moods = {k: float(v) for k, v in (vibe_raw.get("combined") or {}).items()}
    for r in clip.get("reactions") or []:
        moods[r] = max(moods.get(r, 0.0), 1.0)
    use_when = list(clip.get("use_when") or [])
    description = (clip.get("description") or "").strip()
    folk = list(clip.get("folk_names") or [])
    return {
        "id": clip.get("id"),
        "title": title, "caption": caption, "dialogue": dialogue, "description": description,
        # The clip's own joke: the caption frames a situation, the clip answers it.
        "setup": caption or ("; ".join(use_when) if use_when else ""),
        "punchline": dialogue or description or title,
        "about": ". ".join(p for p in (title, ", ".join(folk), dialogue, description,
                                        "; ".join(use_when), caption) if p),
        "moods": moods,
        "measured_moods": {k: float(v) for k, v in (vibe_raw.get("combined") or {}).items()},
        "face_moods": vibe_raw.get("faces") or {},
        "scene_moods": vibe_raw.get("scene") or {},
        "dialogue_moods": vibe_raw.get("dialogue") or {},
        "visual_gap": (raw.get("humor_visual") or {}).get("gap"),
        "topics": list(clip.get("topics") or []),
        "reactions": list(clip.get("reactions") or []),
        "use_when": use_when,
        "folk_names": folk,
        "people": list(clip.get("people") or []),
        "source_title": clip.get("source_title") or "",
        "comments": list(clip.get("comments") or []),
        "popularity": int(clip.get("shares") or 0) + int(clip.get("duplicates_seen") or 1) - 1,
    }


# --------------------------------------------------------------------------- #
# Valence: is a message / a clip positive or negative?
# --------------------------------------------------------------------------- #

def mood_valence(moods: dict[str, float]) -> float:
    total = sum(max(0.0, s) for m, s in moods.items() if m in MOOD_VALENCE)
    if total <= 0:
        return 0.0
    return sum(MOOD_VALENCE[m] * max(0.0, s) for m, s in moods.items() if m in MOOD_VALENCE) / total


def text_valence(text: str, space: Space | None = None, vec=None) -> float:
    """-1 (negative) .. +1 (positive). Word lists for the obvious cases, the embedding
    for the rest. Laughing emoji are not counted as sad: 😭 and 💀 mean "so funny"."""
    if not (text or "").strip():
        return 0.0
    lex = count(POSITIVE_WORDS, text) - count(NEGATIVE_WORDS, text)
    lexical = math.tanh(0.8 * lex)
    if space is None:
        return lexical
    zs = space.z_groups(vec if vec is not None else space.vec(text), VALENCE)
    semantic = math.tanh(0.6 * (zs["positive"] - zs["negative"]))
    return max(-1.0, min(1.0, 0.5 * lexical + 0.5 * semantic if lex else semantic))


# --------------------------------------------------------------------------- #
# Mechanisms (pure functions of vectors and facts)
# --------------------------------------------------------------------------- #

def incongruity(space: Space, s, p, bridges: dict[str, list[str]] | None = None, k: int = 3) -> dict:
    """Incongruity-resolution between setup vector s and punchline vector p.

    surprise   = 1 - sigmoid(z(s, p) - 1): far apart in meaning -> near 1
    resolution = how well one situation explains BOTH sides: min(z(s, b), z(p, b))
                 for the best bridge b among the setup's own top-k readings and the
                 punchline's top-k (the setup evokes a script, the punchline is
                 re-read inside it). Only 2k candidates, not every bridge: the best
                 of 50 noisy matches would "resolve" anything by chance.
    score      = sqrt(surprise * resolution); absurdity = surprise * (1 - resolution)
    """
    bridges = bridges or BRIDGES
    rel = space.z(s, p)
    surprise = 1.0 - sigmoid(rel - 1.0)
    zs, zp = space.z_groups(s, bridges), space.z_groups(p, bridges)
    candidates = sorted(zs, key=lambda b: -zs[b])[:k] + sorted(zp, key=lambda b: -zp[b])[:k]
    best = max(candidates, key=lambda b: min(zs[b], zp[b]))
    resolution = sigmoid(1.5 * (min(zs[best], zp[best]) - 1.5))
    return {"score": round(math.sqrt(surprise * resolution), 3),
            "absurdity": round(surprise * (1.0 - resolution), 3),
            "surprise": round(surprise, 3), "resolution": round(resolution, 3),
            "relatedness_z": round(rel, 2), "bridge": best}


def script_opposition(space: Space, s, p) -> dict:
    """The opposition pair whose poles split setup and punchline the most."""
    best = {"score": 0.0, "pair": None, "setup_side": None, "punchline_side": None}
    for name, (a, a_s), (b, b_s) in SCRIPT_OPPOSITIONS:
        poles = {a: a_s, b: b_s}
        zs, zp = space.z_groups(s, poles), space.z_groups(p, poles)
        # setup leans to one pole, punchline to the other, and each really matches its pole
        forward = sigmoid(2 * (zs[a] - zs[b])) * sigmoid(2 * (zp[b] - zp[a])) * match(min(zs[a], zp[b]))
        backward = sigmoid(2 * (zs[b] - zs[a])) * sigmoid(2 * (zp[a] - zp[b])) * match(min(zs[b], zp[a]))
        score = max(forward, backward)
        if score > best["score"]:
            first, second = (a, b) if forward >= backward else (b, a)
            best = {"score": round(score, 3), "pair": name, "setup_side": first, "punchline_side": second}
    return best


def tonal_mismatch(facts: dict) -> dict:
    """Face/scene says one thing, words say another (deadpan, irony)."""
    look = facts["face_moods"] or facts["scene_moods"]
    words = facts["dialogue_moods"]
    js = js_divergence(look, words) if look and words else 0.0
    v_look, v_words = mood_valence(look), mood_valence(words)
    flip = abs(v_look - v_words) / 2 if v_look * v_words < 0 else 0.0
    gap = facts.get("visual_gap") or 0.0
    return {"score": round(max(js * 0.9, flip, gap), 3), "js": round(js, 3),
            "look_valence": round(v_look, 2), "words_valence": round(v_words, 2),
            "visual_gap": round(gap, 3)}


def _mood(facts: dict, *names: str) -> float:
    return max((facts["moods"].get(n, 0.0) for n in names), default=0.0)


def mechanisms(facts: dict, setup: str, punchline: str, extra_text: str = "",
               space: Space | None = None) -> dict:
    """Every mechanism's score (0..1) plus the details behind it."""
    space = space or get_space()
    all_text = " ".join(t for t in (setup, punchline, extra_text, facts["caption"], facts["title"]) if t)
    whole = space.vec(". ".join(t for t in (setup, extra_text, facts["about"]) if t) or all_text)
    out: dict[str, dict] = {}

    if setup.strip() and punchline.strip():
        s, p = space.vecs([setup, punchline])
        inc = incongruity(space, s, p)
        out["incongruity"] = inc
        out["absurdity"] = {"score": inc["absurdity"]}
        out["script_opposition"] = script_opposition(space, s, p)
    else:   # a bare template: no setup to be surprised by
        out["incongruity"] = {"score": 0.0, "note": "no setup"}
        out["absurdity"] = {"score": 0.0}
        out["script_opposition"] = {"score": 0.0, "pair": None}

    out["tonal_mismatch"] = tonal_mismatch(facts)

    z = lambda protos: next(iter(space.z_groups(whole, protos).values()))   # noqa: E731
    violation = max(match(z(VIOLATION)), _mood(facts, "disgusted", "angry") * 0.8, _mood(facts, "shocked") * 0.5)
    benign = max(match(z(BENIGN)), _mood(facts, "laughing", "happy", "dancing") * 0.8,
                 0.6 if count(LAUGH, all_text) else 0.0)
    out["benign_violation"] = {"score": round(math.sqrt(violation * benign), 3),
                               "violation": round(violation, 3), "benign": round(benign, 3)}

    roast = 0.9 if {"roast", "sarcasm"} & set(facts["topics"]) else 0.0
    out["superiority"] = {"score": round(max(match(z(SUPERIORITY)), roast, _mood(facts, "smug") * 0.7), 3)}

    tension = max(match(z(TENSION)), _mood(facts, "scared", "helpless", "in a hurry"))
    release = max(match(z(RELEASE)), _mood(facts, "laughing", "happy", "dancing"))
    out["relief"] = {"score": round(math.sqrt(tension * release), 3),
                     "tension": round(tension, 3), "release": round(release, 3)}

    intensity = max(facts["measured_moods"].values(), default=0.0)   # models' scores, not hand tags
    out["exaggeration"] = {"score": round(max(saturate(count(EXAGGERATION, all_text)),
                                              max(0.0, intensity - 0.7) / 0.3 * 0.6), 3)}

    fmt = count(RELATABLE_FORMAT, " \n".join([setup, extra_text, facts["caption"], facts["title"]]))
    out["relatability"] = {"score": round(max(saturate(fmt, 1.2), match(z(RELATABLE))), 3),
                           "format": bool(fmt)}

    first = count(FIRST_PERSON, " ".join([extra_text, facts["caption"], facts["title"]]))
    negative = max(0.0, -mood_valence(facts["moods"]))
    out["self_deprecation"] = {"score": round(max(match(z(SELF_DEPRECATION)),
                                                  saturate(first) * negative), 3)}

    refs = [*facts["folk_names"], *facts["people"], *([facts["source_title"]] if facts["source_title"] else [])]
    known = 0.4 * bool(facts["folk_names"]) + 0.3 * bool(facts["people"] or facts["source_title"])
    known += 0.3 * min(1.0, math.log1p(facts["popularity"]) / math.log(20))
    out["recognition"] = {"score": round(known, 3), "refs": refs[:4]}

    laughs = sum(1 for c in facts["comments"] if LAUGH.search(c))
    ratio = laughs / len(facts["comments"]) if facts["comments"] else 0.0
    out["laughter"] = {"score": round(max(ratio, _mood(facts, "laughing") * 0.8, saturate(laughs, 0.4)), 3),
                       "laughing_comments": laughs}
    return out


# --------------------------------------------------------------------------- #
# Funniness: a logistic model over the mechanisms
# --------------------------------------------------------------------------- #

def feature_vector(mech: dict) -> dict[str, float]:
    return {m: float(mech[m]["score"]) for m in MECHANISMS}


def funniness(features: dict[str, float], weights: dict[str, float] | None = None) -> float:
    w = weights or PRIOR_WEIGHTS
    return sigmoid(w["bias"] + sum(w.get(k, 0.0) * v for k, v in features.items()))


def fit_funny(examples: list[tuple[dict[str, float], float]], prior: dict[str, float] | None = None,
              strength: float = 4.0, steps: int = 400, lr: float = 0.5) -> dict[str, float]:
    """MAP logistic regression: the weights that best predict the feedback labels
    (1 funny / 0 not), pulled towards the prior weights by a Gaussian prior of
    precision `strength`. With no or little feedback the prior wins; with lots of
    feedback the data does. Plain gradient descent: a dozen features, any n."""
    import numpy as np

    prior = prior or PRIOR_WEIGHTS
    if not examples:
        return dict(prior)
    keys = ["bias", *MECHANISMS]
    X = np.array([[1.0, *[f.get(k, 0.0) for k in MECHANISMS]] for f, _ in examples])
    y = np.array([float(lbl) for _, lbl in examples])
    w0 = np.array([prior[k] for k in keys])
    w = w0.copy()
    n = len(examples)
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        grad = X.T @ (p - y) / n + strength / n * (w - w0)
        w -= lr * grad
    return {k: round(float(v), 4) for k, v in zip(keys, w)}


# --------------------------------------------------------------------------- #
# Plain-language reasons (templates, not generated text)
# --------------------------------------------------------------------------- #

def _short(text: str, n: int = 60) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def reasons(mech: dict, setup: str, punchline: str, in_chat: bool) -> list[dict]:
    """The mechanisms that clearly fire, strongest first, each with one sentence."""
    lead = "the message" if in_chat else "the caption"
    said = []
    m = mech
    inc = m["incongruity"]
    if inc["score"] >= 0.45:
        said.append(("incongruity", inc["score"],
                     f"Incongruity: {lead} (\"{_short(setup)}\") and the clip (\"{_short(punchline)}\") "
                     f"are far apart, yet both fit \"{inc['bridge']}\" — the gap snaps shut, that is the joke."))
    elif m["absurdity"]["score"] >= 0.55:
        said.append(("absurdity", m["absurdity"]["score"],
                     f"Absurdity: the clip has almost nothing to do with {lead}; the randomness is the joke."))
    so = m["script_opposition"]
    if so["score"] >= 0.35 and so["pair"]:
        said.append(("script_opposition", so["score"],
                     f"Script opposition ({so['pair']}): {lead} sets up \"{so['setup_side']}\", "
                     f"the clip flips it to \"{so['punchline_side']}\"."))
    tm = m["tonal_mismatch"]
    if tm["score"] >= 0.4:
        said.append(("tonal_mismatch", tm["score"],
                     "Tonal mismatch: the face says one thing and the words another (deadpan / irony)."))
    bv = m["benign_violation"]
    if bv["score"] >= 0.45:
        said.append(("benign_violation", bv["score"],
                     "Benign violation: something is wrong or rude, but clearly harmless — safe to laugh at."))
    simple = {
        "superiority": "Superiority: someone is the butt of the joke (a roast or a fail).",
        "relief": "Relief: tension builds (stress, fear, a deadline) and the clip releases it.",
        "exaggeration": "Exaggeration: the reaction is way bigger than the situation deserves.",
        "relatability": "Relatability: an everyday experience everyone recognises (\"me when…\").",
        "self_deprecation": "Self-deprecation: the joke is on the person sending it.",
        "recognition": "Recognition: a known meme/person/scene — the shared reference itself is funny.",
        "laughter": "Social proof: people already react to it with laughter.",
    }
    for key, sentence in simple.items():
        if m[key]["score"] >= 0.45:
            if key == "recognition" and m[key].get("refs"):
                sentence = sentence[:-1] + f" ({', '.join(m[key]['refs'][:2])})."
            said.append((key, m[key]["score"], sentence))
    said.sort(key=lambda t: -t[1])
    return [{"mechanism": k, "score": round(s, 3), "why": w} for k, s, w in said]


def analyse(facts: dict, context: list[str] | None = None, caption: str = "",
            space: Space | None = None, weights: dict[str, float] | None = None) -> dict:
    """What makes this clip funny, on its own or as a reply in `context`."""
    context = [c for c in (context or []) if c and c.strip()]
    in_chat = bool(context)
    if in_chat:
        setup = " ".join(context[-2:])                      # the last turns set it up
        punchline = ". ".join(t for t in (caption, facts["punchline"], facts["description"]) if t)
    else:
        setup, punchline = facts["setup"], facts["punchline"]
    mech = mechanisms(facts, setup, punchline, extra_text=caption, space=space)
    feats = feature_vector(mech)
    score = funniness(feats, weights)
    why = reasons(mech, setup, punchline, in_chat)
    if not in_chat and mech["incongruity"].get("note") == "no setup":
        why.append({"mechanism": "template", "score": 0.0,
                    "why": "On its own this is a reaction template: no caption sets it up, so the joke "
                           "only lands when it answers a message."})
    return {"funny": round(score, 3), "in_chat": in_chat, "reasons": why,
            "features": {k: round(v, 3) for k, v in feats.items()}, "mechanisms": mech}


# --------------------------------------------------------------------------- #
# Ingest-time signal: does the picture match the words? (CLIP, optional)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _clip_text():
    from fastembed import TextEmbedding

    from . import config

    return TextEmbedding(model_name=config.VIBE_CLIP_TEXT, cache_dir=str(config.MODELS_DIR / "fastembed"))


def visual_gap(frames, text: str) -> dict:
    """Cross-modal incongruity: CLIP similarity between the keyframes and the
    clip's words, turned into a gap (1 = picture and words unrelated). CLIP's text
    side is English-only, so this is only meaningful for English-ish captions."""
    if not frames or not (text or "").strip():
        return {}
    from .semspace import unit

    img_model, _ = vibe._clip()
    t = unit(next(iter(_clip_text().embed([text[:300]]))))
    sims = [float(unit(v) @ t) for v in img_model.embed([str(f) for f in frames])]
    best = max(sims)
    # CLIP ViT-B/32: matching photo/caption ~0.30, unrelated ~0.15.
    return {"clip_sim": round(best, 3), "gap": round(min(1.0, max(0.0, (0.30 - best) / 0.15)), 3)}
