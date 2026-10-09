"""What a meme says when someone sends it in a chat. No LLM.

A meme in a chat is a reply: the same clip means "congrats!" after good news and
"yeah right" after a brag. So the reading is a product of experts (Hinton 2002),
each giving a probability for every INTENT:

  meme      what the clip itself tends to say: its moods, topics, use-when lines
            and title vs the intent prototypes, plus how people have actually
            used this clip before (a Dirichlet posterior over past feedback)
  context   what the last messages call for. Each message is classified into a
            dialogue act (good news, brag, confession ...) and pushed through a
            transition matrix of adjacency pairs (conversation analysis: news ->
            congratulations, request -> refusal, confession -> call-out). The
            matrix starts from TRANSITIONS and is updated from feedback.
  valence   sentiment contrast between message and meme (Riloff et al. 2013:
            sarcasm = positive words about a negative situation): a cheerful
            message answered with a grim face reads as irony, a sad one answered
            with a sad one reads as sympathy.
  caption   what the sender typed with the meme ("me rn", "you 😂").

Pooling is geometric (p ∝ Π expert ** weight), so the winner must be plausible to
every expert that has an opinion; each expert has a floor so none can veto alone.
"""
from __future__ import annotations

import math

from . import humor
from .semspace import (Space, entropy_confidence, get_space, mix, normalise_dist,
                       product_of_experts, sigmoid)

INTENTS: dict[str, dict] = {
    "agree": {"protos": ["exactly, so true", "I agree one hundred percent", "yes, this is it"],
              "gloss": "\"So true / exactly this.\" Agreeing with what was said.", "target": "message"},
    "disagree": {"protos": ["no way, that's wrong", "I completely disagree", "absolutely not true"],
                 "gloss": "\"Nope, not buying it.\" Disagreeing with the message.", "target": "message"},
    "mock": {"protos": ["making fun of you", "look at this idiot", "roasting you hard"],
             "gloss": "Roasting the other person: laughing at them (usually affectionately).",
             "target": "recipient"},
    "sarcasm": {"protos": ["oh wow, what a genius idea", "yeah right, sure", "saying the opposite of what I mean"],
                "gloss": "Irony: the reaction means the opposite of what it shows.", "target": "message"},
    "celebrate": {"protos": ["congratulations, let's party", "we did it, celebrating", "great news, dancing"],
                  "gloss": "Celebrating and hyping it up.", "target": "message"},
    "sympathise": {"protos": ["I feel you, that's rough", "same here, we are in this together",
                              "so sorry, sending hugs"],
                   "gloss": "\"I feel you.\" Sympathy or shared pain.", "target": "recipient"},
    "complain": {"protos": ["I am so fed up with this", "this is so annoying", "why does this always happen"],
                 "gloss": "Venting frustration about the situation.", "target": "situation"},
    "disbelief": {"protos": ["I can't believe this", "what just happened", "are you serious right now"],
                  "gloss": "Shock and disbelief at what was just said.", "target": "message"},
    "call_out": {"protos": ["who did this", "caught you red-handed", "explain yourself right now"],
                 "gloss": "Calling someone out: \"who did this?\" / \"caught you\".", "target": "recipient"},
    "refuse": {"protos": ["no, I'm not doing that", "I'm out, bye", "not happening"],
               "gloss": "A refusal: \"not doing it\" / \"I'm out\".", "target": "message"},
    "confused": {"protos": ["I don't understand anything", "what is going on", "this makes no sense"],
                 "gloss": "Confusion: \"what are you even saying?\"", "target": "message"},
    "self_deprecate": {"protos": ["that's so me, I'm a mess", "me failing again", "my life is a joke"],
                       "gloss": "Laughing at themselves: \"that's me\".", "target": "self"},
    "told_you_so": {"protos": ["I told you so", "I was right all along", "who's laughing now"],
                    "gloss": "\"Told you so.\" Smug vindication.", "target": "recipient"},
    "awkward": {"protos": ["this is so awkward", "cringe, let me leave", "pretending I didn't see that"],
                "gloss": "Awkward / cringe: \"I'll just slowly back away\".", "target": "situation"},
    "panic": {"protos": ["we are in big trouble", "run, everything is on fire", "the deadline is now"],
              "gloss": "Panic: \"we're doomed\".", "target": "situation"},
    "lecture": {"protos": ["listen to my advice", "here is a life lesson", "you should have known better"],
                "gloss": "Giving (mock-)serious advice or a lecture.", "target": "recipient"},
}
INTENT_PROTOS = {k: v["protos"] for k, v in INTENTS.items()}

# Dialogue acts a message can perform. Hinglish examples on purpose: the clips are.
CONTEXT_ACTS: dict[str, list[str]] = {
    "good_news": ["I got the job!", "we won the match", "finally passed my exam", "mujhe promotion mil gaya"],
    "bad_news": ["I failed the exam", "my flight got cancelled", "I lost my phone", "exam mein fail ho gaya"],
    "complaint": ["my boss is so annoying", "I'm so tired of this", "traffic is terrible again",
                  "yeh kya bakwas hai"],
    "opinion": ["I think pineapple belongs on pizza", "this is the best movie ever",
                "honestly cats are better than dogs"],
    "brag": ["I'm the best at this", "look at my new car", "I finished everything in one day, easy"],
    "question": ["who ate my food?", "what are you doing?", "why is nobody replying?", "kaun tha?"],
    "request": ["can you cover my shift tomorrow", "lend me some money", "please do my homework",
                "bhai paise de de"],
    "confession": ["I broke the vase", "I forgot your birthday", "I ate the last piece",
                   "maine galti se delete kar diya"],
    "plan": ["let's go out tonight", "party at my place", "chalo trip pe chalte hain"],
    "joke": ["haha that's hilarious", "lol", "I'm dead 😂"],
    "lecture": ["you should always save money", "let me explain how the economy works",
                "wake up early, it's good for you"],
}

# Adjacency pairs: after a message doing X, a reply usually does Y. Relative weights.
TRANSITIONS: dict[str, dict[str, float]] = {
    "good_news": {"celebrate": 3, "disbelief": 1.5, "sarcasm": 0.7, "mock": 0.5, "agree": 0.5},
    "bad_news": {"sympathise": 3, "disbelief": 1, "mock": 1, "complain": 0.5, "panic": 0.5,
                 "self_deprecate": 0.5},
    "complaint": {"sympathise": 2, "agree": 2, "complain": 1.5, "mock": 0.7, "lecture": 0.5},
    "opinion": {"agree": 2, "disagree": 2, "sarcasm": 1, "disbelief": 0.7, "mock": 0.7, "confused": 0.5},
    "brag": {"mock": 2, "sarcasm": 2, "disbelief": 1, "celebrate": 1, "told_you_so": 0.3},
    "question": {"confused": 1.5, "refuse": 1, "call_out": 1, "awkward": 1, "agree": 0.5, "disagree": 0.5},
    "request": {"refuse": 3, "awkward": 1, "sarcasm": 1, "agree": 0.7},
    "confession": {"call_out": 2.5, "disbelief": 2, "mock": 1.5, "told_you_so": 1, "sympathise": 0.5},
    "plan": {"celebrate": 2, "agree": 1.5, "refuse": 1.5, "panic": 0.3},
    "joke": {"agree": 1, "mock": 1, "self_deprecate": 0.7, "celebrate": 0.5, "disbelief": 0.5, "awkward": 0.5},
    "lecture": {"sarcasm": 1.5, "confused": 1, "refuse": 1, "mock": 1, "awkward": 0.7, "agree": 0.7,
                "lecture": 0.5},
}

MOOD_TO_INTENT: dict[str, dict[str, float]] = {
    "happy": {"celebrate": 1, "agree": 0.6},
    "laughing": {"mock": 1, "agree": 0.5, "celebrate": 0.4},
    "dancing": {"celebrate": 1.5},
    "shocked": {"disbelief": 1.5, "call_out": 0.4},
    "sad": {"sympathise": 0.8, "self_deprecate": 0.8, "complain": 0.5},
    "angry": {"complain": 1, "call_out": 0.8, "disagree": 0.6},
    "scared": {"panic": 1.3, "awkward": 0.4},
    "disgusted": {"disagree": 0.8, "awkward": 0.8, "mock": 0.4},
    "smug": {"told_you_so": 1.3, "mock": 0.7, "sarcasm": 0.5},
    "confused": {"confused": 1.5},
    "helpless": {"self_deprecate": 0.8, "complain": 0.6, "panic": 0.4},
    "in a hurry": {"panic": 1.0, "refuse": 0.5},
    "preachy": {"lecture": 1.5, "sarcasm": 0.3},
}

TOPIC_TO_INTENT: dict[str, dict[str, float]] = {
    "sarcasm": {"sarcasm": 1.5}, "roast": {"mock": 1.5},
    "fail": {"self_deprecate": 0.6, "mock": 0.6, "sympathise": 0.3},
    "celebration": {"celebrate": 1.5}, "confusion": {"confused": 1.5}, "cringe": {"awkward": 1.5},
    "late": {"panic": 1.0}, "speech": {"lecture": 1.0}, "motivation": {"lecture": 0.8, "celebrate": 0.3},
    "wholesome": {"sympathise": 0.8, "celebrate": 0.4}, "work": {"complain": 0.5, "panic": 0.3},
    "study": {"panic": 0.5, "complain": 0.4}, "money": {"complain": 0.4, "refuse": 0.4},
    "politics": {"sarcasm": 0.5}, "family": {"call_out": 0.4, "lecture": 0.4},
}

MECH_TO_INTENT: dict[str, dict[str, float]] = {
    "superiority": {"mock": 1.0}, "self_deprecation": {"self_deprecate": 1.0},
    "tonal_mismatch": {"sarcasm": 1.0}, "relatability": {"agree": 0.5, "self_deprecate": 0.5},
}

# (message sign, meme sign) -> what the contrast suggests.
VALENCE_PAIRS: dict[tuple[str, str], dict[str, float]] = {
    ("+", "+"): {"celebrate": 2, "agree": 1},
    ("+", "-"): {"sarcasm": 1.5, "disbelief": 1.2, "mock": 0.8},
    ("-", "-"): {"sympathise": 1.5, "agree": 1, "complain": 1, "self_deprecate": 0.5},
    ("-", "+"): {"mock": 1.5, "sarcasm": 1, "celebrate": 0.3},
}

# How a clip is used against its face value, by the sign of its mood.
IRONIC_USE: dict[str, dict[str, float]] = {
    "+": {"sarcasm": 1.0, "mock": 1.0},
    "-": {"self_deprecate": 1.0, "sympathise": 0.5, "complain": 0.5},
    "0": {"sarcasm": 0.5, "confused": 0.5, "awkward": 0.5},
}
IRONY_SHARE = 0.25
UNSURE_BELOW = 0.2   # confidence under which the reading says it is only a guess

W_MEME, W_CONTEXT, W_CAPTION, W_VALENCE = 1.0, 0.8, 0.9, 0.6
CLIP_PRIOR_STRENGTH = 4.0     # pseudo-counts behind the clip's own reading
TRANSITION_PRIOR_STRENGTH = 6.0


def _spread(table: dict[str, dict[str, float]], weights: dict[str, float]) -> dict[str, float]:
    """Σ weight(key) * table[key] -> unnormalised scores per intent."""
    out: dict[str, float] = {}
    for key, w in weights.items():
        for intent, a in table.get(key, {}).items():
            out[intent] = out.get(intent, 0.0) + w * a
    return out


# --------------------------------------------------------------------------- #
# Learning from feedback (counts in, distributions out)
# --------------------------------------------------------------------------- #

def clip_posterior(prior: dict[str, float], counts: dict[str, float] | None,
                   strength: float = CLIP_PRIOR_STRENGTH) -> dict[str, float]:
    """Dirichlet-multinomial update: how this clip has been read before, with the
    content-based reading as `strength` pseudo-observations. One label nudges it,
    twenty labels dominate it."""
    counts = {k: v for k, v in (counts or {}).items() if k in INTENTS}
    n = sum(counts.values())
    if n <= 0:
        return prior
    return normalise_dist({k: (counts.get(k, 0.0) + strength * prior.get(k, 0.0)) / (n + strength)
                           for k in INTENTS}, INTENTS, floor=0.0)


def transition_matrix(act_counts: dict[str, dict[str, float]] | None = None,
                      strength: float = TRANSITION_PRIOR_STRENGTH) -> dict[str, dict[str, float]]:
    """TRANSITIONS as probabilities, each row updated with feedback soft counts
    {act: {intent: weight}} (Dirichlet again, `strength` pseudo-counts per row)."""
    out = {}
    for act in CONTEXT_ACTS:
        prior = normalise_dist(TRANSITIONS.get(act, {}), INTENTS, floor=0.01)
        counts = (act_counts or {}).get(act, {})
        n = sum(counts.values())
        out[act] = {k: (counts.get(k, 0.0) + strength * prior[k]) / (n + strength) for k in INTENTS}
    return out


# --------------------------------------------------------------------------- #
# Experts
# --------------------------------------------------------------------------- #

def meme_expert(facts: dict, mech: dict | None, space: Space, about_vec=None) -> tuple[dict[str, float], list[str]]:
    """What the clip says on its own: a mixture of four weak readings."""
    evidence = []
    v = about_vec if about_vec is not None else space.vec(facts["about"] or facts["title"] or " ")
    text = space.classify(v, INTENT_PROTOS, temperature=0.6) if facts["about"] else {}
    moods = {m: s for m, s in facts["moods"].items() if s >= 0.3}
    by_mood = _spread(MOOD_TO_INTENT, moods)
    by_topic = _spread(TOPIC_TO_INTENT, {t: 1.0 for t in facts["topics"]})
    by_mech = _spread(MECH_TO_INTENT, {k: mech[k]["score"] for k in MECH_TO_INTENT if mech and mech[k]["score"] >= 0.4})
    if moods:
        evidence.append("the clip's mood: " + ", ".join(sorted(moods, key=lambda m: -moods[m])[:3]))
    if by_topic:
        evidence.append("it is about: " + ", ".join(t for t in facts["topics"] if t in TOPIC_TO_INTENT))
    literal = mix([(1.0, text), (1.3, normalise_dist(by_mood, INTENTS, 0) if by_mood else {}),
                   (1.0, normalise_dist(by_topic, INTENTS, 0) if by_topic else {}),
                   (0.6, normalise_dist(by_mech, INTENTS, 0) if by_mech else {})], INTENTS)
    # Memes are polysemous: a happy clip is also the classic way to mock or be
    # sarcastic, a sad one the classic way to laugh at yourself. Keep that mass.
    valence = humor.mood_valence(facts["moods"])
    ironic = IRONIC_USE["+" if valence > 0.2 else "-" if valence < -0.2 else "0"]
    dist = mix([(1 - IRONY_SHARE, literal), (IRONY_SHARE, normalise_dist(ironic, INTENTS, 0))], INTENTS)
    return dist, evidence


def context_acts(messages: list[str], space: Space, vecs=None) -> dict[str, float]:
    """Dialogue-act distribution of the chat so far; the last message counts most."""
    if not messages:
        return {}
    vecs = vecs if vecs is not None else space.vecs(messages)
    total: dict[str, float] = {a: 0.0 for a in CONTEXT_ACTS}
    weight_sum = 0.0
    for age, (msg, v) in enumerate(zip(reversed(messages), reversed(list(vecs)))):
        w = 0.5 ** age
        acts = space.classify(v, CONTEXT_ACTS, temperature=0.5)
        # Surface cues the embedding can miss in two-word chat lines.
        cues: dict[str, float] = {}
        if "?" in msg:
            cues["question"] = 1.0
        if humor.count(humor.LAUGH, msg):
            cues["joke"] = 1.0
        if humor.count(humor.POSITIVE_WORDS, msg) > humor.count(humor.NEGATIVE_WORDS, msg):
            cues["good_news"] = 0.6
        elif humor.count(humor.NEGATIVE_WORDS, msg):
            cues["bad_news"], cues["complaint"] = 0.4, 0.4
        if cues:
            acts = mix([(1.0, acts), (0.6, normalise_dist(cues, CONTEXT_ACTS, 0))], CONTEXT_ACTS)
        for a, p in acts.items():
            total[a] += w * p
        weight_sum += w
    return {a: p / weight_sum for a, p in total.items()}


def context_expert(acts: dict[str, float], matrix: dict[str, dict[str, float]]) -> dict[str, float]:
    """P(intent | chat) = Σ_act P(act) · P(intent | act)."""
    if not acts:
        return {}
    return {i: sum(p * matrix[a][i] for a, p in acts.items()) for i in INTENTS}


def valence_expert(v_ctx: float, v_meme: float) -> tuple[dict[str, float], float]:
    """Soft 2x2 sentiment contrast. Returns the reading and how much it should count
    (nothing when either side is neutral)."""
    pc, pm = (1 + v_ctx) / 2, (1 + v_meme) / 2
    sides = {"+": (pc, pm), "-": (1 - pc, 1 - pm)}
    scores: dict[str, float] = {}
    for (c, m), table in VALENCE_PAIRS.items():
        w = sides[c][0] * sides[m][1]
        for i, a in table.items():
            scores[i] = scores.get(i, 0.0) + w * a
    strength = math.sqrt(abs(v_ctx) * abs(v_meme))
    return normalise_dist(scores, INTENTS, 0), strength


def caption_expert(caption: str, space: Space) -> dict[str, float]:
    if not (caption or "").strip():
        return {}
    dist = space.classify(space.vec(caption), INTENT_PROTOS, temperature=0.6)
    cues: dict[str, float] = {}
    if humor.count(humor.FIRST_PERSON, caption) and not humor.count(humor.SECOND_PERSON, caption):
        cues.update(self_deprecate=1.0, agree=0.4)
    if humor.count(humor.SECOND_PERSON, caption):
        cues.update(mock=1.0, call_out=0.6)
    if humor.count(humor.SARCASM_MARK, caption):
        cues["sarcasm"] = cues.get("sarcasm", 0) + 1.5
    if humor.count(humor.LAUGH, caption):
        cues["mock"] = cues.get("mock", 0) + 0.5
    if humor.count(humor.RELATABLE_FORMAT, caption):
        cues["self_deprecate"] = cues.get("self_deprecate", 0) + 0.5
    return mix([(1.0, dist), (1.2, normalise_dist(cues, INTENTS, 0) if cues else {})], INTENTS)


# --------------------------------------------------------------------------- #
# The reading
# --------------------------------------------------------------------------- #

def _top(d: dict[str, float], n: int) -> list[dict]:
    return [{"label": k, "p": round(v, 3)} for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:n]]


def read(facts: dict, context: list[str] | None = None, caption: str = "", mech: dict | None = None,
         clip_counts: dict[str, float] | None = None, act_counts: dict[str, dict[str, float]] | None = None,
         space: Space | None = None) -> dict:
    """What this clip communicates, sent after `context` (oldest first) with `caption`."""
    space = space or get_space()
    context = [c.strip() for c in (context or []) if c and c.strip()]
    about_vec = space.vec(facts["about"] or facts["title"] or " ")

    meme_prior, evidence = meme_expert(facts, mech, space, about_vec)
    meme = clip_posterior(meme_prior, clip_counts)
    if clip_counts:
        n = int(sum(clip_counts.values()))
        evidence.append(f"{n} earlier reading{'s' if n != 1 else ''} of this clip")
    v_text = humor.text_valence(facts["about"], space, about_vec)
    v_meme = (0.5 * humor.mood_valence(facts["moods"]) + 0.5 * v_text) if facts["moods"] else v_text

    experts = [(W_MEME, meme)]
    acts, v_ctx, fit = {}, 0.0, None
    if context:
        vecs = space.vecs(context)
        acts = context_acts(context, space, vecs)
        experts.append((W_CONTEXT, context_expert(acts, transition_matrix(act_counts))))
        last = context[-1]
        v_ctx = humor.text_valence(last, space, vecs[-1])
        val, strength = valence_expert(v_ctx, v_meme)
        experts.append((W_VALENCE * strength, val))
        fit = sigmoid(space.z(vecs[-1], about_vec) - 1.0)
        top_act = max(acts, key=acts.get)
        evidence.append(f"the message reads as {top_act.replace('_', ' ')} ({acts[top_act]:.0%})")
        if strength >= 0.25:
            sign = lambda v: "positive" if v > 0 else "negative"   # noqa: E731
            evidence.append(f"a {sign(v_ctx)} message answered by a {sign(v_meme)} clip")
    cap = caption_expert(caption, space)
    if cap:
        experts.append((W_CAPTION, cap))
        evidence.append(f"the sender's caption: \"{caption.strip()[:60]}\"")

    final = product_of_experts(experts, INTENTS)
    best = max(final, key=final.get)
    target = INTENTS[best]["target"]
    if caption:
        if humor.count(humor.FIRST_PERSON, caption) and not humor.count(humor.SECOND_PERSON, caption):
            target = "self"
        elif humor.count(humor.SECOND_PERSON, caption):
            target = "recipient"
    if fit is not None and fit < 0.25:
        evidence.append("the clip is only loosely related to the message: a generic reaction")
    confidence = final[best] * (0.5 + 0.5 * entropy_confidence(final))
    reading = INTENTS[best]["gloss"]
    if confidence < UNSURE_BELOW:
        reading = f"No clear message. Best guess: {reading[0].lower()}{reading[1:]}"
    return {
        "intent": best,
        "reading": reading,
        "aimed_at": target,
        "confidence": round(confidence, 3),
        "intents": _top(final, 5),
        "evidence": evidence,
        "experts": {"meme": _top(meme, 3), "context_acts": _top(acts, 3) if acts else [],
                    "valence": {"message": round(v_ctx, 2), "meme": round(v_meme, 2)},
                    "fit": round(fit, 3) if fit is not None else None},
        "acts": {k: round(v, 4) for k, v in acts.items()},   # stored with feedback, for learning
    }
