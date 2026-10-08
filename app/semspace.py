"""Shared vector maths for the meme understanding engine (humor.py, intent.py).

Everything is measured in the sentence-embedding space the search already uses
(multilingual MiniLM, app/embed.py). No LLM: sentences go in, unit vectors come
out, and every judgement is a comparison between vectors.

Raw cosine similarities are not comparable across models or sentence lengths
(MiniLM puts two unrelated sentences at ~0.1, two related ones at ~0.5, other
models elsewhere). So similarities are read as z-scores against BACKGROUND, a
fixed set of ordinary chat sentences: "how much closer is x to y than to an
everyday sentence?". z = 0 is unrelated, z = 2 is clearly related, whatever
model is behind it.

Prototype groups ({name: [sentence, ...]}) are embedded once and averaged into
one unit vector per name.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Iterable

import numpy as np

Embed = Callable[[list[str]], list[list[float]]]

# Ordinary, mixed-topic chat lines: the "nothing in particular" baseline that
# z-scores are measured against. Kept varied on purpose; do not make it all memes.
BACKGROUND = [
    "what time is the meeting tomorrow",
    "I am going to the market to buy vegetables",
    "the train leaves at seven in the morning",
    "can you send me the document",
    "it is raining heavily today",
    "my phone battery is almost dead",
    "let's have dinner at eight",
    "the movie starts in ten minutes",
    "I need to pay the electricity bill",
    "she is reading a book in the garden",
    "the shop is closed on sunday",
    "he bought a new laptop last week",
    "please call me when you reach home",
    "the weather is nice this evening",
    "we are watching the cricket match",
    "my cousin is visiting next month",
    "I forgot my umbrella at the office",
    "the coffee here is good",
    "traffic on the highway is slow",
    "the exam results come out on friday",
    "kal office jaldi jana hai",
    "khana ban gaya hai aa jao",
    "main ghar pe hoon abhi",
    "bhai kal milte hain",
    "the dog is sleeping on the sofa",
    "I am cooking rice and dal",
    "the package was delivered this morning",
    "we moved to a new apartment",
    "turn off the lights before leaving",
    "the printer is out of paper",
    "I watched a documentary about oceans",
    "he plays guitar on weekends",
    "the bus was full today",
    "my sister is learning to drive",
    "send the photos from the trip",
    "the food delivery is late",
    "I have a dentist appointment",
    "the wifi password changed",
    "we should water the plants",
    "the store has a sale this week",
]


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def softmax(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    if not scores:
        return {}
    m = max(scores.values())
    e = {k: math.exp((v - m) / temperature) for k, v in scores.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def normalise_dist(d: dict[str, float], keys: Iterable[str], floor: float = 1e-3) -> dict[str, float]:
    """A probability distribution over exactly `keys`; every key gets at least `floor`
    before renormalising, so one silent expert can never veto an option outright."""
    keys = list(keys)
    raw = {k: max(0.0, float(d.get(k, 0.0))) for k in keys}
    total = sum(raw.values())
    if total <= 0:
        return {k: 1.0 / len(keys) for k in keys}
    raw = {k: v / total + floor for k, v in raw.items()}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def mix(dists: list[tuple[float, dict[str, float]]], keys: Iterable[str]) -> dict[str, float]:
    """Weighted average of distributions (a mixture); empty ones are skipped."""
    keys = list(keys)
    parts = [(w, d) for w, d in dists if d and w > 0]
    if not parts:
        return {k: 1.0 / len(keys) for k in keys}
    total = sum(w for w, _ in parts)
    return normalise_dist({k: sum(w * d.get(k, 0.0) for w, d in parts) / total for k in keys}, keys)


def product_of_experts(experts: list[tuple[float, dict[str, float]]], keys: Iterable[str]) -> dict[str, float]:
    """Weighted geometric pooling: p(k) ∝ Π expert_i(k) ** w_i.

    Unlike a mixture, an option must be plausible to every expert that speaks:
    the meme says "shocked", the chat says "good news" -> the reading that fits
    both (disbelief) wins over what fits only one."""
    keys = list(keys)
    parts = [(w, normalise_dist(d, keys)) for w, d in experts if d and w > 0]
    if not parts:
        return {k: 1.0 / len(keys) for k in keys}
    logs = {k: sum(w * math.log(d[k]) for w, d in parts) for k in keys}
    return softmax(logs)


def entropy_confidence(p: dict[str, float]) -> float:
    """1 - normalised entropy: 1.0 = one option certain, 0.0 = all equally likely."""
    if len(p) < 2:
        return 1.0
    h = -sum(v * math.log(v) for v in p.values() if v > 0)
    return max(0.0, 1.0 - h / math.log(len(p)))


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence in bits (0 = same, 1 = disjoint) between two
    score dicts, each renormalised over the union of their keys."""
    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0
    ps, qs = sum(max(0.0, p.get(k, 0.0)) for k in keys), sum(max(0.0, q.get(k, 0.0)) for k in keys)
    if ps <= 0 or qs <= 0:
        return 0.0
    out = 0.0
    for k in keys:
        a, b = max(0.0, p.get(k, 0.0)) / ps, max(0.0, q.get(k, 0.0)) / qs
        m = (a + b) / 2
        if a > 0:
            out += 0.5 * a * math.log2(a / m)
        if b > 0:
            out += 0.5 * b * math.log2(b / m)
    return max(0.0, min(1.0, out))


class Space:
    """Sentences -> unit vectors, plus the z-scored comparisons the engine needs.

    `embed` is any function list[str] -> list[vector]; tests pass a tiny
    deterministic one, production passes app.embed.get_embedder().embed."""

    def __init__(self, embed: Embed):
        self._embed = embed
        self._groups: dict[tuple, tuple[list[str], np.ndarray]] = {}
        self._bg = self.vecs(BACKGROUND)

    def vecs(self, texts: list[str]) -> np.ndarray:
        return np.vstack([unit(v) for v in self._embed([t or " " for t in texts])])

    def vec(self, text: str) -> np.ndarray:
        return self.vecs([text])[0]

    def groups(self, protos: dict[str, list[str]]) -> tuple[list[str], np.ndarray]:
        """One averaged unit vector per prototype group, embedded once per content."""
        key = tuple((n, tuple(ss)) for n, ss in protos.items())
        if key not in self._groups:
            names = list(protos)
            flat = [s for n in names for s in protos[n]]
            vs = self.vecs(flat)
            rows, i = [], 0
            for n in names:
                k = len(protos[n])
                rows.append(unit(vs[i:i + k].mean(axis=0)))
                i += k
            self._groups[key] = (names, np.vstack(rows))
        return self._groups[key]

    def baseline(self, x: np.ndarray) -> tuple[float, float]:
        sims = self._bg @ x
        return float(sims.mean()), float(sims.std()) or 1e-6

    def z(self, x: np.ndarray, y: np.ndarray) -> float:
        """How related x and y are, in standard deviations above x's everyday baseline."""
        mu, sd = self.baseline(x)
        return (float(x @ y) - mu) / sd

    def z_groups(self, x: np.ndarray, protos: dict[str, list[str]]) -> dict[str, float]:
        names, m = self.groups(protos)
        mu, sd = self.baseline(x)
        return {n: (float(s) - mu) / sd for n, s in zip(names, m @ x)}

    def classify(self, x: np.ndarray, protos: dict[str, list[str]], temperature: float = 0.5) -> dict[str, float]:
        """Zero-shot distribution over prototype groups (softmax of z-scores)."""
        return softmax(self.z_groups(x, protos), temperature)


@lru_cache(maxsize=1)
def get_space() -> Space:
    from .embed import get_embedder

    return Space(get_embedder().embed)
