"""Text handling shared by indexing and search.

Two jobs:
  1. to_roman()   - Devanagari -> the casual Roman spelling people type in chats.
  2. normalise()  - squash spelling variation so "rasode me kon tha" and
                    "rasodey mein kaun tha" land on (nearly) the same string.

The SAME normalise() must run on stored text and on the user's query. If you change
a rule here, re-run `python -m app.reindex` so stored text is rebuilt.
"""
from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- #
# Devanagari -> casual Roman
# --------------------------------------------------------------------------- #

_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
}
# consonant + nukta (U+093C)
_NUKTA = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "फ": "f", "ड": "d", "ढ": "dh"}

_MATRAS = {
    "ा": "a", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॅ": "e", "ॉ": "o",
}
_VOWELS = {
    "अ": "a", "आ": "a", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऍ": "e", "ऑ": "o",
}
_SIGNS = {"ं": "n", "ँ": "n", "ः": "h"}
_VIRAMA = "्"
_NUKTA_CHAR = "़"
_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}

_DEVANAGARI_WORD = re.compile(r"[ऀ-ॿ]+")

INHERENT = "\0"  # marker for the unwritten 'a' every bare consonant carries


def _word_to_roman(word: str) -> str:
    word = unicodedata.normalize("NFD", word)
    # Each unit is [consonant, vowel, trailing_sign]
    units: list[list[str]] = []
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        if ch in _CONSONANTS:
            cons = _CONSONANTS[ch]
            if i + 1 < n and word[i + 1] == _NUKTA_CHAR:
                cons = _NUKTA.get(ch, cons)
                i += 1
            vowel = INHERENT
            if i + 1 < n and word[i + 1] == _VIRAMA:
                vowel = ""
                i += 1
            elif i + 1 < n and word[i + 1] in _MATRAS:
                vowel = _MATRAS[word[i + 1]]
                i += 1
            units.append([cons, vowel, ""])
        elif ch in _VOWELS:
            units.append(["", _VOWELS[ch], ""])
        elif ch in _SIGNS:
            if units:
                units[-1][2] += _SIGNS[ch]
        elif ch in _DIGITS:
            units.append(["", _DIGITS[ch], ""])
        # anything else (danda, om sign, stray marks) is dropped
        i += 1

    # Hindi drops the unwritten 'a' in predictable places ("schwa deletion"):
    #   - at the end of a word:           कौन  -> kaun  (not kauna)
    #   - between two sounded syllables:  करना -> karna (not karana)
    # Walk right to left so each decision sees the final state of its neighbour.
    last = len(units) - 1
    for idx in range(last, -1, -1):
        cons, vowel, sign = units[idx]
        if vowel != INHERENT or sign:
            continue
        if idx == last:
            if len(units) > 1:
                units[idx][1] = ""
            continue
        if idx == 0:
            continue
        prev_has_vowel = units[idx - 1][1] != ""
        nxt = units[idx + 1]
        next_has_vowel = nxt[0] != "" and nxt[1] != ""
        if prev_has_vowel and next_has_vowel:
            units[idx][1] = ""

    return "".join(c + ("a" if v == INHERENT else v) + s for c, v, s in units)


def to_roman(text: str) -> str:
    """Transliterate any Devanagari in `text`; everything else is left untouched."""
    return _DEVANAGARI_WORD.sub(lambda m: _word_to_roman(m.group(0)), text)


def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI_WORD.search(text))


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

# Whole-word chat shorthand -> the fuller spelling. Add to this as query logs
# show you new variants; it is the cheapest search-quality lever you have.
TOKEN_VARIANTS = {
    "h": "hai", "hain": "hai", "hy": "hai", "he": "hai",
    "me": "mein", "mei": "mein", "main": "mein", "mai": "mein",
    "nhi": "nahi", "nai": "nahi", "nahin": "nahi", "ni": "nahi",
    "kon": "kaun", "kya": "kya", "kia": "kya", "kyu": "kyun", "kyon": "kyun",
    "k": "ke", "ki": "ki", "ko": "ko",
    "plz": "please", "pls": "please", "plij": "please", "pliz": "please",
    "u": "you", "ur": "your", "r": "are",
    "bhai": "bhai", "bhaiya": "bhai", "bro": "bhai",
    "tha": "tha", "thi": "thi",
    "ha": "haan", "han": "haan",
}

# Sound-alike spellings, applied inside every word, in order.
_PHONETIC = [
    (re.compile(r"ee"), "i"),
    (re.compile(r"oo"), "u"),
    (re.compile(r"(.)\1+"), r"\1"),   # kyaaaa -> kya, accha -> acha
    (re.compile(r"au|ou|ow"), "o"),
    (re.compile(r"ei|ey"), "e"),         # mein/men, rasodey/rasode ("ai" is left alone: bhai, chai)
    (re.compile(r"iy"), "i"),          # jodiye -> jodie
    (re.compile(r"ph"), "f"),
    (re.compile(r"ck"), "k"),
    (re.compile(r"w"), "v"),
    (re.compile(r"z"), "j"),
    (re.compile(r"q"), "k"),
    (re.compile(r"(?<=[tdbkgj])h"), ""),  # haath/hath/haat, bhai/bai
]

_NOT_WORD = re.compile(r"[^a-z0-9]+")


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalise_token(token: str) -> str:
    token = TOKEN_VARIANTS.get(token, token)
    if token.isdigit():
        return token
    for pattern, repl in _PHONETIC:
        token = pattern.sub(repl, token)
    return token


def normalise(text: str | None) -> str:
    """Lowercase, romanise, drop punctuation and emoji, squash spelling variants."""
    if not text:
        return ""
    text = to_roman(text)
    text = _strip_accents(text).lower()
    tokens = [t for t in _NOT_WORD.split(text) if t]
    return " ".join(normalise_token(t) for t in tokens)


# --------------------------------------------------------------------------- #
# Cleaning source metadata
# --------------------------------------------------------------------------- #

NOISE_TAGS = {
    "viral", "fyp", "foryou", "foryoupage", "trending", "reels", "reel", "shorts",
    "explore", "explorepage", "instagram", "insta", "tiktok", "youtube", "ytshorts",
    "follow", "like", "share", "subscribe", "memes", "meme", "funny", "comedy",
    "instagood", "viralvideos", "viralreels", "trendingreels", "reelsinstagram",
    "greenscreen", "template", "memetemplate", "nocopyright", "hd", "4k",
}

_BOILERPLATE = re.compile(
    r"(follow\s+(me|us|for more)|subscribe|like\s+and\s+share|link\s+in\s+bio|"
    r"turn\s+on\s+notifications?|credits?\s*:|dm\s+for|no\s+copyright|"
    r"green\s*screen|meme\s+template|template)",
    re.IGNORECASE,
)
_HASHTAG = re.compile(r"#(\w+)")
_URL = re.compile(r"https?://\S+")
_MENTION = re.compile(r"@\w+")


def clean_caption(text: str | None) -> tuple[str, list[str]]:
    """Split a title or caption into (meaningful text, meaningful hashtags)."""
    if not text:
        return "", []
    tags = [t.lower() for t in _HASHTAG.findall(text)]
    tags = [t for t in dict.fromkeys(tags) if t not in NOISE_TAGS]
    body = _HASHTAG.sub(" ", text)
    body = _URL.sub(" ", body)
    body = _MENTION.sub(" ", body)
    body = _BOILERPLATE.sub(" ", body)
    body = re.sub(r"[|•·\-–—_]+", " ", body)
    body = re.sub(r"\s+", " ", body).strip()
    return body, tags
