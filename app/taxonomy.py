"""The labels a clip is tagged with: one place, four facets, about 50 labels each.

    reply   what the sender is saying when they send the clip
    event   which moment the clip suits
    funny   why it is funny
    topic   what it is about

Each facet is {group: {label: description}}. The description is what Laya and Jev
read when they choose, so it is short (at most 6 words) and plain. Groups exist so a
long list can be asked in two steps (group first, then the label inside it) on a
model that only fits about 20 options per question.

First draft (2026-10-09), to be revisited: check it with `python -m app.check_labels`
against real clips and queries, then split labels that are everywhere, drop labels
nothing uses and add the ones queries keep asking for.
"""
from __future__ import annotations

NONE = "none"

REPLY: dict[str, dict[str, str]] = {
    "support": {
        "agree": "yes, exactly, so true",
        "hype_up": "cheering someone on, you got this",
        "sympathise": "I feel your pain",
        "comfort": "it's okay, don't worry",
        "thank": "thank you, grateful",
        "proud_of_you": "so proud of you",
        "same_here": "me too, same situation",
    },
    "attack": {
        "disagree": "no, that is wrong",
        "mock": "making fun of someone",
        "roast": "a harsh savage insult",
        "call_out": "who did this, explain yourself",
        "blame": "this is your fault",
        "empty_threat": "a threat nobody takes seriously",
        "shut_up": "stop talking, keep quiet",
        "dismiss": "whatever, I don't care",
        "sarcasm": "saying the opposite of meaning",
    },
    "reaction": {
        "disbelief": "I can't believe this",
        "shocked": "stunned by sudden news",
        "mind_blown": "amazed, that is genius",
        "confused": "I don't understand anything",
        "facepalm": "this is so stupid",
        "awkward": "uncomfortable, pretending not there",
        "speechless": "no words left",
        "cringe": "embarrassing to watch",
    },
    "about_me": {
        "self_deprecate": "laughing at my own mess",
        "flex": "showing off what I have",
        "big_talk": "boasting about doing something huge",
        "excuse": "a weak excuse for failing",
        "confess": "admitting what I did",
        "plead": "please, I am begging",
        "panic": "we are in big trouble",
        "done_with_life": "tired, I give up",
        "lazy": "not doing anything today",
    },
    "social_move": {
        "greet": "hello, look who is here",
        "goodbye": "bye, I am leaving",
        "celebrate": "party, we did it",
        "lets_go": "come on, let's do it",
        "refuse": "no, not doing that",
        "apologise": "sorry, my mistake",
        "flirt": "romantic teasing",
        "tease": "playful friendly poking",
        "request": "asking for a favour",
        "warn": "be careful, trouble ahead",
        "lecture": "giving serious life advice",
        "told_you_so": "I was right all along",
        "mic_drop": "winning the argument completely",
        "impatient": "waiting, hurry up",
        "ignore": "leaving someone on read",
    },
}

EVENT: dict[str, dict[str, str]] = {
    "caught_out": {
        "caught": "caught red-handed doing something",
        "parents_find_out": "parents discover what you did",
        "lie_exposed": "a lie gets found out",
        "excuse": "making up an excuse",
        "reveal": "a secret comes out",
    },
    "work_and_study": {
        "exam_result": "exam results come out",
        "deadline": "deadline is close",
        "monday_morning": "start of the week",
        "boss_meeting": "facing the boss",
        "job_interview": "a job interview",
        "payday": "salary arrives",
        "salary_cut": "less money than expected",
    },
    "relationships": {
        "breakup": "a relationship ends",
        "proposal": "asking someone out or marriage",
        "crush_replies": "the crush finally replies",
        "left_on_read": "message seen, no reply",
        "friend_ditches": "a friend backs out",
        "wedding": "a wedding or shaadi",
    },
    "occasions": {
        "birthday": "someone's birthday",
        "festival": "Diwali, Holi, Eid, festivals",
        "new_year": "new year resolutions",
        "weekend": "weekend plans",
        "vacation": "a trip or holiday",
    },
    "daily_life": {
        "late_arrival": "arriving late",
        "wake_up": "waking up in the morning",
        "food_arrives": "food is finally here",
        "diet_fail": "diet or gym plan fails",
        "gym": "working out",
        "traffic": "stuck in traffic",
        "power_cut": "electricity or internet goes",
        "price_hike": "things get expensive",
    },
    "conflict_and_news": {
        "scolding": "getting scolded",
        "fight": "an argument or fight",
        "group_chat_chaos": "group chat goes wild",
        "match_win": "your team wins",
        "match_loss": "your team loses",
        "election_result": "election results",
        "plan_cancelled": "plans get cancelled",
        "showoff": "someone shows off",
        "favour_asked": "someone asks for help",
    },
}

FUNNY: dict[str, dict[str, str]] = {
    "language": {
        "broken_language": "funny broken or mixed English",
        "mispronunciation": "words said wrong",
        "wordplay": "a pun or double meaning",
        "misheard_line": "a line heard wrong",
        "catchphrase": "a famous repeated line",
    },
    "delivery": {
        "deadpan": "said with a straight face",
        "overacting": "dramatic exaggerated acting",
        "bluster": "loud confidence with nothing behind",
        "mock_serious": "serious tone about silly things",
        "awkward_silence": "an uncomfortable pause",
        "timing": "the cut or pause lands",
    },
    "content": {
        "twist": "an unexpected ending",
        "absurd": "makes no sense at all",
        "irony": "the opposite of what's expected",
        "mismatch": "words don't fit the situation",
        "dark": "humour about grim things",
        "naughty": "cheeky or double meaning",
        "wholesome": "sweet and kind",
        "relatable": "everyone has been there",
        "unexpected_honesty": "too honest to be polite",
    },
    "form": {
        "putdown": "a sharp insult",
        "self_roast": "mocking oneself",
        "slapstick": "physical falls and hits",
        "failed_attempt": "trying and failing badly",
        "reference": "points to a film or show",
        "remix_edit": "edited, dubbed or remixed",
        "repetition": "the same thing again and again",
        "over_the_top": "far bigger than needed",
    },
}

TOPIC: dict[str, dict[str, str]] = {
    "entertainment": {
        "bollywood": "Hindi films and film stars",
        "tv_serial": "daily soaps and TV shows",
        "web_series": "streaming series",
        "music": "songs and singers",
        "dance": "dancing",
        "gaming": "video games",
        "celebrity": "famous people, influencers",
    },
    "public_life": {
        "politics": "politicians and government",
        "elections": "voting and campaigns",
        "news": "news anchors and reports",
        "interview": "someone answering questions on camera",
        "speech": "a lecture, sermon or address",
        "police_crime": "police, thieves, crime",
        "court": "lawyers, judges, court",
    },
    "life_stages": {
        "school": "school kids and teachers",
        "college": "college life",
        "hostel": "hostel or PG life",
        "exams": "exams and results",
        "job": "jobs and job hunting",
        "office": "office, boss, colleagues",
        "startup": "startups and tech",
        "marriage": "marriage and in-laws",
        "kids": "children",
        "old_age": "elders and grandparents",
    },
    "people": {
        "family": "parents and home",
        "friends": "friends and gangs",
        "love": "romance and crushes",
        "relatives": "nosy relatives",
        "neighbours": "neighbours",
        "teacher": "teachers and lessons",
    },
    "everyday": {
        "food": "food and eating",
        "money": "money, rich or broke",
        "travel": "trips and transport",
        "traffic": "roads and traffic",
        "weather": "heat, rain, cold",
        "health_gym": "fitness and health",
        "phone": "phones and calls",
        "social_media": "Instagram, WhatsApp, reels",
        "shopping": "buying things, sales",
    },
    "sport_and_culture": {
        "cricket": "cricket and IPL",
        "other_sports": "football and other sports",
        "festival": "festivals and celebrations",
        "religion": "temples, prayers, faith",
        "animals": "pets and animals",
        "motivation": "inspiring, never give up",
    },
}

FACETS: dict[str, dict[str, dict[str, str]]] = {"reply": REPLY, "event": EVENT, "funny": FUNNY, "topic": TOPIC}

QUESTIONS: dict[str, str] = {
    "reply": "If someone sent this clip as a reply in a chat, what would it be saying?",
    "event": "Which moment does this clip show or suit?",
    "funny": "Why is this clip funny?",
    "topic": "What is this clip about?",
}


def labels(facet: str) -> dict[str, str]:
    """Every label of a facet with its description, groups flattened."""
    return {label: desc for group in FACETS[facet].values() for label, desc in group.items()}


def group_of(facet: str, label: str) -> str | None:
    return next((g for g, ls in FACETS[facet].items() if label in ls), None)
