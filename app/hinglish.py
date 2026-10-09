"""English words inside Hindi speech, spelled back in English.

Whisper with the language set to Hindi writes English words in Devanagari
(वीडियो, लाइक, कंट्रोल), and romanising that gives "veediyo", "laik", "kantrol":
nobody searches for those. This maps the romanised form of such a loanword back
to its English spelling, for words that came out of Devanagari only (text already
written in English is never touched).

The keys are not typed by hand: each English word lists how it is commonly written
in Devanagari, and the key is whatever our own romaniser makes of that. So the map
always matches what the pipeline produces. A word is left out when its romanised
form is also a common Hindi word (सर "head" vs sir, पास "near" vs pass, मैन vs मैं).
"""
from __future__ import annotations

# English -> Devanagari spellings seen in transcripts and chats. Add freely; the
# collision test in tests/test_hinglish.py guards against Hindi look-alikes.
LOANWORDS: dict[str, list[str]] = {
    # video / social
    "video": ["वीडियो", "विडियो", "वीडियोज़"], "like": ["लाइक"], "likes": ["लाइक्स"],
    "subscribe": ["सब्सक्राइब"], "channel": ["चैनल"], "comment": ["कमेंट"], "share": ["शेयर"],
    "million": ["मिलियन"], "views": ["व्यूज़", "व्यूज"], "status": ["स्टेटस"], "online": ["ऑनलाइन"],
    "message": ["मैसेज", "मेसेज"], "reply": ["रिप्लाई"], "block": ["ब्लॉक"], "delete": ["डिलीट"],
    "phone": ["फोन", "फ़ोन"], "mobile": ["मोबाइल"],
    # the sample clips
    "control": ["कंट्रोल", "कन्ट्रोल"], "correct": ["करेक्ट", "करैक्ट"],
    "excitement": ["एक्साइटमेंट", "एक्साइटमैंट"],
    "guaranteed": ["गारंटीड", "गैरंटीड", "गैरिंटेड", "गारंटेड"], "guarantee": ["गारंटी", "गैरंटी"],
    "money": ["मनी"], "follow": ["फॉलो", "फोलो"], "follows": ["फॉलोज़", "फोलोज", "फॉलोज"],
    # talk
    "sorry": ["सॉरी", "सोरी"], "please": ["प्लीज़", "प्लीज"], "thank": ["थैंक"], "thanks": ["थैंक्स"],
    "you": ["यू"], "okay": ["ओके"], "hello": ["हेलो", "हैलो"], "yes": ["यस"],
    "actually": ["एक्चुअली"], "basically": ["बेसिकली"], "literally": ["लिटरली"], "really": ["रियली"],
    "seriously": ["सीरियसली"], "serious": ["सीरियस"], "crazy": ["क्रेज़ी", "क्रेजी"], "super": ["सुपर"],
    "best": ["बेस्ट"], "next": ["नेक्स्ट"], "smart": ["स्मार्ट"], "handsome": ["हैंडसम"],
    "level": ["लेवल"], "style": ["स्टाइल"], "attitude": ["एटीट्यूड", "ऐटिट्यूड"], "power": ["पावर"],
    "mood": ["मूड"], "feeling": ["फीलिंग"], "emotional": ["इमोशनल"], "idea": ["आइडिया"],
    "chance": ["चांस"], "point": ["पॉइंट"], "points": ["पॉइंट्स"], "tension": ["टेंशन"], "problem": ["प्रॉब्लम", "प्रोब्लम"],
    "system": ["सिस्टम"], "time": ["टाइम"], "life": ["लाइफ"], "public": ["पब्लिक"],
    # people
    "friend": ["फ्रेंड"], "friends": ["फ्रेंड्स"], "girlfriend": ["गर्लफ्रेंड"], "boyfriend": ["बॉयफ्रेंड"],
    "husband": ["हसबैंड"], "wife": ["वाइफ"], "boss": ["बॉस"], "madam": ["मैडम"], "doctor": ["डॉक्टर"],
    "hero": ["हीरो"], "villain": ["विलेन"], "fan": ["फैन"], "team": ["टीम"], "mummy": ["मम्मी"],
    # work / study
    "office": ["ऑफिस", "ऑफ़िस"], "meeting": ["मीटिंग"], "salary": ["सैलरी"], "job": ["जॉब"],
    "deadline": ["डेडलाइन"], "college": ["कॉलेज"], "exam": ["एग्जाम", "एग्ज़ाम"], "result": ["रिजल्ट", "रिज़ल्ट"],
    "fail": ["फेल"], "topper": ["टॉपर"], "interview": ["इंटरव्यू"], "news": ["न्यूज़", "न्यूज"],
    "politics": ["पॉलिटिक्स"],
    # life
    "party": ["पार्टी"], "plan": ["प्लान"], "cancel": ["कैंसिल"], "weekend": ["वीकेंड"],
    "holiday": ["हॉलिडे"], "birthday": ["बर्थडे"], "gift": ["गिफ्ट"], "date": ["डेट"],
    "movie": ["मूवी"], "picture": ["पिक्चर"], "song": ["सॉन्ग"], "dance": ["डांस"],
    "cricket": ["क्रिकेट"], "match": ["मैच"], "lunch": ["लंच"], "dinner": ["डिनर"],
    "pizza": ["पिज़्ज़ा", "पिज्जा"], "chocolate": ["चॉकलेट"],
}


def build(romanise) -> dict[str, str]:
    """romanised Devanagari spelling -> English, using the pipeline's own romaniser."""
    out: dict[str, str] = {}
    for english, spellings in LOANWORDS.items():
        for s in spellings:
            out.setdefault(romanise(s), english)
    return out
