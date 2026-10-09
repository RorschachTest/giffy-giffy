"""English said inside Hindi comes back in English; Hindi words are never touched. No models."""
from . import conftest  # noqa: F401

from app import hinglish, text

# Common Hindi words (as our romaniser writes them) that a loanword key must never equal.
HINDI = {"main", "mein", "sar", "pas", "bas", "man", "kal", "aj", "hai", "kya", "kar", "par", "ho",
         "ka", "ki", "ke", "ko", "se", "to", "na", "nahin", "naheen", "bat", "dam", "nam", "log",
         "yar", "tum", "ap", "ham", "vo", "ye", "jo", "ab", "tab", "jab", "kab", "sab", "rat", "din",
         "ghar", "pyar", "dil", "mat", "bhee", "hee", "bap", "ma", "beta", "desh", "chal", "lal"}


def test_loanwords_in_devanagari_come_back_in_english():
    # what Whisper actually wrote on the sample clips
    assert text.to_roman("अगर तुम्हें वीडियो पसंद आई तो लाइक मारो") == "agar tumhen video pasand aee to like maro"
    assert text.to_roman("कंट्रोल, कंट्रोल उदे") == "control, control ude"
    assert text.to_roman("मनी फोलोज, नाम या दाम") == "money follows, nam ya dam"
    assert text.to_roman("करेक्ट फैसला किया है") == "correct phaisla kiya hai"
    assert text.to_roman("2 मिलियन लाइक्स") == "2 million likes"


def test_english_already_in_latin_is_untouched():
    assert text.to_roman("like maro yaar, Control") == "like maro yaar, Control"


def test_no_loanword_key_is_a_hindi_word():
    keys = set(hinglish.build(text._word_to_roman))
    assert not (keys & HINDI), keys & HINDI
    assert not (keys & set(text.TOKEN_VARIANTS)), keys & set(text.TOKEN_VARIANTS)
