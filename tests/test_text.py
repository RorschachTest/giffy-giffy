from app.text import clean_caption, normalise, to_roman


def test_to_roman_basic():
    assert to_roman("रसोड़े में कौन था") == "rasode men kaun tha"
    assert to_roman("करना") == "karna"
    assert to_roman("लड़का") == "ladka"
    assert to_roman("समझना") == "samajhna"
    assert to_roman("नमस्ते") == "namaste"
    assert to_roman("कमल") == "kamal"
    assert to_roman("क्या गुंडा बनेगा रे तू") == "kya gunda banega re too"


def test_to_roman_leaves_other_text_alone():
    assert to_roman("hello 123 दोस्त!") == "hello 123 dost!"


def test_variants_collapse_to_same_string():
    target = normalise("रसोड़े में कौन था")
    for typed in [
        "rasode me kon tha",
        "Rasode mein kaun tha?",
        "rasodey mein kaun thaa",
        "RASODE MAIN KON THA!!",
    ]:
        assert normalise(typed) == target, typed


def test_haath_jodiye():
    target = normalise("हाथ जोड़िए")
    for typed in ["haath jodiye", "hath jodiye", "haat jodiye", "Hath jodie"]:
        assert normalise(typed) == target, typed


def test_stretched_letters_and_shorthand():
    assert normalise("kyaaaaa") == normalise("kya")
    assert normalise("nhi h") == normalise("nahi hai")
    assert normalise("plz bolo") == normalise("please bolo")


def test_common_words_survive():
    # normalising must not squash everyday words into something unrecognisable
    assert normalise("bhai") == normalise("bhaiya") == "bai"
    assert normalise("hai") == normalise("h") == normalise("he") == "hai"
    assert normalise("मैं") == normalise("main") == normalise("me") == "men"


def test_empty():
    assert normalise(None) == ""
    assert normalise("🔥🔥🔥") == ""


def test_clean_caption():
    body, tags = clean_caption(
        "Rasode mein kaun tha | Kokilaben meme template #viral #kokilaben #fyp "
        "follow for more https://x.y/z @memepage"
    )
    assert body == "Rasode mein kaun tha Kokilaben"
    assert tags == ["kokilaben"]
