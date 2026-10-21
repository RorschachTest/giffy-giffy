"""Reading the model's reply and retry rules. No network, no model."""
import socket
import urllib.error

import pytest

from . import conftest  # noqa: F401

from app import describe


def _reply(content, finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}], "usage": {"cost": 0.001}}


def test_a_good_reply_is_read():
    assert describe.read_answer(_reply('{"happens": "x", "send_when": ["a"]}'))["happens"] == "x"


@pytest.mark.parametrize("content", [None, "", "   ", '{"happens": "cut of', "[1, 2]"])
def test_an_unreadable_reply_keeps_the_reply(content):
    with pytest.raises(describe.UnreadableReply) as e:
        describe.read_answer(_reply(content, finish="length"))
    assert e.value.reply["usage"]["cost"] == 0.001


def test_what_is_retried():
    def http(code):
        return urllib.error.HTTPError("u", code, "m", {}, None)
    assert describe._retryable(http(429)) and describe._retryable(http(503))
    assert not describe._retryable(http(400)) and not describe._retryable(http(403))
    assert describe._retryable(socket.timeout()) and describe._retryable(urllib.error.URLError("down"))
    assert not describe._retryable(ValueError())


def test_clean_drops_junk():
    row = describe.clean({"send_when": ["  a  ", "", None, "b"], "confidence": "HIGH ", "happens": None})
    assert row == {"happens": "", "poster_meant": "", "send_when": ["a", "b"], "confidence": "high"}
