"""The local backend against a fake Ollama server: request shape, usage, readiness."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from . import conftest  # noqa: F401

from app import config, explain


class _Ollama(BaseHTTPRequestHandler):
    seen: list = []
    models = ["qwen2.5vl:7b"]

    def _send(self, payload):
        out = json.dumps(payload).encode()
        self.send_response(200); self.send_header("content-length", str(len(out))); self.end_headers(); self.wfile.write(out)

    def do_GET(self):  # noqa: N802
        self._send({"models": [{"name": m} for m in type(self).models]})

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).seen.append((self.path, body))
        self._send({"message": {"content": '{"gist": "A man boasts.", "why_funny": "Empty bluster.", "send_when": ["when he talks big"]}'},
                    "prompt_eval_count": 800, "eval_count": 70})

    def log_message(self, *a):
        pass


@pytest.fixture
def ollama(monkeypatch, tmp_path):
    server = HTTPServer(("127.0.0.1", 0), _Ollama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _Ollama.seen = []; _Ollama.models = ["qwen2.5vl:7b"]
    monkeypatch.setattr(config, "EXPLAIN_LOCAL_URL", f"http://127.0.0.1:{server.server_port}")
    from PIL import Image
    p = tmp_path / "f.jpg"; Image.new("RGB", (1280, 720), (10, 80, 30)).save(p)
    yield [p]
    server.shutdown()


def test_request_shape_and_result(ollama):
    out = explain.run("local", ollama, "ravi kishan", "नाम", "naam", "hi")
    assert out["gist"] == "A man boasts." and out["usage"]["input_tokens"] == 800 and out["usage"]["output_tokens"] == 70
    path, body = _Ollama.seen[0]
    assert path == "/api/chat" and body["model"] == "qwen2.5vl:7b" and body["stream"] is False and body["format"] == "json"
    assert body["messages"][0] == {"role": "system", "content": explain.SYSTEM}
    assert len(body["messages"][1]["images"]) == 1 and "Title: ravi kishan" in body["messages"][1]["content"]


def test_ready_explains_what_is_missing(ollama, monkeypatch):
    assert explain.ready("local") == (True, "")
    _Ollama.models = ["llama3.2:3b"]
    ok, why = explain.ready("local")
    assert not ok and "ollama pull qwen2.5vl:7b" in why
    monkeypatch.setattr(config, "EXPLAIN_LOCAL_URL", "http://127.0.0.1:9")      # nothing listens there
    ok, why = explain.ready("local")
    assert not ok and "brew install ollama" in why
    monkeypatch.setattr(config, "EXPLAIN_BACKEND", "local")
    assert explain.available() is False
