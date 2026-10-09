"""Serving clips from a folder or a bucket, with byte ranges for seeking. No real S3."""
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from . import conftest  # noqa: F401

from app import config, storage

app = FastAPI()


@app.api_route("/media/{key}", methods=["GET", "HEAD"])
def media(key: str, request: Request):
    return storage.serve(key, request)


client = TestClient(app)
DATA = bytes(range(256)) * 40          # 10 240 bytes


class FakeBody:
    def __init__(self, data: bytes):
        self.data = data

    def iter_chunks(self, size):
        for i in range(0, len(self.data), size):
            yield self.data[i:i + size]


class FakeS3:
    """Just enough of boto3's S3 client: get/head/upload/delete, with Range."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def get_object(self, Bucket, Key, Range=None):
        from botocore.exceptions import ClientError
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        data = self.objects[Key]
        out = {"ContentType": "video/mp4"}
        if Range:
            start, end = Range.removeprefix("bytes=").split("-")
            start, end = int(start), int(end or len(data) - 1)
            out.update(ContentRange=f"bytes {start}-{end}/{len(data)}", ContentLength=end - start + 1,
                       Body=FakeBody(data[start:end + 1]))
        else:
            out.update(ContentLength=len(data), Body=FakeBody(data))
        return out

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.objects[Key]), "ContentType": "video/mp4"}

    def upload_file(self, path, Bucket, Key, ExtraArgs=None):
        self.objects[Key] = open(path, "rb").read()

    def download_file(self, Bucket, Key, path):
        open(path, "wb").write(self.objects[Key])

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)


def test_local_file_whole_and_range(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STORAGE", "local")
    monkeypatch.setattr(config, "MEDIA_DIR", tmp_path)
    (tmp_path / "abc123.mp4").write_bytes(DATA)
    r = client.get("/media/abc123.mp4")
    assert r.status_code == 200 and r.content == DATA and r.headers["content-type"] == "video/mp4"
    r = client.get("/media/abc123.mp4", headers={"Range": "bytes=100-199"})
    assert r.status_code == 206 and r.content == DATA[100:200]
    assert client.get("/media/missing.mp4").status_code == 404


def test_bad_keys_are_refused(monkeypatch):
    monkeypatch.setattr(config, "STORAGE", "local")
    for key in ("..%2Fsecret", ".env", "a..b"):
        assert client.get(f"/media/{key}").status_code == 404


def test_bucket_whole_range_head_and_missing(monkeypatch):
    fake = FakeS3()
    fake.objects["abc123.mp4"] = DATA
    monkeypatch.setattr(config, "STORAGE", "s3")
    monkeypatch.setattr(storage, "_s3", lambda: fake)
    r = client.get("/media/abc123.mp4")
    assert r.status_code == 200 and r.content == DATA and r.headers["accept-ranges"] == "bytes"
    r = client.get("/media/abc123.mp4", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == DATA[10:20]
    assert r.headers["content-range"] == f"bytes 10-19/{len(DATA)}"
    r = client.head("/media/abc123.mp4")
    assert r.status_code == 200 and r.headers["content-length"] == str(len(DATA))
    assert client.get("/media/nope.mp4").status_code == 404


def test_save_uploads_and_removes_the_local_copy(tmp_path, monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(config, "STORAGE", "s3")
    monkeypatch.setattr(storage, "_s3", lambda: fake)
    f = tmp_path / "abc123.jpg"
    f.write_bytes(b"jpeg")
    assert storage.save(f) == "abc123.jpg" and fake.objects["abc123.jpg"] == b"jpeg" and not f.exists()
    with storage.local_copy("abc123.jpg") as p:
        assert p.read_bytes() == b"jpeg"
    storage.delete("abc123.jpg")
    assert "abc123.jpg" not in fake.objects


def test_local_save_keeps_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STORAGE", "local")
    monkeypatch.setattr(config, "MEDIA_DIR", tmp_path)
    f = tmp_path / "abc123.mp4"
    f.write_bytes(b"x")
    assert storage.save(f) == "abc123.mp4" and f.exists()
