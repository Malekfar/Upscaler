"""Exercise the HTTP pipeline without downloading model weights."""
import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from upscaler import api


@pytest.fixture(autouse=True)
def fake_tagger(monkeypatch):
    class FakeTagger:
        def tag(self, image):
            return [{"label": "استخر", "score": 0.31}]
    monkeypatch.setattr(api, "_tagger", FakeTagger())


class FakeUpscaler:
    def __init__(self, **kwargs):
        pass

    def upscale(self, image):
        return image.resize((image.width * 2, image.height * 2))


def test_pipeline_and_download(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(api, "Upscaler", FakeUpscaler)
    monkeypatch.setattr(api, "_engines", {})
    client = TestClient(api.app)
    source = io.BytesIO()
    Image.new("RGB", (2, 3), (80, 80, 80)).save(source, "PNG")
    response = client.post("/api/pipeline", files={"image": ("input.png", source.getvalue(), "image/png")},
                           data={"adjustments": '{"exposure": 50}'})
    assert response.status_code == 202
    queued = response.json()
    assert queued["status"] == "queued"
    for _ in range(100):
        payload = client.get(queued["status_url"]).json()
        if payload["status"] in ("completed", "failed"):
            break
        time.sleep(0.01)
    assert payload["status"] == "completed"
    assert (payload["width"], payload["height"]) == (4, 6)
    assert payload["download_url"].startswith("https://")
    assert payload["tagging"] is True
    assert payload["tags"] == [{"label": "استخر", "score": 0.31}]
    listed = client.get("/api/jobs").json()["jobs"][0]
    assert listed["download_url"] == payload["download_url"]
    assert listed["tags"] == payload["tags"]
    downloaded = client.get(payload["download_url"])
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "image/png"
    result = Image.open(io.BytesIO(downloaded.content))
    assert result.getpixel((0, 0))[0] > 80
    assert (tmp_path / (payload["id"] + ".png")).exists()


def test_invalid_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "_OUTPUT_DIR", tmp_path)
    client = TestClient(api.app)
    upload = {"image": ("image.png", b"not an image", "image/png")}
    assert client.post("/api/pipeline", files=upload).status_code == 400
    assert client.post("/api/pipeline", files=upload, data={"model": "bad"}).status_code == 422
    assert client.post("/api/pipeline", files=upload, data={"adjustments": '{"exposure": 101}'}).status_code == 422
    assert client.post("/api/pipeline", files=upload, data={"adjustments": '{"unknown": 1}'}).status_code == 422
    assert client.get("/api/images/00000000-0000-0000-0000-000000000000").status_code == 404
    assert client.get("/api/jobs/00000000-0000-0000-0000-000000000000").status_code == 404


def test_failed_job(tmp_path, monkeypatch):
    class BrokenUpscaler:
        def __init__(self, **kwargs):
            raise RuntimeError("disk failure")

    monkeypatch.setattr(api, "_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(api, "Upscaler", BrokenUpscaler)
    monkeypatch.setattr(api, "_engines", {})
    client = TestClient(api.app)
    source = io.BytesIO()
    Image.new("RGB", (2, 3)).save(source, "PNG")
    response = client.post("/api/pipeline", files={"image": ("input.png", source.getvalue(), "image/png")})
    assert response.status_code == 202
    for _ in range(100):
        job = client.get(response.json()["status_url"]).json()
        if job["status"] == "failed":
            break
        time.sleep(0.01)
    assert job["status"] == "failed"
    assert "download_url" not in job
    assert "error" in job


def test_skip_upscale_and_swagger(tmp_path, monkeypatch):
    def fail_if_loaded(**kwargs):
        raise AssertionError("Upscaler must not be loaded")

    monkeypatch.setattr(api, "_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(api, "Upscaler", fail_if_loaded)
    monkeypatch.setattr(api, "_engines", {})
    client = TestClient(api.app)
    source = io.BytesIO()
    Image.new("RGB", (2, 3), (80, 80, 80)).save(source, "PNG")
    response = client.post(
        "/api/pipeline", files={"image": ("input.png", source.getvalue(), "image/png")},
        data={"upscale": "false", "tagging": "false", "adjustments": '{"exposure":50}'},
    )
    assert response.status_code == 202
    for _ in range(100):
        job = client.get(response.json()["status_url"]).json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.01)
    assert job["status"] == "completed"
    assert job["upscale"] is False
    assert job["tagging"] is False
    assert "tags" not in job
    assert job["model"] is None
    assert (job["width"], job["height"]) == (2, 3)
    image = Image.open(io.BytesIO(client.get(job["download_url"]).content))
    assert image.getpixel((0, 0))[0] > 80

    docs = client.get("/api/docs")
    assert docs.status_code == 200
    assert "/api/openapi.json" in docs.text
    schema = client.get("/api/openapi.json").json()
    assert set(schema["paths"]) == {"/api/pipeline", "/api/jobs", "/api/jobs/{job_id}", "/api/images/{image_id}"}
    body_ref = schema["paths"]["/api/pipeline"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    body = schema["components"]["schemas"][body_ref.rsplit("/", 1)[-1]]
    assert body["properties"]["upscale"]["default"] is True
    assert body["properties"]["tagging"]["default"] is True
    assert "tags" in schema["components"]["schemas"]["JobResponse"]["properties"]


def test_fifo_queue_and_error_details(tmp_path, monkeypatch):
    import threading

    entered = threading.Event()
    release = threading.Event()
    seen = []

    class SerialUpscaler:
        def __init__(self, **kwargs):
            pass

        def upscale(self, image):
            value = image.getpixel((0, 0))[0]
            seen.append(value)
            if value == 0:
                entered.set()
                assert release.wait(10)
            if value == 5:
                raise RuntimeError("specific failure")
            return image

    monkeypatch.setattr(api, "_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(api, "Upscaler", SerialUpscaler)
    monkeypatch.setattr(api, "_engines", {})
    client = TestClient(api.app)
    jobs = []
    try:
        for i in range(12):
            source = io.BytesIO()
            Image.new("RGB", (2, 2), (i, 0, 0)).save(source, "PNG")
            response = client.post("/api/pipeline", files={"image": ("in.png", source.getvalue(), "image/png")})
            assert response.status_code == 202
            jobs.append(response.json()["id"])
            if i == 0:
                assert entered.wait(5)
        listing = client.get("/api/jobs?limit=50").json()
        assert listing["total"] == 12
        assert listing["counts"] == {"processing": 1, "queued": 11}
        assert client.get(f"/api/jobs/{jobs[1]}").json()["queue_position"] == 1
        assert client.get(f"/api/jobs/{jobs[-1]}").json()["queue_position"] == 11
        assert seen == [0]
    finally:
        release.set()
    for _ in range(500):
        last = client.get(f"/api/jobs/{jobs[-1]}").json()
        if last["status"] == "completed":
            break
        time.sleep(0.02)
    assert last["status"] == "completed"
    assert seen == list(range(12))
    failure = client.get(f"/api/jobs/{jobs[5]}").json()
    assert failure["status"] == "failed"
    assert failure["stage"] == "upscaling"
    assert "specific failure" in failure["error"]
    assert client.get("/api/queue").status_code == 200


def test_interrupted_job_is_recovered(tmp_path, monkeypatch):
    from uuid import uuid4

    monkeypatch.setattr(api, "_OUTPUT_DIR", tmp_path)
    job_id = str(uuid4())
    (tmp_path / "inputs").mkdir()
    Image.new("RGB", (2, 2)).save(tmp_path / "inputs" / (job_id + ".upload"), "PNG")
    with api._connect(tmp_path) as db:
        db.execute("""INSERT INTO jobs (id,status,stage,created_at,upscale,params)
                      VALUES (?, 'processing', 'saving', ?, 0, '{}')""", (job_id, api._now()))
    client = TestClient(api.app)
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.02)
    assert job["status"] == "completed"
    assert job["tagging"] is False
    assert client.get(job["download_url"]).status_code == 200


def test_tagging_failure_is_reported(tmp_path, monkeypatch):
    class BrokenTagger:
        def tag(self, image):
            raise RuntimeError("tagging failed")

    monkeypatch.setattr(api, "_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(api, "_tagger", BrokenTagger())
    client = TestClient(api.app)
    source = io.BytesIO()
    Image.new("RGB", (3, 3)).save(source, "PNG")
    response = client.post("/api/pipeline", files={"image": ("input.png", source.getvalue(), "image/png")},
                           data={"upscale": "false"})
    for _ in range(200):
        job = client.get(response.json()["status_url"]).json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.01)
    assert job["status"] == "failed"
    assert job["stage"] == "tagging"
    assert "tagging failed" in job["error"]
    assert "download_url" not in job


def test_legacy_database_migration(tmp_path):
    import sqlite3
    db = sqlite3.connect(tmp_path / "jobs.sqlite3")
    db.execute("""CREATE TABLE jobs (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
        status TEXT NOT NULL, stage TEXT NOT NULL, created_at TEXT NOT NULL, started_at TEXT,
        finished_at TEXT, upscale INTEGER NOT NULL, model TEXT, params TEXT NOT NULL,
        width INTEGER, height INTEGER, error TEXT)""")
    db.execute("INSERT INTO jobs (id, status, stage, created_at, upscale, params) VALUES ('old', 'completed', 'completed', '2020', 0, '{}')")
    db.commit()
    db.close()
    with api._connect(tmp_path) as migrated:
        row = migrated.execute("SELECT tagging, tags FROM jobs WHERE id='old'").fetchone()
        assert row['tagging'] == 0
        assert row['tags'] is None
