"""Exercise the HTTP pipeline without downloading model weights."""
import io
import time

from fastapi.testclient import TestClient
from PIL import Image

from upscaler import api


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
        data={"upscale": "false", "adjustments": '{"exposure":50}'},
    )
    assert response.status_code == 202
    for _ in range(100):
        job = client.get(response.json()["status_url"]).json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.01)
    assert job["status"] == "completed"
    assert job["upscale"] is False
    assert job["model"] is None
    assert (job["width"], job["height"]) == (2, 3)
    image = Image.open(io.BytesIO(client.get(job["download_url"]).content))
    assert image.getpixel((0, 0))[0] > 80

    docs = client.get("/api/docs")
    assert docs.status_code == 200
    assert "/api/openapi.json" in docs.text
    schema = client.get("/api/openapi.json").json()
    assert set(schema["paths"]) == {"/api/pipeline", "/api/jobs/{job_id}", "/api/images/{image_id}"}
    body_ref = schema["paths"]["/api/pipeline"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    body = schema["components"]["schemas"][body_ref.rsplit("/", 1)[-1]]
    assert body["properties"]["upscale"]["default"] is True
