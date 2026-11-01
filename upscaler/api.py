"""Durable, sequential HTTP image processing queue."""
from __future__ import annotations
import io
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID, uuid4
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError
from upscaler.adjust import AdjustParams, adjust_image
from upscaler.engine import Upscaler
from upscaler.models.registry import MODELS

app = FastAPI(title="Upscaler Pipeline API", docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
_OUTPUT_DIR = Path(os.environ.get("UPSCALER_API_OUTPUT_DIR", "out/api"))
_MAX_UPLOAD = 20 * 1024 * 1024
_engines = {}
_workers = {}
_workers_lock = threading.Lock()
_logger = logging.getLogger(__name__)

class Adjustments(BaseModel):
    exposure: float = Field(0, ge=-100, le=100)
    contrast: float = Field(0, ge=-100, le=100)
    highlights: float = Field(0, ge=-100, le=100)
    shadows: float = Field(0, ge=-100, le=100)
    black_point: float = Field(0, ge=0, le=50)
    white_point: float = Field(100, ge=50, le=100)
    gamma: float = Field(1, ge=0.2, le=3)
    clarity: float = Field(0, ge=-100, le=100)
    temperature: float = Field(0, ge=-100, le=100)
    tint: float = Field(0, ge=-100, le=100)
    hue: float = Field(0, ge=-180, le=180)
    saturation: float = Field(0, ge=-100, le=100)
    vibrance: float = Field(0, ge=-100, le=100)
    mono: bool = False
    mono_red: float = Field(30, ge=0, le=100)
    mono_green: float = Field(59, ge=0, le=100)
    mono_blue: float = Field(11, ge=0, le=100)
    tone_color: str = Field("#d8b070", pattern=r"^#[0-9a-fA-F]{6}$")
    tone_strength: float = Field(0, ge=0, le=100)

    class Config:
        extra = "forbid"


@contextmanager
def _connect(directory):
    directory.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(directory / "jobs.sqlite3"), timeout=30)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("""CREATE TABLE IF NOT EXISTS jobs (
            seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL, stage TEXT NOT NULL, created_at TEXT NOT NULL,
            started_at TEXT, finished_at TEXT, upscale INTEGER NOT NULL, model TEXT,
            params TEXT NOT NULL, width INTEGER, height INTEGER, error TEXT)""")
        db.commit()
        with db:
            yield db
    finally:
        db.close()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _update(directory, job_id, **fields):
    with _connect(directory) as db:
        db.execute("UPDATE jobs SET " + ", ".join(k + "=?" for k in fields) + " WHERE id=?",
                   (*fields.values(), str(job_id)))


def _worker(directory):
    # The lock serializes consumers even if multiple server processes are started.
    lock_file = open(directory / "worker.lock", "a+b")
    try:
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                time.sleep(1)
        with _connect(directory) as db:
            db.execute("UPDATE jobs SET status='queued', stage='waiting', started_at=NULL WHERE status='processing'")
        while True:
            with _connect(directory) as db:
                db.execute("BEGIN IMMEDIATE")
                job = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY seq LIMIT 1").fetchone()
                if job:
                    db.execute("UPDATE jobs SET status='processing', stage='loading', started_at=? WHERE id=?", (_now(), job['id']))
            if job:
                _process_job(directory, job)
            else:
                time.sleep(0.2)
    finally:
        lock_file.close()


def _ensure_worker(directory):
    directory = directory.resolve()
    with _workers_lock:
        thread = _workers.get(directory)
        if thread is None or not thread.is_alive():
            with _connect(directory):
                pass
            thread = threading.Thread(target=_worker, args=(directory,), daemon=True, name="image-pipeline")
            _workers[directory] = thread
            thread.start()


@app.on_event("startup")
def _start_queue():
    _ensure_worker(_OUTPUT_DIR)


def _process_job(directory, job):
    job_id = job["id"]
    input_path = directory / "inputs" / (job_id + ".upload")
    path = directory / (job_id + ".png")
    terminal = False
    try:
        with Image.open(input_path) as source:
            source.seek(0)
            original = ImageOps.exif_transpose(source).convert("RGB")
        if job["upscale"]:
            _update(directory, job_id, stage="upscaling")
            model = job["model"]
            if model not in _engines:
                _engines[model] = Upscaler(model=model, device="auto", tile=256)
            result = _engines[model].upscale(original)
        else:
            result = original
        _update(directory, job_id, stage="adjusting")
        result = adjust_image(result, AdjustParams(**json.loads(job["params"])))
        _update(directory, job_id, stage="saving")
        result.save(path, "PNG")
        _update(directory, job_id, status="completed", stage="completed", width=result.width,
                height=result.height, finished_at=_now())
        terminal = True
    except Exception as exc:
        path.unlink(missing_ok=True)
        _logger.exception("Image pipeline job %s failed", job_id)
        _update(directory, job_id, status="failed", finished_at=_now(), error=f"{type(exc).__name__}: {exc}")
        terminal = True
    finally:
        if terminal:
            input_path.unlink(missing_ok=True)

@app.post("/api/pipeline", status_code=202)
def pipeline(request: Request,
             image: UploadFile = File(..., description="Image to process (maximum 20 MiB)."),
             model: str = Form("realesrgan-x4plus", description="Upscale model, used when upscale is true."),
             upscale: bool = Form(True, description="Set false to retain the original image dimensions."),
             adjustments: str = Form("{}", description='JSON object, e.g. {"exposure":10,"contrast":8}.')):
    """Queue an image for optional upscaling and color/light adjustments.

    Poll the returned status_url for completion and a download URL.

    The adjustments form field is a JSON object. Exposure, contrast,
    highlights, shadows, clarity, temperature, tint, saturation and vibrance
    range from -100 to 100. Hue ranges from -180 to 180; black_point from
    0 to 50; white_point from 50 to 100; gamma from 0.2 to 3. Set mono to
    true for black and white. mono_red, mono_green, mono_blue and
    tone_strength range from 0 to 100; tone_color is a six-digit hex color.
    """
    if upscale and model not in MODELS:
        raise HTTPException(422, "Unknown upscale model")
    try:
        settings = Adjustments(**json.loads(adjustments))
    except (ValueError, TypeError, ValidationError):
        raise HTTPException(422, "Invalid adjustments JSON or adjustment values")
    data = image.file.read(_MAX_UPLOAD + 1)
    if len(data) > _MAX_UPLOAD:
        raise HTTPException(413, "Image exceeds the 20 MiB upload limit")
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.verify()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, "Invalid or unsupported image")
    job_id = uuid4()
    directory = _OUTPUT_DIR.resolve()
    input_dir = directory / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / (str(job_id) + ".upload")
    params = settings.model_dump() if hasattr(settings, "model_dump") else settings.dict()
    try:
        input_path.write_bytes(data)
        with _connect(directory) as db:
            db.execute("""INSERT INTO jobs (id, status, stage, created_at, upscale, model, params)
                          VALUES (?, 'queued', 'waiting', ?, ?, ?, ?)""",
                       (str(job_id), _now(), int(upscale), model if upscale else None, json.dumps(params)))
    except Exception:
        input_path.unlink(missing_ok=True)
        raise
    _ensure_worker(directory)
    return {"id": str(job_id), "status": "queued",
            "status_url": str(request.url_for("pipeline_status", job_id=str(job_id)))}


def _job_response(request, row, db):
    job = dict(row)
    job.pop("params")
    seq = job.pop("seq")
    job["upscale"] = bool(job["upscale"])
    if job["status"] == "queued":
        job["queue_position"] = db.execute(
            "SELECT COUNT(*) FROM jobs WHERE status='queued' AND seq<=?", (seq,)).fetchone()[0]
    if job["status"] == "completed":
        job["download_url"] = str(request.url_for("download", image_id=job["id"]).replace(scheme="https"))
    return {key: value for key, value in job.items() if value is not None or key == "model"}


@app.get("/api/jobs")
def list_jobs(request: Request, limit: int = 50, offset: int = 0):
    """List jobs with their current state and queue position, newest first."""
    if not 1 <= limit <= 200 or offset < 0:
        raise HTTPException(422, "limit must be 1–200 and offset must be non-negative")
    _ensure_worker(_OUTPUT_DIR)
    with _connect(_OUTPUT_DIR) as db:
        total = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        rows = db.execute("SELECT * FROM jobs ORDER BY seq DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        counts = {row["status"]: row["count"] for row in db.execute(
            "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status")}
        return {"total": total, "counts": counts, "limit": limit, "offset": offset,
                "jobs": [_job_response(request, row, db) for row in rows]}


@app.get("/api/jobs/{job_id}", name="pipeline_status")
def pipeline_status(request: Request, job_id: UUID):
    """Return job status and, when complete, the result download URL."""
    _ensure_worker(_OUTPUT_DIR)
    with _connect(_OUTPUT_DIR) as db:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (str(job_id),)).fetchone()
        if row is None:
            raise HTTPException(404, "Job not found")
        return _job_response(request, row, db)


@app.get("/api/queue", response_class=HTMLResponse, include_in_schema=False)
def queue_page():
    return (Path(__file__).parent / "queue.html").read_text(encoding="utf-8")


@app.get("/api/images/{image_id}", name="download")
def download(image_id: UUID):
    """Download a completed result as a PNG image."""
    path = _OUTPUT_DIR / f"{image_id}.png"
    if not path.is_file():
        raise HTTPException(404, "Image not found")
    return FileResponse(path, media_type="image/png", filename=path.name)
