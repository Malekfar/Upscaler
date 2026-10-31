"""HTTP API for the upscale → color and light pipeline."""
from __future__ import annotations

import io
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError

from upscaler.adjust import AdjustParams, adjust_image
from upscaler.engine import OutputTooLargeError, Upscaler
from upscaler.models.registry import MODELS

app = FastAPI(title="Upscaler Pipeline API", docs_url="/api/docs",
              openapi_url="/api/openapi.json", redoc_url=None)
_OUTPUT_DIR = Path(os.environ.get("UPSCALER_API_OUTPUT_DIR", "out/api"))
_MAX_UPLOAD = 20 * 1024 * 1024
_engines = {}
_jobs = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="image-pipeline")
# Bound both queued and running images to avoid an unbounded memory queue.
_job_slots = threading.BoundedSemaphore(8)
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


def _process_job(job_id: UUID, original: Image.Image, model: str, params: dict, upscale: bool):
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "processing"
        if upscale:
            if model not in _engines:
                _engines[model] = Upscaler(model=model, device="auto", tile=256)
            result = _engines[model].upscale(original)
        else:
            result = original
        result = adjust_image(result, AdjustParams(**params))
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = _OUTPUT_DIR / f"{job_id}.png"
        try:
            result.save(path, "PNG")
        except Exception:
            path.unlink(missing_ok=True)
            raise
        with _jobs_lock:
            _jobs[job_id].update(status="completed", width=result.width, height=result.height)
    except Exception as exc:
        _logger.exception("Image pipeline job %s failed", job_id)
        message = str(exc) if isinstance(exc, OutputTooLargeError) else "Image processing failed; check server logs."
        with _jobs_lock:
            _jobs[job_id].update(status="failed", error=message)
    finally:
        _job_slots.release()


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
            source.seek(0)
            original = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, "Invalid or unsupported image")
    if not _job_slots.acquire(blocking=False):
        raise HTTPException(503, "The processing queue is full; try again later")
    job_id = uuid4()
    with _jobs_lock:
        _jobs[job_id] = {"id": str(job_id), "status": "queued", "upscale": upscale,
                         "model": model if upscale else None}
    params = settings.model_dump() if hasattr(settings, "model_dump") else settings.dict()
    try:
        _executor.submit(_process_job, job_id, original, model, params, upscale)
    except Exception:
        with _jobs_lock:
            _jobs.pop(job_id, None)
        _job_slots.release()
        raise
    return {"id": str(job_id), "status": "queued",
            "status_url": str(request.url_for("pipeline_status", job_id=str(job_id)))}


@app.get("/api/jobs/{job_id}", name="pipeline_status")
def pipeline_status(request: Request, job_id: UUID):
    """Return job status and, when complete, the result download URL."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        response = dict(job)
    if response["status"] == "completed":
        response["download_url"] = str(request.url_for("download", image_id=str(job_id)))
    return response


@app.get("/api/images/{image_id}", name="download")
def download(image_id: UUID):
    """Download a completed result as a PNG image."""
    path = _OUTPUT_DIR / f"{image_id}.png"
    if not path.is_file():
        raise HTTPException(404, "Image not found")
    return FileResponse(path, media_type="image/png", filename=path.name)
