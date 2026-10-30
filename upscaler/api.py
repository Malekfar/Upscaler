"""HTTP API for the upscale → color and light pipeline."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError

from upscaler.adjust import AdjustParams, adjust_image
from upscaler.engine import OutputTooLargeError, Upscaler
from upscaler.models.registry import MODELS

app = FastAPI(title="Upscaler API")
_OUTPUT_DIR = Path(os.environ.get("UPSCALER_API_OUTPUT_DIR", "out/api"))
_MAX_UPLOAD = 20 * 1024 * 1024
_lock = threading.Lock()
_engines = {}


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


@app.post("/api/pipeline", status_code=201)
def pipeline(request: Request, image: UploadFile = File(...),
             model: str = Form("realesrgan-x4plus"),
             adjustments: str = Form("{}")):
    """Upload an image and apply upscaling followed by color/light adjustments."""
    if model not in MODELS:
        raise HTTPException(422, "Unknown upscale model")
    try:
        settings = Adjustments(**json.loads(adjustments))
    except (ValueError, TypeError, ValidationError):
        raise HTTPException(422, "Invalid adjustments JSON or adjustment values")
    data = image.file.read(_MAX_UPLOAD + 1)
    if len(data) > _MAX_UPLOAD:
        raise HTTPException(413, "Image exceeds the 20 MiB upload limit")
    try:
        import io
        with Image.open(io.BytesIO(data)) as source:
            source.seek(0)
            original = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, "Invalid or unsupported image")

    try:
        # Serialize inference: engines and GPU memory are shared between requests.
        with _lock:
            if model not in _engines:
                _engines[model] = Upscaler(model=model, device="auto", tile=256)
            result = _engines[model].upscale(original)
            params = settings.model_dump() if hasattr(settings, "model_dump") else settings.dict()
            result = adjust_image(result, AdjustParams(**params))
            _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            image_id = uuid4()
            path = _OUTPUT_DIR / f"{image_id}.png"
            try:
                result.save(path, "PNG")
            except Exception:
                path.unlink(missing_ok=True)
                raise
    except OutputTooLargeError as exc:
        raise HTTPException(413, str(exc)) from exc
    return {"id": str(image_id), "download_url": str(request.url_for("download", image_id=str(image_id))),
            "width": result.width, "height": result.height, "model": model}


@app.get("/api/images/{image_id}", name="download")
def download(image_id: UUID):
    path = _OUTPUT_DIR / f"{image_id}.png"
    if not path.is_file():
        raise HTTPException(404, "Image not found")
    return FileResponse(path, media_type="image/png", filename=path.name)
