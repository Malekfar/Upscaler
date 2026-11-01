FROM python:3.11-slim

# NVIDIA CUDA-enabled PyTorch wheels. Override at build time if you need a
# different CUDA wheel index (for example, the CPU index on a non-GPU host).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    UPSCALER_HOST=0.0.0.0 \
    UPSCALER_PORT=7860 \
    UPSCALER_WEIGHTS_DIR=/data/weights \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

# Keep downloaded wheels across builds, even when the layer must be rebuilt.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url ${TORCH_INDEX_URL} torch torchvision

# Resolve dependencies before copying application code. Ordinary source changes
# then reuse this layer instead of downloading the ML stack again.
COPY pyproject.toml README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p upscaler && touch upscaler/__init__.py app.py \
    && pip install ".[gui,onnx,face,video,api]"

COPY app.py ./
COPY upscaler ./upscaler
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps .

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data/weights /data/api-output /home/app/.upscaler \
    && chown -R app /data /home/app/.upscaler
USER app
VOLUME ["/data/weights", "/home/app/.upscaler"]

EXPOSE 7860
CMD ["python", "app.py"]
