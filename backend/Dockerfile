# EduVance AI — Backend Dockerfile for Hugging Face Spaces (Docker SDK)
# CPU-only deployment: Phi-3 via HF Inference API, EasyOCR + Whisper on CPU.

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# ffmpeg/ffprobe: required by the Whisper audio-extraction step.
# libgl1/libglib2.0-0: required by opencv-python-headless at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Install CPU-only torch first from the dedicated wheel index — this avoids
# pulling the multi-GB CUDA build, which isn't needed (and won't work anyway;
# HF Spaces CPU Basic has no GPU).
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# main_v7-9.py has a hyphen in its name, which isn't a valid Python import
# target ("import main_v7-9" is a syntax error). uvicorn's module:app string
# loader needs a clean module name, so copy it to app_entry.py at build time
# rather than renaming your actual source file.
RUN cp main_v7-9.py app_entry.py

# Hugging Face Spaces expects the app to listen on port 7860.
ENV PORT=7860
EXPOSE 7860

# Force CPU-safe defaults. These can still be overridden by Space secrets,
# but if someone forgets to set them, the app fails safe onto CPU instead
# of crashing looking for a GPU that doesn't exist.
ENV PHI3_BACKEND=api
ENV EASYOCR_GPU=false
ENV WHISPER_MODEL_SIZE=tiny

CMD ["uvicorn", "app_entry:app", "--host", "0.0.0.0", "--port", "7860"]
