"""RF-DETR detection web demo.

Run: webapp/.venv/bin/python -m uvicorn server:app --app-dir webapp --host 127.0.0.1 --port 8000
"""

import io
import time
from pathlib import Path

from models import DEFAULT_MODEL, MODELS, get_model, infer_lock, weights_path  # sets RF_HOME; import first

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from rfdetr.config import DEVICE

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

app = FastAPI(title="RF-DETR detection demo")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/models")
def list_models() -> dict:
    return {
        "device": str(DEVICE),
        "default": DEFAULT_MODEL,
        "models": [
            {
                "name": name,
                "label": spec.label,
                "class": spec.cls,
                "resolution": spec.resolution,
                "license": spec.license,
                "downloaded": weights_path(name).is_file(),
            }
            for name, spec in MODELS.items()
        ],
    }


@app.post("/api/detect")
def detect(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    threshold: float = Form(0.25),
) -> dict:
    if model not in MODELS:
        raise HTTPException(400, f"Unknown model '{model}'. Choose one of: {', '.join(MODELS)}.")
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(400, "threshold must be between 0 and 1.")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, "That file is not an image this server can read.") from None

    try:
        detector = get_model(model)
    except ImportError as exc:  # PML models without the rfdetr_plus package
        raise HTTPException(503, f"{MODELS[model].label} needs the rfdetr[plus] extra: {exc}") from None
    with infer_lock(model):
        start = time.perf_counter()
        result = detector.predict(image, threshold=threshold)
        inference_ms = (time.perf_counter() - start) * 1000

    names = result.data.get("class_name")
    detections = [
        {
            "label": str(names[i]) if names is not None else str(int(result.class_id[i])),
            "confidence": round(float(result.confidence[i]), 4),
            "box": [round(float(v), 1) for v in result.xyxy[i]],
        }
        for i in range(len(result))
    ]
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return {
        "model": model,
        "label": MODELS[model].label,
        "license": MODELS[model].license,
        "resolution": MODELS[model].resolution,
        "device": str(getattr(detector.model, "device", DEVICE)),
        "width": image.width,
        "height": image.height,
        "inference_ms": round(inference_ms, 1),
        "detections": detections,
    }
