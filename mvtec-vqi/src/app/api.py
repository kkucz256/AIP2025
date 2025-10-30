import base64
import io
import os

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from src.infer.predictor import AnomalyPredictor
from src.utils import viz
from src.utils.common import get_device, load_config, resolve_path


def load_settings():
    config_path = os.getenv("MVTEC_CONFIG", "configs/default.yaml")
    path = resolve_path(config_path)
    if not path.exists():
        raise RuntimeError(f"Nie znaleziono pliku konfiguracyjnego {path}")
    return load_config(path), path


APP_CONFIG, config_path = load_settings()
app = FastAPI()
predictor_cache = {}


def get_predictor(backend, category):
    key = (backend, category)
    if key not in predictor_cache:
        predictor_cache[key] = AnomalyPredictor(
            backend=backend,
            category=category,
            config=APP_CONFIG,
            device=get_device(),
        )
    return predictor_cache[key]


def read_image_from_bytes(data):
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Nieprawidłowy obraz") from exc
    return np.array(image)


@app.post("/infer")
async def infer(request: Request, backend: str = Form(None), category: str = Form(None), file: UploadFile = File(None), image_base64: str = Form(None)):
    payload = {}
    if request.headers.get("content-type", "").startswith("application/json"):
        payload = await request.json()
    backend = backend or payload.get("backend") or "padim_resnet50"
    category = category or payload.get("category") or APP_CONFIG.get("category", "bottle")
    data_bytes = None
    if file is not None:
        data_bytes = await file.read()
    elif image_base64 or payload.get("image_base64"):
        data_str = image_base64 or payload.get("image_base64")
        try:
            data_bytes = base64.b64decode(data_str)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Nie udało się zdekodować base64") from exc
    else:
        raise HTTPException(status_code=400, detail="Brak obrazu w żądaniu")
    image = read_image_from_bytes(data_bytes)
    try:
        predictor = get_predictor(backend, category)
        result = predictor.predict_array(image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    heatmap_b64 = viz.encode_png_base64(result["heatmap"])
    overlay_b64 = viz.encode_png_base64(result["overlay"])
    map_b64 = viz.encode_array_base64(result["anomaly_map"])
    response = {
        "backend": backend,
        "category": category,
        "score": result["score"],
        "heatmap": heatmap_b64,
        "overlay": overlay_b64,
        "map": map_b64,
        "config_path": str(config_path),
    }
    return JSONResponse(response)
