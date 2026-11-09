#!/usr/bin/env python3
"""Beta-only v11 intranet API (FastAPI).

One process, one GPU model instance. HTTP requests are accepted concurrently,
but GPU inference is serialized with a lock (detector -> DA depth -> beta-only
V11). This is the safe mode for a single 11 GB card; for throughput use more
GPUs/processes.

Run:
  export DLEF_DATA_ROOT=...
  export PYTHONPATH=/path/to/DLEF
  /path/to/PolypMeasure/.venv/bin/uvicorn api.server:app --host 0.0.0.0 --port 8000

Endpoints:
  GET  /health
  POST /predict       multipart: file=<image>, bbox="[x1,y1,x2,y2]" (optional)
  POST /predict/json  {"image_base64": "...", "bbox": [...]} (optional)

bbox is in ORIGINAL image pixel coordinates. If omitted, the company detector
runs on the content crop.
"""
from __future__ import annotations

import os

# Must precede transformers / polyp_measurement imports (same quirk as the
# video inference script: NO_PROXY=[::1] breaks httpx URL parsing).
os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import base64
import json
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

POLYP = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/PolypMeasure")
PROJ = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_ep")
DLEF = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/DLEF")
for path in (str(POLYP), str(PROJ / "scripts"), str(PROJ), str(DLEF)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.chdir(POLYP)  # weights/delivery/*.pt are relative to PolypMeasure

from build_company_features import robust_content_box  # noqa: E402
from polyp_measurement.algorithms.measurement.lesion_features import (  # noqa: E402
    align_relative_depth,
    prepare_lesion,
)
from polyp_measurement.domain.models import CoordinateSpace, FramePacket  # noqa: E402
from polyp_measurement.runtime.builder import create_detector, create_pipeline  # noqa: E402
from polyp_measurement.runtime.config import component_config, load_config  # noqa: E402

from core.config import ModelConfig  # noqa: E402
from core.models.v11 import V11Net  # noqa: E402

DEFAULT_RUN = PROJ / (
    "output/runs/beta-regression/v11-company-diranchor40-betaonly/"
    "20261106-075001_f83b8b9b_s20261021_48542a"
)
MODEL_CONFIG = POLYP / "polyp_measurement/configs/polyp_size_delivery_dav2.yaml"
API_KEY = os.environ.get("V11_API_KEY", "")
MAX_IMAGE_BYTES = int(os.environ.get("V11_MAX_IMAGE_MB", "20")) * 1024 * 1024
LOCK_TIMEOUT_S = float(os.environ.get("V11_LOCK_TIMEOUT_S", "30"))

STATE: dict = {}
GPU_LOCK = threading.Lock()


def _load_models() -> None:
    run = Path(os.environ.get("V11_RUN", str(DEFAULT_RUN)))
    if not (run / "checkpoints/final_ema.pt").is_file():
        raise RuntimeError(f"missing checkpoint: {run}/checkpoints/final_ema.pt")
    resolved = json.loads((run / "config.resolved.json").read_text())
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    ckpt = torch.load(run / "checkpoints/final_ema.pt", map_location="cpu",
                      weights_only=False)
    net.load_state_dict(ckpt.model)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the API")
    net = net.cuda().eval()

    cfg = load_config(MODEL_CONFIG)
    pipe = create_pipeline(cfg)
    pipe.depth_estimator.initialize()
    detector = create_detector(component_config(cfg, "detector"))
    detector.initialize()

    STATE.update(
        {
            "net": net,
            "pipe": pipe,
            "detector": detector,
            "run_id": run.name,
            "run_path": str(run),
            "epoch": int(getattr(ckpt, "epoch", 0)),
            "input_size": int(resolved.get("data", {}).get("input_size", 224)),
            "model": "v11-betaonly-s21",
            "device": "cuda",
        }
    )


def _decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise HTTPException(400, "empty image")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"image larger than {MAX_IMAGE_BYTES} bytes")
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "cannot decode image")
    return bgr


def _parse_bbox(raw: str | None) -> list[float] | None:
    if raw is None or not raw.strip():
        return None
    # Swagger UI prefills optional string fields with the literal "string".
    # Treat it as "not provided" so /docs Try-it-out works without clearing.
    if raw.strip().lower() == "string":
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        try:
            value = [float(x) for x in raw.split(",")]
        except ValueError as exc:
            raise HTTPException(400, "bbox must be JSON [x1,y1,x2,y2] or 'x1,y1,x2,y2'") from exc
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise HTTPException(400, "bbox must have 4 numbers")
    try:
        box = [float(x) for x in value]
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "bbox must have 4 numbers") from exc
    if not (0 <= box[0] < box[2] and 0 <= box[1] < box[3]):
        raise HTTPException(400, "bbox must satisfy x1<x2 and y1<y2")
    return box


def _infer(bgr: np.ndarray, bbox_original: list[float] | None) -> dict:
    if GPU_LOCK.acquire(timeout=LOCK_TIMEOUT_S) is False:
        raise HTTPException(503, "GPU busy; retry later")
    queued = time.time()
    try:
        started = time.time()
        crop = robust_content_box(bgr)
        x0, y0, w, h = crop
        rgb = cv2.cvtColor(bgr[y0:y0 + h, x0:x0 + w], cv2.COLOR_BGR2RGB)
        packet = FramePacket(
            image=rgb, frame_id=0, source_id="api",
            coordinate_space=CoordinateSpace.PROCESSED_IMAGE,
            image_color="rgb",
            metadata={"crop_box_xywh": crop,
                      "original_size_wh": (bgr.shape[1], bgr.shape[0])},
        )
        confidence = None
        if bbox_original is None:
            detection = STATE["detector"].detect(packet)
            dets = detection.data.detections if detection.data is not None else []
            if not dets:
                return {"detected": False, "size_mm": None, "bbox_original": None,
                        "bbox_content": None, "confidence": None, "q": None,
                        "model": STATE["model"], "run_id": STATE["run_id"],
                        "elapsed_ms": round((time.time() - started) * 1000, 1)}
            best = max(dets, key=lambda d: d.bbox.width * d.bbox.height)
            bbox_content = [float(best.bbox.x1), float(best.bbox.y1),
                            float(best.bbox.x2), float(best.bbox.y2)]
            confidence = float(best.confidence)
        else:
            bbox_content = [
                min(max(bbox_original[0] - x0, 0.0), w - 1),
                min(max(bbox_original[1] - y0, 0.0), h - 1),
                min(max(bbox_original[2] - x0, 0.0), w - 1),
                min(max(bbox_original[3] - y0, 0.0), h - 1),
            ]
            if not (bbox_content[0] < bbox_content[2] and bbox_content[1] < bbox_content[3]):
                raise HTTPException(422, "bbox is outside the endoscope content area")

        prediction = STATE["pipe"].depth_estimator.predict(packet)
        if prediction.data is None:
            raise HTTPException(422, f"depth failed: {prediction.message}")
        raw = align_relative_depth(packet, prediction.data)
        item = prepare_lesion(rgb, raw, bbox_content)
        batch = {
            "rgb_views": torch.from_numpy(item["rgb"]).unsqueeze(0).cuda(),
            "depth": torch.from_numpy(item["depth"][:, :2]).unsqueeze(0).cuda(),
            "scalars": torch.from_numpy(item["scalars"]).unsqueeze(0).cuda(),
            "D": torch.tensor([float(np.exp(item["scalars"][6]))],
                              dtype=torch.float32, device="cuda"),
            "S": torch.tensor([float(item["span_px"])],
                              dtype=torch.float32, device="cuda"),
        }
        with torch.inference_mode():
            value, _ = STATE["net"](batch)
        size_mm = float(value[0])
        return {
            "detected": True,
            "size_mm": round(size_mm, 3),
            "bbox_original": [round(x0 + bbox_content[0], 1),
                              round(y0 + bbox_content[1], 1),
                              round(x0 + bbox_content[2], 1),
                              round(y0 + bbox_content[3], 1)],
            "bbox_content": [round(v, 1) for v in bbox_content],
            "confidence": None if confidence is None else round(confidence, 4),
            "q": round(float(item["q"]), 5),
            "span_px": round(float(item["span_px"]), 1),
            "model": STATE["model"],
            "run_id": STATE["run_id"],
            "elapsed_ms": round((time.time() - started) * 1000, 1),
        }
    finally:
        GPU_LOCK.release()


async def _require_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.time()
    _load_models()
    STATE["loaded_s"] = round(time.time() - started, 1)
    yield
    try:
        STATE["detector"].close()
    except Exception:  # noqa: BLE001
        pass


app = FastAPI(title="beta-only polyp size API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Browser-friendly root: send people to Swagger instead of a 404."""
    return RedirectResponse("/docs")


@app.get("/health")
def health(_: None = Depends(_require_key)) -> dict:
    return {
        "status": "ok" if "net" in STATE else "loading",
        "model": STATE.get("model"),
        "run_id": STATE.get("run_id"),
        "epoch": STATE.get("epoch"),
        "device": STATE.get("device"),
        "loaded_s": STATE.get("loaded_s"),
    }


@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    bbox: str | None = Form(default=None),
    _: None = Depends(_require_key),
) -> dict:
    bgr = _decode_image(file.file.read())
    return _infer(bgr, _parse_bbox(bbox))


class PredictJson(BaseModel):
    image_base64: str = Field(min_length=1)
    bbox: list[float] | None = None


@app.post("/predict/json")
def predict_json(req: PredictJson, _: None = Depends(_require_key)) -> dict:
    try:
        data = base64.b64decode(req.image_base64, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, "invalid base64 image") from exc
    bgr = _decode_image(data)
    bbox = None if req.bbox is None else _parse_bbox(json.dumps(req.bbox))
    return _infer(bgr, bbox)
