# Beta-only polyp size API

Single-process FastAPI service for the adopted `beta-only` v11 model
(`company-diranchor40-betaonly` s21, `fusion: none`).

- HTTP requests are concurrent; **GPU inference is serialized** by one lock
  (detector -> DA-V2 depth -> beta-only V11). One 11 GB GPU is the bottleneck.
- Model input matches training: content crop, company detector (or client bbox),
  DA-V2 relative depth, 10 geometric scalars, RGB whole + 1.25x tight views.
- `bbox` is in **original image pixel coordinates** `[x1,y1,x2,y2]`.
  If omitted, the company detector runs. The literal `bbox=string` that Swagger
  UI prefills is treated as omitted.

## Run

```bash
cd /media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_ep
export DLEF_DATA_ROOT=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data
export PYTHONPATH=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/DLEF
/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/PolypMeasure/.venv/bin/uvicorn api.server:app \
  --host 0.0.0.0 --port 8000 --workers 1
```

Intranet: `http://192.168.0.115:8000/` (Swagger at `/docs`).

Optional env:
- `V11_API_KEY=...` require `X-API-Key` on all endpoints.
- `V11_LOCK_TIMEOUT_S=30` return 503 instead of waiting indefinitely.
- `V11_MAX_IMAGE_MB=20` upload limit.
- `V11_RUN=/path/to/run` load another DA-trained beta-only checkpoint
  (do **not** point this at an EndoStreamDepth run: the depth pipeline here is DA).

## Endpoints

### `GET /health`

```json
{"status":"ok","model":"v11-betaonly-s21","run_id":"20261106-075001...","epoch":40,"device":"cuda","loaded_s":42.1}
```

### `POST /predict` (multipart)

```bash
curl -s -X POST http://192.168.0.115:8000/predict \
  -F "file=@/path/frame.jpg"
# with a client-side detection box in original coordinates:
curl -s -X POST http://192.168.0.115:8000/predict \
  -F "file=@/path/frame.jpg" \
  -F 'bbox=[798,254,946,424]'
```

### `POST /predict/json`

```json
{"image_base64": "...", "bbox": [798, 254, 946, 424]}
```

Response:

```json
{
  "detected": true,
  "size_mm": 6.42,
  "bbox_original": [798.0, 254.0, 946.0, 424.0],
  "bbox_content": [104.0, 224.0, 252.0, 394.0],
  "confidence": 0.782,
  "q": 1.18,
  "span_px": 170.0,
  "model": "v11-betaonly-s21",
  "run_id": "20261106-075001_f83b8b9b_s20261021_48542a",
  "elapsed_ms": 180.4
}
```

No detection returns `{"detected": false, "size_mm": null, ...}` with HTTP 200.

## Notes

- **Research use only**: the model was trained on company-v1; cross-device
  absolute scale can be biased. Do not use as a clinical measurement.
- One GPU process: for higher throughput run multiple API processes on multiple
  GPUs (each loads its own models), or add a request queue.
- The first request after startup is warm because models are loaded in the
  FastAPI lifespan; subsequent inference is ~0.2-0.6 s on an RTX 2080 Ti.
- No files are written per request; images stay in memory.
