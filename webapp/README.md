# RF-DETR detection web demo

Pick a model (Atto / Femto / Pico / Nano / Small / Medium / XL / 2XL), upload an image, see the detected objects. Runs locally with PyTorch (MPS on Apple Silicon). No Roboflow API key needed.

Run from the repo root:

```bash
uv venv webapp/.venv --python 3.12
uv pip install --python webapp/.venv/bin/python -r webapp/requirements.txt
webapp/.venv/bin/python webapp/download_models.py     # all 8 models, ~2.3 GB; or name some: atto nano
webapp/.venv/bin/python -m uvicorn server:app --app-dir webapp --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000.

Atto, Femto, Pico, XL and 2XL come from the `rfdetr_plus` package (installed by `rfdetr[plus]`) under the Platform Model License 1.0. They download and run without an API key, but the license requires a Roboflow platform plan (free or paid) in good standing. Nano, Small and Medium are Apache-2.0.

This venv (latest rfdetr) is separate from the CoreML export environment, which needs rfdetr 1.5–1.6. Do not mix them.

API: `GET /api/models`, `POST /api/detect` (multipart `file`, `model`, `threshold`).
