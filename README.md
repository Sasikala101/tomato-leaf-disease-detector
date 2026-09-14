# Tomato Leaf Disease Detection System

A portfolio-ready web application that screens tomato leaf photographs across 10 PlantVillage-style classes, reports model confidence, and provides responsible disease-management guidance.

## Features

- FastAPI prediction API with interactive Swagger docs
- Responsive drag-and-drop frontend
- JPEG, PNG, and WebP validation with upload-size limits
- MobileNetV2 transfer-learning pipeline for 10 classes
- Confidence score, remedy guidance, and diagnostic disclaimer
- Explicit demo mode for testing the UI before training

## Local setup (Python 3.8)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Copy the environment example, then start the app:

```bash
cp .env.example .env
DEMO_MODE=true uvicorn app.main:app --reload --port 8001
```

Open <http://127.0.0.1:8001>. Demo predictions are deterministic placeholders and are visibly labeled; they are not agricultural diagnoses.

## Dataset layout and training

Download a properly licensed tomato-leaf dataset and arrange it as follows. Folder names must exactly match the keys in `app/diseases.py`.

```text
data/
  train/
    Tomato___Bacterial_spot/
    ...
    Tomato___healthy/
  validation/
    Tomato___Bacterial_spot/
    ...
    Tomato___healthy/
```

Train and save the model:

```bash
python train.py
```

The server automatically loads `models/tomato_leaf_model.keras`. Start it without demo mode for genuine inference:

```bash
uvicorn app.main:app --reload
```

## API

- `GET /api/health` — server and model readiness
- `POST /api/predict` — multipart image upload under the `file` field
- `GET /docs` — interactive API documentation

## Responsible use

This is a screening and educational tool. Field symptoms can overlap, image datasets can contain bias, and remedies differ by region. Confirm consequential crop decisions with a local agricultural extension service or plant pathologist.

## Before publishing results

Report the exact dataset source and license, class distribution, train/validation/test split, test accuracy, per-class precision and recall, confusion matrix, and model limitations. Do not claim the example image totals or model accuracy until they are reproduced locally.
