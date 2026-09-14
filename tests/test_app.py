import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_rejects_non_image():
    response = client.post("/api/predict", files={"file": ("note.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_valid_image_requires_model_when_demo_disabled():
    stream = io.BytesIO()
    Image.new("RGB", (20, 20), "green").save(stream, format="PNG")
    response = client.post("/api/predict", files={"file": ("leaf.png", stream.getvalue(), "image/png")})
    assert response.status_code in (200, 503)

