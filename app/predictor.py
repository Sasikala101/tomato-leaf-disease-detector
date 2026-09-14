import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

from .diseases import CLASS_NAMES, DISEASES


class ModelUnavailableError(RuntimeError):
    pass


class Predictor:
    def __init__(self) -> None:
        self.model_path = Path(os.getenv("MODEL_PATH", "models/tomato_leaf_model.keras"))
        self.demo_mode = os.getenv("DEMO_MODE", "false").lower() == "true"
        self.model = None
        self.load_error = None
        if self.model_path.exists():
            try:
                from tensorflow.keras.models import load_model

                self.model = load_model(str(self.model_path))
            except Exception as exc:  # surfaced through /api/health
                self.load_error = str(exc)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        if self.model is None:
            if not self.demo_mode:
                raise ModelUnavailableError(
                    "No trained model is loaded. Train the model or set DEMO_MODE=true to test the interface."
                )
            return self._demo_prediction(image)

        array = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
        probabilities = np.asarray(self.model.predict(array[None, ...], verbose=0)[0])
        if len(probabilities) != len(CLASS_NAMES):
            raise RuntimeError("Model output count does not match the configured disease classes.")
        index = int(np.argmax(probabilities))
        return self._response(index, float(probabilities[index]), demo=False)

    def _demo_prediction(self, image: Image.Image) -> Dict[str, Any]:
        # Deterministic placeholder for UI testing; never presented as a real diagnosis.
        pixels = np.asarray(image.convert("RGB").resize((32, 32)), dtype=np.uint8)
        index = int(pixels.mean()) % len(CLASS_NAMES)
        return self._response(index, 0.50, demo=True)

    @staticmethod
    def _response(index: int, confidence: float, demo: bool) -> Dict[str, Any]:
        key = CLASS_NAMES[index]
        details = DISEASES[key]
        return {
            "class_key": key,
            "disease": details["name"],
            "confidence": round(confidence * 100, 2),
            "remedy": details["remedy"],
            "demo": demo,
            "disclaimer": "This result is an AI screening aid, not a definitive agricultural diagnosis.",
        }

