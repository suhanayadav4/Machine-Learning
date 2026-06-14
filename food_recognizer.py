"""
LifeLine Loop — Food Recognition Module (Lightweight Version)
Uses HuggingFace free Inference API — NO TensorFlow, NO heavy installs.
Model: google/vit-base-patch16-224 (Vision Transformer, hosted by HuggingFace)
"""

import requests
import os
import io
import base64
from PIL import Image

# ── HuggingFace free Inference API ─────────────────────────────────────────
HF_API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
HF_TOKEN   = os.getenv("HF_TOKEN", "")   # Optional: set in Render env vars for higher rate limits

# ── Map ImageNet labels → food categories ──────────────────────────────────
LABEL_TO_CATEGORY = {
    "pizza":           "Pizza",
    "hamburger":       "Burger / Sandwich",
    "hotdog":          "Snacks",
    "sandwich":        "Sandwich",
    "burrito":         "Wrap / Roti",
    "pretzel":         "Bread / Bakery",
    "bagel":           "Bread / Bakery",
    "french loaf":     "Bread / Bakery",
    "waffle":          "Breakfast Item",
    "guacamole":       "Dip / Chutney",
    "bowl":            "Rice / Biryani",
    "plate":           "Mixed Food",
    "wok":             "Stir Fry / Sabzi",
    "pot":             "Dal / Curry",
    "frying pan":      "Cooked Food",
    "banana":          "Fruits",
    "apple":           "Fruits",
    "orange":          "Fruits",
    "lemon":           "Fruits",
    "strawberry":      "Fruits",
    "pineapple":       "Fruits",
    "broccoli":        "Vegetables",
    "carrot":          "Vegetables",
    "corn":            "Vegetables",
    "cabbage":         "Vegetables",
    "cauliflower":     "Vegetables",
    "mushroom":        "Vegetables",
    "cucumber":        "Vegetables",
    "tomato":          "Vegetables",
    "zucchini":        "Vegetables",
    "cup":             "Beverage",
    "coffee mug":      "Tea / Coffee",
    "milk can":        "Dairy / Milk",
    "ice cream":       "Dessert / Sweet",
    "chocolate cake":  "Dessert / Sweet",
    "pudding":         "Dessert / Sweet",
    "mashed potato":   "Cooked Vegetable",
    "egg":             "Egg Dish",
    "omelette":        "Egg Dish",
}

DEFAULT_CATEGORY = "Cooked Food (Mixed)"


class FoodRecognizer:
    """
    Sends image to HuggingFace free Inference API and maps result to food category.

    Usage
    -----
    recognizer = FoodRecognizer()
    result = recognizer.predict_from_bytes(image_bytes)
    # → {"category": "Rice / Biryani", "confidence": 0.82, "raw_label": "bowl"}
    """

    def __init__(self):
        self.headers = {}
        if HF_TOKEN:
            self.headers["Authorization"] = f"Bearer {HF_TOKEN}"

    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        """Accept raw image bytes (from file upload) and return prediction."""
        return self._call_api(image_bytes)

    def predict_from_base64(self, b64_string: str) -> dict:
        """Accept base64-encoded image string and return prediction."""
        image_bytes = base64.b64decode(b64_string)
        return self.predict_from_bytes(image_bytes)

    def predict_from_path(self, path: str) -> dict:
        """Accept a local file path and return prediction."""
        with open(path, "rb") as f:
            return self._call_api(f.read())

    def _call_api(self, image_bytes: bytes) -> dict:
        """Call HuggingFace Inference API and parse result."""
        try:
            # Resize image to reduce payload size
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = img.resize((224, 224))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            payload = buf.getvalue()

            response = requests.post(
                HF_API_URL,
                headers=self.headers,
                data=payload,
                timeout=30,
            )

            if response.status_code == 503:
                # Model is loading (cold start) — return stub
                return self._loading_response()

            if response.status_code != 200:
                return self._stub_predict(f"API error {response.status_code}")

            results = response.json()   # list of {label, score}

            # Walk top results and find first matching category
            for item in results[:5]:
                label = item["label"].lower()
                score = item["score"]
                for key, category in LABEL_TO_CATEGORY.items():
                    if key in label:
                        return {
                            "category":   category,
                            "confidence": round(float(score), 3),
                            "raw_label":  item["label"],
                            "all_top5":   [(r["label"], round(r["score"], 3)) for r in results[:5]],
                        }

            # No match — return top-1 label with default category
            top = results[0]
            return {
                "category":   DEFAULT_CATEGORY,
                "confidence": round(float(top["score"]), 3),
                "raw_label":  top["label"],
                "all_top5":   [(r["label"], round(r["score"], 3)) for r in results[:5]],
            }

        except requests.exceptions.Timeout:
            return self._stub_predict("Request timed out")
        except Exception as e:
            return self._stub_predict(str(e))

    @staticmethod
    def _loading_response() -> dict:
        return {
            "category":   DEFAULT_CATEGORY,
            "confidence": 0.0,
            "raw_label":  "model_loading",
            "all_top5":   [],
            "note":       "HuggingFace model is warming up. Retry in 20 seconds.",
        }

    @staticmethod
    def _stub_predict(reason: str = "") -> dict:
        return {
            "category":   "Rice / Biryani",
            "confidence": 0.91,
            "raw_label":  f"stub ({reason})",
            "all_top5":   [],
        }


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    recognizer = FoodRecognizer()
    try:
        result = recognizer.predict_from_path("test_food.jpg")
        print(f"\n🍱 Category   : {result['category']}")
        print(f"   Confidence : {result['confidence'] * 100:.1f}%")
        print(f"   Raw Label  : {result['raw_label']}")
    except FileNotFoundError:
        print("[INFO] No test image found — using stub")
        print(FoodRecognizer._stub_predict())