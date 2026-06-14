"""
LifeLine Loop — Food Recognition Module
Uses MobileNetV2 (pretrained on ImageNet) to classify food from an image.
Maps ImageNet labels → Indian/common food categories.
"""

import numpy as np
from PIL import Image
import io
import base64

# ── Try to load TensorFlow; fall back to a rule-based stub if unavailable ──
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    print("[WARN] TensorFlow not found. Food recognizer running in stub mode.")


# ── Mapping: ImageNet synset → friendly food category ──────────────────────
IMAGENET_TO_FOOD = {
    # Cooked / prepared
    "pizza":           "Pizza",
    "hamburger":       "Burger / Sandwich",
    "hotdog":          "Snacks",
    "sandwich":        "Sandwich",
    "burrito":         "Wrap / Roti",
    "taco":            "Wrap / Roti",
    "pretzel":         "Bread / Bakery",
    "bagel":           "Bread / Bakery",
    "french_loaf":     "Bread / Bakery",
    "croissant":       "Bread / Bakery",
    "waffle":          "Breakfast Item",
    "pancake":         "Breakfast Item",
    "omelette":        "Egg Dish",
    "pot_pie":         "Cooked Curry / Gravy",
    "stew":            "Dal / Curry",
    "soup":            "Soup / Dal",
    "cheeseburger":    "Burger / Sandwich",
    "ice_cream":       "Dessert / Sweet",
    "chocolate_cake":  "Dessert / Sweet",
    "guacamole":       "Dip / Chutney",
    "mashed_potato":   "Cooked Vegetable",

    # Rice / biryani proxies
    "bowl":            "Rice / Biryani",
    "plate":           "Mixed Food",
    "wok":             "Stir Fry / Sabzi",

    # Fruits
    "banana":          "Fruits",
    "apple":           "Fruits",
    "orange":          "Fruits",
    "lemon":           "Fruits",
    "pomegranate":     "Fruits",
    "fig":             "Fruits",
    "strawberry":      "Fruits",
    "pineapple":       "Fruits",
    "mango":           "Fruits",
    "watermelon":      "Fruits",
    "grape":           "Fruits",

    # Vegetables
    "broccoli":        "Vegetables",
    "carrot":          "Vegetables",
    "corn":            "Vegetables",
    "head_cabbage":    "Vegetables",
    "cauliflower":     "Vegetables",
    "pumpkin":         "Vegetables",
    "mushroom":        "Vegetables",
    "bell_pepper":     "Vegetables",
    "cucumber":        "Vegetables",
    "tomato":          "Vegetables",
    "eggplant":        "Vegetables",
    "spinach":         "Vegetables",
    "artichoke":       "Vegetables",
    "zucchini":        "Vegetables",

    # Grains / legumes
    "acorn":           "Pulses / Dal",
    "bean_curd":       "Paneer / Tofu",

    # Beverages
    "cup":             "Beverage",
    "coffee_mug":      "Tea / Coffee",
    "milk_can":        "Dairy / Milk",
    "pitcher":         "Beverage",
    "water_bottle":    "Water / Beverage",
}

# Default when nothing matches
DEFAULT_CATEGORY = "Cooked Food (Mixed)"


class FoodRecognizer:
    """
    Wraps MobileNetV2 for food image recognition.

    Usage
    -----
    recognizer = FoodRecognizer()
    result = recognizer.predict_from_bytes(image_bytes)
    # → {"category": "Rice / Biryani", "confidence": 0.82, "raw_label": "bowl"}
    """

    def __init__(self):
        self.model = None
        if _TF_AVAILABLE:
            print("[INFO] Loading MobileNetV2 weights …")
            self.model = MobileNetV2(weights="imagenet")
            print("[INFO] MobileNetV2 ready.")

    # ── public API ────────────────────────────────────────────────────────

    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        """Accept raw image bytes (from file upload) and return prediction."""
        img = self._bytes_to_pil(image_bytes)
        return self._predict(img)

    def predict_from_base64(self, b64_string: str) -> dict:
        """Accept a base64-encoded image string and return prediction."""
        image_bytes = base64.b64decode(b64_string)
        return self.predict_from_bytes(image_bytes)

    def predict_from_path(self, path: str) -> dict:
        """Accept a local file path and return prediction."""
        img = Image.open(path).convert("RGB")
        return self._predict(img)

    # ── internals ─────────────────────────────────────────────────────────

    def _bytes_to_pil(self, data: bytes) -> Image.Image:
        return Image.open(io.BytesIO(data)).convert("RGB")

    def _predict(self, pil_img: Image.Image) -> dict:
        if not _TF_AVAILABLE or self.model is None:
            return self._stub_predict()

        # Resize & preprocess for MobileNetV2 (224×224)
        pil_img = pil_img.resize((224, 224))
        arr = np.array(pil_img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)

        preds = self.model.predict(arr, verbose=0)
        top5  = decode_predictions(preds, top=5)[0]   # list of (class_id, name, prob)

        # Walk top-5 predictions and find the first one we can map
        for _, label, prob in top5:
            label_lower = label.lower()
            for key, category in IMAGENET_TO_FOOD.items():
                if key in label_lower:
                    return {
                        "category":   category,
                        "confidence": round(float(prob), 3),
                        "raw_label":  label,
                        "all_top5":   [(n, round(float(p), 3)) for _, n, p in top5],
                    }

        # Fallback: return the top-1 label unmapped
        _, top_label, top_prob = top5[0]
        return {
            "category":   DEFAULT_CATEGORY,
            "confidence": round(float(top_prob), 3),
            "raw_label":  top_label,
            "all_top5":   [(n, round(float(p), 3)) for _, n, p in top5],
        }

    @staticmethod
    def _stub_predict() -> dict:
        """Returns a fake prediction when TensorFlow is not installed."""
        return {
            "category":   "Rice / Biryani",
            "confidence": 0.91,
            "raw_label":  "bowl (stub)",
            "all_top5":   [("bowl", 0.91), ("plate", 0.05)],
        }


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    recognizer = FoodRecognizer()

    # Test with a local image (change path as needed)
    try:
        result = recognizer.predict_from_path("test_food.jpg")
        print("\n🍱 Food Recognition Result")
        print(f"   Category   : {result['category']}")
        print(f"   Confidence : {result['confidence'] * 100:.1f}%")
        print(f"   Raw Label  : {result['raw_label']}")
    except FileNotFoundError:
        print("[INFO] No test image found. Running stub …")
        print(FoodRecognizer._stub_predict())
