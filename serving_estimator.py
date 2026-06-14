"""
LifeLine Loop — Serving Estimator Module
Predicts approximate number of servings from food metadata.

Model  : Random Forest Regressor (scikit-learn)
Input  : food_type, weight_kg, container_type, is_cooked
Output : estimated_servings (int)

The model is trained on synthetic but realistic data.
Swap train_model() with real logged data once you have it.
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── Category definitions ────────────────────────────────────────────────────
FOOD_TYPES = [
    "rice",
    "biryani",
    "dal",
    "sabzi",
    "roti",
    "bread",
    "soup",
    "salad",
    "dessert",
    "snacks",
    "fruits",
    "mixed",
]

CONTAINER_TYPES = [
    "bucket",      # ~20 L — hostel / wedding
    "large_tray",  # ~10 L — restaurant
    "vessel",      # ~5 L  — home cooking
    "box",         # ~2 L  — packed meal
    "packet",      # ~0.5 L
]

MODEL_PATH = "serving_estimator_model.pkl"
ENCODER_PATH = "serving_label_encoders.pkl"


# ── Synthetic training data generator ──────────────────────────────────────
def generate_training_data(n_samples: int = 3000) -> tuple:
    """
    Generates realistic (food_type, weight_kg, container_type, is_cooked)
    → servings training pairs based on domain knowledge.
    """
    rng = np.random.default_rng(42)

    # Avg serving size per food type (grams per person)
    serving_grams = {
        "rice":    200,
        "biryani": 250,
        "dal":     150,
        "sabzi":   120,
        "roti":     60,   # per piece ~60g, typically 2–3 pieces
        "bread":    80,
        "soup":    250,
        "salad":   150,
        "dessert": 100,
        "snacks":   80,
        "fruits":  150,
        "mixed":   200,
    }

    records = []
    targets = []

    for _ in range(n_samples):
        food_type  = rng.choice(FOOD_TYPES)
        container  = rng.choice(CONTAINER_TYPES)
        is_cooked  = rng.integers(0, 2)           # 0 or 1

        # Weight based on container type (with noise)
        base_weight = {
            "bucket":     18.0,
            "large_tray": 9.0,
            "vessel":     4.5,
            "box":        1.8,
            "packet":     0.4,
        }[container]
        weight_kg = max(0.2, base_weight + rng.normal(0, base_weight * 0.15))

        # Servings = weight / per-person grams (+ small noise)
        grams_per_serving = serving_grams[food_type]
        # Cooked food is slightly denser after cooking; raw ingredients yield fewer servings
        if not is_cooked:
            grams_per_serving *= 0.75
        servings = (weight_kg * 1000) / grams_per_serving
        servings = max(1, round(servings + rng.normal(0, servings * 0.1)))

        records.append([food_type, weight_kg, container, is_cooked])
        targets.append(servings)

    return records, targets


# ── Label encoders ─────────────────────────────────────────────────────────
def build_encoders() -> dict:
    le_food      = LabelEncoder().fit(FOOD_TYPES)
    le_container = LabelEncoder().fit(CONTAINER_TYPES)
    return {"food": le_food, "container": le_container}


def encode_features(records: list, encoders: dict) -> np.ndarray:
    """Convert raw records → numeric feature matrix."""
    rows = []
    for food_type, weight_kg, container, is_cooked in records:
        f = encoders["food"].transform([food_type])[0]
        c = encoders["container"].transform([container])[0]
        rows.append([f, weight_kg, c, int(is_cooked)])
    return np.array(rows, dtype=np.float32)


# ── Train & persist ─────────────────────────────────────────────────────────
def train_model(save: bool = True) -> tuple:
    """Train the Random Forest and optionally save to disk."""
    print("[INFO] Generating training data …")
    records, targets = generate_training_data(n_samples=3000)
    encoders = build_encoders()

    X = encode_features(records, encoders)
    y = np.array(targets, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[INFO] Training Random Forest Regressor …")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"[INFO] MAE = {mae:.1f} servings | R² = {r2:.3f}")

    if save:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(encoders, f)
        print(f"[INFO] Saved model → {MODEL_PATH}")

    return model, encoders


def load_model() -> tuple:
    """Load a previously saved model from disk (or train fresh if missing)."""
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            encoders = pickle.load(f)
        return model, encoders
    print("[INFO] No saved model found — training from scratch …")
    return train_model(save=True)


# ── Public API ──────────────────────────────────────────────────────────────
class ServingEstimator:
    """
    Estimates the number of servings from food metadata.

    Usage
    -----
    estimator = ServingEstimator()
    result = estimator.predict(
        food_type="biryani",
        weight_kg=8.0,
        container_type="large_tray",
        is_cooked=True
    )
    # → {"estimated_servings": 32, "confidence_range": "28–36"}
    """

    def __init__(self):
        self.model, self.encoders = load_model()

    def predict(
        self,
        food_type: str    = "mixed",
        weight_kg: float  = 5.0,
        container_type: str = "vessel",
        is_cooked: bool   = True,
    ) -> dict:
        # Normalise inputs
        food_type      = food_type.lower() if food_type.lower() in FOOD_TYPES else "mixed"
        container_type = container_type.lower() if container_type.lower() in CONTAINER_TYPES else "vessel"

        record = [[food_type, weight_kg, container_type, int(is_cooked)]]
        X = encode_features(record, self.encoders)

        # Get predictions from all trees for confidence range
        tree_preds = np.array([
            tree.predict(X)[0]
            for tree in self.model.estimators_
        ])
        mean_pred = int(round(float(np.mean(tree_preds))))
        std_pred  = float(np.std(tree_preds))

        lower = max(1, int(mean_pred - std_pred))
        upper = int(mean_pred + std_pred)

        return {
            "estimated_servings": mean_pred,
            "confidence_range":   f"{lower}–{upper}",
            "food_type":          food_type,
            "weight_kg":          weight_kg,
            "container_type":     container_type,
            "is_cooked":          is_cooked,
        }


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    est = ServingEstimator()

    test_cases = [
        dict(food_type="biryani",  weight_kg=10.0, container_type="large_tray", is_cooked=True),
        dict(food_type="roti",     weight_kg=3.0,  container_type="vessel",      is_cooked=True),
        dict(food_type="dal",      weight_kg=5.0,  container_type="vessel",      is_cooked=True),
        dict(food_type="fruits",   weight_kg=2.0,  container_type="box",         is_cooked=False),
        dict(food_type="biryani",  weight_kg=50.0, container_type="bucket",      is_cooked=True),
    ]

    print("\n🍽️  Serving Estimator Results")
    print("-" * 55)
    for tc in test_cases:
        r = est.predict(**tc)
        print(
            f"  {r['food_type']:<10} | {r['weight_kg']:>5} kg | "
            f"~{r['estimated_servings']:>3} servings  ({r['confidence_range']})"
        )
