"""
LifeLine Loop — Expiry Risk Predictor Module
Classifies a food donation's urgency as High / Medium / Low priority.

Model  : Random Forest Classifier (scikit-learn)
Input  : food_type, hours_remaining, is_cooked, temperature_c, is_perishable
Output : priority ("High" | "Medium" | "Low") + confidence + reason

Risk logic (domain rules embedded in training data):
  High   → expires in < 2 h, or cooked perishables in heat
  Medium → expires in 2–6 h
  Low    → expires in > 6 h, or dry / packed food
"""

import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Constants ───────────────────────────────────────────────────────────────
PRIORITY_LABELS = ["High", "Medium", "Low"]

FOOD_PERISHABILITY = {
    # (is_perishable, typical_max_hours_at_room_temp)
    "rice":    (True,  4),
    "biryani": (True,  4),
    "dal":     (True,  5),
    "sabzi":   (True,  5),
    "roti":    (True,  8),
    "bread":   (False, 24),
    "soup":    (True,  4),
    "salad":   (True,  3),
    "dessert": (True,  6),
    "snacks":  (False, 48),
    "fruits":  (False, 12),
    "mixed":   (True,  4),
}

MODEL_PATH   = "expiry_model.pkl"
ENCODER_PATH = "expiry_encoders.pkl"

FOOD_TYPES = list(FOOD_PERISHABILITY.keys())


# ── Priority assignment rule (used for label generation) ───────────────────
def _assign_priority(
    food_type: str,
    hours_remaining: float,
    is_cooked: int,
    temperature_c: float,
) -> str:
    is_perishable, max_hours = FOOD_PERISHABILITY.get(food_type, (True, 4))

    # Temperature penalty: every 5°C above 25 halves safe window
    temp_excess   = max(0.0, temperature_c - 25)
    temp_factor   = 2 ** (temp_excess / 5.0)          # 1.0 at 25°C, 2.0 at 30°C …
    effective_hrs = hours_remaining / temp_factor

    if is_cooked and is_perishable:
        if effective_hrs < 1.5:
            return "High"
        elif effective_hrs < 4.0:
            return "Medium"
        else:
            return "Low"
    elif is_perishable:
        if effective_hrs < 2.0:
            return "High"
        elif effective_hrs < 6.0:
            return "Medium"
        else:
            return "Low"
    else:
        # Non-perishable
        if hours_remaining < 1.0:
            return "High"
        elif hours_remaining < 8.0:
            return "Medium"
        else:
            return "Low"


# ── Synthetic training data ─────────────────────────────────────────────────
def generate_training_data(n_samples: int = 4000) -> tuple:
    rng = np.random.default_rng(7)
    records, labels = [], []

    for _ in range(n_samples):
        food_type       = rng.choice(FOOD_TYPES)
        hours_remaining = round(float(rng.exponential(scale=5.0)), 2)   # 0–30 h range
        hours_remaining = min(hours_remaining, 48.0)
        is_cooked       = int(rng.integers(0, 2))
        temperature_c   = float(rng.uniform(18, 42))                    # 18–42°C
        is_perishable   = int(FOOD_PERISHABILITY[food_type][0])

        label = _assign_priority(food_type, hours_remaining, is_cooked, temperature_c)

        records.append([food_type, hours_remaining, is_cooked, temperature_c, is_perishable])
        labels.append(label)

    return records, labels


# ── Encoders ────────────────────────────────────────────────────────────────
def build_encoders() -> dict:
    le_food     = LabelEncoder().fit(FOOD_TYPES)
    le_priority = LabelEncoder().fit(PRIORITY_LABELS)
    return {"food": le_food, "priority": le_priority}


def encode_X(records: list, encoders: dict) -> np.ndarray:
    rows = []
    for food_type, hours_remaining, is_cooked, temperature_c, is_perishable in records:
        f = encoders["food"].transform([food_type])[0]
        rows.append([f, hours_remaining, is_cooked, temperature_c, is_perishable])
    return np.array(rows, dtype=np.float32)


# ── Train & persist ─────────────────────────────────────────────────────────
def train_model(save: bool = True) -> tuple:
    print("[INFO] Generating training data …")
    records, labels = generate_training_data(4000)
    encoders = build_encoders()

    X = encode_X(records, encoders)
    y = encoders["priority"].transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[INFO] Training Random Forest Classifier …")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    label_names = encoders["priority"].classes_
    print(classification_report(y_test, y_pred, target_names=label_names))

    if save:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(encoders, f)
        print(f"[INFO] Saved model → {MODEL_PATH}")

    return model, encoders


def load_model() -> tuple:
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            encoders = pickle.load(f)
        return model, encoders
    print("[INFO] No saved model — training from scratch …")
    return train_model(save=True)


# ── Human-readable reason generator ────────────────────────────────────────
def _build_reason(
    priority: str,
    food_type: str,
    hours_remaining: float,
    temperature_c: float,
    is_cooked: bool,
) -> str:
    reasons = []
    if hours_remaining < 1:
        reasons.append(f"only {int(hours_remaining * 60)} minutes left before expiry")
    elif hours_remaining < 3:
        reasons.append(f"{hours_remaining:.1f} hours until expiry")
    else:
        reasons.append(f"{hours_remaining:.1f} hours remaining")

    if temperature_c > 35:
        reasons.append(f"high ambient temperature ({temperature_c:.0f}°C accelerates spoilage)")
    if is_cooked and FOOD_PERISHABILITY.get(food_type, (True,))[0]:
        reasons.append("cooked perishable food degrades quickly")

    return "; ".join(reasons).capitalize() + "."


# ── Public API ──────────────────────────────────────────────────────────────
class ExpiryPredictor:
    """
    Predicts food expiry risk (priority) given metadata.

    Usage
    -----
    predictor = ExpiryPredictor()

    result = predictor.predict(
        food_type="biryani",
        available_until="21:00",      # or pass hours_remaining directly
        is_cooked=True,
        temperature_c=32.0,
    )
    # → {
    #     "priority": "High",
    #     "confidence": 0.95,
    #     "hours_remaining": 1.5,
    #     "badge_color": "#ef4444",
    #     "reason": "..."
    #   }
    """

    BADGE_COLORS = {
        "High":   "#ef4444",   # red
        "Medium": "#f97316",   # orange
        "Low":    "#22c55e",   # green
    }

    def __init__(self):
        self.model, self.encoders = load_model()

    def predict(
        self,
        food_type: str        = "mixed",
        hours_remaining: float = None,
        available_until: str  = None,      # "HH:MM" (24-h, today)
        is_cooked: bool       = True,
        temperature_c: float  = 28.0,
    ) -> dict:

        # ── Resolve hours_remaining ──────────────────────────────────────
        if hours_remaining is None:
            if available_until:
                now  = datetime.now()
                h, m = map(int, available_until.split(":"))
                target = now.replace(hour=h, minute=m, second=0, microsecond=0)
                if target < now:
                    target = target.replace(day=target.day + 1)   # next day
                hours_remaining = max(0.0, (target - now).total_seconds() / 3600)
            else:
                hours_remaining = 3.0   # default fallback

        # ── Normalise food type ──────────────────────────────────────────
        food_type = food_type.lower() if food_type.lower() in FOOD_TYPES else "mixed"

        # ── Feature vector ───────────────────────────────────────────────
        is_perishable = int(FOOD_PERISHABILITY[food_type][0])
        record = [[food_type, hours_remaining, int(is_cooked), temperature_c, is_perishable]]
        X = encode_X(record, self.encoders)

        # ── Prediction + confidence ──────────────────────────────────────
        proba     = self.model.predict_proba(X)[0]
        pred_idx  = int(np.argmax(proba))
        priority  = self.encoders["priority"].classes_[pred_idx]
        confidence = round(float(proba[pred_idx]), 3)

        # ── All probabilities ────────────────────────────────────────────
        all_probs = {
            cls: round(float(p), 3)
            for cls, p in zip(self.encoders["priority"].classes_, proba)
        }

        reason = _build_reason(priority, food_type, hours_remaining, temperature_c, is_cooked)

        return {
            "priority":        priority,
            "confidence":      confidence,
            "all_probabilities": all_probs,
            "hours_remaining": round(hours_remaining, 2),
            "badge_color":     self.BADGE_COLORS[priority],
            "reason":          reason,
            "food_type":       food_type,
            "temperature_c":   temperature_c,
        }


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = ExpiryPredictor()

    test_cases = [
        dict(food_type="biryani", hours_remaining=0.5, is_cooked=True,  temperature_c=35),
        dict(food_type="dal",     hours_remaining=3.0, is_cooked=True,  temperature_c=28),
        dict(food_type="snacks",  hours_remaining=20,  is_cooked=False, temperature_c=25),
        dict(food_type="salad",   hours_remaining=1.0, is_cooked=False, temperature_c=38),
        dict(food_type="roti",    hours_remaining=6.0, is_cooked=True,  temperature_c=30),
    ]

    print("\n⏰  Expiry Risk Predictor Results")
    print("-" * 65)
    for tc in test_cases:
        r = predictor.predict(**tc)
        bar = "🔴" if r["priority"]=="High" else ("🟠" if r["priority"]=="Medium" else "🟢")
        print(
            f"  {bar} [{r['priority']:<6}] {r['food_type']:<10} | "
            f"{r['hours_remaining']:>4.1f}h | {r['temperature_c']:.0f}°C | "
            f"conf={r['confidence']:.0%}"
        )
    print()
