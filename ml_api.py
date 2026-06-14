"""
LifeLine Loop — ML FastAPI Service
Exposes three endpoints for the Next.js frontend to call.

Endpoints
---------
POST /api/ml/recognize-food     → food category from image
POST /api/ml/estimate-servings  → serving count from metadata
POST /api/ml/expiry-risk        → High / Medium / Low priority

Run locally:
    pip install fastapi uvicorn python-multipart tensorflow scikit-learn pillow
    uvicorn ml_api:app --reload --port 8000

Then call from Next.js:
    fetch("http://localhost:8000/api/ml/expiry-risk", { method: "POST", ... })

Deploy to Railway / Render / Fly.io for production.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import base64

from food_recognizer  import FoodRecognizer
from serving_estimator import ServingEstimator
from expiry_predictor import ExpiryPredictor

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LifeLine Loop ML Service",
    description="Food recognition, serving estimation, and expiry risk prediction.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict to your Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models at startup (warm-up) ────────────────────────────────────────
print("[STARTUP] Loading ML models …")
recognizer  = FoodRecognizer()
estimator   = ServingEstimator()
predictor   = ExpiryPredictor()
print("[STARTUP] All models ready ✓")


# ── Request / Response schemas ───────────────────────────────────────────────

class ServingRequest(BaseModel):
    food_type:      str   = Field("mixed",    description="Type of food (rice, biryani, dal …)")
    weight_kg:      float = Field(5.0,        description="Estimated weight in kg")
    container_type: str   = Field("vessel",   description="Container size: bucket/large_tray/vessel/box/packet")
    is_cooked:      bool  = Field(True,       description="Is the food already cooked?")

class ServingResponse(BaseModel):
    estimated_servings: int
    confidence_range:   str
    food_type:          str
    weight_kg:          float
    container_type:     str
    is_cooked:          bool


class ExpiryRequest(BaseModel):
    food_type:        str            = Field("mixed", description="Type of food")
    hours_remaining:  Optional[float]= Field(None,   description="Hours until expiry (overrides available_until)")
    available_until:  Optional[str]  = Field(None,   description="Time in HH:MM format (24h, today)")
    is_cooked:        bool           = Field(True,   description="Is the food cooked?")
    temperature_c:    float          = Field(28.0,   description="Ambient temperature in °C")

class ExpiryResponse(BaseModel):
    priority:          str
    confidence:        float
    all_probabilities: dict
    hours_remaining:   float
    badge_color:       str
    reason:            str
    food_type:         str
    temperature_c:     float


class FoodImageBase64Request(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image string")

class FoodRecognitionResponse(BaseModel):
    category:   str
    confidence: float
    raw_label:  str
    all_top5:   list


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "LifeLine Loop ML"}


# 1️⃣  Food Recognition — multipart upload ─────────────────────────────────
@app.post(
    "/api/ml/recognize-food",
    response_model=FoodRecognitionResponse,
    summary="Classify food from an uploaded image",
)
async def recognize_food_upload(file: UploadFile = File(...)):
    """
    Upload a food image (jpg/png) and get back:
    - category  : e.g. "Rice / Biryani"
    - confidence: probability 0–1
    - raw_label : original ImageNet label
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg/png).")
    image_bytes = await file.read()
    result = recognizer.predict_from_bytes(image_bytes)
    return FoodRecognitionResponse(**result)


# 1️⃣b Food Recognition — base64 (for JS fetch without FormData) ───────────
@app.post(
    "/api/ml/recognize-food-b64",
    response_model=FoodRecognitionResponse,
    summary="Classify food from a base64 image string",
)
async def recognize_food_b64(payload: FoodImageBase64Request):
    try:
        result = recognizer.predict_from_base64(payload.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
    return FoodRecognitionResponse(**result)


# 2️⃣  Serving Estimation ────────────────────────────────────────────────────
@app.post(
    "/api/ml/estimate-servings",
    response_model=ServingResponse,
    summary="Estimate number of servings from food metadata",
)
def estimate_servings(payload: ServingRequest):
    """
    Send food metadata and get estimated servings:
    - estimated_servings : e.g. 40
    - confidence_range   : e.g. "35–45"
    """
    result = estimator.predict(
        food_type      = payload.food_type,
        weight_kg      = payload.weight_kg,
        container_type = payload.container_type,
        is_cooked      = payload.is_cooked,
    )
    return ServingResponse(**result)


# 3️⃣  Expiry Risk Prediction ─────────────────────────────────────────────────
@app.post(
    "/api/ml/expiry-risk",
    response_model=ExpiryResponse,
    summary="Predict food expiry urgency (High / Medium / Low)",
)
def expiry_risk(payload: ExpiryRequest):
    """
    Send donation timing + food details and get:
    - priority     : "High" | "Medium" | "Low"
    - badge_color  : hex color for the UI badge
    - reason       : human-readable explanation
    - confidence   : 0–1
    """
    result = predictor.predict(
        food_type       = payload.food_type,
        hours_remaining = payload.hours_remaining,
        available_until = payload.available_until,
        is_cooked       = payload.is_cooked,
        temperature_c   = payload.temperature_c,
    )
    return ExpiryResponse(**result)


# ── Combo endpoint: all three in one call ────────────────────────────────────
class ComboResponse(BaseModel):
    recognition: dict
    servings:    dict
    expiry:      dict

@app.post(
    "/api/ml/analyze-donation",
    summary="Run all three ML models in one request (image + metadata)",
)
async def analyze_donation(
    file:           UploadFile = File(...),
    weight_kg:      float      = 5.0,
    container_type: str        = "vessel",
    is_cooked:      bool       = True,
    available_until: str       = None,
    hours_remaining: float     = None,
    temperature_c:  float      = 28.0,
):
    """
    One-shot endpoint: upload image + pass metadata, get back
    food category, serving estimate, and expiry risk together.
    """
    image_bytes = await file.read()

    recognition = recognizer.predict_from_bytes(image_bytes)
    food_type   = recognition["category"].lower().split("/")[0].strip()

    servings = estimator.predict(
        food_type=food_type,
        weight_kg=weight_kg,
        container_type=container_type,
        is_cooked=is_cooked,
    )
    expiry = predictor.predict(
        food_type=food_type,
        hours_remaining=hours_remaining,
        available_until=available_until,
        is_cooked=is_cooked,
        temperature_c=temperature_c,
    )

    return {"recognition": recognition, "servings": servings, "expiry": expiry}
