"""
Titanic Survival Prediction API
FastAPI app serving the trained SVC classification pipeline.
"""

import re
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Resolve paths relative to this file so they work regardless of CWD ───────
_ROOT = Path(__file__).resolve().parent
PIPELINE      = joblib.load(_ROOT / "models" / "final_pipeline.pkl")
LABEL_ENCODER = joblib.load(_ROOT / "models" / "label_encoder.pkl")

MODEL_NAME = type(PIPELINE.named_steps["model"]).__name__
MODEL_ACCURACY = 0.8436  # Test-set accuracy from training run

# ── Feature columns expected by the pipeline ─────────────────────────────────
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title"]
PIPELINE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ── Title extraction helpers (mirror src/preprocess.py) ──────────────────────
_TITLE_REGEX = re.compile(r" ([A-Za-z]+)\.")
_KNOWN_TITLES = {"Mr", "Mrs", "Miss", "Master"}


def _extract_title(name: Optional[str]) -> str:
    """Extract and normalise title from passenger name."""
    if not name:
        return "Mr"  # safe default
    match = _TITLE_REGEX.search(name)
    if not match:
        return "Mr"
    title = match.group(1)
    return title if title in _KNOWN_TITLES else "Rare"


# ── Pydantic models ───────────────────────────────────────────────────────────
class PassengerInput(BaseModel):
    PassengerId: Optional[int] = None
    Name: Optional[str] = None
    Pclass: int
    Sex: str
    Age: Optional[float] = None
    SibSp: int
    Parch: int
    Ticket: Optional[str] = None
    Fare: Optional[float] = None
    Cabin: Optional[str] = None
    Embarked: Optional[str] = None


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predicts Titanic passenger survival using a trained SVC pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper: build feature DataFrame from passenger input ─────────────────────
def _build_feature_df(passenger: PassengerInput) -> pd.DataFrame:
    title = _extract_title(passenger.Name)
    row = {
        "Pclass": passenger.Pclass,
        "Age": passenger.Age,
        "SibSp": passenger.SibSp,
        "Parch": passenger.Parch,
        "Fare": passenger.Fare,
        "Sex": passenger.Sex,
        "Embarked": passenger.Embarked,
        "Title": title,
    }
    return pd.DataFrame([row], columns=PIPELINE_FEATURES)


def _predict_one(df: pd.DataFrame) -> PredictionResponse:
    pred_class = int(PIPELINE.predict(df)[0])
    proba = float(PIPELINE.predict_proba(df)[0][pred_class])
    label = str(LABEL_ENCODER.inverse_transform([pred_class])[0])
    return PredictionResponse(prediction=pred_class, label=label, probability=round(proba, 4))


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "accuracy": MODEL_ACCURACY}


@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: PassengerInput):
    try:
        df = _build_feature_df(passenger)
        return _predict_one(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(passengers: List[PassengerInput]):
    if not passengers:
        raise HTTPException(status_code=400, detail="Passenger list must not be empty.")
    try:
        rows = [_build_feature_df(p).iloc[0].to_dict() for p in passengers]
        df = pd.DataFrame(rows, columns=PIPELINE_FEATURES)
        pred_classes = PIPELINE.predict(df).tolist()
        probas = PIPELINE.predict_proba(df)
        results = []
        for i, pred_class in enumerate(pred_classes):
            proba = float(probas[i][pred_class])
            label = str(LABEL_ENCODER.inverse_transform([pred_class])[0])
            results.append(
                PredictionResponse(
                    prediction=int(pred_class),
                    label=label,
                    probability=round(proba, 4),
                )
            )
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
