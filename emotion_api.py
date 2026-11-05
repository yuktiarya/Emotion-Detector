"""FastAPI service that loads and serves the trained emotion pipeline."""

from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).parent / "Emotion_Detector_Model.pkl"
app = FastAPI(title="Emotion Detector API")


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class EmotionScore(BaseModel):
    emotion: str
    confidence: float


class PredictionResponse(BaseModel):
    emotion: str
    confidence: float
    alternatives: list[EmotionScore]


@lru_cache(maxsize=1)
def load_model():
    """Load the notebook-created model once when the API first needs it."""
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "Trained model not found. Run all cells in Train_Model.ipynb "
                "to create Emotion_Detector_Model.pkl."
            ),
        )

    model = joblib.load(MODEL_PATH)
    required_attributes = ("predict", "predict_proba", "classes_")
    if not all(hasattr(model, attribute) for attribute in required_attributes):
        raise HTTPException(
            status_code=503,
            detail=(
                "The pickle file is not a trained prediction pipeline. "
                "Run all cells in Train_Model.ipynb to replace it."
            ),
        )
    return model


@app.get("/health")
def health_check() -> dict[str, str]:
    """Report ready only when a trained model can be loaded."""
    load_model()
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(request: PredictionRequest) -> PredictionResponse:
    """Predict an emotion and return the most likely alternatives."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Please enter some text.")

    model = load_model()
    probabilities = model.predict_proba([text])[0]
    ranked_emotions = sorted(
        zip(model.classes_, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )
    emotion, confidence = ranked_emotions[0]

    return PredictionResponse(
        emotion=str(emotion),
        confidence=float(confidence),
        alternatives=[
            EmotionScore(emotion=str(label), confidence=float(score))
            for label, score in ranked_emotions[1:4]
        ],
    )
