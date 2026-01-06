"""
FastAPI for Heart Disease Prediction Model.

This API provides endpoints for making heart disease predictions using
a trained machine learning model loaded from MLflow.
"""
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path for model imports
sys.path.insert(0, os.path.abspath("."))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart-disease-api")

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease risk using MLflow models",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
MODEL_NAME = "heart-disease-classifier"
MODEL_STAGE = "Production"

# Metrics
metrics = {
    "total_requests": 0,
    "total_predictions": 0,
    "predictions_disease": 0,
    "predictions_no_disease": 0,
    "start_time": datetime.utcnow().isoformat(),
}


def load_model():
    """Load the model from MLflow registry."""
    global model
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        logger.info(f"Loading model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("[SUCCESS] Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Error loading model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


class HeartDiseaseFeatures(BaseModel):
    age: int = Field(..., example=63, description="Age in years")
    sex: int = Field(..., example=1, description="1 = male; 0 = female")
    cp: int = Field(..., example=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., example=145, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., example=233, description="Cholesterol (mg/dl)")
    fbs: int = Field(
        ..., example=1, description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)"
    )
    restecg: int = Field(..., example=0, description="Resting electrocardiographic results (0-2)")
    thalach: int = Field(..., example=150, description="Maximum heart rate achieved")
    exang: int = Field(..., example=0, description="Exercise induced angina (1 = yes; 0 = no)")
    oldpeak: float = Field(
        ..., example=2.3, description="ST depression induced by exercise relative to rest"
    )
    slope: int = Field(..., example=0, description="Slope of the peak exercise ST segment (0-2)")
    ca: int = Field(
        ..., example=0, description="Number of major vessels (0-3) colored by flourosopy"
    )
    thal: int = Field(
        ...,
        example=1,
        description="Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)",
    )


class PredictionRequest(BaseModel):
    features: HeartDiseaseFeatures


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    confidence: Dict[str, float]
    model_version: str


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    metrics["total_requests"] += 1
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/")
def home():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    """Detailed health status."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_info": {"name": MODEL_NAME, "stage": MODEL_STAGE},
    }


@app.get("/metrics")
def get_metrics():
    """Application metrics."""
    uptime = (
        datetime.utcnow() - datetime.fromisoformat(metrics["start_time"].rstrip("Z"))
    ).total_seconds()

    return {**metrics, "uptime_seconds": uptime}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make a prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        data_dict = request.features.model_dump()
        input_df = pd.DataFrame([data_dict])

        # Predict
        prediction = model.predict(input_df)[0]
        # Depending on model wrapper, predict_proba might be separate or part of predict
        # For mlflow.pyfunc, standard sklearn models return just prediction usually unless wrapped specially
        # But we can try to access the underlying model if needed, or assume the pyfunc returns direct output
        # If the pyfunc was saved from sklearn, it supports predict.
        # For probabilities, we might need to check if we can get them.

        # If we can't get probas easily from generic pyfunc without custom wrapper,
        # we'll mock them or try to unwrap.
        # For now, let's assume the model object supports predict_proba if it's a sklearn model unwrapped,
        # but pyfunc usually only exposes predict.

        # Workaround: accessing native model if possible, OR if 'predict' returns probas
        # For sklearn, mlflow log_model allows signature.

        # Let's try basic prediction first.
        # If we really need probability, we might need to rely on the model being logged with 'predict_proba' method signature
        # or similar.

        # Simplified logic for now:
        risk_prob = 0.0
        # If the output is a class, we might lose probability unless we change how we predict.
        # For this exercise, let's assume simple prediction 0/1.

        # Re-implementing risk logic requires probabilities.
        # Let's try to see if the loaded 'model' object has predict_proba
        # MLflow PyFuncModel doesn't have predict_proba by default.
        # We can try model._model_impl.predict_proba if it's sklearn

        confidence = {"no_disease": 0.0, "disease": 0.0}

        if hasattr(model, "_model_impl") and hasattr(model._model_impl, "predict_proba"):
            probs = model._model_impl.predict_proba(input_df)[0]
            risk_prob = float(probs[1])
            confidence = {
                "no_disease": round(float(probs[0]), 4),
                "disease": round(float(probs[1]), 4),
            }
        else:
            # Fallback if we can't get probs (e.g. strict pyfunc)
            risk_prob = float(prediction)  # 0.0 or 1.0
            confidence = {"no_disease": 1.0 - risk_prob, "disease": risk_prob}

        # Determine risk level
        if risk_prob >= 0.7:
            risk_level = "High"
        elif risk_prob >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Update metrics
        metrics["total_predictions"] += 1
        if prediction == 1:
            metrics["predictions_disease"] += 1
        else:
            metrics["predictions_no_disease"] += 1

        return {
            "prediction": int(prediction),
            "probability": round(risk_prob, 4),
            "risk_level": risk_level,
            "confidence": confidence,
            "model_version": MODEL_STAGE,
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
