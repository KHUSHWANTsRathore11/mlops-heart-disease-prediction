"""
<<<<<<< HEAD
FastAPI for Heart Disease Prediction Model.

This API provides endpoints for making heart disease predictions using
a trained machine learning model loaded from MLflow.
"""
=======
Flask API for Heart Disease Prediction Model.

This API provides endpoints for making heart disease predictions using
a trained machine learning model.
"""
import json
>>>>>>> origin/develop
import logging
import os
import sys
import time
from datetime import datetime
<<<<<<< HEAD
from typing import Dict

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
=======

import joblib
import pandas as pd
from flask import Flask, g, jsonify, request
from flask_cors import CORS
>>>>>>> origin/develop

# Add project root to path for model imports
sys.path.insert(0, os.path.abspath("."))

<<<<<<< HEAD
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
=======
# Configure structured logging


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        return json.dumps(log_data)


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler with JSON formatting
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Metrics tracking
>>>>>>> origin/develop
metrics = {
    "total_requests": 0,
    "total_predictions": 0,
    "predictions_disease": 0,
    "predictions_no_disease": 0,
<<<<<<< HEAD
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
=======
    "total_errors": 0,
    "total_duration_ms": 0,
    "start_time": datetime.utcnow().isoformat(),
}

# Load model and preprocessor at startup
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")

try:
    logger.info(f"Looking for model at: {os.path.abspath(MODEL_PATH)}")
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    logger.info(f"[SUCCESS] Model loaded successfully: {type(model).__name__}")
    logger.info("[SUCCESS] Preprocessor loaded successfully")
except Exception as e:
    logger.error(f"[ERROR] Error loading model: {e}")
    model = None
    preprocessor = None

# Expected feature names
EXPECTED_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


# Request/Response logging middleware
@app.before_request
def before_request():
    """Log incoming request and start timer"""
    g.start_time = time.time()
    metrics["total_requests"] += 1

    log_record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Incoming request",
        args=(),
        exc_info=None,
    )
    log_record.extra_data = {
        "event": "request_started",
        "method": request.method,
        "path": request.path,
        "remote_addr": request.remote_addr,
    }
    logger.handle(log_record)


@app.after_request
def after_request(response):
    """Log response and duration"""
    if hasattr(g, "start_time"):
        duration = (time.time() - g.start_time) * 1000  # Convert to ms
        metrics["total_duration_ms"] += duration

        log_record = logging.LogRecord(
            name=logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Request completed",
            args=(),
            exc_info=None,
        )
        log_record.extra_data = {
            "event": "request_completed",
            "method": request.method,
            "path": request.path,
            "status_code": response.status_code,
            "duration_ms": round(duration, 2),
        }
        logger.handle(log_record)

    return response


@app.route("/", methods=["GET"])
def home():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "message": "Heart Disease Prediction API",
            "version": "1.0.0",
            "model_loaded": model is not None,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """Detailed health status"""
    return jsonify(
        {
            "status": "healthy" if model is not None else "unhealthy",
            "model_loaded": model is not None,
            "preprocessor_loaded": preprocessor is not None,
            "model_type": type(model).__name__ if model else None,
        }
    )


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Prometheus-compatible metrics endpoint"""
    avg_duration = (
        (metrics["total_duration_ms"] / metrics["total_requests"])
        if metrics["total_requests"] > 0
        else 0
    )

>>>>>>> origin/develop
    uptime = (
        datetime.utcnow() - datetime.fromisoformat(metrics["start_time"].rstrip("Z"))
    ).total_seconds()

<<<<<<< HEAD
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
=======
    return jsonify(
        {
            "total_requests": metrics["total_requests"],
            "total_predictions": metrics["total_predictions"],
            "predictions_disease": metrics["predictions_disease"],
            "predictions_no_disease": metrics["predictions_no_disease"],
            "total_errors": metrics["total_errors"],
            "average_duration_ms": round(avg_duration, 2),
            "uptime_seconds": round(uptime, 2),
            "error_rate": round((metrics["total_errors"] / metrics["total_requests"] * 100), 2)
            if metrics["total_requests"] > 0
            else 0,
            "model_loaded": model is not None,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint.

    Expected JSON format:
    {
        "features": {
            "age": 63,
            "sex": 1,
            "cp": 3,
            ...
        }
    }
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Get JSON data
        data = request.get_json()

        if not data or "features" not in data:
            return (
                jsonify(
                    {
                        "error": "Invalid input format",
                        "expected": {"features": {feat: "value" for feat in EXPECTED_FEATURES}},
                    }
                ),
                400,
            )

        features = data["features"]

        # Validate all required features are present
        missing_features = [f for f in EXPECTED_FEATURES if f not in features]
        if missing_features:
            return jsonify({"error": "Missing required features", "missing": missing_features}), 400

        # Create DataFrame with features in correct order
        feature_data = {feat: [features[feat]] for feat in EXPECTED_FEATURES}
        input_df = pd.DataFrame(feature_data)

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        # Determine risk level
        risk_prob = float(probability[1])
>>>>>>> origin/develop
        if risk_prob >= 0.7:
            risk_level = "High"
        elif risk_prob >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

<<<<<<< HEAD
=======
        # Return prediction
        response = {
            "prediction": int(prediction),
            "probability": round(risk_prob, 4),
            "risk_level": risk_level,
            "confidence": {
                "no_disease": round(float(probability[0]), 4),
                "disease": round(float(probability[1]), 4),
            },
            "model_version": "1.0.0",
            "model_type": type(model).__name__,
        }

>>>>>>> origin/develop
        # Update metrics
        metrics["total_predictions"] += 1
        if prediction == 1:
            metrics["predictions_disease"] += 1
        else:
            metrics["predictions_no_disease"] += 1

<<<<<<< HEAD
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
=======
        # Log prediction with details
        log_record = logging.LogRecord(
            name=logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Prediction made",
            args=(),
            exc_info=None,
        )
        log_record.extra_data = {
            "event": "prediction",
            "prediction": int(prediction),
            "probability": round(risk_prob, 4),
            "risk_level": risk_level,
            "model_type": type(model).__name__,
        }
        logger.handle(log_record)

        return jsonify(response), 200

    except Exception as e:
        metrics["total_errors"] += 1

        log_record = logging.LogRecord(
            name=logger.name,
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Prediction error",
            args=(),
            exc_info=None,
        )
        log_record.extra_data = {
            "event": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "endpoint": "/predict",
        }
        logger.handle(log_record)

        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
>>>>>>> origin/develop
