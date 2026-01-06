"""
Flask API for Heart Disease Prediction Model.

This API provides endpoints for making heart disease predictions using
a trained machine learning model.
"""
import json
import logging
import os
import sys
import time
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, g, jsonify, request
from flask_cors import CORS

# Add project root to path for model imports
sys.path.insert(0, os.path.abspath("."))

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
metrics = {
    "total_requests": 0,
    "total_predictions": 0,
    "predictions_disease": 0,
    "predictions_no_disease": 0,
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

    uptime = (
        datetime.utcnow() - datetime.fromisoformat(metrics["start_time"].rstrip("Z"))
    ).total_seconds()

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
        if risk_prob >= 0.7:
            risk_level = "High"
        elif risk_prob >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

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

        # Update metrics
        metrics["total_predictions"] += 1
        if prediction == 1:
            metrics["predictions_disease"] += 1
        else:
            metrics["predictions_no_disease"] += 1

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
