"""Test model inference capability"""
import sys

import joblib
import pandas as pd


def test_model_inference():
    """Test that model can make predictions on sample data"""
    try:
        # Load model and preprocessor
        model = joblib.load("models/model.pkl")
        preprocessor = joblib.load("models/preprocessor.pkl")

        # Get the timestamp that the preprocessor expects
        if "timestamp" in preprocessor.label_encoders:
            trained_timestamp = preprocessor.label_encoders["timestamp"].classes_[0]
        else:
            trained_timestamp = "2026-01-06 01:39:15.441070"

        # Create sample input with all features the preprocessor expects
        sample = pd.DataFrame(
            [
                {
                    "age": 63,
                    "sex": 1,
                    "cp": 3,
                    "trestbps": 145,
                    "chol": 233,
                    "fbs": 1,
                    "restecg": 0,
                    "thalach": 150,
                    "exang": 0,
                    "oldpeak": 2.3,
                    "slope": 0,
                    "ca": 0,
                    "thal": 1,
                    "timestamp": trained_timestamp,
                    "patient_id": 1,
                }
            ]
        )

        # Preprocess the sample
        sample_processed = preprocessor.transform(sample)

        # Drop columns not expected by the model
        model_features = model.feature_names_in_
        sample_for_model = sample_processed[model_features]

        # Make prediction
        pred = model.predict(sample_for_model)
        prob = model.predict_proba(sample_for_model)

        print(f"Prediction: {pred[0]}")
        print(f"Probability: {prob[0]}")
        print("[SUCCESS] Model inference working")

        return 0
    except Exception as e:
        print(f"[ERROR] Model inference failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(test_model_inference())
