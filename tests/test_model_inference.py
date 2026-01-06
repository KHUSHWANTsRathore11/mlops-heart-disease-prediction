"""Test model inference capability"""
import sys

<<<<<<< HEAD
=======
import joblib
>>>>>>> origin/develop
import pandas as pd


def test_model_inference():
<<<<<<< HEAD
    """Test that model can make predictions on sample data (using temp model)"""
    try:
        from sklearn.linear_model import LogisticRegression

        from src.data.preprocessing import HeartDiseasePreprocessor

        # Train a dummy model
        data = {
            "age": [25, 30, 45, 50, 60],
            "sex": [1, 0, 1, 0, 1],
            "cp": [1, 2, 3, 1, 2],
            "trestbps": [120, 130, 140, 150, 160],
            "chol": [200, 220, 240, 260, 280],
            "fbs": [0, 1, 0, 1, 0],
            "restecg": [0, 1, 0, 1, 0],
            "thalach": [150, 160, 170, 180, 190],
            "exang": [0, 1, 0, 1, 0],
            "oldpeak": [0.0, 1.0, 0.5, 2.0, 1.5],
            "slope": [1, 2, 1, 2, 1],
            "ca": [0, 1, 0, 1, 0],
            "thal": [3, 6, 3, 7, 3],
            "target": [0, 1, 0, 1, 0],
        }
        df = pd.DataFrame(data)

        preprocessor = HeartDiseasePreprocessor()
        preprocessor.fit(df, target_col="target")
        X = preprocessor.transform(df, target_col="target")
        y = df["target"]
        X = X.drop(columns=["target"])

        model = LogisticRegression()
        model.fit(X, y)
=======
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
>>>>>>> origin/develop

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
<<<<<<< HEAD
                    # "timestamp" not needed for basic preprocessor
                    # "patient_id": 1,
=======
                    "timestamp": trained_timestamp,
                    "patient_id": 1,
>>>>>>> origin/develop
                }
            ]
        )

        # Preprocess the sample
        sample_processed = preprocessor.transform(sample)

<<<<<<< HEAD
        # Make prediction
        pred = model.predict(sample_processed)
        prob = model.predict_proba(sample_processed)
=======
        # Drop columns not expected by the model
        model_features = model.feature_names_in_
        sample_for_model = sample_processed[model_features]

        # Make prediction
        pred = model.predict(sample_for_model)
        prob = model.predict_proba(sample_for_model)
>>>>>>> origin/develop

        print(f"Prediction: {pred[0]}")
        print(f"Probability: {prob[0]}")
        print("[SUCCESS] Model inference working")

<<<<<<< HEAD
        assert len(pred) == 1
=======
>>>>>>> origin/develop
        return 0
    except Exception as e:
        print(f"[ERROR] Model inference failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(test_model_inference())
