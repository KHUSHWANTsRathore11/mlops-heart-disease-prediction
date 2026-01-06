"""Test model inference capability"""
import joblib
import pandas as pd
import sys


def test_model_inference():
    """Test that model can make predictions on sample data"""
    try:
        # Load model and preprocessor
        model = joblib.load('models/model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        
        # Create sample input
        sample = pd.DataFrame([{
            'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145,
            'chol': 233, 'fbs': 1, 'restecg': 0, 'thalach': 150,
            'exang': 0, 'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
        }])
        
        # Preprocess and make prediction
        sample_processed = preprocessor.transform(sample)
        pred = model.predict(sample_processed)
        prob = model.predict_proba(sample_processed)
        
        print(f'Prediction: {pred[0]}')
        print(f'Probability: {prob[0]}')
        print('[SUCCESS] Model inference working')
        
        return 0
    except Exception as e:
        print(f'[ERROR] Model inference failed: {e}')
        return 1


if __name__ == '__main__':
    sys.exit(test_model_inference())
