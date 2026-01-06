"""
Pytest configuration and fixtures for heart disease prediction tests.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_heart_data():
    """
    Create sample heart disease data for testing.

    Returns:
        pd.DataFrame: Sample data with typical heart disease features
    """
    np.random.seed(42)
    n_samples = 100

    data = {
        'age': np.random.randint(25, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(1, 5, n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(120, 400, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(70, 200, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples),
        'slope': np.random.randint(1, 4, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.choice([3, 6, 7], n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_features_and_target(sample_heart_data):
    """
    Split sample data into features and target.

    Returns:
        tuple: (X, y) - Features and target
    """
    X = sample_heart_data.drop('target', axis=1)
    y = sample_heart_data['target']
    return X, y


@pytest.fixture
def data_paths():
    """
    Provide paths to test data files.

    Returns:
        dict: Dictionary of data file paths
    """
    return {
        'raw_data': Path('data/raw/heart_disease_combined.csv'),
        'clean_data': Path('data/processed/heart_disease_clean.csv'),
        'train_features': Path('data/processed/features_train.csv'),
        'test_features': Path('data/processed/features_test.csv'),
        'preprocessor': Path('models/preprocessor.pkl'),
        'model': Path('models/model.pkl')
    }


@pytest.fixture
def temp_model_path(tmp_path):
    """
    Create a temporary path for saving models during tests.

    Returns:
        Path: Temporary directory path
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir / "test_model.pkl"
