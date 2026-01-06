"""
Tests for feature engineering pipeline
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path BEFORE importing src modules
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.data.preprocessing import (
    HeartDiseasePreprocessor,
    clean_data,
    prepare_train_test_split,
)


class TestHeartDiseasePreprocessor:
    """Test cases for HeartDiseasePreprocessor class"""

    def test_preprocessor_fit_transform(self):
        """Test that preprocessor can fit and transform data"""
        # Create sample data
        data = {
            "age": [25, 30, 45, 50, 60],
            "sex": [1, 0, 1, 0, 1],
            "cp": [1, 2, 3, 1, 2],
            "trestbps": [120, 130, 140, 150, 160],
            "chol": [200, 220, 240, 260, 280],
            "target": [0, 1, 0, 1, 0],
        }
        df = pd.DataFrame(data)

        # Initialize and fit preprocessor
        preprocessor = HeartDiseasePreprocessor()
        transformed = preprocessor.fit_transform(df, target_col="target")

        # Check that data is transformed
        assert transformed is not None
        assert len(transformed) == len(df)
        assert "target" in transformed.columns

        # Check that numerical features are scaled (mean ~= 0, std ~= 1)
        # Note: For small samples, std won't be exactly 1.0, so we use a reasonable tolerance
        numerical_cols = ["age", "trestbps", "chol"]
        for col in numerical_cols:
            mean_val = transformed[col].mean()
            std_val = transformed[col].std()
            assert abs(mean_val) < 1e-10  # Very close to 0
            assert 0.5 < std_val < 1.5  # Reasonably close to 1 for small samples

    def test_preprocessor_save_load(self, tmp_path):
        """Test that preprocessor can be saved and loaded"""
        # Create sample data
        data = {"age": [25, 30, 45], "sex": [1, 0, 1], "target": [0, 1, 0]}
        df = pd.DataFrame(data)

        # Fit preprocessor
        preprocessor = HeartDiseasePreprocessor()
        preprocessor.fit(df, target_col="target")

        # Save preprocessor
        save_path = tmp_path / "test_preprocessor.pkl"
        preprocessor.save(save_path)

        # Load preprocessor
        loaded_preprocessor = HeartDiseasePreprocessor.load(save_path)

        # Check that feature names are preserved
        assert loaded_preprocessor.feature_names == preprocessor.feature_names
        assert loaded_preprocessor.numerical_features == preprocessor.numerical_features

    def test_preprocessor_consistency(self):
        """Test that preprocessor transforms train and test consistently"""
        # Create train and test data
        train_data = {
            "age": [25, 30, 45, 50, 60],
            "sex": [1, 0, 1, 0, 1],
            "target": [0, 1, 0, 1, 0],
        }
        test_data = {"age": [35, 55], "sex": [1, 0], "target": [1, 0]}
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)

        # Fit on train, transform both
        preprocessor = HeartDiseasePreprocessor()
        preprocessor.fit(train_df, target_col="target")

        train_transformed = preprocessor.transform(train_df, target_col="target")
        test_transformed = preprocessor.transform(test_df, target_col="target")

        # Both should have the same columns
        assert set(train_transformed.columns) == set(test_transformed.columns)


class TestCleanData:
    """Test cases for clean_data function"""

    def test_clean_data_no_missing_values(self):
        """Test that cleaned data has no missing values"""
        # Load actual data
        data_path = Path("data/raw/heart_disease_combined.csv")
        if data_path.exists():
            df = pd.read_csv(data_path)
            df_clean = clean_data(df)

            # Check no missing values
            assert df_clean.isnull().sum().sum() == 0

    def test_clean_data_target_binary(self):
        """Test that target is converted to binary"""
        # Create sample data with multi-class target
        data = {"age": [25, 30, 45], "sex": [1, 0, 1], "target": [0, 1, 2]}  # Multi-class
        df = pd.DataFrame(data)
        df_clean = clean_data(df)

        # Target should be binary (0 or 1)
        assert set(df_clean["target"].unique()).issubset({0, 1})


class TestTrainTestSplit:
    """Test cases for train_test_split function"""

    def test_split_ratio(self):
        """Test that split produces correct ratio"""
        # Create sample data
        data = {
            "age": list(range(100)),
            "sex": [i % 2 for i in range(100)],
            "target": [i % 2 for i in range(100)],
        }
        df = pd.DataFrame(data)

        X_train, X_test, y_train, y_test = prepare_train_test_split(df, test_size=0.2)

        # Check sizes
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_split_stratification(self):
        """Test that split preserves class distribution"""
        # Create imbalanced data
        data = {
            "age": list(range(100)),
            "sex": [i % 2 for i in range(100)],
            "target": [1] * 70 + [0] * 30,  # 70% class 1, 30% class 0
        }
        df = pd.DataFrame(data)

        X_train, X_test, y_train, y_test = prepare_train_test_split(df, test_size=0.2)

        # Check class distribution is preserved
        train_ratio = y_train.sum() / len(y_train)
        test_ratio = y_test.sum() / len(y_test)

        # Ratios should be similar (within 10%)
        assert abs(train_ratio - test_ratio) < 0.1


class TestFeatureEngineeringIntegration:
    """Integration tests for the complete feature engineering pipeline"""

    def test_full_pipeline_exists(self):
        """Test that all pipeline outputs exist"""
        train_path = Path("data/processed/features_train.csv")
        test_path = Path("data/processed/features_test.csv")
        preprocessor_path = Path("models/preprocessor.pkl")

        assert train_path.exists(), "Training features not found"
        assert test_path.exists(), "Test features not found"
        assert preprocessor_path.exists(), "Preprocessor not found"

    def test_features_have_correct_shape(self):
        """Test that generated features have expected columns"""
        train_df = pd.read_csv("data/processed/features_train.csv")
        test_df = pd.read_csv("data/processed/features_test.csv")

        # Both should have the same columns
        assert set(train_df.columns) == set(test_df.columns)

        # Should have target column
        assert "target" in train_df.columns
        assert "target" in test_df.columns

    def test_features_are_scaled(self):
        """Test that numerical features are properly scaled"""
        train_df = pd.read_csv("data/processed/features_train.csv")

        # Check that numerical features have reasonable scaled values
        numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

        for col in numerical_features:
            if col in train_df.columns:
                mean_val = train_df[col].mean()
                # Scaled values should have mean close to 0
                assert abs(mean_val) < 0.1, f"{col} mean is {mean_val}, expected close to 0"

    def test_no_data_leakage(self):
        """Test that no samples appear in both train and test"""
        train_df = pd.read_csv("data/processed/features_train.csv")
        test_df = pd.read_csv("data/processed/features_test.csv")

        # If patient_id exists, check for duplicates
        if "patient_id" in train_df.columns and "patient_id" in test_df.columns:
            train_ids = set(train_df["patient_id"])
            test_ids = set(test_df["patient_id"])

            # No overlap allowed
            assert len(train_ids.intersection(test_ids)) == 0, "Data leakage detected!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
