"""
Tests for model training functionality.
"""
import joblib
<<<<<<< HEAD
=======
import pandas as pd
>>>>>>> origin/develop
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


class TestModelInitialization:
    """Test cases for model initialization"""

    def test_logistic_regression_init(self):
        """Test Logistic Regression initialization"""
        model = LogisticRegression(random_state=42)
        assert model is not None
        assert model.random_state == 42

    def test_random_forest_init(self):
        """Test Random Forest initialization"""
        model = RandomForestClassifier(random_state=42)
        assert model is not None
        assert model.random_state == 42


class TestModelTraining:
    """Test cases for model training"""

    def test_model_can_fit(self, sample_features_and_target):
        """Test that model can be trained"""
        X, y = sample_features_and_target
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        assert hasattr(model, "coef_")

    def test_model_training_with_different_models(self, sample_features_and_target):
        """Test training with different model types"""
        X, y = sample_features_and_target

        models = [
            LogisticRegression(random_state=42),
            RandomForestClassifier(n_estimators=10, random_state=42),
        ]

        for model in models:
            model.fit(X, y)
            assert hasattr(model, "predict")


class TestModelPredictions:
    """Test cases for model predictions"""

    def test_model_can_predict(self, sample_features_and_target):
        """Test that trained model can make predictions"""
        X, y = sample_features_and_target
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert all(p in [0, 1] for p in predictions)

    def test_model_can_predict_proba(self, sample_features_and_target):
        """Test that model can predict probabilities"""
        X, y = sample_features_and_target
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        probas = model.predict_proba(X)
        assert probas.shape == (len(X), 2)
        assert all(0 <= p <= 1 for row in probas for p in row)
        # Check that probabilities sum to 1
        assert all(abs(sum(row) - 1.0) < 0.01 for row in probas)


class TestModelMetrics:
    """Test cases for metrics calculation"""

    def test_accuracy_calculation(self, sample_features_and_target):
        """Test accuracy metric calculation"""
        from sklearn.metrics import accuracy_score

        X, y = sample_features_and_target
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        accuracy = accuracy_score(y, predictions)
        assert 0 <= accuracy <= 1

    def test_multiple_metrics(self, sample_features_and_target):
        """Test calculation of multiple metrics"""
        from sklearn.metrics import f1_score, precision_score, recall_score

        X, y = sample_features_and_target
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        precision = precision_score(y, predictions, zero_division=0)
        recall = recall_score(y, predictions, zero_division=0)
        f1 = f1_score(y, predictions, zero_division=0)

        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1


class TestModelPersistence:
    """Test cases for model saving and loading"""

    def test_model_can_be_saved(self, sample_features_and_target, temp_model_path):
        """Test that model can be saved to disk"""
        X, y = sample_features_and_target
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        joblib.dump(model, temp_model_path)
        assert temp_model_path.exists()

    def test_model_can_be_loaded(self, sample_features_and_target, temp_model_path):
        """Test that saved model can be loaded"""
        X, y = sample_features_and_target
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Save model
        joblib.dump(model, temp_model_path)

        # Load model
        loaded_model = joblib.load(temp_model_path)
        assert loaded_model is not None
        assert hasattr(loaded_model, "predict")

    def test_loaded_model_makes_same_predictions(self, sample_features_and_target, temp_model_path):
        """Test that loaded model makes same predictions as original"""
        X, y = sample_features_and_target
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        original_predictions = model.predict(X)

        # Save and load
        joblib.dump(model, temp_model_path)
        loaded_model = joblib.load(temp_model_path)
        loaded_predictions = loaded_model.predict(X)

        assert all(original_predictions == loaded_predictions)


class TestCrossValidation:
    """Test cases for cross-validation"""

    def test_cross_validation_runs(self, sample_features_and_target):
        """Test that cross-validation executes successfully"""
        X, y = sample_features_and_target
        model = LogisticRegression(random_state=42)

        scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
        assert len(scores) == 3
        assert all(0 <= score <= 1 for score in scores)

    def test_cross_validation_with_different_metrics(self, sample_features_and_target):
        """Test cross-validation with different scoring metrics"""
        X, y = sample_features_and_target
        model = LogisticRegression(random_state=42)

        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        for metric in metrics:
            scores = cross_val_score(model, X, y, cv=3, scoring=metric)
            assert len(scores) == 3
            assert all(0 <= score <= 1 for score in scores)
<<<<<<< HEAD
=======


class TestSavedModel:
    """Test cases for the actual saved project model"""

    def test_project_model_exists(self, data_paths):
        """Test that the project's trained model exists"""
        assert data_paths["model"].exists(), f"Model not found at {data_paths['model']}"

    def test_project_model_can_load(self, data_paths):
        """Test that the project model can be loaded"""
        if data_paths["model"].exists():
            model = joblib.load(data_paths["model"])
            assert model is not None
            assert hasattr(model, "predict")

    def test_project_model_can_predict(self, data_paths):
        """Test that project model can make predictions on test data"""
        if data_paths["model"].exists() and data_paths["test_features"].exists():
            model = joblib.load(data_paths["model"])
            test_df = pd.read_csv(data_paths["test_features"])

            # Remove target and metadata if present
            X_test = test_df.drop(columns=["target", "timestamp", "patient_id"], errors="ignore")

            predictions = model.predict(X_test)
            assert len(predictions) == len(X_test)
            assert all(p in [0, 1] for p in predictions)
>>>>>>> origin/develop
