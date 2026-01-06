"""
Model training script for Heart Disease Prediction.

This script trains multiple classification models with cross-validation,
tracks experiments with MLflow, and saves the best model.

Usage:
    python src/models/train_model.py
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.data
import mlflow.sklearn
import pandas as pd

# Import evaluation utilities
from evaluation import (
    calculate_metrics,
    cross_validate_model,
    generate_classification_report,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    print_metrics,
)

# Import MLflow configuration
from mlflow_config import (
    log_artifact,
    log_metrics,
    log_model,
    log_params,
    print_run_info,
    setup_mlflow,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_git_revision_hash():
    """Get the current git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("ascii")
    except Exception:
        return "unknown"


def log_dataset_to_mlflow(df, path, context="training"):
    """
    Log a dataset to MLflow with DVC versioning info.

    Args:
        df: Pandas DataFrame
        path: Path to the dataset file
        context: Context of usage ('training' or 'validation')
    """
    try:
        # Get git revision
        git_rev = get_git_revision_hash()

        # Create MLflow dataset
        dataset = mlflow.data.from_pandas(
            df, source=path, name=os.path.basename(path), targets="target"
        )

        # Log input to MLflow
        mlflow.log_input(dataset, context=context)

        # Log DVC tags
        mlflow.set_tag(f"dvc.{context}.path", path)
        mlflow.set_tag(f"dvc.{context}.git_commit", git_rev)

    except Exception as e:
        print(f"Warning: Failed to log dataset to MLflow: {e}")


def load_data(train_path, test_path):
    """
    Load training and test data.

    Args:
        train_path: Path to training features CSV
        test_path: Path to test features CSV

    Returns:
        tuple: (X_train, y_train, X_test, y_test, feature_names)
    """
    print(f"Loading data from {train_path} and {test_path}...")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Assume 'target' is the column name
    target_col = "target"

    # Drop non-feature columns
    drop_cols = ["patient_id", "timestamp"]
    drop_cols = [c for c in drop_cols if c in train_df.columns]

    # Separate features and target
    feature_cols = [c for c in train_df.columns if c != target_col and c not in drop_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {', '.join(feature_cols)}")

    return X_train, y_train, X_test, y_test, feature_cols


def get_model_config(model_type):
    """
    Get model configuration.

    Args:
        model_type: Type of model ('logistic_regression' or 'random_forest')

    Returns:
        tuple: (model, params_dict)
    """
    if model_type == "logistic_regression":
        params = {"C": 1.0, "solver": "liblinear", "max_iter": 1000, "random_state": 42}
        model = LogisticRegression(**params)

    elif model_type == "random_forest":
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }
        model = RandomForestClassifier(**params)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, params


def train_single_model(model_name, X_train, y_train, X_test, y_test, feature_names, output_dir):
    """
    Train a single model with cross-validation and evaluation.

    Args:
        model_name: Name of the model ('logistic_regression' or 'random_forest')
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names
        output_dir: Directory to save artifacts

    Returns:
        dict: Results including model, metrics, and artifacts
    """
    print("\n" + "=" * 80)
    print(f"Training {model_name.replace('_', ' ').title()}")
    print("=" * 80)

    # Get model configuration
    model, params = get_model_config(model_name)

    # Add metadata to params
    params_to_log = {
        **params,
        "model_type": model_name,
        "n_train_samples": X_train.shape[0],
        "n_test_samples": X_test.shape[0],
        "n_features": X_train.shape[1],
    }

    # Perform cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_results = cross_validate_model(model, X_train, y_train, cv=5)

    # Train on full training set
    print("\nTraining on full training set...")
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

    print_metrics(train_metrics, "Training Metrics")
    print_metrics(test_metrics, "Test Metrics")

    # Create output directory
    model_output_dir = Path(output_dir) / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    artifacts = {}

    # Confusion matrix
    cm_path = model_output_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        y_test,
        y_test_pred,
        save_path=cm_path,
        title=f'{model_name.replace("_", " ").title()} - Confusion Matrix',
    )
    artifacts["confusion_matrix"] = str(cm_path)

    # ROC curve
    roc_path = model_output_dir / "roc_curve.png"
    plot_roc_curve(
        y_test,
        y_test_proba,
        save_path=roc_path,
        title=f'{model_name.replace("_", " ").title()} - ROC Curve',
    )
    artifacts["roc_curve"] = str(roc_path)

    # Feature importance (for tree-based models)
    if hasattr(model, "feature_importances_"):
        fi_path = model_output_dir / "feature_importance.png"
        plot_feature_importance(
            model,
            feature_names,
            save_path=fi_path,
            title=f'{model_name.replace("_", " ").title()} - Feature Importance',
        )
        artifacts["feature_importance"] = str(fi_path)

    # Classification report
    report_path = model_output_dir / "classification_report.json"
    generate_classification_report(y_test, y_test_pred, save_path=report_path)
    artifacts["classification_report"] = str(report_path)

    # Combine all metrics for MLflow logging
    all_metrics = {
        # CV metrics
        **{k: v for k, v in cv_results.items() if "cv_mean" in k or "cv_std" in k},
        # Test metrics
        **{f"test_{k}": v for k, v in test_metrics.items()},
        # Train metrics
        **{f"train_{k}": v for k, v in train_metrics.items()},
    }

    return {
        "model": model,
        "params": params_to_log,
        "metrics": all_metrics,
        "artifacts": artifacts,
        "test_auc": test_metrics["roc_auc"],
    }


def train_all_models(
    X_train, y_train, X_test, y_test, feature_names, output_dir="models/experiments"
):
    """
    Train all models and track with MLflow.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names
        output_dir: Directory to save artifacts

    Returns:
        dict: Results for all models
    """
    # Setup MLflow
    setup_mlflow()

    # Define models to train
    models_to_train = ["logistic_regression", "random_forest"]

    results = {}
    best_model = None
    best_auc = 0.0

    # Train each model
    for model_name in models_to_train:
        with mlflow.start_run(run_name=model_name.replace("_", " ").title()):
            print_run_info()

            # Log datasets
            log_dataset_to_mlflow(
                pd.concat([X_train, y_train], axis=1),
                "data/processed/features_train.csv",
                "training",
            )
            log_dataset_to_mlflow(
                pd.concat([X_test, y_test], axis=1),
                "data/processed/features_test.csv",
                "validation",
            )

            # Train model
            result = train_single_model(
                model_name, X_train, y_train, X_test, y_test, feature_names, output_dir
            )

            # Log to MLflow
            log_params(result["params"])
            log_metrics(result["metrics"])

            # Log artifacts
            for artifact_name, artifact_path in result["artifacts"].items():
                if os.path.exists(artifact_path):
                    log_artifact(artifact_path)

            # Log model
            log_model(result["model"], artifact_path="model")

            # Store result
            results[model_name] = result

            # Track best model
            if result["test_auc"] > best_auc:
                best_auc = result["test_auc"]
                best_model = model_name

            print(f"\n[SUCCESS] {model_name.replace('_', ' ').title()} training complete!")
            print(f"   Test ROC-AUC: {result['test_auc']:.4f}")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best Model: {best_model.replace('_', ' ').title()}")
    print(f"Best ROC-AUC: {best_auc:.4f}")
    print("=" * 80)

    return results, best_model


def save_best_model(results, best_model_name, model_path, metrics_path):
    """
    Save the best model and its metrics.

    Args:
        results: Results from all models
        best_model_name: Name of the best model
        model_path: Path to save the model
        metrics_path: Path to save metrics
    """
    best_result = results[best_model_name]

    # Save model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_result["model"], model_path)
    print(f"\n[SUCCESS] Best model saved to {model_path}")

    # Save metrics
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    metrics_to_save = {
        "model_type": best_model_name,
        "test_accuracy": best_result["metrics"]["test_accuracy"],
        "test_precision": best_result["metrics"]["test_precision"],
        "test_recall": best_result["metrics"]["test_recall"],
        "test_f1": best_result["metrics"]["test_f1"],
        "test_roc_auc": best_result["metrics"]["test_roc_auc"],
        "cv_accuracy_mean": best_result["metrics"]["accuracy_cv_mean"],
        "cv_accuracy_std": best_result["metrics"]["accuracy_cv_std"],
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_to_save, f, indent=4)
    print(f"[SUCCESS] Metrics saved to {metrics_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Heart Disease Prediction Models")
    parser.add_argument(
        "--train-path",
        default="data/processed/features_train.csv",
        help="Path to training features",
    )
    parser.add_argument(
        "--test-path", default="data/processed/features_test.csv", help="Path to test features"
    )
    parser.add_argument("--output-path", default="models/model.pkl", help="Path to save best model")
    parser.add_argument(
        "--metrics-path", default="metrics/training_metrics.json", help="Path to save metrics"
    )
    parser.add_argument(
        "--experiment-dir", default="models/experiments", help="Directory for experiment artifacts"
    )

    args = parser.parse_args()

    # Load data
    X_train, y_train, X_test, y_test, feature_names = load_data(args.train_path, args.test_path)

    # Train all models
    results, best_model = train_all_models(
        X_train, y_train, X_test, y_test, feature_names, args.experiment_dir
    )

    # Save best model
    save_best_model(results, best_model, args.output_path, args.metrics_path)

    print("\n" + "=" * 80)
    print("[SUCCESS] Model training pipeline completed successfully!")
    print("=" * 80)
    print("\nTo view experiments, run: mlflow ui")
    print("Then open http://localhost:5000 in your browser")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
