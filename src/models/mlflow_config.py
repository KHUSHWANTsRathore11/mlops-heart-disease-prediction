"""
MLflow configuration for experiment tracking.

This module centralizes MLflow configuration and provides helper functions
for experiment tracking in the Heart Disease Prediction project.
"""
import os
import mlflow


# MLflow Configuration
MLFLOW_TRACKING_URI = f"file://{os.path.abspath('mlruns')}"
EXPERIMENT_NAME = "heart-disease-model-training"


def setup_mlflow():
    """
    Initialize MLflow with project configuration.

    Returns:
        experiment_id: The MLflow experiment ID
    """
    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    # Set or create experiment
    try:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"Created new experiment: {EXPERIMENT_NAME}")
    except Exception:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {EXPERIMENT_NAME}")

    mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment_id


def log_params(params):
    """
    Log parameters to MLflow.

    Args:
        params: Dictionary of parameters
    """
    mlflow.log_params(params)


def log_metrics(metrics):
    """
    Log metrics to MLflow.

    Args:
        metrics: Dictionary of metrics
    """
    mlflow.log_metrics(metrics)


def log_artifact(filepath):
    """
    Log an artifact to MLflow.

    Args:
        filepath: Path to the artifact file
    """
    if os.path.exists(filepath):
        mlflow.log_artifact(filepath)
        print(f"Logged artifact: {filepath}")
    else:
        print(f"Warning: Artifact not found: {filepath}")


def log_model(model, artifact_path="model", **kwargs):
    """
    Log a scikit-learn model to MLflow.

    Args:
        model: Trained sklearn model
        artifact_path: Path within the run's artifact directory
        **kwargs: Additional arguments for mlflow.sklearn.log_model
    """
    mlflow.sklearn.log_model(model, artifact_path, **kwargs)
    print(f"Model logged to MLflow at {artifact_path}")


def get_run_info():
    """
    Get information about the current MLflow run.

    Returns:
        dict: Run information including run_id, experiment_id, etc.
    """
    run = mlflow.active_run()
    if run:
        return {
            'run_id': run.info.run_id,
            'experiment_id': run.info.experiment_id,
            'run_name': run.info.run_name,
            'status': run.info.status
        }
    return None


def print_run_info():
    """
    Print information about the current MLflow run.
    """
    info = get_run_info()
    if info:
        print("\n" + "=" * 60)
        print("MLflow Run Information")
        print("=" * 60)
        for key, value in info.items():
            print(f"{key:15s}: {value}")
        print("=" * 60 + "\n")
    else:
        print("No active MLflow run")
