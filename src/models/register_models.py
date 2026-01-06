"""
Register trained models to MLflow Model Registry.

This script registers the best models from MLflow runs to the Model Registry
for easier model management and deployment.
"""
import mlflow
from mlflow.tracking import MlflowClient
import os


def setup_mlflow():
    """Setup MLflow connection."""
    mlflow_uri = f"file://{os.path.abspath('mlruns')}"
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"MLflow tracking URI: {mlflow_uri}")
    return MlflowClient()


def get_best_runs(client, experiment_name="heart-disease-model-training"):
    """Get the best run for each model type."""
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found!")
        return {}
    
    experiment_id = experiment.experiment_id
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.test_roc_auc DESC"]
    )
    
    # Group by model type and get best for each
    best_runs = {}
    for run in runs:
        model_type = run.data.params.get('model_type', 'unknown')
        if model_type not in best_runs:
            best_runs[model_type] = run
    
    return best_runs


def register_model(client, run_id, model_name, model_type):
    """Register a model from a run to the Model Registry."""
    try:
        # Get the model URI from the run
        model_uri = f"runs:/{run_id}/model"
        
        # Register the model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        # Add description and tags
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=f"Heart Disease Prediction model using {model_type}"
        )
        
        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="model_type",
            value=model_type
        )
        
        print(f"[SUCCESS] Registered {model_name} version {model_version.version}")
        return model_version
        
    except Exception as e:
        print(f"[ERROR] Error registering {model_name}: {e}")
        return None


def main():
    """Main function to register models."""
    print("="*70)
    print("MLflow Model Registration")
    print("="*70)
    
    # Setup MLflow
    client = setup_mlflow()
    
    # Get best runs
    print("\nFinding best models...")
    best_runs = get_best_runs(client)
    
    if not best_runs:
        print("No runs found!")
        return
    
    print(f"\nFound {len(best_runs)} model(s) to register:")
    for model_type, run in best_runs.items():
        test_auc = run.data.metrics.get('test_roc_auc', 0)
        print(f"  - {model_type}: ROC-AUC = {test_auc:.4f} (Run: {run.info.run_id[:8]}...)")
    
    # Register each model
    print("\nRegistering models...")
    registered = []
    
    for model_type, run in best_runs.items():
        # Create model name
        model_name = f"heart-disease-{model_type.replace('_', '-')}"
        
        # Register model
        version = register_model(
            client,
            run.info.run_id,
            model_name,
            model_type
        )
        
        if version:
            registered.append({
                'name': model_name,
                'version': version.version,
                'type': model_type,
                'run_id': run.info.run_id
            })
    
    # Summary
    print("\n" + "="*70)
    print("Registration Summary")
    print("="*70)
    if registered:
        for model in registered:
            print(f"[SUCCESS] {model['name']} v{model['version']} ({model['type']})")
        
        print(f"\n[SUCCESS] Successfully registered {len(registered)} model(s)!")
        print("\nView registered models in MLflow UI:")
        print("  1. Go to http://localhost:5000")
        print("  2. Click on 'Models' tab in the top menu")
        print("  3. You'll see all registered models")
    else:
        print("[ERROR] No models were registered")
    
    print("="*70)


if __name__ == "__main__":
    main()
