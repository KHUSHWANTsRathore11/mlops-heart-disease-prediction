import sys
import yaml
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import FeatureSet, FeatureSetSpecification
from azure.identity import DefaultAzureCredential

# Add project root to path to import src modules
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import load_config

def deploy_feature_set():
    print("Loading project configuration...")
    config = load_config()
    
    # Azure connection details
    subscription_id = config['azure']['subscription_id']
    resource_group = config['azure']['resource_group']
    workspace_name = config['azure']['workspace_name']
    
    # Feature Store details
    fs_name = config['feature_store']['name']
    fs_version = config['feature_store']['version']
    
    print(f"Connecting to Azure ML Workspace: {workspace_name}...")
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id,
        resource_group,
        workspace_name
    )
    
    # Load Feature Spec
    feature_spec_path = Path("feature_specs/heart_disease_features.yaml")
    print(f"Loading feature specification from {feature_spec_path}...")
    
    with open(feature_spec_path, "r") as f:
        feature_spec = yaml.safe_load(f)
        
    print(f"Registering Feature Set: {fs_name} (Version: {fs_version})...")
    
    # Define Feature Set
    # Note: In a real scenario, you typically point to the path containing the transformation code (code=...)
    # For now, we are registering the specification and lineage.
    
    heart_disease_features = FeatureSet(
        name=fs_name,
        version=fs_version,
        description="Heart Disease Prediction Features",
        entities=["age", "sex"], # Ideally defined as entities in Azure ML, keeping simple for now
        specification=FeatureSetSpecification(path="feature_specs/heart_disease_features.yaml"),
        tags={"project": "heart-disease", "created_by": "mlops-pipeline"}
    )
    
    # Create or Update
    poller = ml_client.feature_sets.begin_create_or_update(heart_disease_features)
    print("Operation started. Waiting for completion...")
    poller.result()
    
    print(f"Feature Set {fs_name}:{fs_version} successfully registered!")

if __name__ == "__main__":
    try:
        deploy_feature_set()
    except Exception as e:
        print(f"Error registering feature set: {e}")
        # Don't fail the pipeline yet if it's just auth/config missing in dev
        print("Note: Ensure you have 'azure-ai-ml' installed and valid Azure credentials.")
        sys.exit(1)
