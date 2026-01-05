import sys
import yaml
import tempfile
import shutil
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import FeatureSet, FeatureSetSpecification
from azure.identity import DefaultAzureCredential

# Add project root to path to import src modules
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import load_config

def get_dvc_hash(file_path):
    """Parses dvc.lock to find the md5 hash of a specific file."""
    with open("dvc.lock", "r") as f:
        lock_data = yaml.safe_load(f)
    
    for stage in lock_data.get('stages', {}).values():
        for out in stage.get('outs', []):
            if out.get('path') == file_path:
                return out.get('md5')
    return None

def get_cloud_path(config, file_path):
    """Constructs the Azure Blob URL for a DVC-tracked file."""
    md5_hash = get_dvc_hash(file_path)
    if not md5_hash:
        raise ValueError(f"Could not find hash for {file_path} in dvc.lock")
    
    account_name = config['storage']['account_name']
    container_name = config['storage']['container_name']
    
    # DVC default structure: files/md5/ab/cdef...
    blob_path = f"files/md5/{md5_hash[:2]}/{md5_hash[2:]}"
    
    # Construct WASBS URL (optimized for Azure ML)
    # wasbs://<container>@<account>.blob.core.windows.net/<path>
    url = f"wasbs://{container_name}@{account_name}.blob.core.windows.net/{blob_path}"
    return url

def deploy_feature_set():
    print("Loading project configuration...")
    config = load_config()
    
    # Azure connection details
    subscription_id = config['azure']['subscription_id']
    resource_group = config['azure']['resource_group']
    
    # Use Feature Store resource if defined, otherwise default workspace
    workspace_name = config['feature_store'].get('resource_name', config['azure']['workspace_name'])
    
    # Feature Store details
    fs_name = config['feature_store']['name']
    fs_version = config['feature_store']['version']
    
    print(f"Connecting to Feature Store/Workspace: {workspace_name}...")
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id,
        resource_group,
        workspace_name
    )
    
    # Load Feature Spec and Update Source dynamically
    feature_spec_path = Path("feature_specs/FeatureSetSpec.yaml")
    print(f"Loading feature specification from {feature_spec_path}...")
    
    with open(feature_spec_path, "r") as f:
        feature_spec = yaml.safe_load(f)
    
    # Dynamic Source Update
    print("Resolving Cloud URI for features_train.csv from DVC...")
    cloud_url = get_cloud_path(config, "data/processed/features_train.csv")
    print(f"Resolved Cloud URL: {cloud_url}")
    
    feature_spec['source']['type'] = 'uri_file'
    feature_spec['source']['path'] = cloud_url
    
    
    # Use temporary directory for the spec folder
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_spec_path = Path(temp_dir) / "FeatureSetSpec.yaml"
        
        # Save the modified spec to the temp dir with the correct name
        with open(temp_spec_path, "w") as f:
            yaml.dump(feature_spec, f)
            
        print(f"Registering Feature Set: {fs_name} (Version: {fs_version})...")
        
        # Define Feature Set
        # The specification requires the PATH to the FOLDER containing FeatureSetSpec.yaml
        heart_disease_features = FeatureSet(
            name=fs_name,
            version=fs_version,
            description="Heart Disease Prediction Features",
            entities=["age", "sex"], # Ideally defined as entities in Azure ML, keeping simple for now
            specification=FeatureSetSpecification(path=temp_dir),
            tags={"project": "heart-disease", "created_by": "mlops-pipeline", "dvc_source": cloud_url}
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
