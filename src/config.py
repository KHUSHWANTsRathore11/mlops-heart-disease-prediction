import yaml
import os
from pathlib import Path

def load_config():
    """
    Load project configuration from project_config.yaml.
    Overrides with environment variables if present.
    """
    config_path = Path(__file__).resolve().parents[1] / "project_config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Override with env vars (GitHub Secrets pattern)
    if os.environ.get("AZURE_SUBSCRIPTION_ID"):
        config['azure']['subscription_id'] = os.environ.get("AZURE_SUBSCRIPTION_ID")
        
    return config
