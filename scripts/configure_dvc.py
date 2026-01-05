import sys
import subprocess
from pathlib import Path

# Add project root to path to import src modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config

def run_dvc_command(command):
    """Run a DVC command via subprocess."""
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"Successfully executed: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {command}: {e}")
        sys.exit(1)

def configure_dvc():
    """Configure DVC remote based on project_config.yaml."""
    print("Loading configuration...")
    config = load_config()
    
    storage_account = config['storage']['account_name']
    container_name = config['storage']['container_name']
    
    remote_name = "azure-remote"
    remote_url = f"azure://{container_name}/dvc-data"
    
    print(f"Configuring DVC remote '{remote_name}'...")
    print(f"  Storage Account: {storage_account}")
    print(f"  Container: {container_name}")
    
    # Check if remote exists
    try:
        subprocess.run(f"dvc remote list | grep {remote_name}", shell=True, check=True, stdout=subprocess.DEVNULL)
        print(f"Remote '{remote_name}' already exists. Updating settings...")
    except subprocess.CalledProcessError:
        print(f"Adding new remote '{remote_name}'...")
        run_dvc_command(f"dvc remote add -d {remote_name} {remote_url}")

    # Update URL in case it changed
    run_dvc_command(f"dvc remote modify {remote_name} url {remote_url}")
    
    # Set account name
    run_dvc_command(f"dvc remote modify {remote_name} account_name {storage_account}")
    
    print("\nDVC configuration completed successfully!")
    print("You can now run 'dvc push' to upload data to Azure Blob Storage.")

if __name__ == "__main__":
    configure_dvc()
