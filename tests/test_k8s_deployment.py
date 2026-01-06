"""Validate Kubernetes deployment configuration"""
import sys

import yaml


def test_deployment_config():
    """Test that deployment configuration is valid"""
    try:
        with open("k8s/deployment.yaml") as f:
            config = yaml.safe_load(f)
            print(f"Replicas: {config['spec']['replicas']}")
            print(
                f"Image: {config['spec']['template']['spec']['containers'][0]['image']}"
            )
            print("[SUCCESS] Deployment config validated")
        return 0
    except Exception as e:
        print(f"[ERROR] Deployment validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(test_deployment_config())
