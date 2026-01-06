"""Validate Kubernetes service configuration"""
import sys

import yaml


def test_service_config():
    """Test that service configuration is valid"""
    try:
        with open("k8s/service.yaml") as f:
            config = yaml.safe_load(f)
            print(f"Service type: {config['spec']['type']}")
            print(f"Port: {config['spec']['ports'][0]['port']}")
            print("[SUCCESS] Service config validated")
        return 0
    except Exception as e:
        print(f"[ERROR] Service validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(test_service_config())
