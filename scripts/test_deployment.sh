#!/bin/bash

# Script to test and compare Docker and Kubernetes deployments
# Usage: ./scripts/test_deployment.sh

# Configuration
DOCKER_URL="http://localhost:8000/predict"
KUBE_URL="http://localhost:30080/predict"
CONTENT_TYPE="Content-Type: application/json"

# Payload from test_api_request.sh
PAYLOAD='{
  "features": {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }
}'

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting Deployment Tests...${NC}"
echo "----------------------------------------"

# Function to test an endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local output_file=$3

    echo -e "Testing ${name} at ${url}..."
    
    # Determine kubectl path
    KUBECTL_CMD="kubectl"
    if ! command -v kubectl &> /dev/null; then
        if [ -f "./.bin/kubectl" ]; then
            KUBECTL_CMD="./.bin/kubectl"
        fi
    fi

    # Check if endpoint is reachable first
    if ! curl -s --head "$url" > /dev/null; then
        echo -e "${RED}[ERROR] Could not connect to ${name} at ${url}${NC}"
        echo "Make sure the service is running and exposed."
        if [ "$name" == "Kubernetes" ]; then
            echo "For Kubernetes (Kind/Minikube), you might need to use port-forwarding:"
            echo "$KUBECTL_CMD port-forward svc/heart-disease-api 30080:8000 -n heart-disease"
        fi
        return 1
    fi

    # Make the prediction request
    response=$(curl -s -X POST "$url" -H "$CONTENT_TYPE" -d "$PAYLOAD")
    
    if [ -z "$response" ]; then
        echo -e "${RED}[FAIL] Empty response from ${name}${NC}"
        return 1
    fi

    echo "$response" > "$output_file"
    echo -e "${GREEN}[SUCCESS] Received response from ${name}${NC}"
    # Use jq/python to pretty print if possible, else cat
    if command -v jq &> /dev/null; then
        echo "$response" | jq .
    else
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    fi
    return 0
}

# Create temporary files for outputs
DOCKER_OUT=$(mktemp)
KUBE_OUT=$(mktemp)

# Test Docker
echo "1. Testing Docker Container..."
test_endpoint "Docker" "$DOCKER_URL" "$DOCKER_OUT"
DOCKER_STATUS=$?

echo "Done."
