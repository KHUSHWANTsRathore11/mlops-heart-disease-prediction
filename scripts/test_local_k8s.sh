#!/bin/bash
set -e

# Directory for local binaries
BIN_DIR="$(pwd)/.bin"
mkdir -p "$BIN_DIR"
export PATH="$BIN_DIR:$PATH"

# OS detection for downloading binaries
OS="$(uname | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
if [ "$ARCH" = "x86_64" ]; then ARCH="amd64"; fi
if [ "$ARCH" = "aarch64" ]; then ARCH="arm64"; fi

echo "========================================="
echo "Setting up local Kubernetes environment"
echo "OS: $OS, Arch: $ARCH"
echo "========================================="

# 1. Download kubectl if not present
if ! command -v kubectl &> /dev/null; then
    echo "Downloading kubectl..."
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/$OS/$ARCH/kubectl"
    chmod +x kubectl
    mv kubectl "$BIN_DIR/"
else
    echo "kubectl already installed"
fi

# 2. Download kind if not present
if ! command -v kind &> /dev/null; then
    echo "Downloading kind..."
    curl -Lo ./kind "https://kind.sigs.k8s.io/dl/v0.20.0/kind-$OS-$ARCH"
    chmod +x ./kind
    mv ./kind "$BIN_DIR/"
else
    echo "kind already installed"
fi

# 3. Create cluster
CLUSTER_NAME="heart-disease-cluster"
if kind get clusters | grep -q "$CLUSTER_NAME"; then
    echo "Cluster $CLUSTER_NAME already exists"
else
    echo "Creating kind cluster..."
    kind create cluster --name "$CLUSTER_NAME"
fi

# 4. Build Docker image
IMAGE_NAME="heart-disease-api:test"
echo "Building Docker image: $IMAGE_NAME..."
docker build -t "$IMAGE_NAME" .

# 5. Load image into kind
echo "Loading image into kind cluster..."
kind load docker-image "$IMAGE_NAME" --name "$CLUSTER_NAME"

# 6. Apply manifests
# We need to update the image in deployment.yaml to use our local tag and pullPolicy
# Creating a temporary deployment file
echo "Applying manifests..."
sed "s|image: .*|image: $IMAGE_NAME|g" k8s/deployment.yaml > k8s/deployment_test.yaml
sed -i 's|imagePullPolicy: .*|imagePullPolicy: Never|g' k8s/deployment_test.yaml

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment_test.yaml
kubectl apply -f k8s/service.yaml

# Clean up temp file
rm k8s/deployment_test.yaml

# 7. Wait for rollout
echo "Waiting for deployment..."
# Corrected namespace and deployment name
kubectl rollout status deployment/heart-disease-api -n heart-disease --timeout=120s

# 8. Function to clean up on exit
cleanup() {
    echo ""
    echo "Clean up? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Deleting cluster..."
        kind delete cluster --name "$CLUSTER_NAME"
    fi
}
trap cleanup EXIT

# 9. Test endpoint
echo "========================================="
echo "Testing application..."
echo "Forwarding port 8000..."

# Start port forwarding in background
# Corrected service name and namespace. Service maps 8000->8000
# Using port 8081 locally to avoid conflict with running docker container
kubectl port-forward svc/heart-disease-api 8081:8000 -n heart-disease &
PF_PID=$!

# Give it a moment to establish
sleep 5

echo "Checking health endpoint..."
if curl -s http://localhost:8081/health | grep -q "healthy"; then
    echo "[SUCCESS] Health check passed!"
    # Print pods check
    kubectl get pods -n heart-disease
else
    echo "[FAILURE] Health check failed"
    kubectl get pods -n heart-disease
    kubectl logs -l app=heart-disease-api -n heart-disease
fi

kill $PF_PID
echo "========================================="
echo "Test complete."
