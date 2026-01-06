# Kubernetes Local Deployment Guide

## Overview
This directory contains Kubernetes manifests for deploying the Heart Disease Prediction API to a local Kubernetes cluster.

## Prerequisites

- **Kubernetes Cluster**: minikube, kind, or Docker Desktop with Kubernetes enabled
- **kubectl**: Kubernetes command-line tool
- **Docker**: For building the container image

## Quick Start

### 1. Start Local Kubernetes Cluster

**Using Minikube**:
```bash
minikube start
```

**Using kind**:
```bash
kind create cluster --name heart-disease
```

**Using Docker Desktop**: Enable Kubernetes in settings

### 2. Build and Load Docker Image

**For Minikube**:
```bash
# Build image
docker build -t heart-disease-api:latest ../

# Load image into minikube
minikube image load heart-disease-api:latest
```

**For kind**:
```bash
# Build image
docker build -t heart-disease-api:latest ../

# Load image into kind
kind load docker-image heart-disease-api:latest --name heart-disease
```

**For Docker Desktop**:
```bash
# Just build - Docker Desktop shares images with K8s
docker build -t heart-disease-api:latest ../
```

### 3. Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f .

# Or use kustomize
kubectl apply -k .
```

### 4. Verify Deployment

```bash
# Check namespace
kubectl get namespaces

# Check pods
kubectl get pods -n heart-disease

# Check service
kubectl get svc -n heart-disease

# Check deployment status
kubectl rollout status deployment/heart-disease-api -n heart-disease
```

### 5. Access the API

**Option 1: Using minikube**:
```bash
minikube service heart-disease-api -n heart-disease
```

**Option 2: Using port-forward**:
```bash
kubectl port-forward -n heart-disease svc/heart-disease-api 8000:8000
```

**Option 3: Using NodePort** (if accessible):
```bash
# Get node IP
kubectl get nodes -o wide

# Access at: http://<NODE_IP>:30080
```

### 6. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
```

## Manifests Overview

### namespace.yaml
- Creates isolated namespace: `heart-disease`

### configmap.yaml
- Configuration data for the API
- Model version, logging level, feature names

### deployment.yaml
- Deployment with 2 replicas
- Resource limits: 256Mi-512Mi memory, 250m-500m CPU
- Health probes (liveness & readiness)
- Environment variables from ConfigMap

### service.yaml
- NodePort service on port 30080
- Routes traffic to pods on port 8000

### kustomization.yaml
- Kustomize configuration for managing resources

## Scaling

```bash
# Scale deployment
kubectl scale deployment heart-disease-api -n heart-disease --replicas=3

# Check status
kubectl get pods -n heart-disease
```

## Updating the Deployment

```bash
# After rebuilding image
minikube image load heart-disease-api:latest

# Restart deployment
kubectl rollout restart deployment/heart-disease-api -n heart-disease
```

## Monitoring

```bash
# View logs
kubectl logs -f -n heart-disease -l app=heart-disease-api

# Describe pod
kubectl describe pod -n heart-disease -l app=heart-disease-api

# Get events
kubectl get events -n heart-disease --sort-by='.lastTimestamp'
```

## Cleanup

```bash
# Delete all resources
kubectl delete -f .

# Or using kustomize
kubectl delete -k .

# Stop minikube (if using)
minikube stop

# Or delete kind cluster
kind delete cluster --name heart-disease
```

## Troubleshooting

### Pods not starting
```bash
# Check pod status
kubectl get pods -n heart-disease

# Check pod events
kubectl describe pod <pod-name> -n heart-disease

# Check logs
kubectl logs <pod-name> -n heart-disease
```

### Image pull errors
```bash
# Verify image is loaded
minikube image ls | grep heart-disease-api
# OR
docker exec -it kind-control-plane crictl images | grep heart-disease-api
```

### Service not accessible
```bash
# Check service endpoints
kubectl get endpoints -n heart-disease

# Verify pods are running and ready
kubectl get pods -n heart-disease -o wide
```

## Production Considerations

For production deployment, consider:
- Using a proper container registry
- Implementing HorizontalPodAutoscaler
- Adding Ingress for external access
- Implementing persistent storage for models
- Setting up monitoring (Prometheus/Grafana)
- Configuring resource quotas
- Implementing network policies
