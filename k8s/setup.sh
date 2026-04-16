#!/bin/bash
set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  MLOps Kubernetes Deployment"
echo "═══════════════════════════════════════════════════════════════"

echo ""
echo "[1/9] Starting Minikube..."
if minikube status | grep -q "Running"; then
    echo "  Minikube is already running."
else
    minikube start --memory=4096 --cpus=2
fi

echo ""
echo "[2/9] Enabling NGINX Ingress controller..."
minikube addons enable ingress --alsologtostderr 2>/dev/null || true
echo "  Waiting for ingress controller pod to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=controller \
    -n ingress-nginx --timeout=120s 2>/dev/null || \
kubectl wait --for=condition=ready pod -l app=ingress-nginx \
    -n ingress-nginx --timeout=120s 2>/dev/null || \
echo "  Warning: Ingress controller not ready yet — it may still be starting."

echo ""
echo "[3/9] Configuring Docker to use Minikube's daemon..."
eval $(minikube docker-env)

echo ""
echo "[4/9] Building Docker images (this may take a few minutes)..."

echo "  Building mlops-mlflow..."
docker build -t mlops-mlflow:latest -f k8s/docker/Dockerfile.mlflow .

echo "  Building mlops-training..."
docker build -t mlops-training:latest -f k8s/docker/Dockerfile.training .

echo "  Building mlops-predict..."
docker build -t mlops-predict:latest -f k8s/docker/Dockerfile.predict .

echo "  All images built."

echo ""
echo "[5/9] Creating namespace and PVC..."
kubectl apply -f k8s/manifests/namespace.yaml
kubectl apply -f k8s/manifests/pvc.yaml

echo ""
echo "[6/9] Deploying MLflow server..."
kubectl apply -f k8s/manifests/mlflow-deployment.yaml
echo "  Waiting for MLflow server to be ready..."
kubectl wait --for=condition=available deployment/mlflow-server -n mlops --timeout=120s
echo "  MLflow server is ready."

echo ""
echo "[7/9] Starting training job (this takes several minutes)..."
# Delete previous job if it exists (jobs are not rerunnable)
kubectl delete job training-job -n mlops --ignore-not-found=true
kubectl apply -f k8s/manifests/training-job.yaml
echo "  Waiting for training to complete..."
echo "  (You can watch logs with: kubectl logs -f job/training-job -n mlops)"
kubectl wait --for=condition=complete job/training-job -n mlops --timeout=600s
echo "  Training complete!"

echo ""
echo "[8/9] Deploying prediction app..."
kubectl apply -f k8s/manifests/predict-deployment.yaml
echo "  Waiting for prediction app to be ready..."
kubectl wait --for=condition=available deployment/predict-app -n mlops --timeout=180s
echo "  Prediction app is ready."

echo ""
echo "[9/9] Applying Ingress..."
kubectl apply -f k8s/manifests/ingress.yaml

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  DEPLOYMENT COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""

kubectl get all -n mlops

MINIKUBE_IP=$(minikube ip 2>/dev/null || echo "127.0.0.1")

echo ""
echo "───────────────────────────────────────────────────────────────"
echo "  ACCESS YOUR SERVICES"
echo "───────────────────────────────────────────────────────────────"
echo "    Port-forward (works on macOS Docker driver)"
echo "    kubectl port-forward svc/predict-service 8080:8080 -n mlops"
echo "    kubectl port-forward svc/mlflow-service 5000:5000 -n mlops"
echo "    Then open:  http://localhost:8080  and  http://localhost:5000"

