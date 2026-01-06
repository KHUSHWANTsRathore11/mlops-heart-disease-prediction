# Screenshot Collection Guide

This guide provides step-by-step instructions for capturing all screenshots needed for the final report.

## Setup
Before capturing screenshots:
```bash
cd /home/ksr11/workspace/M_TECH/sem3/MLOPS/Project2/mlops-heart-disease-prediction
source .venv/bin/activate
```

---

## Task 1: Data Acquisition & EDA

### [SCREENSHOT 1.1] - EDA Notebook Output
**How to capture**:
1. Open Jupyter/VS Code with notebook
2. Open `notebooks/00_eda.ipynb`
3. Capture:
   - Data distribution plots (histograms)
   - Correlation heatmap
   - Feature distributions by target
   - Target balance (pie/bar chart)

### [SCREENSHOT 1.2] - Data Download and Cleaning
**Command**:
```bash
python src/data/download_data.py
```
**Capture**: Terminal output showing download and cleaning summary

---

## Task 2: Feature Engineering & Model Development

### [SCREENSHOT 2.1] - DVC Pipeline DAG
**Command**:
```bash
.venv/bin/dvc dag
```
**Capture**: Terminal output showing pipeline stages

### [SCREENSHOT 2.2] - Model Training Output
**Command**:
```bash
python src/models/train_model.py
```
**Capture**:
- Cross-validation results for both models
- Model comparison table
- Best model selection message

### [SCREENSHOT 2.3] - Saved Model Artifacts
**Command**:
```bash
ls -lh models/
```
**Capture**: File listing showing model.pkl and preprocessor.pkl with sizes

---

## Task 3: Experiment Tracking

### [SCREENSHOT 3.1] - MLflow Experiments Dashboard
**Steps**:
1. Start MLflow: `.venv/bin/mlflow ui --port 5001`
2. Open browser: http://localhost:5001
3. Capture:
   - Experiments list view
   - Run comparison table
   - Metrics comparison (sort by ROC-AUC)

### [SCREENSHOT 3.2] - MLflow Run Details
**Steps**:
1. In MLflow UI, click on a run (Random Forest)
2. Capture:
   - Parameters section
   - Metrics section
   - Artifacts list

### [SCREENSHOT 3.3] - MLflow Artifacts
**Steps**:
1. In run details, click on Artifacts
2. Capture:
   - Confusion matrix plot
   - ROC curve
   - Feature importance plot

---

## Task 4: Model Packaging & Reproducibility

### [SCREENSHOT 4.1] - Model Loading Test
**Command**:
```bash
python -c "import joblib; model = joblib.load('models/model.pkl'); print(f'Model loaded: {type(model).__name__}')"
```
**Capture**: Terminal output with success message

### [SCREENSHOT 4.2] - Requirements.txt
**Command**:
```bash
cat requirements.txt | head -20
echo "..."
wc -l requirements.txt
```
**Capture**: First 20 lines + total line count (35 packages)

---

## Task 5: CI/CD Pipeline & Automated Testing

### [SCREENSHOT 5.1] - GitHub Actions Workflow
**Steps**:
1. Go to GitHub repository
2. Click on "Actions" tab
3. Capture:
   - Workflow runs list (showing successful runs)
   - Click on latest run
   - Capture workflow details showing all jobs passing

### [SCREENSHOT 5.2] - Test Results
**Command**:
```bash
pytest tests/ -v --tb=short
```
**Capture**: Terminal output showing all 40 tests and summary

### [SCREENSHOT 5.3] - Coverage Report
**Command**:
```bash
pytest tests/ --cov=src --cov-report=term-missing
```
**Capture**: Coverage report showing >80% coverage

---

## Task 6: Model Containerization

### [SCREENSHOT 6.1] - API Health Check
**Commands**:
```bash
# Start API in background
python app/main.py > /tmp/api.log 2>&1 &

# Wait and test
sleep 3
curl http://localhost:8000/health | python -m json.tool
```
**Capture**: JSON response showing model_loaded: true

### [SCREENSHOT 6.2] - API Prediction Request/Response
**Command**:
```bash
bash test_api_request.sh
```
**Or**:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features":{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}}' | python -m json.tool
```
**Capture**: JSON response with prediction, probability, risk_level

### [SCREENSHOT 6.3] - Docker Build/Dockerfile
**Option 1** (if Docker available):
```bash
docker build -t heart-disease-api:latest .
```
**Capture**: Build output showing successful build

**Option 2** (if Docker not available):
```bash
cat Dockerfile
```
**Capture**: Dockerfile content

---

## Task 7: Production Deployment

### [SCREENSHOT 7.1] - Kubernetes Manifests
**Command**:
```bash
ls -lah k8s/
```
**Capture**: Directory listing showing all YAML files

### [SCREENSHOT 7.2] - Deployment YAML
**Command**:
```bash
cat k8s/deployment.yaml
```
**Capture**: YAML content, highlight:
- replicas: 2
- resources section
- livenessProbe/readinessProbe

### [SCREENSHOT 7.3] - Kubernetes Resources
**Option 1** (if cluster available):
```bash
kubectl apply -f k8s/
kubectl get all -n heart-disease
```
**Capture**: Resources created and pod status

**Option 2** (if cluster not available):
```bash
cat k8s/service.yaml
```
**Capture**: Service manifest showing NodePort configuration

---

## Task 8: Monitoring & Logging

### [SCREENSHOT 8.1] - Structured JSON Logs
**Commands**:
```bash
# Generate some traffic
for i in {1..5}; do
  curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features":{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}}' > /dev/null 2>&1
done

# View logs
cat /tmp/api.log | tail -20
```
**Capture**: JSON-formatted log entries

### [SCREENSHOT 8.2] - Metrics Endpoint
**Command**:
```bash
curl http://localhost:8000/metrics | python -m json.tool
```
**Capture**: JSON showing all metrics

### [SCREENSHOT 8.3] - Log Analysis Output
**Command**:
```bash
python scripts/analyze_logs.py /tmp/api.log
```
**Capture**: Statistics report with request counts, response times, etc.

---

## Task 9: Documentation

**No specific screenshots needed** - this report itself is the deliverable!

---

## Tips

1. **Use high-quality screenshots** - Use native screenshot tools (not photos)
2. **Highlight important parts** - Use rectangles/arrows to draw attention
3. **Keep them readable** - Ensure text is legible
4. **Consistent sizing** - Try to keep similar aspect ratios
5. **Save with descriptive names** - e.g., `screenshot_3_1_mlflow_dashboard.png`

## Screenshot Storage

Create a screenshots directory:
```bash
mkdir -p docs/screenshots
```

Save screenshots as:
- `docs/screenshots/task1_1_eda_plots.png`
- `docs/screenshots/task2_1_dvc_dag.png`
- `docs/screenshots/task3_1_mlflow_dashboard.png`
- etc.

Then embed in FINAL_REPORT.md as:
```markdown
![Screenshot Description](screenshots/task1_1_eda_plots.png)
```
