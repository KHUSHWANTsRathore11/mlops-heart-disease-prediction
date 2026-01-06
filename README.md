# MLOps Heart Disease Prediction


End-to-end machine learning project for predicting heart disease risk using MLOps best practices.

## Project Structure

```
mlops-heart-disease-prediction/
<<<<<<< HEAD
   .github/workflows/     # CI/CD pipelines
   app/                   # Flask API source
   data/                  # Data files (DVC tracked)
   deployment/            # Infrastructure as Code (Terraform)
   docs/                  # Documentation & Screenshots
   k8s/                   # Kubernetes manifests
   metrics/               # Model training metrics
   mlruns/                # MLflow tracking store
   models/                # Serialized models & registry
   notebooks/             # EDA & experimental notebooks
   scripts/               # Utility & automation scripts
   src/                   # Core source code
      data/             # Data processing modules
      features/         # Feature engineering modules
      models/           # Model training & evaluation
   tests/                 # Automated tests (pytest)
   dvc.yaml              # DVC pipeline configuration
   Dockerfile            # Container build definition
   project_config.yaml   # Project configuration
   pyproject.toml        # Build system configuration
   requirements.txt      # Python dependencies
   test_api_request.sh   # API testing script
   README.md             # Project overview
```


=======
   data/
      raw/              # Raw data files
      processed/        # Cleaned and processed data
   notebooks/            # Jupyter notebooks for exploration
   src/
      data/            # Data processing modules
   deployment/
      terraform/       # Infrastructure as Code
   docs/                # Documentation
   requirements.txt     # Python dependencies
   README.md
```

>>>>>>> origin/develop
## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd mlops-heart-disease-prediction
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Data

```bash
python src/data/download_data.py
```

## Task Progress

<<<<<<< HEAD
### Task 1: Data Acquisition & EDA 
=======
### Task 1: Data Acquisition & EDA [5 marks]
- Data download script
>>>>>>> origin/develop
- Data cleaning utilities
- EDA notebook with visualizations
- Execute and verify results

<<<<<<< HEAD
### Task 2: Feature Engineering & Model Development 
=======
### Task 2: Feature Engineering & Model Development [8 marks]
>>>>>>> origin/develop
- Feature engineering pipeline
  - Data cleaning and preprocessing
  - StandardScaler for numerical features
  - Train/test split (80/20, stratified)
  - Preprocessor serialization
  - DVC pipeline integration
  - Comprehensive test suite (11 tests)
- Model training scripts (2 models: Logistic Regression + Random Forest)
  - 5-fold stratified cross-validation
  - Multiple evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
  - Automatic best model selection
  - Visualization artifacts (confusion matrix, ROC curve, feature importance)
- Model evaluation with cross-validation
  - Logistic Regression: 89.40% ROC-AUC
  - Random Forest: 92.75% ROC-AUC (Best Model)

<<<<<<< HEAD
### Task 3: Experiment Tracking 
=======
### Task 3: Experiment Tracking [5 marks]
>>>>>>> origin/develop
- MLflow integration
  - Local tracking server setup
  - Experiment creation and management
- Experiment logging
  - Parameters tracking
  - Metrics tracking (CV and test)
  - Artifacts logging (plots, reports)
  - Model registry

<<<<<<< HEAD
### Task 4: Model Packaging & Reproducibility 
=======
### Task 4: Model Packaging & Reproducibility [7 marks]
>>>>>>> origin/develop
- Model serialization
  - Best model saved as `models/model.pkl` (Random Forest, 1.3MB)
  - Model can be loaded and used for predictions
- Preprocessing pipeline
  - Preprocessor saved as `models/preprocessor.pkl`
  - Ensures consistent feature transformation
- Requirements & reproducibility
  - Clean `requirements.txt` (35 essential packages)
  - DVC pipeline for full reproducibility
  - Metrics tracked in `metrics/training_metrics.json`

<<<<<<< HEAD
### Task 5: CI/CD Pipeline & Automated Testing 
=======
### Task 5: CI/CD Pipeline & Automated Testing [8 marks]
>>>>>>> origin/develop
- Unit tests
  - Expanded test suite (40 tests total)
  - Data loading tests (13 tests)
  - Model training tests (16 tests)
  - Feature engineering tests (11 tests)
  - Test coverage > 80%
- GitHub Actions workflow
  - Automated linting (flake8, black, isort)
  - Automated testing with coverage reports
  - DVC pipeline verification
  - Artifact upload (coverage reports)

<<<<<<< HEAD
### Task 6: Model Containerization 
=======
### Task 6: Model Containerization [5 marks]
>>>>>>> origin/develop
- Dockerfile
  - Multi-stage build with Python 3.11-slim
  - Optimized with .dockerignore
  - Health check configured
  - Gunicorn WSGI server
- API development
  - Flask API with endpoints: /, /health, /predict
  - JSON input validation
  - Error handling
  - Returns prediction with confidence scores
  - Running on port 8000

<<<<<<< HEAD
### Task 7: Production Deployment 
=======
### Task 7: Production Deployment [7 marks]
>>>>>>> origin/develop
- Local kubernetes deployment
  - Namespace for isolation
  - ConfigMap for configuration
  - Deployment with 2 replicas
  - Health probes (liveness & readiness)
  - Resource limits (CPU & memory)
- Kubernetes manifests
  - deployment.yaml
  - service.yaml (NodePort on 30080)
  - configmap.yaml  
  - namespace.yaml
  - kustomization.yaml
  - Comprehensive deployment README

<<<<<<< HEAD
### Task 8: Monitoring & Logging 
=======
### Task 8: Monitoring & Logging [3 marks]
>>>>>>> origin/develop
- Logging implementation
  - Structured JSON logging
  - Request/response logging with duration tracking
  - Prediction logging with confidence scores
  - Error tracking with details
- Monitoring dashboard
  - `/metrics` endpoint with real-time stats
  - Log analysis script (scripts/analyze_logs.py)
  - Comprehensive monitoring documentation

<<<<<<< HEAD
### Task 9: Documentation & Reporting 
=======
### Task 9: Documentation & Reporting [2 marks]
>>>>>>> origin/develop
- Final report
  - Executive summary
  - All tasks and results documented
  - Architecture overview
  - Deployment instructions
  - Future improvements outlined
- Complete project documentation (see docs/FINAL_REPORT.md)

## Usage

### Data Processing

```bash
# Download data
python src/data/download_data.py

# Clean and preprocess
python src/data/preprocessing.py
```

### Exploratory Data Analysis

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/01_data_acquisition_eda.ipynb
```

## Dataset Information

**Source:** UCI Machine Learning Repository - Heart Disease Dataset

**Features:**
- age: Age in years
- sex: Sex (1 = male, 0 = female)
- cp: Chest pain type
- trestbps: Resting blood pressure
- chol: Serum cholesterol
- fbs: Fasting blood sugar
- restecg: Resting ECG results
- thalach: Maximum heart rate achieved
- exang: Exercise induced angina
- oldpeak: ST depression
- slope: Slope of peak exercise ST segment
- ca: Number of major vessels
- thal: Thalassemia
- target: Heart disease presence (0 = no, 1 = yes)

## License

See LICENSE file for details.

<<<<<<< HEAD
=======
## Contributors

- [Your Name]
>>>>>>> origin/develop

## Assignment Details

- Course: MLOps (S1-25_AIMLCZG523)
- Assignment: I
<<<<<<< HEAD
=======
- Total Marks: 50
>>>>>>> origin/develop
