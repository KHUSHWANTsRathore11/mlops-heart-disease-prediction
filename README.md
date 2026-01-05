# MLOps Heart Disease Prediction

End-to-end machine learning project for predicting heart disease risk using MLOps best practices.

## Project Structure

```
mlops-heart-disease-prediction/
├── data/
│   ├── raw/              # Raw data files
│   └── processed/        # Cleaned and processed data
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   └── data/            # Data processing modules
├── deployment/
│   └── terraform/       # Infrastructure as Code
├── docs/                # Documentation
├── requirements.txt     # Python dependencies
└── README.md
```

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

### Task 1: Data Acquisition & EDA [5 marks]
- [x] Data download script
- [x] Data cleaning utilities
- [x] EDA notebook with visualizations
- [x] Execute and verify results

### Task 2: Feature Engineering & Model Development [8 marks]
- [x] Feature engineering pipeline
- [ ] Model training scripts
- [ ] Model evaluation

### Task 3: Experiment Tracking [5 marks]
- [ ] MLflow integration
- [ ] Experiment logging

### Task 4: Model Packaging & Reproducibility [7 marks]
- [ ] Model serialization
- [ ] Preprocessing pipeline

### Task 5: CI/CD Pipeline & Automated Testing [8 marks]
- [ ] Unit tests
- [ ] GitHub Actions workflow

### Task 6: Model Containerization [5 marks]
- [ ] Dockerfile
- [ ] API development

### Task 7: Production Deployment [7 marks]
- [ ] Cloud deployment
- [ ] Kubernetes manifests

### Task 8: Monitoring & Logging [3 marks]
- [ ] Logging implementation
- [ ] Monitoring dashboard

### Task 9: Documentation & Reporting [2 marks]
- [ ] Final report
- [ ] Demo video

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

## Contributors

- [Your Name]

## Assignment Details

- Course: MLOps (S1-25_AIMLCZG523)
- Assignment: I
- Total Marks: 50
