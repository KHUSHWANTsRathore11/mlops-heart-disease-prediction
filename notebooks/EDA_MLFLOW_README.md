# EDA with MLflow Logging - Heart Disease Prediction

This notebook performs exploratory data analysis and logs all statistics and plots to MLflow for tracking.

## Instructions

Instead of modifying the Jupyter notebook directly, I'll create a Python script that:
1. Performs all EDA analysis
2. Generates all plots
3. Logs everything to MLflow

You can then run this script, and all EDA will be tracked in MLflow!

## To Use

1. Run the script:
```bash
python scripts/eda_with_mlflow.py
```

2. View in MLflow UI:
```bash
mlflow ui --port 5001
```

3. Look for experiment: "heart-disease-eda"

The script will log:
- Dataset statistics (shape, missing values, data types)
- Feature distributions
- Correlation matrix
- Target distribution
- All visualization plots as artifacts
