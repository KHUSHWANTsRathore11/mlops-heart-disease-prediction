import argparse
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

import sys
sys.path.append('src')
from config import load_config

def setup_mlflow(config):
    """Configure MLflow with Azure ML tracking URI."""
    print("Setting up MLflow tracking...")
    
    # Connect to Azure ML
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential,
        config['azure']['subscription_id'],
        config['azure']['resource_group'],
        config['azure']['workspace_name']
    )
    
    # Get workspace details to retrieve tracking URI
    ws = ml_client.workspaces.get(config['azure']['workspace_name'])
    tracking_uri = ws.mlflow_tracking_uri
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("heart-disease-training")
    print(f"MLflow tracking URI set to: {tracking_uri}")

def train(train_path, test_path, output_path):
    # Load data
    print(f"Loading data from {train_path} and {test_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Assume 'target' is the column name. Adjust if different via preprocessing.
    # Looking at preprocessing.py, the target column name should be preserved.
    target_col = 'target'
    
    # Drop non-feature columns if present (e.g., patient_id, timestamp if added by Feature Store logic)
    # Using simple drop of known metadata columns
    drop_cols = ['patient_id', 'timestamp'] if 'patient_id' in train_df.columns else []
    
    X_train = train_df.drop(columns=[target_col] + [c for c in drop_cols if c in train_df.columns], errors='ignore')
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=[target_col] + [c for c in drop_cols if c in test_df.columns], errors='ignore')
    y_test = test_df[target_col]
    
    # Start MLflow run
    with mlflow.start_run():
        print("Training Logistic Regression model...")
        # Hyperparameters (can be externalized later)
        params = {"C": 1.0, "solver": "liblinear", "random_state": 42}
        
        # Log params
        mlflow.log_params(params)
        
        # Train
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds)
        auc = roc_auc_score(y_test, test_probs)
        
        print(f"Metrics: Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        
        # Log metrics
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "auc": auc})
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save locally for DVC
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(model, output_path)
        print(f"Model saved locally to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/processed/features_train.csv")
    parser.add_argument("--test-path", default="data/processed/features_test.csv")
    parser.add_argument("--output-path", default="models/model.pkl")
    args = parser.parse_args()
    
    # Load project config
    config = load_config()
    
    try:
        setup_mlflow(config)
    except Exception as e:
        print(f"Warning: Could not set up Azure MLflow ({e}). Continuing with local MLflow/no logging.")
        
    train(args.train_path, args.test_path, args.output_path)
