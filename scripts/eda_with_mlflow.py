#!/usr/bin/env python3
"""
EDA with MLflow Logging - Heart Disease Prediction

This script performs comprehensive exploratory data analysis and logs
all statistics, metrics, and visualizations to MLflow.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Paths
DATA_PATH = Path("data/raw/heart_disease_combined.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load the heart disease dataset"""
    df = pd.read_csv(DATA_PATH)
    print(f"[OK] Data loaded: {df.shape}")
    return df

def log_dataset_stats(df):
    """Log basic dataset statistics to MLflow"""
    stats = {
        "total_samples": len(df),
        "total_features": len(df.columns) - 1,  # Excluding target
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_0_count": int((df['target'] == 0).sum()),
        "target_1_count": int((df['target'] == 1).sum()),
        "target_balance_ratio": float((df['target'] == 1).sum() / len(df))
    }
    
    mlflow.log_params(stats)
    print("[INFO] Dataset statistics logged")
    return stats

def log_feature_stats(df):
    """Log feature-level statistics"""
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('target')
    
    for feature in numerical_features:
        mlflow.log_metrics({
            f"{feature}_mean": float(df[feature].mean()),
            f"{feature}_std": float(df[feature].std()),
            f"{feature}_min": float(df[feature].min()),
            f"{feature}_max": float(df[feature].max())
        })
    
    print(f"[INFO] Statistics logged for {len(numerical_features)} features")

def create_and_log_correlation_heatmap(df):
    """Create and log correlation heatmap"""
    plt.figure(figsize=(14, 12))
    
    # Calculate correlation
    corr = df.corr()
    
    # Create heatmap
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save and log
    output_path = OUTPUT_DIR / "correlation_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(str(output_path))
    plt.close()
    
    # Log top correlations with target
    target_corr = corr['target'].abs().sort_values(ascending=False)[1:6]
    for i, (feature, corr_value) in enumerate(target_corr.items(), 1):
        mlflow.log_metric(f"top{i}_correlation_with_target", float(corr_value))
        mlflow.log_param(f"top{i}_correlated_feature", feature)
    
    print("[INFO] Correlation heatmap created and logged")

def create_and_log_feature_distributions(df):
    """Create and log feature distribution plots"""
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('target')
    
    # Create subplots
    n_features = len(numerical_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()
    
    for idx, feature in enumerate(numerical_features):
        axes[idx].hist(df[feature], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{feature} Distribution', fontweight='bold')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Feature Distributions', fontsize=18, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save and log
    output_path = OUTPUT_DIR / "feature_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(str(output_path))
    plt.close()
    
    print("[INFO] Feature distributions created and logged")

def create_and_log_boxplots(df):
    """Create and log boxplots for features by target"""
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('target')
    
    # Create subplots
    n_features = len(numerical_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()
    
    for idx, feature in enumerate(numerical_features):
        df.boxplot(column=feature, by='target', ax=axes[idx])
        axes[idx].set_title(f'{feature} by Target', fontweight='bold')
        axes[idx].set_xlabel('Target')
        axes[idx].set_ylabel(feature)
    
    # Hide empty subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Feature Boxplots by Target', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save and log
    output_path = OUTPUT_DIR / "feature_boxplots.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(str(output_path))
    plt.close()
    
    print("[INFO] Boxplots created and logged")

def create_and_log_target_distribution(df):
    """Create and log target distribution plot"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count plot
    target_counts = df['target'].value_counts().sort_index()
    axes[0].bar(target_counts.index, target_counts.values, 
                color=['#3498db', '#e74c3c'], edgecolor='black', alpha=0.7)
    axes[0].set_title('Target Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Target')
    axes[0].set_ylabel('Count')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['No Disease', 'Disease'])
    axes[0].grid(alpha=0.3)
    
    # Add value labels
    for idx, label in enumerate(target_counts.index):
        axes[0].text(label, target_counts.iloc[idx] + 10, str(target_counts.iloc[idx]), 
                    ha='center', fontweight='bold')
    
    # Pie chart
    colors = ['#3498db', '#e74c3c']
    sizes = [target_counts.iloc[0], target_counts.iloc[1]]
    labels = [f'No Disease\n({sizes[0]})', f'Disease\n({sizes[1]})']
    axes[1].pie(sizes, labels=labels, 
                autopct='%1.1f%%', colors=colors, startangle=90,
                explode=(0.05, 0.05), shadow=True)
    axes[1].set_title('Target Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save and log
    output_path = OUTPUT_DIR / "target_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(str(output_path))
    plt.close()
    
    print("[INFO] Target distribution created and logged")

def main():
    """Main EDA function with MLflow logging"""
    
    # Set MLflow experiment
    mlflow.set_experiment("heart-disease-eda")
    
    # Start MLflow run
    with mlflow.start_run(run_name="EDA_Analysis"):
        
        print("\n" + "="*60)
        print("[START] HEART DISEASE EDA WITH MLFLOW LOGGING")
        print("="*60 + "\n")
        
        # Load data
        df = load_data()
        
        # Log dataset info
        mlflow.log_param("dataset_path", str(DATA_PATH))
        mlflow.log_param("dataset_name", "Heart Disease UCI Combined")
        
        # Log dataset statistics
        print("\n[INFO] Logging dataset statistics...")
        stats = log_dataset_stats(df)
        
        # Log feature-level statistics
        print("\n[INFO] Logging feature statistics...")
        log_feature_stats(df)
        
        # Create and log visualizations
        print("\n[PLOT] Creating and logging visualizations...")
        print("-" * 60)
        
        create_and_log_correlation_heatmap(df)
        create_and_log_feature_distributions(df)
        create_and_log_boxplots(df)
        create_and_log_target_distribution(df)
        
        # Log summary
        print("\n" + "="*60)
        print("[OK] EDA COMPLETE!")
        print("="*60)
        print(f"\n[INFO] Logged to MLflow:")
        print(f"   - {stats['total_samples']} samples")
        print(f"   - {stats['total_features']} features")
        print(f"   - 4 visualization plots")
        print(f"   - Feature-level statistics")
        print(f"   - Top feature correlations")
        print(f"\n[VIEW] View in MLflow UI: http://localhost:5001")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()
