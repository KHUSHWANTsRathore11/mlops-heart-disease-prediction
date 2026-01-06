"""
Model evaluation utilities for Heart Disease Prediction.

This module provides reusable functions for evaluating classification models,
generating visualizations, and performing cross-validation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from pathlib import Path
import json


def plot_confusion_matrix(y_true, y_pred, save_path=None, title='Confusion Matrix'):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot (optional)
        title: Plot title

    Returns:
        Path to saved plot or None
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        return save_path

    plt.close()
    return None


def plot_roc_curve(y_true, y_proba, save_path=None, title='ROC Curve'):
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        save_path: Path to save the plot (optional)
        title: Plot title

    Returns:
        tuple: (fpr, tpr, auc_score, save_path)
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")

    plt.close()
    return fpr, tpr, roc_auc, save_path


def plot_feature_importance(model, feature_names, save_path=None,
                            title='Feature Importance', top_n=15):
    """
    Plot feature importance for tree-based models.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        save_path: Path to save the plot (optional)
        title: Plot title
        top_n: Number of top features to display

    Returns:
        Path to saved plot or None
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return None

    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    # Create dataframe for plotting
    importance_df = pd.DataFrame({
        'feature': [feature_names[i] for i in indices],
        'importance': importances[indices]
    })

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
        return save_path

    plt.close()
    return None


def generate_classification_report(y_true, y_pred, save_path=None):
    """
    Generate classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the report as JSON (optional)

    Returns:
        dict: Classification report as dictionary
    """
    report_dict = classification_report(y_true, y_pred,
                                        target_names=['No Disease', 'Disease'],
                                        output_dict=True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report_dict, f, indent=4)
        print(f"Classification report saved to {save_path}")

    # Also print to console
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=['No Disease', 'Disease']))

    return report_dict


def cross_validate_model(model, X, y, cv=5, scoring=None):
    """
    Perform cross-validation with multiple metrics.

    Args:
        model: Sklearn model
        X: Features
        y: Target
        cv: Number of folds (default: 5)
        scoring: List of scoring metrics (optional)

    Returns:
        dict: Cross-validation results with mean and std for each metric
    """
    if scoring is None:
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    # Use stratified K-fold for balanced splits
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Perform cross-validation
    cv_results = cross_validate(model, X, y, cv=cv_strategy,
                                scoring=scoring, return_train_score=True)

    # Calculate statistics
    results_summary = {}
    for metric in scoring:
        test_key = f'test_{metric}'
        train_key = f'train_{metric}'

        results_summary[f'{metric}_cv_mean'] = cv_results[test_key].mean()
        results_summary[f'{metric}_cv_std'] = cv_results[test_key].std()
        results_summary[f'{metric}_train_mean'] = cv_results[train_key].mean()
        results_summary[f'{metric}_train_std'] = cv_results[train_key].std()

    # Print summary
    print("\n" + "=" * 60)
    print("Cross-Validation Results (5-fold)")
    print("=" * 60)
    for metric in scoring:
        mean_score = results_summary[f'{metric}_cv_mean']
        std_score = results_summary[f'{metric}_cv_std']
        print(f"{metric.upper():15s}: {mean_score:.4f} (+/- {std_score:.4f})")
    print("=" * 60)

    return results_summary


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate all evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities for positive class (optional)

    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)

    return metrics


def print_metrics(metrics, title="Metrics"):
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics display
    """
    print(f"\n{title}")
    print("=" * 40)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.upper():15s}: {metric_value:.4f}")
    print("=" * 40)
