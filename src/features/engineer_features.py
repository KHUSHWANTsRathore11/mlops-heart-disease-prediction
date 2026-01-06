import argparse
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd

from src.data.preprocessing import HeartDiseasePreprocessor, prepare_train_test_split

# Add project root to path to import src modules
sys.path.append(str(Path(__file__).resolve().parents[2]))


def load_data(input_path):
    """Load cleaned data from CSV."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return pd.read_csv(input_path)


def engineer_features(input_path, output_dir, test_size=0.2, random_state=42):
    """
    Main feature engineering logic.
    1. Load data
    2. Split train/test
    3. Fit preprocessor
    4. Transform data
    5. Save artifacts
    """
    print(f"Loading data from {input_path}...")
    df = load_data(input_path)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split data
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    # Combine X and y for preprocessing convenience (if needed by preprocessor class structure)
    # The existing HeartDiseasePreprocessor expects a DataFrame defined by 'target' column for exclusion during fit
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Initialize and fit preprocessor
    print("Fitting preprocessor...")
    preprocessor = HeartDiseasePreprocessor()
    preprocessor.fit(train_df, target_col="target")

    # Transform data
    print("Transforming data...")
    train_transformed = preprocessor.transform(train_df, target_col="target")
    test_transformed = preprocessor.transform(test_df, target_col="target")

    # Log to MLflow if active run exists
    if mlflow.active_run():
        print("Logging to MLflow...")
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("train_samples", len(train_transformed))
        mlflow.log_metric("test_samples", len(test_transformed))
        mlflow.log_metric("n_features", len(preprocessor.feature_names))

    # Save outputs
    train_path = output_dir / "features_train.csv"
    test_path = output_dir / "features_test.csv"
    preprocessor_path = Path("models/preprocessor.pkl")
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving train features to {train_path}")
    train_transformed.to_csv(train_path, index=False)

    print(f"Saving test features to {test_path}")
    test_transformed.to_csv(test_path, index=False)

    print(f"Saving preprocessor to {preprocessor_path}")
    preprocessor.save(preprocessor_path)

    return train_path, test_path, preprocessor_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engineering for Heart Disease Prediction")
    parser.add_argument(
        "--input",
        default="data/processed/heart_disease_clean.csv",
        help="Path to input cleaned data",
    )
    parser.add_argument(
        "--output-dir", default="data/processed", help="Directory to save engineered features"
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    try:
        engineer_features(args.input, args.output_dir, args.test_size, args.random_state)
        print("Feature engineering completed successfully.")
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        sys.exit(1)
