"""
Script to download the Heart Disease UCI dataset.
"""
import sys
import pandas as pd
import urllib.request
from pathlib import Path


def download_heart_disease_data(output_dir="data/raw"):
    """
    Download the Heart Disease UCI dataset from the UCI ML Repository.

    Args:
        output_dir (str): Directory to save the downloaded data
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # UCI Heart Disease Dataset URLs
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"

    # Dataset files
    files = {
        "processed.cleveland.data": "processed.cleveland.data",
        "processed.hungarian.data": "processed.hungarian.data",
        "processed.switzerland.data": "processed.switzerland.data",
        "processed.va.data": "processed.va.data"
    }

    # Column names for the dataset
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    print("Starting Heart Disease dataset download...")
    print(f"Output directory: {output_path.absolute()}")

    # Download and combine datasets
    all_data = []

    for filename, save_name in files.items():
        url = base_url + filename
        save_path = output_path / save_name

        try:
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, save_path)
            print(f"Saved to {save_path}")

            # Read the data
            df = pd.read_csv(save_path, names=column_names, na_values="?")
            all_data.append(df)

        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            continue

    if all_data:
        # Combine all datasets
        combined_df = pd.concat(all_data, ignore_index=True)

        # Save combined dataset
        combined_path = output_path / "heart_disease_combined.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"\nCombined dataset saved to {combined_path}")
        print(f"Total records: {len(combined_df)}")
        print(f"Total features: {len(combined_df.columns)}")

        # Display basic info
        print("\nDataset Info:")
        print(f"Shape: {combined_df.shape}")
        print(f"Missing values: {combined_df.isnull().sum().sum()}")
        print(f"Target distribution:\n{combined_df['target'].value_counts().sort_index()}")

        return combined_path
    else:
        print("Failed to download any datasets")
        return None


def download_from_kaggle(output_dir="data/raw"):
    """
    Alternative method: Download from a preprocessed source or Kaggle.
    This is a backup method if UCI repository is unavailable.

    Args:
        output_dir (str): Directory to save the downloaded data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Alternative URL for preprocessed heart disease data
    url = "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"
    save_path = output_path / "heart_disease.csv"

    try:
        print("Downloading heart disease dataset from alternative source...")
        urllib.request.urlretrieve(url, save_path)
        print(f"Dataset saved to {save_path}")

        df = pd.read_csv(save_path)
        print("\nDataset Info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        return save_path

    except Exception as e:
        print(f"Error downloading from alternative source: {str(e)}")
        return None


if __name__ == "__main__":
    # Try primary source first
    result = download_heart_disease_data()

    # If primary fails, try alternative
    if result is None:
        print("\nTrying alternative download source...")
        result = download_from_kaggle()

    if result:
        print("\nData download completed successfully!")
        sys.exit(0)
    else:
        print("\nData download failed!")
        sys.exit(1)
