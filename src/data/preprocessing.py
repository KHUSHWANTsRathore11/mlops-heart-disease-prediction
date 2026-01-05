"""
Data preprocessing utilities for Heart Disease dataset.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path


class HeartDiseasePreprocessor:
    """
    Preprocessor for Heart Disease dataset with fit/transform interface.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.numerical_features = []
        self.categorical_features = []
        
    def fit(self, df, target_col='target'):
        """
        Fit the preprocessor on training data.
        
        Args:
            df (pd.DataFrame): Training dataframe
            target_col (str): Name of target column
        """
        # Store feature names
        self.feature_names = [col for col in df.columns if col != target_col]
        
        # Identify numerical and categorical features
        self.numerical_features = df[self.feature_names].select_dtypes(
            include=[np.number]
        ).columns.tolist()
        
        self.categorical_features = df[self.feature_names].select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        # Fit scaler on numerical features
        if self.numerical_features:
            self.scaler.fit(df[self.numerical_features])
        
        # Fit label encoders on categorical features
        for col in self.categorical_features:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le
        
        return self
    
    def transform(self, df, target_col='target'):
        """
        Transform data using fitted preprocessor.
        
        Args:
            df (pd.DataFrame): Dataframe to transform
            target_col (str): Name of target column
            
        Returns:
            pd.DataFrame: Transformed dataframe
        """
        df_copy = df.copy()
        
        # Scale numerical features
        if self.numerical_features:
            df_copy[self.numerical_features] = self.scaler.transform(
                df_copy[self.numerical_features]
            )
        
        # Encode categorical features
        for col in self.categorical_features:
            if col in self.label_encoders:
                df_copy[col] = self.label_encoders[col].transform(
                    df_copy[col].astype(str)
                )
        
        return df_copy
    
    def fit_transform(self, df, target_col='target'):
        """
        Fit and transform in one step.
        
        Args:
            df (pd.DataFrame): Dataframe to fit and transform
            target_col (str): Name of target column
            
        Returns:
            pd.DataFrame: Transformed dataframe
        """
        self.fit(df, target_col)
        return self.transform(df, target_col)
    
    def save(self, filepath):
        """
        Save the preprocessor to disk.
        
        Args:
            filepath (str): Path to save the preprocessor
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """
        Load a preprocessor from disk.
        
        Args:
            filepath (str): Path to the saved preprocessor
            
        Returns:
            HeartDiseasePreprocessor: Loaded preprocessor
        """
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor


def clean_data(df, target_col='target'):
    """
    Clean the raw heart disease dataset.
    
    Args:
        df (pd.DataFrame): Raw dataframe
        target_col (str): Name of target column
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("Starting data cleaning...")
    print(f"Initial shape: {df.shape}")
    print(f"Initial missing values: {df.isnull().sum().sum()}")
    
    df_clean = df.copy()
    
    # Convert target to binary (0: no disease, 1: disease present)
    if target_col in df_clean.columns:
        df_clean[target_col] = (df_clean[target_col] > 0).astype(int)
    
    # Add dummy timestamp and index for Feature Store compatibility
    if 'timestamp' not in df_clean.columns:
        df_clean['timestamp'] = pd.Timestamp.now()
    if 'patient_id' not in df_clean.columns:
        df_clean['patient_id'] = range(1, len(df_clean) + 1)
    
    # Handle missing values
    # For numerical columns, fill with median
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"Filled {col} missing values with median: {median_val}")
    
    # For categorical columns, fill with mode
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_val, inplace=True)
            print(f"Filled {col} missing values with mode: {mode_val}")
    
    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    removed_duplicates = initial_rows - len(df_clean)
    if removed_duplicates > 0:
        print(f"Removed {removed_duplicates} duplicate rows")
    
    print(f"Final shape: {df_clean.shape}")
    print(f"Final missing values: {df_clean.isnull().sum().sum()}")
    print("Data cleaning completed")
    
    return df_clean


def prepare_train_test_split(df, target_col='target', test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        target_col (str): Name of target column
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print(f"\nSplitting data into train and test sets...")
    print(f"Test size: {test_size * 100}%")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Training target distribution:\n{y_train.value_counts()}")
    print(f"Test target distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Load data
    data_path = Path("data/raw/heart_disease_combined.csv")
    if not data_path.exists():
        data_path = Path("data/raw/heart_disease.csv")
    
    if data_path.exists():
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Clean data
        df_clean = clean_data(df)
        
        # Save cleaned data
        output_path = Path("data/processed/heart_disease_clean.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        print(f"\nCleaned data saved to {output_path}")
        
    else:
        print(f"Data file not found at {data_path}")
        print("Please run download_data.py first")
        sys.exit(1)
