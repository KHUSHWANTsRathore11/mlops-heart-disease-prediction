"""
Tests for data loading and validation.
"""
import pandas as pd


class TestDataLoading:
    """Test cases for data loading functionality"""

    def test_raw_data_exists(self, data_paths):
        """Test that raw data file exists"""
        assert data_paths['raw_data'].exists(), f"Raw data not found at {data_paths['raw_data']}"

    def test_raw_data_loads(self, data_paths):
        """Test that raw data can be loaded"""
        if data_paths['raw_data'].exists():
            df = pd.read_csv(data_paths['raw_data'])
            assert df is not None
            assert len(df) > 0

    def test_raw_data_schema(self, data_paths):
        """Test that raw data has expected columns"""
        if data_paths['raw_data'].exists():
            df = pd.read_csv(data_paths['raw_data'])
            expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'target']
            for col in expected_columns:
                assert col in df.columns, f"Expected column '{col}' not found"


class TestDataValidation:
    """Test cases for data validation"""

    def test_no_missing_target(self, sample_heart_data):
        """Test that target column has no missing values"""
        assert sample_heart_data['target'].isnull().sum() == 0

    def test_target_is_binary(self, sample_heart_data):
        """Test that target is binary (0 or 1)"""
        unique_values = sample_heart_data['target'].unique()
        assert set(unique_values).issubset({0, 1})

    def test_age_range(self, sample_heart_data):
        """Test that age values are in reasonable range"""
        assert sample_heart_data['age'].min() >= 0
        assert sample_heart_data['age'].max() <= 150

    def test_data_not_empty(self, sample_heart_data):
        """Test that data has rows"""
        assert len(sample_heart_data) > 0

    def test_data_has_features(self, sample_heart_data):
        """Test that data has feature columns"""
        assert len(sample_heart_data.columns) > 1


class TestDataSplitting:
    """Test cases for train/test splitting"""

    def test_split_maintains_size(self, sample_heart_data):
        """Test that splitting doesn't lose data"""
        from sklearn.model_selection import train_test_split

        X = sample_heart_data.drop('target', axis=1)
        y = sample_heart_data['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        total = len(X_train) + len(X_test)
        assert total == len(sample_heart_data)

    def test_split_ratio(self, sample_heart_data):
        """Test that test/train split ratio is correct"""
        from sklearn.model_selection import train_test_split

        X = sample_heart_data.drop('target', axis=1)
        y = sample_heart_data['target']

        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        actual_ratio = len(X_test) / (len(X_train) + len(X_test))
        assert abs(actual_ratio - test_size) < 0.02  # Within 2%


class TestProcessedData:
    """Test cases for processed/engineered features"""

    def test_processed_features_exist(self, data_paths):
        """Test that processed feature files exist"""
        assert data_paths['train_features'].exists(), "Training features not found"
        assert data_paths['test_features'].exists(), "Test features not found"

    def test_processed_features_have_same_columns(self, data_paths):
        """Test that train and test have same columns"""
        if data_paths['train_features'].exists() and data_paths['test_features'].exists():
            train_df = pd.read_csv(data_paths['train_features'])
            test_df = pd.read_csv(data_paths['test_features'])

            assert set(train_df.columns) == set(test_df.columns)

    def test_features_are_scaled(self, data_paths):
        """Test that numerical features appear to be scaled"""
        if data_paths['train_features'].exists():
            train_df = pd.read_csv(data_paths['train_features'])

            # Scaled features should have values roughly in range of -3 to 3
            numerical_cols = ['age', 'trestbps', 'chol', 'thalach']
            for col in numerical_cols:
                if col in train_df.columns:
                    assert train_df[col].min() > -10, f"{col} might not be scaled"
                    assert train_df[col].max() < 10, f"{col} might not be scaled"
