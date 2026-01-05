# Feature Store CI/CD Integration Guide with DVC

## Overview

This guide explains how to leverage Azure ML Feature Store with GitHub Actions CI/CD and DVC (Data Version Control) for the heart disease prediction project.

**Why DVC?** We use DVC to track data versions using hashes instead of storing large data files in Git. This ensures reproducibility, data integrity, and efficient version control.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   GitHub Repository (Git)                    │
│  • Source code                                              │
│  • Feature specs                                            │
│  • .dvc files (data hash pointers)                         │
│  • DVC pipeline definitions                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓ dvc pull/push
┌─────────────────────────────────────────────────────────────┐
│              Azure Blob Storage (DVC Remote)                 │
│  • Raw data files (versioned by hash)                      │
│  • Processed features (versioned by hash)                  │
│  • Cached pipeline outputs                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓ CI/CD Pipeline
┌─────────────────────────────────────────────────────────────┐
│                     GitHub Actions CI/CD                     │
│                                                             │
│  ├─── DVC Setup & Data Pull                                │
│  │    └─► Download data from Azure Blob                    │
│  │    └─► Verify data hashes                               │
│  │                                                          │
│  ├─── Feature Validation                                   │
│  │    └─► Schema checks                                    │
│  │    └─► Data quality tests                               │
│  │                                                          │
│  ├─── Feature Engineering                                  │
│  │    └─► DVC pipeline execution                           │
│  │    └─► Transform raw data                               │
│  │    └─► Generate features                                │
│  │                                                          │
│  ├─── Feature Store Deployment                             │
│  │    └─► Register features with DVC metadata              │
│  │    └─► Track data lineage (DVC hash → Feature Store)    │
│  │                                                          │
│  └─── DVC Push                                             │
│       └─► Upload processed data to Azure Blob              │
│       └─► Update .dvc files in Git                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Azure ML Feature Store                          │
│  • Feature sets with DVC lineage metadata                  │
│  • Links to DVC-tracked data                               │
└─────────────────────────────────────────────────────────────┘
```

## Workflow Files

### 1. Feature Store CI/CD Pipeline with DVC
**File**: `.github/workflows/feature-store-cicd.yml`

**Triggers**:
- Push to `main` or `develop` branches
- Changes to `src/data/**`, `src/features/**`, `feature_specs/**`, `*.dvc`, `dvc.yaml`
- Manual workflow dispatch

**Jobs**:
1. **DVC Setup**: Configure DVC and pull data from Azure Blob
2. **Feature Validation**: Validates feature specifications and data integrity (using DVC hashes)
3. **Feature Engineering**: Run DVC pipeline to transform data and create features
4. **Feature Store Deployment**: Registers features to Azure ML Feature Store with DVC metadata
5. **DVC Push**: Upload processed data back to Azure Blob
6. **Feature Monitoring**: Sets up drift detection and quality monitoring

**Key DVC Integration Steps**:
```yaml
- name: Setup DVC
  uses: iterative/setup-dvc@v1

- name: Configure DVC Azure Remote
  env:
    AZURE_STORAGE_CONNECTION_STRING: ${{ secrets.AZURE_STORAGE_CONNECTION_STRING }}
  run: |
    dvc remote modify azure-storage connection_string "$AZURE_STORAGE_CONNECTION_STRING"

- name: Pull data from DVC remote
  run: dvc pull
  # ✓ Downloads data from Azure Blob
  # ✓ Verifies MD5 hashes automatically
  # ✓ Fails if data corrupted

- name: Run DVC pipeline
  run: dvc repro
  # ✓ Executes only changed stages
  # ✓ Caches unchanged outputs
  # ✓ Reproducible pipeline

- name: Push processed data to DVC remote
  run: dvc push
```

### 2. Model Training Pipeline
**File**: `.github/workflows/model-training-cicd.yml`

**Triggers**:
- Push to `main` branch
- Changes to `src/models/**`, `src/training/**`
- Manual workflow dispatch

**Jobs**:
1. **Retrieve Features**: Fetches features from Feature Store
2. **Train Model**: Trains ML model with versioned features
3. **Register Model**: Registers model with feature lineage
4. **Deploy Model**: Deploys model to production endpoint

## DVC Setup

### Prerequisites
1. **DVC installed locally** (for development):
   ```bash
   pip install dvc[azure]
   ```

2. **Azure Blob Storage** for DVC remote (created via Terraform or manually)

### Initialize DVC in Your Repository

```bash
# Initialize DVC
dvc init

# Add Azure Blob as remote storage
dvc remote add -d azure-storage azure://dvc-container/heart-disease-project

# Configure DVC to use Azure authentication
dvc remote modify azure-storage account_name <your-storage-account>
```

### Track Data with DVC

```bash
# Track raw data
dvc add data/raw/heart_disease_combined.csv
# This creates: data/raw/heart_disease_combined.csv.dvc

# Track processed data
dvc add data/processed/heart_disease_clean.csv
# This creates: data/processed/heart_disease_clean.csv.dvc

# Commit .dvc files to Git (NOT the actual data)
git add data/raw/.gitignore data/raw/*.dvc
git add data/processed/.gitignore data/processed/*.dvc
git commit -m "Track data with DVC"

# Push data to Azure Blob
dvc push
```

### DVC Pipeline Definition

Create `dvc.yaml` in the project root:

```yaml
stages:
  download_data:
    cmd: python src/data/download_data.py
    deps:
      - src/data/download_data.py
    outs:
      - data/raw/heart_disease_combined.csv
  
  engineer_features:
    cmd: python src/features/engineer_features.py --input-data data/raw/heart_disease_combined.csv --output-path data/processed/features.parquet
    deps:
      - src/features/engineer_features.py
      - data/raw/heart_disease_combined.csv
    outs:
      - data/processed/features.parquet
    metrics:
      - metrics/feature_stats.json:
          cache: false
```

## Required GitHub Secrets

Add these secrets to your GitHub repository:

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `AZURE_CLIENT_ID` | Service Principal Client ID | `12345678-1234-...` |
| `AZURE_CLIENT_SECRET` | Service Principal Secret | `abcd1234...` |
| `AZURE_SUBSCRIPTION_ID` | Azure Subscription ID | `87654321-4321-...` |
| `AZURE_TENANT_ID` | Azure Tenant ID | `11111111-1111-...` |
| `AZURE_RESOURCE_GROUP` | Resource Group Name | `mlops-heart-disease-rg` |
| `FEATURE_STORE_NAME_DEV` | Dev Feature Store Name | `heart-disease-fs-dev` |
| `FEATURE_STORE_NAME_PROD` | Prod Feature Store Name | `heart-disease-fs-prod` |
| `AZURE_ML_WORKSPACE` | Azure ML Workspace Name | `mlops-heart-disease-ws` |
| `AZURE_STORAGE_ACCOUNT` | Storage Account for DVC | `mlopsdvcdata` |
| `AZURE_STORAGE_CONNECTION_STRING` | Connection string for DVC remote | `DefaultEndpointsProtocol=https;...` |

## Environment Strategy

### Development Environment
- **Branch**: `develop`
- **Feature Store**: `heart-disease-fs-dev`
- **Purpose**: Test feature engineering changes
- **Automatic Deployment**: Yes

### Production Environment
- **Branch**: `main`
- **Feature Store**: `heart-disease-fs-prod`
- **Purpose**: Production features for model training/inference
- **Automatic Deployment**: Yes (after validation)

## Feature Versioning Strategy

### Semantic Versioning
- **Major (x.0.0)**: Breaking changes to feature schema
- **Minor (1.x.0)**: New features added (backward compatible)
- **Patch (1.0.x)**: Bug fixes, data quality improvements

### Auto-versioning in CI/CD
```yaml
- name: Register Feature Set
  run: |
    python src/features/deploy_feature_store.py \
      --version-increment auto  # Automatically increments version
```

## Feature Store Operations with DVC

### 1. Creating a New Feature Set

**Step 1**: Define feature specification
```yaml
# feature_specs/new_feature_set.yaml
name: new_feature_set
version: "1.0"
description: Description of features

features:
  - name: feature_1
    type: float
    description: Feature description
```

**Step 2**: Implement feature engineering
```python
# src/features/engineer_features.py
def _engineer_new_features(self, df: pd.DataFrame):
    # Add feature engineering logic
    df['new_feature'] = ...
    return df
```

**Step 3**: Update DVC pipeline (if needed)
```yaml
# dvc.yaml
stages:
  new_feature_stage:
    cmd: python src/features/engineer_features.py --feature-type new
    deps:
      - src/features/engineer_features.py
      - data/processed/base_features.parquet
    outs:
      - data/processed/new_features.parquet
```

**Step 4**: Test locally with DVC
```bash
# Run the pipeline locally
dvc repro

# Track new output with DVC
dvc add data/processed/new_features.parquet

# Push data to Azure Blob
dvc push
```

**Step 5**: Commit and push to GitHub
```bash
git add feature_specs/new_feature_set.yaml src/features/ dvc.yaml data/processed/new_features.parquet.dvc
git commit -m "Add new feature set with DVC tracking"
git push origin develop
```

**Step 6**: CI/CD automatically:
- Pulls data from DVC remote (Azure Blob)
- Validates DVC hashes for data integrity
- Validates the feature specification
- Runs DVC pipeline to engineer features
- Registers to dev Feature Store with DVC metadata
- Pushes processed data back to Azure Blob
- Runs quality checks

### 2. Updating Existing Features

**Backward Compatible Changes** (Minor version):
- Adding new features
- Improving feature quality
- Bug fixes

**Breaking Changes** (Major version):
- Removing features
- Changing feature types
- Changing feature definitions

### 3. Feature Retrieval in Training

**With DVC Lineage Tracking**:
```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Initialize client
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="...",
    resource_group_name="...",
    workspace_name="heart-disease-fs-prod"
)

# Retrieve features from Feature Store
features = ml_client.feature_sets.get(
    name="cardiac_indicators",
    version="latest"  # Or specific version like "1.2.0"
)

# Get feature data
feature_data = features.to_pandas()

# Access DVC metadata for full reproducibility
dvc_metadata = features.tags.get('dvc_hash')
source_data_hash = features.tags.get('source_dvc_hash')

# Full lineage tracking
print(f"Feature DVC hash: {dvc_metadata}")
print(f"Source data DVC hash: {source_data_hash}")
# Can reproduce exact training with: git checkout <commit> && dvc checkout
```

**Reproduce Training from DVC**:
```bash
# Checkout specific commit where model was trained
git checkout <training-commit-hash>

# Get exact data version from that commit
dvc checkout
dvc pull

# Data is now identical to training time
python train.py  # Reproducible training
```

## Monitoring and Alerts

### Feature Drift Detection
- **Schedule**: Every 6 hours (configurable)
- **Metrics**: Distribution changes, statistical drift
- **Action**: Alert on significant drift

### Data Quality Checks
- **Frequency**: On every feature registration
- **Checks**:
  - Missing value percentage
  - Value range validation
  - Distribution consistency
  - Schema compliance

### Lineage Tracking
- Automatic tracking of:
  - Source data → Feature engineering → Feature sets
  - Feature sets → Models
  - Models → Deployments

## Best Practices

### 1. Feature Organization
```
feature_specs/
├── cardiac_features.yaml       # Cardiac indicators
├── clinical_features.yaml      # Clinical tests
├── demographic_features.yaml   # Patient demographics
└── derived_features.yaml       # Engineered features
```

### 2. Testing Strategy
- **Unit Tests**: Test feature engineering functions
- **Integration Tests**: Test Feature Store operations
- **Data Tests**: Validate feature distributions

### 3. Documentation
- Document each feature in YAML specs
- Include validation rules
- Specify feature dependencies

### 4. Rollback Strategy
```bash
# Rollback to previous feature version
python src/features/deploy_feature_store.py \
  --feature-set-name cardiac_indicators \
  --version 1.1.0  # Previous stable version
```

## Troubleshooting

### Common Issues

**Issue**: Feature registration fails
```
Solution: Check feature specification YAML syntax and validation rules
```

**Issue**: Schema mismatch
```
Solution: Increment major version for breaking changes
```

**Issue**: Authentication fails
```
Solution: Verify GitHub secrets are correctly set
```

## CI/CD Pipeline Metrics

Track these metrics in your CI/CD pipeline:
- Feature registration success rate
- Average feature engineering time
- Data quality score
- Feature drift incidents
- Model training time with features

## DVC Workflow Examples

### Example 1: Developer Updates Features Locally

```bash
# 1. Clone repo and get data
git clone <repo-url>
cd mlops-heart-disease-prediction
dvc pull  # Downloads data from Azure Blob using .dvc files

# 2. Create feature branch
git checkout -b feature/new-cardiac-metric

# 3. Modify feature engineering
vim src/features/engineer_features.py

# 4. Run DVC pipeline locally
dvc repro  # Only reruns changed stages
# Output: data/processed/features.parquet updated

# 5. Track updated data with DVC
dvc add data/processed/features.parquet
# Creates/updates: data/processed/features.parquet.dvc (hash pointer)

# 6. Push data to Azure Blob
dvc push

# 7. Commit code + .dvc file (NOT the actual data)
git add src/features/ data/processed/features.parquet.dvc dvc.lock
git commit -m "Add new cardiac metric"
git push origin feature/new-cardiac-metric

# 8. Create PR to develop
```

### Example 2: CI/CD Pipeline Execution

```bash
# When PR is created, GitHub Actions runs:

# 1. Setup DVC
- uses: iterative/setup-dvc@v1

# 2. Pull data from Azure Blob (using .dvc files)
- run: dvc pull
  # ✓ Downloads data based on hash from .dvc files
  # ✓ Verifies integrity automatically

# 3. Run pipeline
- run: dvc repro
  # ✓ Executes feature engineering
  # ✓ Uses cached results for unchanged stages

# 4. Validate features
- run: pytest tests/test_features.py

# 5. Register to Feature Store (with DVC metadata)
- run: python src/features/deploy_feature_store.py

# 6. Push processed data back to Azure Blob
- run: dvc push
```

### Example 3: Reproduce Training from Any Commit

```bash
# Get exact environment from 2 weeks ago
git checkout abc123def  # Commit from 2 weeks ago

# Get exact data version from that time
dvc checkout  # Updates working directory to match .dvc files
dvc pull      # Downloads data from Azure Blob

# Data and code now match exactly from 2 weeks ago
python src/training/train_model.py  # Reproducible!
```

### Example 4: Complete Feature Store Deployment

```bash
# 1. Developer creates new feature locally
git checkout -b feature/new-cardiac-metric
# ... edit src/features/engineer_features.py
dvc repro
dvc add data/processed/features.parquet
dvc push
git add src/ data/processed/*.dvc
git commit -m "Add new cardiac metric"
git push origin feature/new-cardiac-metric

# 2. Create PR to develop
# GitHub Actions runs:
#   - DVC pull (get data from Azure Blob)
#   - Feature validation
#   - DVC pipeline execution
#   - Unit tests
#   - DVC push (update Azure Blob)

# 3. Merge to develop
# GitHub Actions automatically:
#   - DVC pull
#   - Engineers features (dvc repro)
#   - Registers to dev Feature Store with DVC hash metadata
#   - DVC push
#   - Runs quality checks

# 4. Test in dev environment
# Validate feature quality and model performance

# 5. Merge to main
# GitHub Actions automatically:
#   - DVC pull from prod branch
#   - Deploys to prod Feature Store
#   - Tags features with DVC hashes for lineage
#   - Triggers model retraining with versioned data
#   - Updates production endpoint
#   - DVC push
```

## DVC Commands Reference

### Essential DVC Commands

```bash
# Initialize DVC in repository
dvc init

# Track data files
dvc add data/raw/dataset.csv

# Pull data from remote storage
dvc pull

# Push data to remote storage
dvc push

# Run/reproduce pipeline
dvc repro

# Check pipeline status
dvc status

# Show pipeline DAG
dvc dag

# Checkout specific data version
dvc checkout

# List tracked files
dvc list . data/

# Get file from remote without downloading
dvc get <repo-url> data/file.csv

# Import data from another DVC repo
dvc import <repo-url> data/file.csv
```

### DVC with Azure Blob Storage

```bash
# Add Azure Blob remote
dvc remote add -d myremote azure://container-name/path

# Configure authentication
dvc remote modify myremote account_name <account-name>
dvc remote modify myremote connection_string "<connection-string>"

# Or use environment variables
export AZURE_STORAGE_CONNECTION_STRING="..."
```

## Troubleshooting

### Common DVC Issues

**Issue**: `dvc pull` fails with authentication error
```bash
Solution: Check AZURE_STORAGE_CONNECTION_STRING is set correctly
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;..."
```

**Issue**: Data hash mismatch
```bash
Solution: Data file was modified. Either:
1. Revert changes: dvc checkout data/file.csv
2. Or update hash: dvc add data/file.csv && dvc push
```

**Issue**: `dvc repro` doesn't detect changes
```bash
Solution: Force pipeline run:
dvc repro --force
```

**Issue**: Large data file is slow to push/pull
```bash
Solution: Enable caching and use multiple threads:
dvc remote modify azure-storage jobs 4
```

## Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [DVC with Azure](https://dvc.org/doc/user-guide/data-management/remote-storage/azure-blob-storage)
- [Azure ML Feature Store Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/concept-what-is-managed-feature-store)
- [GitHub Actions for Azure](https://github.com/Azure/actions)
- [DVC GitHub Action](https://github.com/iterative/setup-dvc)
- [Feature Store Best Practices](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-feature-store)
