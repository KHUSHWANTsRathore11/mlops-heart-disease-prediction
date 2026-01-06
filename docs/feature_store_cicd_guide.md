# DVC-Based Feature Store & CI/CD Guide

## Overview

This guide explains how to use **DVC (Data Version Control)** as a lightweight Feature Store. By abandoning complex proprietary feature stores, we leverage DVC's versioning capabilities to treat data exactly like code.

**Philosophy:** "Data as Code." Features are versioned, reproducible, and tied directly to the Git commit hash.

## Architecture

We use DVC to track feature files (Parquet/CSV) and store them in Azure Blob Storage. Git tracks the specific versions (hashes) of these files.

```
+-------------------------------------------------------------+
|                  GitHub Repository (Git)                    |
|  - Source code (feature engineering logic)                  |
|  - dvc.yaml (pipeline definition)                           |
|  - dvc.lock (exact version snapshot)                        |
|  - .dvc files (pointers to blob storage)                    |
+-------------------------------------------------------------+
                              |
                              v (dvc pull / push)
+-------------------------------------------------------------+
|             Azure Blob Storage (DVC Remote)                 |
|  - Raw datasets (versioned by hash)                         |
|  - Processed feature sets (versioned by hash)               |
|  - Implementation Agnostic Storage                          |
+-------------------------------------------------------------+
```

## CI/CD Workflow

**File**: `.github/workflows/feature-store-cicd.yml`

This pipeline ensures that every change to the feature engineering code produces valid, tested, and versioned feature data.

### Pipeline Steps

1.  **Checkout & Setup**:
    *   Clones the repository.
    *   Installs Python dependencies and DVC.
    *   Configures Azure credentials for DVC remote access.

2.  **DVC Pull**:
    *   Downloads the *required* data versions from Azure Blob.
    *   Ensures the environment has the data corresponding to the current `dvc.lock`.

3.  **Reproduction (Feature Engineering)**:
    *   Runs `dvc repro`.
    *   DVC checks if inputs (code or raw data) have changed.
    *   **If changed**: Re-runs the feature engineering script.
    *   **If unchanged**: Skips execution (using cache).

4.  **Integration & Logic Tests**:
    *   Runs `pytest` to verify the generated feature data (schema, distributions, null checks).

5.  **DVC Push** (On Merge/Deploy):
    *   Uploads the newly generated feature data to Azure Blob.
    *   This effectively "publishes" the new feature version.

## How to Use "DVC as a Feature Store"

### 1. Consuming Features (Training)

To use features in a training script or another project, you simply "checkout" the data version you need.

**Option A: Same Repository (Monorepo)**
```bash
# Get the exact data version tied to this commit
dvc pull

# Load data in Python
import pandas as pd
df = pd.read_parquet("data/processed/features.parquet")
```

**Option B: External Consumption (DVC Import)**
If your training code is in a different repo, use `dvc get` or `dvc import` to fetch features from this "Feature Store" repo.

```bash
# Download features from the 'main' branch of the Feature Store repo
dvc get https://github.com/YourOrg/mlops-feature-store \
    data/processed/features.parquet

# Or track it (so you can update it later)
dvc import https://github.com/YourOrg/mlops-feature-store \
    data/processed/features.parquet
```

### 2. Versioning Features

*   **Feature Version** = **Git Commit Hash**.
*   Since `dvc.lock` is committed to Git, every Git commit corresponds to an immutable snapshot of the data.
*   **Tagging**: You can use Git Tags (e.g., `v1.0.0`) to mark specific stable versions of features.

### 3. Creating/Updating Features

1.  **Modify Code**: Edit `src/features/engineer_features.py`.
2.  **Run Pipeline**:
    ```bash
    dvc repro
    ```
    This updates `data/processed/features.parquet` and `dvc.lock`.
3.  **Validate**:
    ```bash
    pytest tests/
    ```
4.  **Commit & Share**:
    ```bash
    git add dvc.lock src/features/engineer_features.py
    git commit -m "Add new rolling_window feature"
    dvc push  # Uploads the new data blob
    git push
    ```

## Advantages of this Approach

1.  **Simplicity**: No complex Feature Store service to maintain (features are files).
2.  **Cost**: Only paying for Blob Storage (cheap).
3.  **Reproducibility**: Strongest possible guarantee. Code + Data are locked together in `dvc.lock`.
4.  **Framework Agnostic**: The output is just a file (Parquet/CSV). Can be read by Pandas, Polars, Spark, Databricks, etc.

## Troubleshooting

*   **Permission Denied**: Ensure `AZURE_CLIENT_ID` (Service Principal) has `Storage Blob Data Contributor` role on the storage account.
*   **Data Missing**: Run `dvc pull` to fetch the blobs referenced in your current `.dvc` files.
*   **Lock File Conflict**: If `dvc.lock` has merge conflicts, run `dvc repro` to regenerate it based on the current state.
