# MLOps Experiments 4, 5, and 6: Walmart Sales Forecasting

This repository contains production-grade implementations for **Experiment 4**, **Experiment 5**, and **Experiment 6** built for the Walmart Weekly Sales Forecasting problem.

---

## Quick Overview of Experiments

| Experiment | Title | Core Technologies | Primary Artifacts |
| :--- | :--- | :--- | :--- |
| **Experiment 4** | **Experiment Tracking & Reproducibility Workflows** | MLflow, Scikit-Learn | Parameter & metric logging, SHA-256 data lineage, residual plots, preprocessors, multi-run comparison table, deterministic seed check |
| **Experiment 5** | **Data Validation & Reproducible Pipelines** | Pandera, Scikit-Learn | Schema checks, corrupt record detection, quarantine pipeline, custom `DateFeatureExtractor`, zero-drift idempotency verification |
| **Experiment 6** | **Model Registry & Version Management** | MLflow Registry, Scikit-Learn | Model versioning, `Staging` / `Production` stages, automated promotion criteria, deployment smoke test, automated rollback, lineage trace |

---

## Repository Structure

```text
mokshmlops/
├── Walmart.csv                                      # Source dataset
├── requirements.txt                                 # Pinned dependencies (MLflow, Pandera, Scikit-Learn, etc.)
├── run_all_experiments.py                           # Master orchestrator script (runs Exp 4, 5, 6 in sequence)
├── mlflow.db                                        # Local SQLite database for MLflow tracking & Model Registry
│
├── src/                                             # Reusable core modules
│   ├── __init__.py
│   ├── data_validation.py                           # Pandera schemas, anomaly detection, quarantine logic
│   ├── pipeline.py                                  # DateFeatureExtractor, ColumnTransformer, idempotency test
│   └── registry_manager.py                          # Model Registry promotion, rollback, smoke testing
│
├── experiments/                                     # Standalone executable Python scripts
│   ├── exp4_tracking_reproducibility.py             # Script for Experiment 4
│   ├── exp5_data_validation_pipeline.py             # Script for Experiment 5
│   └── exp6_model_registry_lifecycle.py             # Script for Experiment 6
│
└── notebooks/                                       # Interactive Jupyter Notebooks for VS Code
    ├── Experiment_4_Tracking_and_Reproducibility.ipynb
    ├── Experiment_5_Data_Validation_and_Pipelines.ipynb
    └── Experiment_6_Model_Registry_and_Lifecycle.ipynb
```

---

## How to Execute in Visual Studio Code

### 1. Select the Virtual Environment in VS Code
1. Open this repository folder in VS Code (`File` -> `Open Folder...` -> `mokshmlops`).
2. Press `Ctrl + Shift + P` (or `Cmd + Shift + P` on macOS) and type:
   ```text
   Python: Select Interpreter
   ```
3. Choose the virtual environment located at:
   ```text
   .\.venv\Scripts\python.exe
   ```

---

### 2. Option A: Run Scripts Directly in VS Code Terminal

Open the integrated terminal in VS Code (`Ctrl + ~`) and execute:

#### Run All 3 Experiments Sequentially (Recommended):
```powershell
.\.venv\Scripts\python.exe run_all_experiments.py
```

#### Or Run Each Experiment Individually:
- **Experiment 4**:
  ```powershell
  .\.venv\Scripts\python.exe experiments/exp4_tracking_reproducibility.py
  ```
- **Experiment 5**:
  ```powershell
  .\.venv\Scripts\python.exe experiments/exp5_data_validation_pipeline.py
  ```
- **Experiment 6**:
  ```powershell
  .\.venv\Scripts\python.exe experiments/exp6_model_registry_lifecycle.py
  ```

---

### 3. Option B: Run Interactive Jupyter Notebooks in VS Code

1. In the VS Code file explorer, open any notebook in the `notebooks/` folder:
   - `notebooks/Experiment_4_Tracking_and_Reproducibility.ipynb`
   - `notebooks/Experiment_5_Data_Validation_and_Pipelines.ipynb`
   - `notebooks/Experiment_6_Model_Registry_and_Lifecycle.ipynb`
2. At the top-right corner of the notebook editor, click **Select Kernel**.
3. Select **Python Environments...** -> Choose **Python 3.11 (.venv)**.
4. Click **Run All** or execute cells individually with `Shift + Enter`.

---

### 4. How to Launch and View the MLflow Dashboard UI

To inspect all recorded runs, parameter charts, metrics comparison, artifacts, and registered model versions:

1. In the VS Code terminal, run:
   ```powershell
   .\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db
   ```
2. Open your web browser and navigate to:
   ```text
   http://localhost:5000
   ```
3. Explore:
   - **Experiments Tab**: View and compare runs for `Experiment_4_Tracking_and_Reproducibility` and `Experiment_6_Model_Registry_Lifecycle`.
   - **Artifacts**: Inspect logged feature plots, residual distributions, and serialized models.
   - **Models Tab**: View the registered `WalmartSalesForecastingModel`, its version history, `Production` and `Staging` stages, and aliases (`champion`, `challenger`).
