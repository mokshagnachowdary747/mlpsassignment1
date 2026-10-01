"""
Experiment 4: Experiment Tracking and Reproducibility Workflows
---------------------------------------------------------------
AIM:
Implement ML experiment tracking and reproducibility workflows using MLflow
by recording parameters, metrics, preprocessing configurations, datasets, and
trained artifacts. Compare multiple experiment runs, analyze the effects of
preprocessing, hyperparameter tuning, and feature selection, and maintain
experiment lineage and reproducibility across repeated executions.
"""

import os
import sys

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import hashlib
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path to import src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.pipeline import build_full_model_pipeline


def compute_file_hash(filepath: str) -> str:
    """Computes SHA-256 hash of dataset file for lineage tracking."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def evaluate_predictions(y_true, y_pred, prefix="test") -> dict:
    """Computes RMSE, MAE, and R2 evaluation metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        f"{prefix}_rmse": rmse,
        f"{prefix}_mae": mae,
        f"{prefix}_r2": r2
    }


def generate_and_log_plots(y_true, y_pred, model_pipeline, run_name: str, temp_dir: str):
    """Generates and logs performance and residual visualization artifacts."""
    sns.set_theme(style="whitegrid")
    
    # 1. Actual vs Predicted Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true / 1e3, y_pred / 1e3, alpha=0.35, color="royalblue", edgecolors="none")
    min_val = min(y_true.min(), y_pred.min()) / 1e3
    max_val = max(y_true.max(), y_pred.max()) / 1e3
    ax.plot([min_val, max_val], [min_val, max_val], color="crimson", linestyle="--", linewidth=2, label="Ideal Fit (y=x)")
    ax.set_title(f"Actual vs Predicted Weekly Sales (in $K) - {run_name}", fontsize=12)
    ax.set_xlabel("Actual Weekly Sales ($K)", fontsize=11)
    ax.set_ylabel("Predicted Weekly Sales ($K)", fontsize=11)
    ax.legend()
    plot_path = os.path.join(temp_dir, f"{run_name}_actual_vs_predicted.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(plot_path, artifact_path="visualizations")

    # 2. Residual Distribution Plot
    residuals = (y_true - y_pred) / 1e3
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, kde=True, ax=ax, color="darkorange", bins=40)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_title(f"Residuals Distribution ($K) - {run_name}", fontsize=12)
    ax.set_xlabel("Residual (Actual - Predicted) [$K]", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    res_path = os.path.join(temp_dir, f"{run_name}_residuals_distribution.png")
    fig.tight_layout()
    fig.savefig(res_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(res_path, artifact_path="visualizations")


def run_experiment_4():
    print("=" * 80)
    print("EXPERIMENT 4: MLflow Tracking and Reproducibility Workflows")
    print("=" * 80)

    # 1. Setup MLflow Tracking with SQLite Backend
    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "Experiment_4_Tracking_and_Reproducibility"
    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] Backend Tracking URI: {tracking_uri}")
    print(f"[MLflow] Active Experiment: {experiment_name}")

    # 2. Load Dataset and Track Dataset Lineage
    data_path = os.path.join(parent_dir, "Walmart.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    raw_df = pd.read_csv(data_path)
    data_hash = compute_file_hash(data_path)
    print(f"[Data] Loaded {len(raw_df)} records from Walmart.csv")
    print(f"[Data] Dataset SHA-256 Digest: {data_hash[:16]}...{data_hash[-8:]}")

    # Feature and Target Split
    y = raw_df["Weekly_Sales"]
    X = raw_df.drop(columns=["Weekly_Sales"])

    # Fixed seed train-test split for strict reproducibility
    SEED = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )
    print(f"[Split] Train samples: {len(X_train)} | Test samples: {len(X_test)} (Seed={SEED})")

    # 3. Define Candidate Pipeline Runs
    experiment_configs = [
        {
            "run_name": "Run_1_Ridge_Baseline",
            "model_type": "Ridge",
            "regressor": Ridge(alpha=1.0, random_state=SEED),
            "scaler_type": "standard",
            "with_feature_selection": False,
            "k_features": 11,
            "hyperparams": {"alpha": 1.0, "solver": "auto"}
        },
        {
            "run_name": "Run_2_Random_Forest_Default",
            "model_type": "RandomForest",
            "regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1),
            "scaler_type": "standard",
            "with_feature_selection": False,
            "k_features": 11,
            "hyperparams": {"n_estimators": 100, "max_depth": 12, "criterion": "squared_error"}
        },
        {
            "run_name": "Run_3_Random_Forest_Feature_Selection",
            "model_type": "RandomForest",
            "regressor": RandomForestRegressor(n_estimators=150, max_depth=16, random_state=SEED, n_jobs=-1),
            "scaler_type": "robust",
            "with_feature_selection": True,
            "k_features": 8,
            "hyperparams": {"n_estimators": 150, "max_depth": 16, "k_features_selected": 8}
        },
        {
            "run_name": "Run_4_Gradient_Boosting_Tuned",
            "model_type": "GradientBoosting",
            "regressor": GradientBoostingRegressor(n_estimators=140, learning_rate=0.08, max_depth=6, random_state=SEED),
            "scaler_type": "robust",
            "with_feature_selection": True,
            "k_features": 9,
            "hyperparams": {"n_estimators": 140, "learning_rate": 0.08, "max_depth": 6, "k_features_selected": 9}
        }
    ]

    run_summary_records = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for config in experiment_configs:
            run_name = config["run_name"]
            print(f"\n---> Executing Run: {run_name} ...")

            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id

                # Build end-to-end pipeline
                pipeline = build_full_model_pipeline(
                    regressor=config["regressor"],
                    scaler_type=config["scaler_type"],
                    with_feature_selection=config["with_feature_selection"],
                    k_features=config["k_features"]
                )

                # Log Parameters
                mlflow.log_param("model_type", config["model_type"])
                mlflow.log_param("scaler_type", config["scaler_type"])
                mlflow.log_param("with_feature_selection", config["with_feature_selection"])
                mlflow.log_param("k_features", config["k_features"])
                mlflow.log_param("random_seed", SEED)
                mlflow.log_param("test_size", 0.20)
                mlflow.log_params(config["hyperparams"])

                # Log Dataset Lineage Tags
                mlflow.set_tag("dataset_file", "Walmart.csv")
                mlflow.set_tag("dataset_sha256", data_hash)
                mlflow.set_tag("data_total_rows", len(raw_df))
                mlflow.set_tag("experiment_type", "regression_tracking")

                # Train Pipeline
                pipeline.fit(X_train, y_train)

                # Predict & Evaluate
                y_train_pred = pipeline.predict(X_train)
                y_test_pred = pipeline.predict(X_test)

                train_metrics = evaluate_predictions(y_train, y_train_pred, prefix="train")
                test_metrics = evaluate_predictions(y_test, y_test_pred, prefix="test")

                # Log Metrics to MLflow
                mlflow.log_metrics(train_metrics)
                mlflow.log_metrics(test_metrics)

                # Generate and Log Visual Artifacts
                generate_and_log_plots(y_test, y_test_pred, pipeline, run_name, temp_dir)

                # Save & Log Preprocessor Standalone Artifact
                preprocessor_path = os.path.join(temp_dir, f"{run_name}_preprocessor.joblib")
                joblib.dump(pipeline.named_steps["preprocessor"], preprocessor_path)
                mlflow.log_artifact(preprocessor_path, artifact_path="preprocessors")

                # Log Trained Sklearn Pipeline with Signature and Input Example
                input_example = X_train.head(3)
                signature = infer_signature(input_example, y_train.head(3))
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                    serialization_format="cloudpickle"
                )

                print(f"     [Metrics] Test RMSE: ${test_metrics['test_rmse']:,.2f} | Test R2: {test_metrics['test_r2']:.4f} | Test MAE: ${test_metrics['test_mae']:,.2f}")
                print(f"     [Artifacts] Logged model, preprocessor, and plots to run ID: {run_id}")

                run_summary_records.append({
                    "Run Name": run_name,
                    "Model": config["model_type"],
                    "Scaler": config["scaler_type"],
                    "Feature Selection": "Yes" if config["with_feature_selection"] else "No",
                    "Test RMSE": round(test_metrics["test_rmse"], 2),
                    "Test MAE": round(test_metrics["test_mae"], 2),
                    "Test R²": round(test_metrics["test_r2"], 4),
                    "Run ID": run_id
                })

    # 4. Multi-Run Comparison Analysis Table
    summary_df = pd.DataFrame(run_summary_records)
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: MULTI-RUN COMPARISON SUMMARY TABLE")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    best_run = summary_df.sort_values(by="Test RMSE").iloc[0]
    print(f"\n[Analysis] Best Performing Configuration: {best_run['Run Name']}")
    print(f"           Lowest Test RMSE: ${best_run['Test RMSE']:,.2f} | Highest Test R²: {best_run['Test R²']:.4f}")

    # 5. Lineage & Reproducibility Verification
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: REPRODUCIBILITY VALIDATION TEST")
    print("=" * 80)
    print("Executing a second identical training run with fixed Seed=42 to verify deterministic reproducibility...")

    repro_config = experiment_configs[1] # Run_2_Random_Forest_Default
    repro_pipeline_1 = build_full_model_pipeline(
        regressor=RandomForestRegressor(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1),
        scaler_type="standard"
    )
    repro_pipeline_2 = build_full_model_pipeline(
        regressor=RandomForestRegressor(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1),
        scaler_type="standard"
    )

    repro_pipeline_1.fit(X_train, y_train)
    preds_1 = repro_pipeline_1.predict(X_test)

    repro_pipeline_2.fit(X_train, y_train)
    preds_2 = repro_pipeline_2.predict(X_test)

    # Check exact numerical parity
    diff = np.max(np.abs(preds_1 - preds_2))
    rmse_1 = np.sqrt(mean_squared_error(y_test, preds_1))
    rmse_2 = np.sqrt(mean_squared_error(y_test, preds_2))

    print(f"Run A Test RMSE: ${rmse_1:,.6f}")
    print(f"Run B Test RMSE: ${rmse_2:,.6f}")
    print(f"Maximum Absolute Prediction Difference: {diff:.10f}")

    if diff < 1e-6:
        print("[Status] REPRODUCIBILITY VERIFICATION: PASSED (Deterministic Parity within Floating-Point Tolerance < 1e-6)")
    else:
        print(f"[Status] REPRODUCIBILITY VERIFICATION: Numerical variance detected: {diff}")

    print("\nExperiment 4 execution completed successfully.")
    return summary_df


if __name__ == "__main__":
    run_experiment_4()
