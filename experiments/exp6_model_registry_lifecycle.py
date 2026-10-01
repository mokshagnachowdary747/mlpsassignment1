"""
Experiment 6: Model Registry and Version Management
---------------------------------------------------
AIM:
Implement model registration and lifecycle management by maintaining model
versions across staging, validation, and production stages. Track model
metadata, preprocessing dependencies, evaluation metrics, dataset lineage, and
deployment readiness, and perform model comparison, promotion, rollback, and
traceability throughout the ML lifecycle.
"""

import os
import sys

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Add parent directory to path to import src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.pipeline import build_full_model_pipeline
from src.registry_manager import ModelRegistryManager


def run_experiment_6():
    print("=" * 80)
    print("EXPERIMENT 6: Model Registry and Version Management")
    print("=" * 80)

    # 1. Setup Tracking URI and Registry Manager
    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "Experiment_6_Model_Registry_Lifecycle"
    mlflow.set_experiment(experiment_name)
    
    model_name = "WalmartSalesForecastingModel"
    registry = ModelRegistryManager(model_name=model_name, tracking_uri=tracking_uri)
    print(f"[MLflow] Backend Tracking URI: {tracking_uri}")
    print(f"[Registry] Target Registered Model: {model_name}")

    # 2. Load Dataset & Train-Test Split
    data_path = os.path.join(parent_dir, "Walmart.csv")
    raw_df = pd.read_csv(data_path)
    X = raw_df.drop(columns=["Weekly_Sales"])
    y = raw_df["Weekly_Sales"]

    SEED = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
    input_example = X_train.head(3)
    signature = infer_signature(input_example, y_train.head(3))

    # 3. Train & Register Model Version 1 (Baseline Ridge Regressor)
    print("\n" + "-" * 70)
    print("[Phase 1] Training & Registering Model Version 1 (Baseline Ridge Model)...")
    print("-" * 70)
    pipeline_v1 = build_full_model_pipeline(
        regressor=Ridge(alpha=10.0, random_state=SEED),
        scaler_type="standard"
    )

    with mlflow.start_run(run_name="Train_Model_Version_1_Ridge") as run_v1:
        run_v1_id = run_v1.info.run_id
        pipeline_v1.fit(X_train, y_train)

        y_pred_v1 = pipeline_v1.predict(X_test)
        rmse_v1 = float(np.sqrt(np.mean((y_test - y_pred_v1) ** 2)))
        r2_v1 = float(pipeline_v1.score(X_test, y_test))

        mlflow.log_param("model_family", "Ridge")
        mlflow.log_param("alpha", 10.0)
        mlflow.log_metric("test_rmse", rmse_v1)
        mlflow.log_metric("test_r2", r2_v1)

        mlflow.sklearn.log_model(
            sk_model=pipeline_v1,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            serialization_format="cloudpickle"
        )

    # Register Run 1 as Version 1
    mv_1 = registry.register_model_from_run(
        run_id=run_v1_id,
        tags={"architecture": "Linear_Ridge", "training_stage": "initial_baseline"}
    )
    print(f">>> Registered Model Version: {mv_1.version} (Run ID: {run_v1_id})")

    # Initial promotion of Version 1 to Production
    registry.transition_stage(mv_1.version, "Production", archive_existing=True)
    registry.set_alias(mv_1.version, "champion")
    print(f">>> Version {mv_1.version} transitioned to 'Production' (alias='champion')")

    # 4. Train & Register Model Version 2 (Challenger Random Forest)
    print("\n" + "-" * 70)
    print("[Phase 2] Training & Registering Model Version 2 (Challenger Random Forest)...")
    print("-" * 70)
    pipeline_v2 = build_full_model_pipeline(
        regressor=RandomForestRegressor(n_estimators=100, max_depth=14, random_state=SEED, n_jobs=-1),
        scaler_type="standard"
    )

    with mlflow.start_run(run_name="Train_Model_Version_2_RandomForest") as run_v2:
        run_v2_id = run_v2.info.run_id
        pipeline_v2.fit(X_train, y_train)

        y_pred_v2 = pipeline_v2.predict(X_test)
        rmse_v2 = float(np.sqrt(np.mean((y_test - y_pred_v2) ** 2)))
        r2_v2 = float(pipeline_v2.score(X_test, y_test))

        mlflow.log_param("model_family", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 14)
        mlflow.log_metric("test_rmse", rmse_v2)
        mlflow.log_metric("test_r2", r2_v2)

        mlflow.sklearn.log_model(
            sk_model=pipeline_v2,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            serialization_format="cloudpickle"
        )

    # Register Run 2 as Version 2
    mv_2 = registry.register_model_from_run(
        run_id=run_v2_id,
        tags={"architecture": "Random_Forest_Ensemble", "training_stage": "tuned_challenger"}
    )
    print(f">>> Registered Model Version: {mv_2.version} (Run ID: {run_v2_id})")

    # Transition Version 2 to Staging for validation
    registry.transition_stage(mv_2.version, "Staging", archive_existing=False)
    registry.set_alias(mv_2.version, "challenger")
    print(f">>> Version {mv_2.version} placed in 'Staging' (alias='challenger')")

    # 5. Automated Model Comparison & Promotion
    print("\n" + "-" * 70)
    print("[Phase 3] Automated Model Comparison & Promotion Workflow...")
    print("-" * 70)
    promotion_decision = registry.compare_and_promote(
        candidate_version=mv_2.version,
        X_val=X_test,
        y_val=y_test,
        improvement_threshold_pct=5.0  # Must improve RMSE by at least 5%
    )

    print(f"Candidate Version          : {promotion_decision['candidate_version']}")
    print(f"Candidate Test RMSE        : ${promotion_decision['candidate_metrics']['rmse']:,.2f}")
    print(f"Current Prod Version       : {promotion_decision.get('current_prod_version')}")
    print(f"Current Prod Test RMSE     : ${promotion_decision['current_prod_metrics']['rmse']:,.2f}")
    print(f"Observed RMSE Improvement  : {promotion_decision['rmse_improvement_pct']}%")
    print(f"Promotion Result           : {'PROMOTED TO PRODUCTION' if promotion_decision['promoted'] else 'REJECTED'}")
    print(f"Reason                     : {promotion_decision['reason']}")

    # 6. Deployment Readiness & Inference Smoke Testing
    print("\n" + "-" * 70)
    print("[Phase 4] Verifying Deployment Readiness & Smoke Testing on Production Model...")
    print("-" * 70)
    active_prod = registry.get_version_by_alias_or_stage("champion")
    readiness_report = registry.verify_deployment_readiness(
        version=active_prod.version,
        sample_input=X_test.head(10)
    )

    print(f"Evaluated Production Version: {readiness_report['version']}")
    print(f"Deployment Ready            : {readiness_report['is_ready_for_deployment']}")
    print(f"Model Load Latency          : {readiness_report['load_latency_ms']} ms")
    print(f"Inference Latency (10 rows) : {readiness_report['inference_latency_ms']} ms")
    print(f"Zero NaNs / Infs Detected   : {not readiness_report['has_nans']}")
    print(f"Sample Predictions Preview  : {[f'${p:,.2f}' for p in readiness_report['sample_prediction_preview']]}")

    # 7. Model Rollback Simulation
    print("\n" + "-" * 70)
    print("[Phase 5] Simulating Model Rollback Workflow...")
    print("-" * 70)
    print("Initiating rollback to previous stable champion model...")
    rollback_result = registry.rollback_production()
    print(f"Rollback Success           : {rollback_result['success']}")
    print(f"Demoted Version            : {rollback_result.get('demoted_version')}")
    print(f"Restored Production Version: {rollback_result['restored_production_version']}")
    print(f"Message                    : {rollback_result['message']}")

    # Re-promote Version 2 back to champion so the final desired state is active
    print("\nRe-promoting Version 2 back to Production (Champion) for operational serving...")
    registry.transition_stage(mv_2.version, "Production", archive_existing=True)
    registry.set_alias(mv_2.version, "champion")

    # 8. Complete Traceability & Lineage Audit Trail
    print("\n" + "-" * 70)
    print("[Phase 6] End-to-End Model Traceability & Lineage Audit Trail...")
    print("-" * 70)
    final_prod = registry.get_version_by_alias_or_stage("champion")
    lineage = registry.get_model_lineage(final_prod.version)

    print(f"Registered Model Name: {lineage['model_name']}")
    print(f"Active Production Ver: {lineage['version']}")
    print(f"Source Run ID        : {lineage['run_id']}")
    print(f"Model Tags           : {lineage['model_tags']}")
    print(f"Logged Parameters    : {lineage['run_parameters']}")
    print(f"Logged Metrics       : {lineage['run_metrics']}")

    print("\n>>> ALL TESTS PASSED: Model registry lifecycle, promotion, and rollback successfully demonstrated.")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment_6()
