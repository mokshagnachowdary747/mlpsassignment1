"""
Model Registry and Lifecycle Management Module (Experiment 6).
Handles:
- Model registration into MLflow Registry
- Version lifecycle management across stages (Staging, Validation, Production, Archived) and aliases (champion, challenger)
- Model metadata, metrics, and dataset lineage tracking
- Automated model comparison and criteria-based promotion
- Safe rollback mechanism to previous production versions
- Deployment readiness smoke tests and inference benchmarking
"""

import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelRegistryManager:
    """
    Manages the lifecycle, stages, promotion, and rollback for models in the MLflow Model Registry.
    """
    def __init__(self, model_name: str = "WalmartSalesForecastingModel", tracking_uri: str = "sqlite:///mlflow.db"):
        self.model_name = model_name
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self._ensure_registered_model_exists()

    def _ensure_registered_model_exists(self):
        """Ensures the registered model entity is initialized in the MLflow store."""
        try:
            self.client.get_registered_model(self.model_name)
        except Exception:
            self.client.create_registered_model(
                self.model_name,
                description="Production ML model for Walmart weekly sales forecasting.",
                tags={"framework": "scikit-learn", "domain": "retail_forecasting", "dataset": "Walmart.csv"}
            )

    def register_model_from_run(
        self,
        run_id: str,
        artifact_path: str = "model",
        tags: Optional[Dict[str, str]] = None
    ) -> ModelVersion:
        """
        Registers a model artifact from an active or finished MLflow run into the registry.
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        reg_version = mlflow.register_model(model_uri=model_uri, name=self.model_name)
        
        # Add metadata tags
        self.client.set_model_version_tag(self.model_name, reg_version.version, "registered_by", "mlops_pipeline")
        self.client.set_model_version_tag(self.model_name, reg_version.version, "run_id", run_id)
        if tags:
            for k, v in tags.items():
                self.client.set_model_version_tag(self.model_name, reg_version.version, str(k), str(v))
                
        return reg_version

    def transition_stage(self, version: str, stage: str, archive_existing: bool = True) -> ModelVersion:
        """
        Transitions a model version to a lifecycle stage:
        'Staging', 'Production', or 'Archived'.
        """
        # Modern MLflow supports transition_model_version_stage
        try:
            updated_version = self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing
            )
        except Exception as e:
            # Fallback to aliases and tags if stage transitions are restricted
            self.client.set_model_version_tag(self.model_name, version, "stage", stage)
            updated_version = self.client.get_model_version(self.model_name, version)
            
        self.client.set_model_version_tag(
            self.model_name,
            version,
            "last_stage_transition_time",
            time.strftime("%Y-%m-%d %H:%M:%S")
        )
        return updated_version

    def set_alias(self, version: str, alias: str):
        """Sets an alias (e.g. 'champion', 'challenger', 'staging_candidate') for a model version."""
        try:
            self.client.set_registered_model_alias(self.model_name, alias, version)
        except Exception:
            self.client.set_model_version_tag(self.model_name, version, "alias", alias)

    def get_version_by_alias_or_stage(self, identifier: str) -> Optional[ModelVersion]:
        """
        Retrieves a model version by stage name (e.g. 'Production') or alias (e.g. 'champion').
        """
        # 1. Try alias
        try:
            model = self.client.get_model_version_by_alias(self.model_name, identifier)
            if model:
                return model
        except Exception:
            pass

        # 2. Try stage
        try:
            versions = self.client.get_latest_versions(self.model_name, stages=[identifier])
            if versions:
                return versions[0]
        except Exception:
            pass

        # 3. Search tags
        all_versions = self.client.search_model_versions(f"name='{self.model_name}'")
        for v in all_versions:
            if v.tags.get("stage") == identifier or v.tags.get("alias") == identifier:
                return v

        return None

    def evaluate_model_version(
        self,
        version: str,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Loads a specific model version and computes standardized evaluation metrics.
        """
        model_uri = f"models:/{self.model_name}/{version}"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        preds = loaded_model.predict(X_test)
        
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        
        return {
            "version": version,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

    def compare_and_promote(
        self,
        candidate_version: str,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        improvement_threshold_pct: float = 0.0
    ) -> Dict[str, Any]:
        """
        Compares candidate version against current Production model.
        Promotes candidate to 'Production' and sets alias 'champion' if it meets criteria.
        Archives previous champion and tags it for potential rollback.
        """
        candidate_metrics = self.evaluate_model_version(candidate_version, X_val, y_val)
        current_prod = self.get_version_by_alias_or_stage("Production")

        decision_log = {
            "candidate_version": candidate_version,
            "candidate_metrics": candidate_metrics,
            "promoted": False,
            "reason": ""
        }

        if current_prod is None:
            # No production model exists yet; promote directly
            self.transition_stage(candidate_version, "Production", archive_existing=True)
            self.set_alias(candidate_version, "champion")
            self.client.set_model_version_tag(self.model_name, candidate_version, "deployment_status", "initial_champion")
            decision_log["promoted"] = True
            decision_log["reason"] = "No previous production model found. Promoted as initial champion."
            return decision_log

        prod_metrics = self.evaluate_model_version(current_prod.version, X_val, y_val)
        decision_log["current_prod_version"] = current_prod.version
        decision_log["current_prod_metrics"] = prod_metrics

        # Lower RMSE is better
        prod_rmse = prod_metrics["rmse"]
        cand_rmse = candidate_metrics["rmse"]
        rmse_diff = prod_rmse - cand_rmse
        rmse_improvement_pct = (rmse_diff / prod_rmse) * 100.0

        decision_log["rmse_improvement_pct"] = round(rmse_improvement_pct, 3)

        if rmse_improvement_pct >= improvement_threshold_pct:
            # Promote candidate
            # Record current prod as rollback target
            self.set_alias(current_prod.version, "previous_champion")
            self.client.set_model_version_tag(self.model_name, current_prod.version, "rollback_ready", "true")

            # Promote candidate
            self.transition_stage(candidate_version, "Production", archive_existing=True)
            self.set_alias(candidate_version, "champion")
            self.client.set_model_version_tag(self.model_name, candidate_version, "deployment_status", "promoted_champion")
            
            decision_log["promoted"] = True
            decision_log["reason"] = f"Candidate achieved {round(rmse_improvement_pct, 2)}% RMSE improvement over production version {current_prod.version}."
        else:
            # Reject promotion; keep in staging or challenger
            self.transition_stage(candidate_version, "Staging", archive_existing=False)
            self.set_alias(candidate_version, "challenger")
            decision_log["promoted"] = False
            decision_log["reason"] = f"Candidate RMSE ({cand_rmse:.2f}) did not beat Production RMSE ({prod_rmse:.2f}) by threshold {improvement_threshold_pct}%."

        return decision_log

    def rollback_production(self) -> Dict[str, Any]:
        """
        Executes a safety rollback: Demotes current production version and restores
        the previous champion marked with alias 'previous_champion'.
        """
        current_prod = self.get_version_by_alias_or_stage("Production")
        previous_champion = self.get_version_by_alias_or_stage("previous_champion")

        if not previous_champion:
            return {
                "success": False,
                "message": "No previous champion model available for rollback."
            }

        # Demote current
        if current_prod:
            self.transition_stage(current_prod.version, "Archived", archive_existing=False)
            self.client.set_model_version_tag(self.model_name, current_prod.version, "demoted_reason", "manual_or_automated_rollback")

        # Restore previous
        self.transition_stage(previous_champion.version, "Production", archive_existing=True)
        self.set_alias(previous_champion.version, "champion")
        self.client.set_model_version_tag(self.model_name, previous_champion.version, "deployment_status", "restored_champion")

        return {
            "success": True,
            "demoted_version": current_prod.version if current_prod else None,
            "restored_production_version": previous_champion.version,
            "message": f"Successfully rolled back Production to version {previous_champion.version}."
        }

    def verify_deployment_readiness(self, version: str, sample_input: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs deployment readiness sanity check:
        - Model loading latency
        - Inference execution on sample payload
        - Absence of NaNs / Infs in predictions
        - Prediction shape verification
        """
        start_time = time.time()
        model_uri = f"models:/{self.model_name}/{version}"
        model = mlflow.pyfunc.load_model(model_uri)
        load_latency_ms = (time.time() - start_time) * 1000

        infer_start = time.time()
        preds = model.predict(sample_input)
        infer_latency_ms = (time.time() - infer_start) * 1000

        preds_arr = np.asarray(preds)
        has_nans = bool(np.isnan(preds_arr).any())
        has_infs = bool(np.isinf(preds_arr).any())
        non_negative = bool((preds_arr >= 0).all())

        is_ready = (not has_nans) and (not has_infs) and (len(preds_arr) == len(sample_input))

        readiness_report = {
            "version": version,
            "is_ready_for_deployment": is_ready,
            "load_latency_ms": round(load_latency_ms, 2),
            "inference_latency_ms": round(infer_latency_ms, 2),
            "sample_records_tested": len(sample_input),
            "has_nans": has_nans,
            "all_predictions_non_negative": non_negative,
            "sample_prediction_preview": [float(p) for p in preds_arr[:3]]
        }

        # Log readiness tag
        self.client.set_model_version_tag(
            self.model_name,
            version,
            "deployment_readiness",
            "VERIFIED" if is_ready else "FAILED"
        )
        return readiness_report

    def get_model_lineage(self, version: str) -> Dict[str, Any]:
        """
        Retrieves end-to-end audit lineage for a registered model version:
        Run ID -> Parameters -> Metrics -> Tags -> Registered Version.
        """
        mv = self.client.get_model_version(self.model_name, version)
        run = self.client.get_run(mv.run_id)

        return {
            "model_name": self.model_name,
            "version": version,
            "current_stage": mv.current_stage,
            "run_id": mv.run_id,
            "created_timestamp": mv.creation_timestamp,
            "model_tags": mv.tags,
            "run_parameters": run.data.params,
            "run_metrics": run.data.metrics,
            "run_tags": run.data.tags
        }
