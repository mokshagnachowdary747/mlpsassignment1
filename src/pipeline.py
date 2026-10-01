"""
Pipeline Module for MLOps Pipeline (Experiments 4 & 5).
Builds reusable, idempotent, and schema-compliant scikit-learn preprocessing and modeling pipelines.
Includes:
- DateFeatureExtractor: Custom transformer for temporal feature engineering
- Preprocessing Pipeline with Imputation and Feature Scaling
- Pipeline Idempotency and Reproducibility Validation Utilities
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer to extract calendar and temporal features
    from raw date strings in format DD-MM-YYYY.
    Ensures idempotent feature engineering.
    """
    def __init__(self, date_column: str = "Date", drop_original: bool = True):
        self.date_column = date_column
        self.drop_original = drop_original
        self.engineered_feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if self.date_column in X_out.columns:
            dates = pd.to_datetime(X_out[self.date_column], format="%d-%m-%Y", errors="coerce")
            
            X_out["Year"] = dates.dt.year.fillna(2010).astype(int)
            X_out["Month"] = dates.dt.month.fillna(1).astype(int)
            X_out["WeekOfYear"] = dates.dt.isocalendar().week.fillna(1).astype(int)
            X_out["Day"] = dates.dt.day.fillna(1).astype(int)
            X_out["DayOfWeek"] = dates.dt.dayofweek.fillna(0).astype(int)

            self.engineered_feature_names = ["Year", "Month", "WeekOfYear", "Day", "DayOfWeek"]

            if self.drop_original:
                X_out = X_out.drop(columns=[self.date_column])
        return X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.engineered_feature_names)
        features = [f for f in input_features if f != self.date_column]
        return np.array(features + self.engineered_feature_names)


def build_preprocessor(
    scaler_type: str = "standard",
    with_feature_selection: bool = False,
    k_features: int = 8
) -> Pipeline:
    """
    Constructs an end-to-end preprocessing pipeline.
    
    Parameters:
        scaler_type: 'standard' or 'robust'
        with_feature_selection: Whether to include SelectKBest
        k_features: Number of top features to retain if selection is enabled
    """
    scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
    
    # Base numeric features
    numeric_features = [
        "Store", "Holiday_Flag", "Temperature", "Fuel_Price",
        "CPI", "Unemployment", "Year", "Month", "WeekOfYear", "Day", "DayOfWeek"
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler)
    ])

    column_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)
        ],
        remainder="drop"
    )

    pipeline_steps = [
        ("date_extractor", DateFeatureExtractor(date_column="Date")),
        ("col_transform", column_preprocessor)
    ]

    if with_feature_selection:
        pipeline_steps.append(("select_k_best", SelectKBest(score_func=f_regression, k=k_features)))

    return Pipeline(steps=pipeline_steps)


def build_full_model_pipeline(
    regressor,
    scaler_type: str = "standard",
    with_feature_selection: bool = False,
    k_features: int = 8
) -> Pipeline:
    """
    Constructs a complete Scikit-Learn training and inference pipeline
    combining preprocessing and a regressor model.
    """
    preprocessor = build_preprocessor(
        scaler_type=scaler_type,
        with_feature_selection=with_feature_selection,
        k_features=k_features
    )
    
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])


def verify_pipeline_idempotency(
    pipeline: Pipeline,
    sample_df: pd.DataFrame,
    y: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Verifies that the preprocessing pipeline is strictly idempotent:
    Running transform(X) multiple times on identical input yields bit-for-bit identical outputs.
    """
    fitted_pipeline = pipeline.fit(sample_df, y)
    
    transform_1 = fitted_pipeline.transform(sample_df)
    transform_2 = fitted_pipeline.transform(sample_df)
    
    # Handle sparse or dense outputs
    arr1 = transform_1.toarray() if hasattr(transform_1, "toarray") else np.asarray(transform_1)
    arr2 = transform_2.toarray() if hasattr(transform_2, "toarray") else np.asarray(transform_2)
    
    # Exact equality check
    is_identical = np.array_equal(arr1, arr2)
    max_abs_diff = float(np.max(np.abs(arr1 - arr2)))
    
    # Check for NaN / Inf leaks
    has_nans = bool(np.isnan(arr1).any())
    has_infs = bool(np.isinf(arr1).any())
    
    return {
        "is_idempotent": is_identical and (max_abs_diff == 0.0),
        "max_absolute_difference": max_abs_diff,
        "output_shape": list(arr1.shape),
        "contains_nans": has_nans,
        "contains_infs": has_infs,
        "status": "PASSED" if (is_identical and not has_nans and not has_infs) else "FAILED"
    }
