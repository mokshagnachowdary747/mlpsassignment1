"""
Data Validation Module for MLOps Pipeline (Experiment 5).
Implements schema-aware validation using Pandera to detect:
- Datatype mismatches
- Schema inconsistencies and missing columns
- Invalid categorical values and out-of-range values
- Corrupted/null records
- Quarantining corrupted records
- Pre- and post-transformation feature validation
"""

import os
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"

from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd

try:
    import pandera.pandas as pa
except ImportError:
    import pandera as pa
from pandera import Column, Check, DataFrameSchema, errors


def get_raw_data_schema(strict: bool = True) -> DataFrameSchema:
    """
    Returns the Pandera DataFrameSchema for the raw Walmart dataset.
    Validates column datatypes, nullability, and numerical range constraints.
    """
    return DataFrameSchema(
        columns={
            "Store": Column(
                pa.Int64,
                checks=[
                    Check.greater_than_or_equal_to(1, error="Store ID must be >= 1"),
                    Check.less_than_or_equal_to(100, error="Store ID must be <= 100")
                ],
                nullable=False,
                coerce=True,
                description="Walmart Store ID"
            ),
            "Date": Column(
                pa.String,
                checks=[
                    Check(
                        lambda s: pd.to_datetime(s, format="%d-%m-%Y", errors="coerce").notnull(),
                        element_wise=False,
                        error="Date must be a parseable DD-MM-YYYY format string"
                    )
                ],
                nullable=False,
                coerce=False,
                description="Sales record date (DD-MM-YYYY)"
            ),
            "Weekly_Sales": Column(
                pa.Float64,
                checks=[
                    Check.greater_than_or_equal_to(0.0, error="Weekly sales cannot be negative")
                ],
                nullable=False,
                coerce=True,
                description="Weekly sales dollar amount"
            ),
            "Holiday_Flag": Column(
                pa.Int64,
                checks=[
                    Check.isin([0, 1], error="Holiday Flag must be binary 0 or 1")
                ],
                nullable=False,
                coerce=True,
                description="Binary indicator for holiday week (0 or 1)"
            ),
            "Temperature": Column(
                pa.Float64,
                checks=[
                    Check.in_range(-50.0, 150.0, error="Temperature must be between -50°F and 150°F")
                ],
                nullable=False,
                coerce=True,
                description="Average temperature in the region (°F)"
            ),
            "Fuel_Price": Column(
                pa.Float64,
                checks=[
                    Check.greater_than(0.0, error="Fuel price must be greater than 0")
                ],
                nullable=False,
                coerce=True,
                description="Fuel price in the region"
            ),
            "CPI": Column(
                pa.Float64,
                checks=[
                    Check.greater_than(0.0, error="CPI must be greater than 0")
                ],
                nullable=False,
                coerce=True,
                description="Consumer Price Index"
            ),
            "Unemployment": Column(
                pa.Float64,
                checks=[
                    Check.greater_than_or_equal_to(0.0, error="Unemployment rate must be >= 0")
                ],
                nullable=False,
                coerce=True,
                description="Unemployment rate percentage"
            ),
        },
        strict=strict,
        ordered=False,
        name="WalmartRawSalesSchema"
    )


def get_transformed_features_schema(num_expected_features: int = 11) -> DataFrameSchema:
    """
    Schema for validating preprocessed feature matrices prior to model ingestion.
    Verifies that all values are finite float64 and no NaNs or Infs leaked through.
    """
    return DataFrameSchema(
        columns={
            f"feature_{i}": Column(
                pa.Float64,
                checks=[
                    Check(lambda s: ~np.isnan(s).any(), error=f"Feature {i} contains NaN values"),
                    Check(lambda s: ~np.isinf(s).any(), error=f"Feature {i} contains Inf values")
                ],
                nullable=False
            )
            for i in range(num_expected_features)
        },
        strict=True,
        name="WalmartTransformedFeaturesSchema"
    )


def validate_dataset(df: pd.DataFrame, lazy: bool = True) -> Tuple[bool, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Validates a pandas DataFrame against the raw data schema.
    
    Parameters:
        df: Input DataFrame to validate.
        lazy: If True, collects all schema errors instead of halting on first violation.
        
    Returns:
        Tuple of (is_valid, validated_df, failure_report_df)
    """
    schema = get_raw_data_schema()
    try:
        validated_df = schema.validate(df, lazy=lazy)
        return True, validated_df, None
    except errors.SchemaErrors as err:
        failure_cases = err.failure_cases
        return False, None, failure_cases
    except errors.SchemaError as err:
        failure_cases = pd.DataFrame([{
            "schema_context": err.schema.name if err.schema else "Column",
            "column": err.schema.name if hasattr(err.schema, "name") else str(err),
            "check": str(err.check),
            "check_index": None,
            "failure_case": str(err)
        }])
        return False, None, failure_cases


def generate_corrupted_samples(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a controlled test set with deliberate data quality anomalies:
    1. Negative sales record
    2. Corrupted date format string
    3. Invalid categorical value (Holiday_Flag = 99)
    4. Out-of-bounds temperature (-100°F)
    5. Datatype mismatch (string in Fuel_Price)
    6. Missing value in critical column (Unemployment is NaN)
    """
    corrupt_df = clean_df.head(6).copy().astype(object)
    
    # 1. Negative Weekly Sales
    corrupt_df.loc[corrupt_df.index[0], "Weekly_Sales"] = -125000.50
    
    # 2. Corrupted Date
    corrupt_df.loc[corrupt_df.index[1], "Date"] = "2020/99/99_INVALID"
    
    # 3. Invalid categorical value
    corrupt_df.loc[corrupt_df.index[2], "Holiday_Flag"] = 99
    
    # 4. Out-of-bounds temperature
    corrupt_df.loc[corrupt_df.index[3], "Temperature"] = -105.0
    
    # 5. String datatype in numeric field
    corrupt_df.loc[corrupt_df.index[4], "Fuel_Price"] = "three_dollars"
    
    # 6. Null / NaN in mandatory column
    corrupt_df.loc[corrupt_df.index[5], "Unemployment"] = np.nan
    
    return corrupt_df


def quarantine_corrupted_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Row-level validation and quarantine mechanism:
    Splits the dataset into clean (valid) rows that proceed down the pipeline,
    and a quarantined dataset with audit metadata for investigation.
    """
    clean_mask = pd.Series(True, index=df.index)
    rejection_reasons = {}

    # Check 1: Mandatory columns present
    required_cols = ["Store", "Date", "Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Fatal schema inconsistency: Missing required columns {missing_cols}")

    # Check 2: Row-level null values
    null_rows = df[required_cols].isnull().any(axis=1)
    clean_mask &= (~null_rows)
    for idx in df[null_rows].index:
        rejection_reasons.setdefault(idx, []).append("Contains null values in required fields")

    # Check 3: Valid date format
    parsed_dates = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    invalid_dates = parsed_dates.isnull()
    clean_mask &= (~invalid_dates)
    for idx in df[invalid_dates].index:
        rejection_reasons.setdefault(idx, []).append("Invalid date format (must be DD-MM-YYYY)")

    # Check 4: Numeric ranges
    # Sales >= 0
    numeric_sales = pd.to_numeric(df["Weekly_Sales"], errors="coerce")
    neg_sales = (numeric_sales < 0) | numeric_sales.isnull()
    clean_mask &= (~neg_sales)
    for idx in df[neg_sales].index:
        rejection_reasons.setdefault(idx, []).append("Weekly_Sales < 0 or non-numeric")

    # Holiday flag in {0, 1}
    numeric_holiday = pd.to_numeric(df["Holiday_Flag"], errors="coerce")
    invalid_holiday = (~numeric_holiday.isin([0, 1])) | numeric_holiday.isnull()
    clean_mask &= (~invalid_holiday)
    for idx in df[invalid_holiday].index:
        rejection_reasons.setdefault(idx, []).append("Holiday_Flag not in {0, 1}")

    # Temperature in [-50, 150]
    numeric_temp = pd.to_numeric(df["Temperature"], errors="coerce")
    invalid_temp = (numeric_temp < -50.0) | (numeric_temp > 150.0) | numeric_temp.isnull()
    clean_mask &= (~invalid_temp)
    for idx in df[invalid_temp].index:
        rejection_reasons.setdefault(idx, []).append("Temperature out of bounds [-50, 150]")

    # Fuel price > 0
    numeric_fuel = pd.to_numeric(df["Fuel_Price"], errors="coerce")
    invalid_fuel = (numeric_fuel <= 0) | numeric_fuel.isnull()
    clean_mask &= (~invalid_fuel)
    for idx in df[invalid_fuel].index:
        rejection_reasons.setdefault(idx, []).append("Fuel_Price <= 0 or non-numeric")

    # Split into clean and quarantined
    clean_df = df[clean_mask].copy()
    quarantined_df = df[~clean_mask].copy()
    quarantined_df["quarantine_reason"] = quarantined_df.index.map(lambda i: "; ".join(rejection_reasons.get(i, ["Schema failure"])))

    # Ensure clean_df has proper numeric datatypes
    for col in ["Store", "Holiday_Flag"]:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col]).astype(int)
    for col in ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col]).astype(float)

    audit_summary = {
        "total_records": len(df),
        "clean_records": len(clean_df),
        "quarantined_records": len(quarantined_df),
        "quarantine_rate_percent": round((len(quarantined_df) / max(1, len(df))) * 100, 2)
    }

    return clean_df, quarantined_df, audit_summary
