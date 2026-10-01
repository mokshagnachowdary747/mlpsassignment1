"""
Experiment 5: Data Validation and Reproducible ML Pipelines
----------------------------------------------------------
AIM:
Design and validate schema-aware data preprocessing workflows by identifying
datatype mismatches, schema inconsistencies, missing attributes, invalid
categorical values, and corrupted records. Build reusable and idempotent
preprocessing pipelines and validate transformed features, preprocessing outputs,
and dataset consistency across repeated executions.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path to import src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.data_validation import (
    get_raw_data_schema,
    get_transformed_features_schema,
    validate_dataset,
    generate_corrupted_samples,
    quarantine_corrupted_records
)
from src.pipeline import build_preprocessor, verify_pipeline_idempotency


def run_experiment_5():
    print("=" * 80)
    print("EXPERIMENT 5: Data Validation and Reproducible ML Pipelines")
    print("=" * 80)

    # 1. Load Clean Dataset
    data_path = os.path.join(parent_dir, "Walmart.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    raw_df = pd.read_csv(data_path)
    print(f"\n[Step 1] Loaded dataset with {len(raw_df)} rows and {len(raw_df.columns)} columns.")

    # 2. Schema-Aware Validation on Clean Data
    print("\n" + "-" * 70)
    print("[Step 2] Executing Pandera Schema Validation on Clean Data...")
    print("-" * 70)
    is_valid, validated_df, failures = validate_dataset(raw_df, lazy=True)
    if is_valid:
        print(">>> SUCCESS: Raw dataset conforms 100% to schema definitions:")
        print("    - All 8 columns present with expected datatypes")
        print("    - Value constraints satisfied (Weekly_Sales >= 0, Holiday_Flag in {0, 1})")
        print("    - Date format verified as DD-MM-YYYY across all records")
    else:
        print(">>> FAILED: Clean data unexpectedly failed validation:")
        print(failures)

    # 3. Controlled Anomaly & Corruption Testing
    print("\n" + "-" * 70)
    print("[Step 3] Injecting Controlled Data Corruptions for Robustness Testing...")
    print("-" * 70)
    corrupted_sample = generate_corrupted_samples(raw_df)
    
    print("Corrupted Samples Injected:")
    for i, (_, row) in enumerate(corrupted_sample.iterrows()):
        print(f"  Row {i+1}: Date={row['Date']} | Sales={row['Weekly_Sales']} | Holiday={row['Holiday_Flag']} | Temp={row['Temperature']} | Fuel={row['Fuel_Price']} | Unemp={row['Unemployment']}")

    print("\nValidating corrupted records against Pandera schema (lazy=True)...")
    is_corrupt_valid, _, failure_report = validate_dataset(corrupted_sample, lazy=True)

    print(f">>> Validation Outcome: is_valid = {is_corrupt_valid}")
    if failure_report is not None:
        print("\n=== PANDERA SCHEMA FAILURE AUDIT REPORT ===")
        # Select key informative columns from Pandera failure cases
        cols_to_show = [c for c in ["schema_context", "column", "check", "failure_case", "index"] if c in failure_report.columns]
        print(failure_report[cols_to_show].drop_duplicates().to_string(index=False))

    # 4. Production Quarantine Workflow
    print("\n" + "-" * 70)
    print("[Step 4] Running Row-Level Quarantining Workflow...")
    print("-" * 70)
    # Combine clean dataset with corrupted records to test split
    mixed_dataset = pd.concat([corrupted_sample, raw_df.head(200)], ignore_index=True)
    clean_partition, quarantined_partition, audit = quarantine_corrupted_records(mixed_dataset)

    print(f"Total Records Ingested : {audit['total_records']}")
    print(f"Clean Records Accepted : {audit['clean_records']}")
    print(f"Corrupted Quarantined  : {audit['quarantined_records']} ({audit['quarantine_rate_percent']}%)")
    
    print("\nQuarantined Records Audit Sample:")
    print(quarantined_partition[["Store", "Date", "Weekly_Sales", "quarantine_reason"]].to_string())

    # 5. Build Reusable Preprocessing Pipeline
    print("\n" + "-" * 70)
    print("[Step 5] Building Reusable & Idempotent Preprocessing Pipeline...")
    print("-" * 70)
    preprocessor = build_preprocessor(scaler_type="standard", with_feature_selection=False)
    
    X_clean = clean_partition.drop(columns=["Weekly_Sales"])
    y_clean = clean_partition["Weekly_Sales"]

    preprocessor.fit(X_clean, y_clean)
    transformed_features = preprocessor.transform(X_clean)

    print(f"Raw Input Features Shape : {X_clean.shape}")
    print(f"Transformed Output Shape : {transformed_features.shape}")

    # Validate output transformed features
    transformed_df = pd.DataFrame(
        transformed_features,
        columns=[f"feature_{i}" for i in range(transformed_features.shape[1])]
    )
    feature_schema = get_transformed_features_schema(num_expected_features=transformed_features.shape[1])
    try:
        feature_schema.validate(transformed_df)
        print(">>> SUCCESS: Preprocessed feature matrix conforms to TransformedFeaturesSchema.")
        print("    - Zero NaNs, Zero Infs, expected column count verified.")
    except Exception as e:
        print(f">>> FAILED: Preprocessed features schema violation: {e}")

    # 6. Idempotency & Reproducibility Verification Across Executions
    print("\n" + "-" * 70)
    print("[Step 6] Verifying Preprocessing Pipeline Idempotency Across Repeated Executions...")
    print("-" * 70)
    idempotency_result = verify_pipeline_idempotency(preprocessor, X_clean, y_clean)

    print(f"Idempotency Status            : {idempotency_result['status']}")
    print(f"Outputs Strictly Identical    : {idempotency_result['is_idempotent']}")
    print(f"Max Absolute Drift Between Runs: {idempotency_result['max_absolute_difference']}")
    print(f"Transformed Feature Dimension : {idempotency_result['output_shape']}")
    print(f"Contains NaNs                 : {idempotency_result['contains_nans']}")
    print(f"Contains Infs                 : {idempotency_result['contains_infs']}")

    # Final assertion check
    assert idempotency_result["is_idempotent"], "Fatal Error: Preprocessing pipeline failed idempotency assertion!"
    assert not idempotency_result["contains_nans"], "Fatal Error: Preprocessed data contains NaNs!"

    print("\n>>> ALL TESTS PASSED: Data validation and idempotent pipeline workflows successfully verified.")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment_5()
