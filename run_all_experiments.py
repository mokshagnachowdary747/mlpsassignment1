"""
Master Runner Script for MLOps Lab Experiments 4, 5, and 6.
Executes:
1. Experiment 4: Experiment Tracking and Reproducibility Workflows (MLflow)
2. Experiment 5: Data Validation and Reproducible ML Pipelines (Pandera + Idempotent Pipelines)
3. Experiment 6: Model Registry and Version Management (MLflow Model Registry)
"""

import sys
import time
import subprocess
import os

EXPERIMENTS = [
    ("Experiment 4: Tracking & Reproducibility", "experiments/exp4_tracking_reproducibility.py"),
    ("Experiment 5: Data Validation & Pipelines", "experiments/exp5_data_validation_pipeline.py"),
    ("Experiment 6: Model Registry & Lifecycle", "experiments/exp6_model_registry_lifecycle.py"),
]


def run_all():
    print("=" * 80)
    print("      MLOPS LAB: EXPERIMENTS 4, 5, AND 6 AUTOMATED EXECUTION RUNNER      ")
    print("=" * 80)
    
    python_exe = sys.executable
    print(f"Using Python Interpreter: {python_exe}")
    print(f"Working Directory       : {os.getcwd()}\n")

    overall_start = time.time()
    results = {}

    for name, script_relpath in EXPERIMENTS:
        script_full = os.path.abspath(script_relpath)
        print("\n" + "#" * 80)
        print(f"STARTING: {name}")
        print(f"SCRIPT  : {script_relpath}")
        print("#" * 80 + "\n")
        
        start_t = time.time()
        process = subprocess.run([python_exe, script_full], capture_output=False)
        duration = time.time() - start_t
        
        status = "PASSED" if process.returncode == 0 else f"FAILED (Exit Code {process.returncode})"
        results[name] = {"status": status, "duration_seconds": round(duration, 2)}
        
        if process.returncode != 0:
            print(f"\n[ERROR] {name} terminated with errors. Halting execution pipeline.")
            sys.exit(process.returncode)

    total_time = round(time.time() - overall_start, 2)
    print("\n" + "=" * 80)
    print("                     ALL EXPERIMENTS COMPLETED SUCCESSFULLY!                     ")
    print("=" * 80)
    for exp, res in results.items():
        print(f"  * {exp:<45} : {res['status']} ({res['duration_seconds']}s)")
    print(f"\nTotal Pipeline Execution Time: {total_time}s")
    print("=" * 80)
    print("\n[NEXT STEPS]")
    print("1. Launch MLflow Dashboard in your terminal to inspect tracked runs and model registry:")
    print("   mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("   Then open: http://localhost:5000 in your browser.")
    print("2. You can also run individual Jupyter Notebooks inside VS Code in the 'notebooks/' folder.")
    print("=" * 80)


if __name__ == "__main__":
    run_all()
