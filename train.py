"""
QuantumShield — train.py
==========================
Master training script that executes all four stages sequentially:

  Stage 1 → Preprocessing (data loading, cleaning, scaling, balancing)
  Stage 2 → Classical ML  (Logistic Regression + Random Forest)
  Stage 3 → Quantum ML    (ZZFeatureMap + Quantum Kernel + QSVC)
  Stage 4 → Evaluation    (metrics, confusion matrices, comparison charts)

Usage:
    python train.py                 # full training run
    python train.py --skip-quantum  # skip the slow quantum stage
    python train.py --max-q 200     # use 200 samples for quantum kernel

After this script completes, the Streamlit app (app.py) will load the
saved models automatically.
"""

import argparse
import sys
import io
import os
import json
import numpy as np

# Force UTF-8 output on Windows to avoid cp1252 codec errors with special chars
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing  import run_preprocessing_pipeline
from src.classical_model import run_classical_pipeline
from src.evaluation      import run_evaluation_pipeline


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="QuantumShield — Training Pipeline"
    )
    parser.add_argument(
        "--skip-quantum", action="store_true",
        help="Skip the quantum QSVC training stage (much faster)"
    )
    parser.add_argument(
        "--max-q", type=int, default=400,
        help="Maximum training samples for the quantum kernel (default: 400)"
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to creditcard.csv (defaults to data/creditcard.csv)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  QuantumShield -- Quantum ML Fraud Detection System")
    print("  Training Pipeline Starting...")
    print("=" * 60)

    # ── Stage 1: Preprocessing ────────────────────────────────────────────
    data_path_arg = args.data_path or os.path.join(
        os.path.dirname(__file__), "data", "creditcard.csv"
    )
    preproc = run_preprocessing_pipeline(data_path=data_path_arg)

    X_train   = preproc["X_train"]
    X_test    = preproc["X_test"]
    y_train   = preproc["y_train"]
    y_test    = preproc["y_test"]
    X_train_q = preproc["X_train_q"]
    X_test_q  = preproc["X_test_q"]

    # ── Stage 2: Classical ML ─────────────────────────────────────────────
    classical = run_classical_pipeline(X_train, X_test, y_train, y_test)
    lr_preds  = classical["lr_preds"]
    rf_preds  = classical["rf_preds"]

    # ── Stage 3: Quantum ML ───────────────────────────────────────────────
    if args.skip_quantum:
        print("\n[Quantum] --skip-quantum flag set: QSVC metrics will be omitted.")
        qsvc_preds = None
        qsvc_model = None
    else:
        from src.quantum_model import run_quantum_pipeline
        quantum = run_quantum_pipeline(
            X_train_q, X_test_q, y_train, y_test,
            max_samples=args.max_q,
        )
        qsvc_preds = quantum["qsvc_preds"]
        qsvc_model = quantum["qsvc_model"]

    # ── Stage 4: Evaluation ───────────────────────────────────────────────
    eval_results = run_evaluation_pipeline(
        y_test, lr_preds, rf_preds, qsvc_preds,
        save_figures=True,
    )

    # ── Save metadata JSON (read by Streamlit) ────────────────────────────
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    metadata = {
        "quantum_trained": not args.skip_quantum,
        "quantum_features": preproc["quantum_features"],
        "feature_scaling": preproc["feature_scaling"],
        "quantum_scaling": preproc["quantum_scaling"],
        "class_dist_raw": preproc["class_dist_raw"],
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "metrics": [
            {k: (float(v) if isinstance(v, (np.floating, float)) else v)
             for k, v in m.items() if k != "confusion_matrix"}
            for m in eval_results["metrics_list"]
        ],
    }

    metadata_path = os.path.join(models_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[Meta] Metadata saved → {metadata_path}")

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training Complete — Results Summary")
    print("=" * 60)
    print(eval_results["metrics_df"].to_string())

    print("\n\nTo launch the dashboard:")
    print("  streamlit run app.py\n")


if __name__ == "__main__":
    main()
