"""
QuantumShield — Classical ML Module
=====================================
Implements and trains classical machine learning models:
  - Logistic Regression
  - Random Forest

Both models are trained on the full preprocessed feature set.
After training, models are serialised with joblib for later use.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helper — ensure model directory exists
# ---------------------------------------------------------------------------

def _ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.

    Notes:
    - Uses 'liblinear' solver which is fast and reliable for small-to-medium
      datasets after undersampling.
    - class_weight='balanced' applies inverse-frequency weighting as an extra
      guard against any residual imbalance.
    - max_iter=1000 prevents convergence warnings on some datasets.

    Args:
        X_train: Training features (DataFrame or ndarray).
        y_train: Training labels.

    Returns:
        Fitted LogisticRegression model.
    """
    print("[LR] Training Logistic Regression …")
    model = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    print("[LR] Training complete ✓")
    return model


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Notes:
    - n_estimators=100 gives a good accuracy/speed tradeoff.
    - class_weight='balanced_subsample' weights classes per bootstrap sample,
      which is the recommended approach for imbalanced data in ensembles.
    - n_jobs=-1 uses all available CPU cores.

    Args:
        X_train: Training features (DataFrame or ndarray).
        y_train: Training labels.

    Returns:
        Fitted RandomForestClassifier model.
    """
    print("[RF] Training Random Forest …")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,                 # cap depth to prevent overfitting
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("[RF] Training complete ✓")
    return model


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict(model, X_test):
    """
    Generate class predictions.

    Args:
        model: Fitted sklearn estimator.
        X_test: Test feature array.

    Returns:
        1-D ndarray of predicted labels.
    """
    return model.predict(X_test)


def predict_proba(model, X_test):
    """
    Generate probability estimates.

    Args:
        model: Fitted sklearn estimator with predict_proba.
        X_test: Test feature array.

    Returns:
        2-D ndarray of shape (n_samples, n_classes).
        Column 0 = P(normal), Column 1 = P(fraud).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)
    # Decision-function fallback (e.g. SVC without probability calibration)
    df = model.decision_function(X_test)
    # Normalise to [0, 1] via a simple sigmoid
    prob_fraud = 1 / (1 + np.exp(-df))
    return np.column_stack([1 - prob_fraud, prob_fraud])


# ---------------------------------------------------------------------------
# Save / load models
# ---------------------------------------------------------------------------

def save_model(model, name: str):
    """
    Serialise a fitted model to disk using joblib.

    Args:
        model: Fitted sklearn estimator.
        name: File stem (e.g. 'logistic_regression').
    """
    _ensure_models_dir()
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"[Save] Model saved → {path}")
    return path


def load_model(name: str):
    """
    Load a previously saved model from disk.

    Args:
        name: File stem (e.g. 'logistic_regression').

    Returns:
        Fitted sklearn estimator.

    Raises:
        FileNotFoundError if the file does not exist.
    """
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    print(f"[Load] Model loaded ← {path}")
    return model


# ---------------------------------------------------------------------------
# Master training pipeline — called by train.py
# ---------------------------------------------------------------------------

def run_classical_pipeline(X_train, X_test, y_train, y_test):
    """
    Train both classical models, run predictions, and save models to disk.

    Args:
        X_train, X_test, y_train, y_test: Preprocessed split arrays.

    Returns:
        dict with keys:
            lr_model  — fitted LogisticRegression
            rf_model  — fitted RandomForestClassifier
            lr_preds  — LR predictions on X_test
            rf_preds  — RF predictions on X_test
            lr_proba  — LR probability estimates on X_test
            rf_proba  — RF probability estimates on X_test
    """
    print("\n" + "=" * 60)
    print("  QuantumShield — Stage 2: Classical ML Pipeline")
    print("=" * 60 + "\n")

    lr = train_logistic_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train)

    lr_preds = predict(lr, X_test)
    rf_preds = predict(rf, X_test)

    lr_proba = predict_proba(lr, X_test)
    rf_proba = predict_proba(rf, X_test)

    save_model(lr, "logistic_regression")
    save_model(rf, "random_forest")

    print("\n[Pipeline] Stage 2 complete ✓\n")

    return {
        "lr_model": lr,
        "rf_model": rf,
        "lr_preds": lr_preds,
        "rf_preds": rf_preds,
        "lr_proba": lr_proba,
        "rf_proba": rf_proba,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.preprocessing import run_preprocessing_pipeline

    data = run_preprocessing_pipeline()
    results = run_classical_pipeline(
        data["X_train"], data["X_test"],
        data["y_train"], data["y_test"],
    )
    print(f"LR predictions sample : {results['lr_preds'][:10]}")
    print(f"RF predictions sample : {results['rf_preds'][:10]}")
