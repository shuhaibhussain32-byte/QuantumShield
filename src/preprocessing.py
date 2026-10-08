"""
QuantumShield — Preprocessing Module
=====================================
Handles all data loading, cleaning, scaling, and feature selection.

Pipeline:
  1. Load CSV (creditcard.csv) from the data/ directory
  2. Handle missing values
  3. Scale numerical features (StandardScaler)
  4. Address class imbalance with undersampling
  5. Select a small feature subset for the quantum model
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Features to keep for the quantum model.
# We use only a handful because every feature = 1 qubit in ZZFeatureMap,
# and quantum simulation cost grows exponentially with qubit count.
# V1, V4, V11, V14, V17 are historically among the most predictive PCA
# components for credit-card fraud.
QUANTUM_FEATURES = ["V1", "V4", "V11", "V14", "V17"]

# Random seed for reproducibility
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Step 1 — Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the Kaggle credit-card fraud dataset.

    The dataset contains transactions made by European cardholders in Sep 2013.
    Features V1–V28 are PCA-transformed (anonymised).
    'Time' = seconds elapsed since first transaction.
    'Amount' = transaction amount in EUR.
    'Class' = 0 (normal) or 1 (fraud).

    Args:
        path: Absolute or relative path to creditcard.csv.

    Returns:
        Raw DataFrame.

    Raises:
        FileNotFoundError: If the CSV is not present.
    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"Dataset not found at: {abs_path}\n"
            "Please download creditcard.csv from:\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place it in the  data/  directory."
        )
    df = pd.read_csv(abs_path)
    print(f"[Load] Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Missing-value audit
# ---------------------------------------------------------------------------

def audit_missing(df: pd.DataFrame) -> dict:
    """
    Report missing values per column.

    Args:
        df: Raw DataFrame.

    Returns:
        Dict mapping column name → missing count (only for columns with > 0 missing).
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].to_dict()
    if missing:
        print(f"[Audit] Columns with missing values: {missing}")
    else:
        print("[Audit] No missing values detected ✓")
    return missing


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute or drop missing values.

    Strategy:
      - Numerical columns → median imputation (robust to outliers / fraud skew)
      - Rows with missing 'Class' label → dropped (can't train on unlabelled rows)

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame with no missing values.
    """
    # Drop rows without a label
    before = len(df)
    df = df.dropna(subset=["Class"])
    dropped = before - len(df)
    if dropped:
        print(f"[Missing] Dropped {dropped} rows with missing 'Class' label.")

    # Median-impute remaining numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[Missing] Imputed {n_missing} NaN in '{col}' with median={median_val:.4f}")

    return df


# ---------------------------------------------------------------------------
# Step 3 — Feature engineering & scaling
# ---------------------------------------------------------------------------

def scale_features(df: pd.DataFrame, return_scalers: bool = False):
    """Standardise Amount and Time and optionally return reusable fit values.

    The model-serving app must apply these exact training values to new rows;
    fitting a fresh scaler or using guessed constants changes the model input.
    """
    amount_scaler = StandardScaler()
    time_scaler = StandardScaler()
    df = df.copy()
    df["scaled_Amount"] = amount_scaler.fit_transform(df[["Amount"]])
    df["scaled_Time"] = time_scaler.fit_transform(df[["Time"]])
    df.drop(columns=["Time", "Amount"], inplace=True)
    print("[Scale] 'Amount' and 'Time' standardised → 'scaled_Amount', 'scaled_Time'")

    if not return_scalers:
        return df

    scaling = {
        "Amount": {"mean": float(amount_scaler.mean_[0]), "scale": float(amount_scaler.scale_[0])},
        "Time": {"mean": float(time_scaler.mean_[0]), "scale": float(time_scaler.scale_[0])},
    }
    return df, scaling


# Step 4 — Class imbalance handling
# ---------------------------------------------------------------------------

def get_class_distribution(df: pd.DataFrame) -> dict:
    """
    Return the count of each class.

    Args:
        df: DataFrame with 'Class' column.

    Returns:
        Dict {0: count_normal, 1: count_fraud}.
    """
    counts = df["Class"].value_counts().to_dict()
    total = sum(counts.values())
    fraud_pct = counts.get(1, 0) / total * 100
    print(f"[Class] Normal: {counts.get(0, 0):,}  |  Fraud: {counts.get(1, 0):,}  "
          f"({fraud_pct:.2f}% fraud)")
    return counts


def balance_dataset(df: pd.DataFrame, strategy: str = "undersample",
                    ratio: float = 1.0) -> pd.DataFrame:
    """
    Address the severe class imbalance (only ~0.17 % of rows are fraud).

    Strategies
    ----------
    'undersample' (default):
        Randomly subsample the majority class (normal) down to
        ratio × (number of fraud samples). Safe for a demo and avoids
        the risk of synthetic fraud data leaking information.

    Args:
        df: DataFrame with 'Class' column.
        strategy: 'undersample' (only option currently implemented).
        ratio: Majority:minority ratio after balancing (default 1.0 = equal).

    Returns:
        Balanced DataFrame.
    """
    if strategy != "undersample":
        raise ValueError(f"Strategy '{strategy}' not yet implemented.")

    fraud_df = df[df["Class"] == 1]
    normal_df = df[df["Class"] == 0]

    target_n = int(len(fraud_df) * ratio)
    normal_down = resample(normal_df, replace=False, n_samples=target_n,
                           random_state=RANDOM_STATE)

    balanced = pd.concat([normal_down, fraud_df]).sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print(f"[Balance] After undersampling — Normal: {len(normal_down):,}  "
          f"| Fraud: {len(fraud_df):,}  | Total: {len(balanced):,}")
    return balanced


# ---------------------------------------------------------------------------
# Step 5 — Train / test split
# ---------------------------------------------------------------------------

def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """
    Split the balanced dataset into stratified train / test sets.

    Stratification ensures both splits maintain the same class ratio.

    Args:
        df: Balanced DataFrame.
        test_size: Fraction for the test set (default 0.2 = 20 %).

    Returns:
        X_train, X_test, y_train, y_test (all NumPy arrays).
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"[Split] Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Step 6 — Quantum feature subset
# ---------------------------------------------------------------------------

def get_quantum_subset(X_train: pd.DataFrame, X_test: pd.DataFrame,
                       features: list = None):
    """
    Extract a small feature subset for the quantum model and re-scale it to
    [0, π] (suitable for angle encoding into quantum gates).

    Quantum note:
      Each feature maps to a qubit rotation angle; ZZFeatureMap uses pairs of
      qubits so having N features → N qubits → 2^N amplitudes to simulate.
      Keeping N ≤ 5 means at most 32 amplitudes — fast on a laptop.

    Args:
        X_train: Full training feature DataFrame.
        X_test:  Full test feature DataFrame.
        features: Column names to use (defaults to QUANTUM_FEATURES).

    Returns:
        X_train_q, X_test_q (NumPy arrays, scaled to [0, π])
    """
    if features is None:
        features = QUANTUM_FEATURES

    # Validate that all requested features exist
    missing_feat = [f for f in features if f not in X_train.columns]
    if missing_feat:
        raise ValueError(f"Features not found in dataset: {missing_feat}")

    X_train_q = X_train[features].values.astype(float)
    X_test_q = X_test[features].values.astype(float)

    # Rescale each feature to [0, π] using min/max from the training set
    # (min/max is fine here because the features are already PCA-bounded)
    mins = X_train_q.min(axis=0)
    maxs = X_train_q.max(axis=0)
    rng = np.where(maxs - mins == 0, 1.0, maxs - mins)  # avoid div-by-zero

    X_train_q = (X_train_q - mins) / rng * np.pi
    X_test_q = (X_test_q - mins) / rng * np.pi

    print(f"[Quantum] Feature subset: {features}")
    print(f"[Quantum] Scaled to [0, π]  — shape train: {X_train_q.shape}, "
          f"test: {X_test_q.shape}")
    return X_train_q, X_test_q


# ---------------------------------------------------------------------------
# Master pipeline — called by train.py
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(data_path: str = DATA_PATH):
    """
    Execute the full Stage-1 preprocessing pipeline and return all artefacts
    needed by later stages.

    Returns:
        dict with keys:
            df_raw          — original loaded DataFrame
            df_processed    — scaled, balanced DataFrame
            X_train         — classical training features (DataFrame)
            X_test          — classical test features (DataFrame)
            y_train         — training labels (Series)
            y_test          — test labels (Series)
            X_train_q       — quantum training features (ndarray, [0,π])
            X_test_q        — quantum test features (ndarray, [0,π])
            class_dist_raw  — class counts before balancing
            quantum_features — list of feature names used for QML
    """
    print("\n" + "=" * 60)
    print("  QuantumShield — Stage 1: Preprocessing Pipeline")
    print("=" * 60 + "\n")

    # 1. Load
    df_raw = load_dataset(data_path)

    # 2. Audit & fix missing values
    audit_missing(df_raw)
    df = handle_missing(df_raw)

    # 3. Class distribution (before balancing)
    class_dist_raw = get_class_distribution(df)

    # 4. Scale Amount & Time
    df, feature_scaling = scale_features(df, return_scalers=True)

    # 5. Balance
    df_balanced = balance_dataset(df, strategy="undersample", ratio=1.0)

    # 6. Train/test split
    X_train, X_test, y_train, y_test = split_data(df_balanced)

    # 7. Quantum feature subset
    X_train_q, X_test_q = get_quantum_subset(X_train, X_test)

    quantum_values = X_train[QUANTUM_FEATURES].to_numpy(dtype=float)
    quantum_min = quantum_values.min(axis=0)
    quantum_range = quantum_values.max(axis=0) - quantum_min
    quantum_range = np.where(quantum_range == 0, 1.0, quantum_range)
    quantum_scaling = {
        "features": list(QUANTUM_FEATURES),
        "min": quantum_min.tolist(),
        "range": quantum_range.tolist(),
    }

    print("\n[Pipeline] Stage 1 complete ✓\n")

    return {
        "df_raw": df_raw,
        "df_processed": df_balanced,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_q": X_train_q,
        "X_test_q": X_test_q,
        "class_dist_raw": class_dist_raw,
        "quantum_features": QUANTUM_FEATURES,
        "feature_scaling": feature_scaling,
        "quantum_scaling": quantum_scaling,
    }


# ---------------------------------------------------------------------------
# Utilities for Streamlit single-row inference
# ---------------------------------------------------------------------------

def preprocess_single_transaction(
    feature_dict: dict,
    quantum_features: list = None,
) -> tuple:
    """
    Preprocess a single transaction entered via the Streamlit form.

    Args:
        feature_dict: Dict mapping feature name → value (floats).
        quantum_features: Feature names for the QML model.

    Returns:
        (x_classical, x_quantum)
          x_classical — 1-D ndarray for the classical model (all features)
          x_quantum   — 1-D ndarray for the QML model (quantum_features only,
                        values clipped to [0, π])
    """
    if quantum_features is None:
        quantum_features = QUANTUM_FEATURES

    # Build a single-row DataFrame; any missing keys default to 0
    row = pd.DataFrame([feature_dict])

    # Classical: just return the values as-is (already scaled by caller)
    x_classical = row.values.astype(float)

    # Quantum: select subset and clip to [0, π]
    x_quantum_raw = row[[f for f in quantum_features if f in row.columns]].values.astype(float)
    x_quantum = np.clip(x_quantum_raw, 0, np.pi)

    return x_classical, x_quantum


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_preprocessing_pipeline()
    print(f"X_train shape     : {results['X_train'].shape}")
    print(f"X_test shape      : {results['X_test'].shape}")
    print(f"X_train_q shape   : {results['X_train_q'].shape}")
    print(f"Class dist (raw)  : {results['class_dist_raw']}")
    print(f"Quantum features  : {results['quantum_features']}")
