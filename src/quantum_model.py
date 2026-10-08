"""
QuantumShield — Quantum ML Module
===================================
Implements a realistic Quantum Machine Learning (QML) demonstration using
current Qiskit (≥ 1.0) and Qiskit Machine Learning (≥ 0.8) APIs.

Architecture
------------
Classical features (small subset)
    ↓
ZZFeatureMap  — encodes features as quantum rotation angles
    ↓
Quantum Kernel — computes an inner product in Hilbert space using a simulator
    ↓
QSVC (Quantum SVC) — SVM that uses the quantum kernel matrix
    ↓
Prediction: NORMAL or FRAUD

Why ZZFeatureMap?
-----------------
ZZFeatureMap creates entanglement between feature pairs using ZZ interactions,
which means the kernel implicitly captures quadratic correlations between
features in quantum Hilbert space — a region that is exponentially hard to
simulate classically for large qubit counts (though we keep it small here).

Why QSVC?
---------
QSVC is a standard SVC where the kernel matrix K[i,j] is computed by
measuring the overlap between quantum states |φ(xᵢ)⟩ and |φ(xⱼ)⟩.
This is the most straightforward and stable QML model available in
Qiskit Machine Learning.

Performance note:
-----------------
The simulation complexity is O(2^n) in memory where n = number of qubits.
With 5 qubits and a training set of ~400 samples, the kernel matrix
(400×400 evaluations) takes a few minutes on a typical laptop — acceptable
for a hackathon demo.
"""

import os
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Qiskit imports — using current Qiskit ≥ 1.0 API
# ---------------------------------------------------------------------------
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import StatevectorSampler

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Build the quantum components
# ---------------------------------------------------------------------------

def build_feature_map(num_features: int, reps: int = 2):
    """
    Build a ZZFeatureMap circuit.

    ZZFeatureMap structure (for 2 features, reps=2):

      ┌──────────────────────────────────────────────────────┐
      │  H   RZ(2x₀)   CX   RZ(2(π-x₀)(π-x₁))  CX   …     │  ← rep 1
      │  H   RZ(2x₁)                                         │
      └──────────────────────────────────────────────────────┘
      Repeated  `reps`  times.

    Args:
        num_features: Number of qubits / features.
        reps: Number of times the encoding block is repeated.

    Returns:
        ZZFeatureMap circuit.
    """
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=reps)
    print(f"[QML] ZZFeatureMap built — {num_features} qubits, {reps} reps")
    return feature_map


def build_quantum_kernel(feature_map):
    """
    Build a FidelityQuantumKernel using the statevector-based ComputeUncompute
    fidelity estimator.

    The kernel value between two data points x and y is:
        K(x, y) = |⟨φ(x)|φ(y)⟩|²
    where |φ(x)⟩ is the quantum state produced by the ZZFeatureMap for input x.

    Uses Qiskit's Sampler primitive (noiseless simulator by default).

    Args:
        feature_map: A ZZFeatureMap or compatible QuantumCircuit.

    Returns:
        FidelityQuantumKernel instance.
    """
    sampler = StatevectorSampler()                 # noiseless statevector sampler (Qiskit 2.x)
    fidelity = ComputeUncompute(sampler=sampler)  # computes |⟨ψ|φ⟩|²
    kernel = FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=fidelity,
    )
    print("[QML] FidelityQuantumKernel built ✓")
    return kernel


def build_qsvc(kernel) -> QSVC:
    """
    Build a QSVC (Quantum Support Vector Classifier) with the given kernel.

    Args:
        kernel: FidelityQuantumKernel instance.

    Returns:
        QSVC instance (not yet trained).
    """
    # Note: QSVC in qiskit-machine-learning 0.9.x passes **kwargs to sklearn SVC
    # 'probability=True' enables predict_proba but is slow — omit for speed
    qsvc = QSVC(quantum_kernel=kernel, C=1.0)
    print("[QML] QSVC instance created ✓")
    return qsvc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_qsvc(X_train_q: np.ndarray, y_train,
               num_features: int = None,
               reps: int = 2,
               max_samples: int = 400) -> tuple:
    """
    Train the full quantum pipeline (feature map → kernel → QSVC).

    To keep simulation time reasonable on a laptop we cap the number of
    training samples at `max_samples`. The kernel matrix is (n×n), so
    400 samples → 160,000 kernel evaluations — manageable in a few minutes.

    Args:
        X_train_q: Quantum feature array (already scaled to [0, π]).
        y_train:   Training labels.
        num_features: Number of quantum features (inferred from array if None).
        reps: ZZFeatureMap repetitions.
        max_samples: Hard cap on training samples for speed.

    Returns:
        (qsvc_model, feature_map, kernel) tuple.
    """
    if num_features is None:
        num_features = X_train_q.shape[1]

    # Subsample if needed
    n = len(X_train_q)
    if n > max_samples:
        idx = np.random.RandomState(RANDOM_STATE).choice(n, max_samples, replace=False)
        X_train_q = X_train_q[idx]
        y_train_arr = np.array(y_train)[idx]
        print(f"[QML] Subsampled training set: {n} → {max_samples} rows "
              f"(kernel matrix would be {n}×{n} = {n**2:,} evaluations otherwise)")
    else:
        y_train_arr = np.array(y_train)

    print(f"\n[QML] Starting QSVC training on {len(X_train_q)} samples …")
    print("[QML] This may take a few minutes — quantum simulation in progress …\n")

    feature_map = build_feature_map(num_features, reps=reps)
    kernel = build_quantum_kernel(feature_map)
    qsvc = build_qsvc(kernel)

    qsvc.fit(X_train_q, y_train_arr)
    print("\n[QML] QSVC training complete ✓")
    return qsvc, feature_map, kernel


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_qsvc(qsvc: QSVC, X_test_q: np.ndarray) -> np.ndarray:
    """
    Predict classes for test samples using the trained QSVC.

    Args:
        qsvc: Trained QSVC model.
        X_test_q: Test quantum feature array (scaled to [0, π]).

    Returns:
        1-D ndarray of predicted labels (0 or 1).
    """
    print(f"[QML] Predicting on {len(X_test_q)} test samples …")
    preds = qsvc.predict(X_test_q)
    print("[QML] Prediction complete ✓")
    return preds


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_qsvc(qsvc: QSVC, name: str = "qsvc_model"):
    """Save the trained QSVC model to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(qsvc, path)
    print(f"[QML Save] Model saved → {path}")
    return path


def load_qsvc(name: str = "qsvc_model") -> QSVC:
    """Load a previously saved QSVC from disk."""
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"QSVC model not found: {path}")
    model = joblib.load(path)
    print(f"[QML Load] Model loaded ← {path}")
    return model


# ---------------------------------------------------------------------------
# Circuit visualisation helper (for Streamlit)
# ---------------------------------------------------------------------------

def get_circuit_diagram(num_features: int = 5, reps: int = 2) -> str:
    """
    Return a text drawing of the ZZFeatureMap circuit.

    Args:
        num_features: Number of qubits.
        reps: Circuit repetitions.

    Returns:
        Multi-line string containing the ASCII circuit diagram.
    """
    fm = ZZFeatureMap(feature_dimension=num_features, reps=reps)
    return fm.decompose().draw(output="text").single_string()


def get_circuit_figure(num_features: int = 5, reps: int = 2):
    """
    Return a Matplotlib figure of the ZZFeatureMap circuit.

    Args:
        num_features: Number of qubits.
        reps: Circuit repetitions.

    Returns:
        Matplotlib Figure object.
    """
    fm = ZZFeatureMap(feature_dimension=num_features, reps=reps)
    fig = fm.decompose().draw(output="mpl", style="clifford")
    return fig


# ---------------------------------------------------------------------------
# Master quantum pipeline — called by train.py
# ---------------------------------------------------------------------------

def run_quantum_pipeline(X_train_q: np.ndarray, X_test_q: np.ndarray,
                         y_train, y_test,
                         max_samples: int = 400):
    """
    Run the complete quantum training and prediction pipeline.

    Args:
        X_train_q, X_test_q: Quantum feature arrays (scaled to [0, π]).
        y_train, y_test:      Labels.
        max_samples: Cap on training set size for simulator speed.

    Returns:
        dict with keys:
            qsvc_model  — trained QSVC
            feature_map — ZZFeatureMap circuit
            kernel      — FidelityQuantumKernel
            qsvc_preds  — predictions on X_test_q
    """
    print("\n" + "=" * 60)
    print("  QuantumShield — Stage 3: Quantum ML Pipeline")
    print("=" * 60 + "\n")

    qsvc, feature_map, kernel = train_qsvc(
        X_train_q, y_train, max_samples=max_samples
    )
    qsvc_preds = predict_qsvc(qsvc, X_test_q)

    save_qsvc(qsvc)

    print("\n[Pipeline] Stage 3 complete ✓\n")

    return {
        "qsvc_model": qsvc,
        "feature_map": feature_map,
        "kernel": kernel,
        "qsvc_preds": qsvc_preds,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.preprocessing import run_preprocessing_pipeline

    data = run_preprocessing_pipeline()
    results = run_quantum_pipeline(
        data["X_train_q"], data["X_test_q"],
        data["y_train"], data["y_test"],
        max_samples=100,   # use tiny sample for self-test
    )
    print(f"QSVC predictions sample: {results['qsvc_preds'][:10]}")
    print(f"Circuit diagram:\n{get_circuit_diagram(5)}")
