"""
QuantumShield — Evaluation Module
====================================
Computes and packages all evaluation metrics for classical and quantum models.

Metrics computed per model:
  - Accuracy
  - Precision (fraud class)
  - Recall    (fraud class)
  - F1-score  (fraud class)
  - Confusion matrix

Visualisation helpers return Matplotlib figures that Streamlit can display
directly via st.pyplot().
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server / Streamlit use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import warnings

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ---------------------------------------------------------------------------
# Core metric calculation
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, model_name: str = "Model") -> dict:
    """
    Compute all classification metrics for a single model.

    Precision / Recall / F1 are reported for the positive (fraud) class,
    since that is what matters operationally.

    Args:
        y_true:     Ground-truth labels.
        y_pred:     Predicted labels.
        model_name: Human-readable label used in print output.

    Returns:
        dict with keys: model, accuracy, precision, recall, f1, confusion_matrix.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n── {model_name} ──")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    print(classification_report(y_true, y_pred,
                                target_names=["Normal", "Fraud"],
                                zero_division=0))

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }


def compute_all_metrics(y_test, lr_preds, rf_preds, qsvc_preds) -> list:
    """
    Compute metrics for all three models.

    Args:
        y_test:     Ground-truth labels.
        lr_preds:   Logistic Regression predictions.
        rf_preds:   Random Forest predictions.
        qsvc_preds: QSVC predictions.

    Returns:
        List of metric dicts, one per model.
    """
    print("\n" + "=" * 60)
    print("  QuantumShield — Stage 4: Evaluation")
    print("=" * 60)

    metrics = [
        compute_metrics(y_test, lr_preds,   "Logistic Regression"),
        compute_metrics(y_test, rf_preds,   "Random Forest"),
        compute_metrics(y_test, qsvc_preds, "QSVC (Quantum)"),
    ]

    print("\n[Pipeline] Stage 4 complete ✓\n")
    return metrics


# ---------------------------------------------------------------------------
# Visualisation — Bar chart comparison
# ---------------------------------------------------------------------------

def plot_metric_comparison(metrics_list: list, save_path: str = None):
    """
    Create a grouped bar chart comparing Accuracy, Precision, Recall, F1
    across all models.

    Args:
        metrics_list: List of metric dicts (from compute_all_metrics).
        save_path: Optional file path to save the figure as PNG.

    Returns:
        Matplotlib Figure.
    """
    metric_names = ["accuracy", "precision", "recall", "f1"]
    display_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names = [m["model"] for m in metrics_list]

    x = np.arange(len(display_names))
    width = 0.25
    n_models = len(model_names)
    offsets = np.linspace(-(n_models - 1) * width / 2,
                          (n_models - 1) * width / 2, n_models)

    # Colour palette
    colours = ["#6C63FF", "#FF6584", "#43D9AD"]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1a1d2e")

    for i, (m, colour, offset) in enumerate(zip(metrics_list, colours, offsets)):
        values = [m[k] for k in metric_names]
        bars = ax.bar(x + offset, values, width, label=m["model"],
                      color=colour, alpha=0.85, edgecolor="white", linewidth=0.5)
        # Value labels on top of bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8, color="white", fontweight="bold")

    ax.set_xlabel("Metric", color="white", fontsize=13)
    ax.set_ylabel("Score", color="white", fontsize=13)
    ax.set_title("Model Performance Comparison — Classical vs Quantum",
                 color="white", fontsize=15, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, color="white", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#3a3d52")
    ax.legend(facecolor="#1a1d2e", edgecolor="#3a3d52",
              labelcolor="white", fontsize=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    return fig


# ---------------------------------------------------------------------------
# Visualisation — Confusion matrices (side by side)
# ---------------------------------------------------------------------------

def plot_confusion_matrices(metrics_list: list, save_path: str = None):
    """
    Plot confusion matrices for all models side by side.

    Args:
        metrics_list: List of metric dicts.
        save_path: Optional file path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    n = len(metrics_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.patch.set_facecolor("#0e1117")

    for ax, m in zip(axes, metrics_list):
        cm = m["confusion_matrix"]
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Normal", "Fraud"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(m["model"], color="white", fontsize=11, fontweight="bold")
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for text in ax.texts:
            text.set_color("white")

    fig.suptitle("Confusion Matrices", color="white",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    return fig


# ---------------------------------------------------------------------------
# Visualisation — Radar / spider chart
# ---------------------------------------------------------------------------

def plot_radar_chart(metrics_list: list, save_path: str = None):
    """
    Create a radar (spider) chart comparing models across all metrics.

    Args:
        metrics_list: List of metric dicts.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
    keys = ["accuracy", "precision", "recall", "f1"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]   # close the polygon

    colours = ["#6C63FF", "#FF6584", "#43D9AD"]

    fig = plt.figure(figsize=(7, 7))
    fig.patch.set_facecolor("#0e1117")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#1a1d2e")

    for m, colour in zip(metrics_list, colours):
        values = [m[k] for k in keys]
        values += values[:1]   # close
        ax.plot(angles, values, "o-", linewidth=2, color=colour, label=m["model"])
        ax.fill(angles, values, alpha=0.15, color=colour)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="white", fontsize=11)
    ax.set_ylim(0, 1)
    ax.yaxis.set_tick_params(labelcolor="white")
    ax.grid(color="#3a3d52", linewidth=0.8)
    ax.spines["polar"].set_color("#3a3d52")
    ax.set_title("Performance Radar", color="white",
                 fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              facecolor="#1a1d2e", edgecolor="#3a3d52",
              labelcolor="white", fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    return fig


# ---------------------------------------------------------------------------
# Metrics → DataFrame (for Streamlit tables)
# ---------------------------------------------------------------------------

def metrics_to_dataframe(metrics_list: list) -> pd.DataFrame:
    """
    Convert a list of metric dicts into a tidy DataFrame for display.

    Args:
        metrics_list: List of metric dicts.

    Returns:
        DataFrame with model names as rows and metrics as columns.
    """
    rows = []
    for m in metrics_list:
        rows.append({
            "Model": m["model"],
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1-Score": f"{m['f1']:.4f}",
        })
    return pd.DataFrame(rows).set_index("Model")


# ---------------------------------------------------------------------------
# Master evaluation pipeline — called by train.py
# ---------------------------------------------------------------------------

def run_evaluation_pipeline(y_test, lr_preds, rf_preds, qsvc_preds,
                             save_figures: bool = True) -> dict:
    """
    Compute metrics and generate all comparison figures.

    Args:
        y_test:     Ground-truth test labels.
        lr_preds:   Logistic Regression predictions.
        rf_preds:   Random Forest predictions.
        qsvc_preds: QSVC predictions.
        save_figures: Whether to save PNGs to disk.

    Returns:
        dict with keys:
            metrics_list    — list of metric dicts
            metrics_df      — tidy DataFrame
            fig_comparison  — grouped bar chart figure
            fig_cm          — confusion matrix figure
            fig_radar       — radar chart figure
    """
    metrics_list = compute_all_metrics(y_test, lr_preds, rf_preds, qsvc_preds)
    df = metrics_to_dataframe(metrics_list)

    bar_path = os.path.join(MODELS_DIR, "metric_comparison.png") if save_figures else None
    cm_path  = os.path.join(MODELS_DIR, "confusion_matrices.png") if save_figures else None
    rad_path = os.path.join(MODELS_DIR, "radar_chart.png")        if save_figures else None

    fig_bar   = plot_metric_comparison(metrics_list, save_path=bar_path)
    fig_cm    = plot_confusion_matrices(metrics_list, save_path=cm_path)
    fig_radar = plot_radar_chart(metrics_list, save_path=rad_path)

    return {
        "metrics_list": metrics_list,
        "metrics_df": df,
        "fig_comparison": fig_bar,
        "fig_cm": fig_cm,
        "fig_radar": fig_radar,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate dummy predictions for testing the evaluation module
    rng = np.random.default_rng(42)
    y_test = rng.integers(0, 2, size=200)
    lr_preds   = rng.integers(0, 2, size=200)
    rf_preds   = rng.integers(0, 2, size=200)
    qsvc_preds = rng.integers(0, 2, size=200)

    results = run_evaluation_pipeline(y_test, lr_preds, rf_preds, qsvc_preds,
                                      save_figures=False)
    print(results["metrics_df"])
