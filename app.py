"""
QuantumShield — app.py
========================
Streamlit dashboard for the Quantum ML Fraud Detection System.

Pages:
  🏠 Dashboard           — Overview & key metrics
  🔍 Transaction Analysis — Single-transaction prediction form
  📊 Model Comparison    — Side-by-side classical vs quantum metrics
  📈 Dataset Statistics  — Dataset overview and distributions
  ⚛️  Quantum Circuit     — Circuit visualisation and QML explanation
  ℹ️  About              — Project info and how QML works

Run:
    streamlit run app.py
"""

import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="QuantumShield — Fraud Detection",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1529 50%, #0e1117 100%);
}

/* Sidebar */
.css-1d391kg, [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1529 0%, #111827 100%);
    border-right: 1px solid #1e293b;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #1e2540 100%);
    border: 1px solid #2d3561;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(108, 99, 255, 0.1);
}
.metric-card:hover {
    border-color: #6C63FF;
    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.25);
    transform: translateY(-2px);
}
.metric-label {
    font-size: 12px;
    font-weight: 600;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(90deg, #6C63FF, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-sub {
    font-size: 11px;
    color: #64748b;
    margin-top: 4px;
}

/* Hero section */
.hero-banner {
    background: linear-gradient(135deg, #1a1d2e 0%, #1e2540 50%, #0d1b3e 100%);
    border: 1px solid #2d3561;
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(108,99,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(90deg, #6C63FF 0%, #a78bfa 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 16px;
    color: #8892b0;
    margin-top: 12px;
    font-weight: 400;
    max-width: 600px;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-normal  { background: rgba(67, 217, 173, 0.15); color: #43D9AD; border: 1px solid #43D9AD; }
.badge-fraud   { background: rgba(255, 101, 132, 0.15); color: #FF6584; border: 1px solid #FF6584; }
.badge-quantum { background: rgba(108, 99, 255, 0.15);  color: #6C63FF; border: 1px solid #6C63FF; }

/* Section headers */
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 4px;
    padding-bottom: 8px;
    border-bottom: 2px solid #6C63FF;
    display: inline-block;
}

/* Info boxes */
.info-box {
    background: linear-gradient(135deg, #1a1d2e, #1e2540);
    border-left: 4px solid #6C63FF;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #cbd5e1;
    font-size: 14px;
    line-height: 1.6;
}
.warning-box {
    background: linear-gradient(135deg, #1a1d2e, #1e2540);
    border-left: 4px solid #FF6584;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #cbd5e1;
    font-size: 14px;
}
.success-box {
    background: linear-gradient(135deg, #1a1d2e, #1e2540);
    border-left: 4px solid #43D9AD;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #cbd5e1;
    font-size: 14px;
}

/* Pipeline diagram */
.pipeline-step {
    background: linear-gradient(135deg, #1a1d2e, #1e2540);
    border: 1px solid #2d3561;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    color: #e2e8f0;
    font-weight: 600;
    font-size: 14px;
    transition: all 0.3s;
}
.pipeline-step:hover {
    border-color: #6C63FF;
    box-shadow: 0 0 20px rgba(108,99,255,0.2);
}

/* Result cards */
.result-normal {
    background: linear-gradient(135deg, rgba(67,217,173,0.1), rgba(67,217,173,0.05));
    border: 2px solid #43D9AD;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.result-fraud {
    background: linear-gradient(135deg, rgba(255,101,132,0.1), rgba(255,101,132,0.05));
    border: 2px solid #FF6584;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}

/* Table styling */
.dataframe {
    background: #1a1d2e !important;
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# State & model loading helpers
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "creditcard.csv")


def load_classical_models():
    """Load pre-trained classical models from disk."""
    import joblib
    lr_path = os.path.join(MODELS_DIR, "logistic_regression.joblib")
    rf_path = os.path.join(MODELS_DIR, "random_forest.joblib")
    models = {}
    if os.path.exists(lr_path):
        models["lr"] = joblib.load(lr_path)
    if os.path.exists(rf_path):
        models["rf"] = joblib.load(rf_path)
    return models


def load_quantum_model():
    """Load pre-trained QSVC from disk if available."""
    import joblib
    qsvc_path = os.path.join(MODELS_DIR, "qsvc_model.joblib")
    if os.path.exists(qsvc_path):
        return joblib.load(qsvc_path)
    return None


@st.cache_data(show_spinner="Loading dataset …")
def load_dataset_cached():
    """Load the raw dataset for statistics."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


def load_metadata():
    """Load training metadata JSON."""
    path = os.path.join(MODELS_DIR, "training_metadata.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def models_trained() -> bool:
    """Return True if the training has been completed."""
    return os.path.exists(os.path.join(MODELS_DIR, "logistic_regression.joblib"))


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:40px;'>⚛️</div>
        <div style='font-size:20px; font-weight:800; background: linear-gradient(90deg,#6C63FF,#a78bfa);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text;'>QuantumShield</div>
        <div style='font-size:11px; color:#64748b; margin-top:4px;'>
            Quantum ML Fraud Detection
        </div>
    </div>
    <hr style='border-color:#1e293b; margin:12px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        options=[
            "🏠  Dashboard",
            "🔍  Transaction Analysis",
            "📊  Model Comparison",
            "📈  Dataset Statistics",
            "⚛️   Quantum Circuit",
            "ℹ️   About & QML Guide",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#1e293b;'>", unsafe_allow_html=True)

    # Training status indicator
    if models_trained():
        st.markdown("""
        <div class='success-box'>
        ✅ <strong>Models Trained</strong><br>
        <small>All models ready for inference</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='warning-box'>
        ⚠️ <strong>Models Not Trained</strong><br>
        <small>Run: <code>python train.py</code></small>
        </div>
        """, unsafe_allow_html=True)

    metadata = load_metadata()
    if metadata:
        st.markdown(f"""
        <div class='info-box'>
        📦 <strong>Training Info</strong><br>
        Train: {metadata.get('train_size','—')} samples<br>
        Test:  {metadata.get('test_size','—')} samples<br>
        QML:   {'✅ Yes' if metadata.get('quantum_trained') else '⚠️ Skipped'}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='position:absolute; bottom:20px; left:0; right:0; text-align:center;
                font-size:10px; color:#374151;'>
        QuantumShield v1.0 · Hackathon 2026
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# Page: Dashboard
# ===========================================================================

if "Dashboard" in page:
    # Hero banner
    st.markdown("""
    <div class='hero-banner'>
        <p class='hero-title'>⚛️ QuantumShield</p>
        <p class='hero-subtitle'>
            Quantum Machine Learning meets Financial Fraud Detection.
            Comparing classical and quantum algorithms on real credit-card transaction data.
        </p>
        <div style='margin-top:16px;'>
            <span class='badge badge-quantum'>ZZFeatureMap</span>&nbsp;
            <span class='badge badge-quantum'>Quantum Kernel</span>&nbsp;
            <span class='badge badge-quantum'>QSVC</span>&nbsp;
            <span class='badge badge-normal'>Random Forest</span>&nbsp;
            <span class='badge badge-normal'>Logistic Regression</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not models_trained():
        st.warning("⚠️  Models have not been trained yet. Run `python train.py` first, then refresh this page.")
    else:
        metadata = load_metadata()
        metrics_list = metadata.get("metrics", []) if metadata else []

        # ── KPI row ─────────────────────────────────────────────────────
        if metrics_list:
            st.markdown("<p class='section-title'>📌 Key Results</p>", unsafe_allow_html=True)
            cols = st.columns(4)
            best = max(metrics_list, key=lambda x: x.get("f1", 0))

            kpis = [
                ("Best F1-Score", f"{best.get('f1', 0):.3f}", best.get('model', '—')),
                ("Best Accuracy", f"{max(m.get('accuracy',0) for m in metrics_list):.3f}", ""),
                ("Best Recall",   f"{max(m.get('recall',0)   for m in metrics_list):.3f}", "Fraud Detection Rate"),
                ("Models Compared", str(len(metrics_list)), "Classical + Quantum"),
            ]
            for col, (label, value, sub) in zip(cols, kpis):
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>{label}</div>
                        <div class='metric-value'>{value}</div>
                        <div class='metric-sub'>{sub}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Metrics overview table ───────────────────────────────────────
        if metrics_list:
            st.markdown("<p class='section-title'>📋 Model Metrics Overview</p>",
                        unsafe_allow_html=True)
            rows = []
            for m in metrics_list:
                rows.append({
                    "Model": m["model"],
                    "Accuracy":  f"{m.get('accuracy',0):.4f}",
                    "Precision": f"{m.get('precision',0):.4f}",
                    "Recall":    f"{m.get('recall',0):.4f}",
                    "F1-Score":  f"{m.get('f1',0):.4f}",
                })
            df_table = pd.DataFrame(rows)
            st.dataframe(df_table, use_container_width=True, hide_index=True)

        # ── Comparison bar chart (Plotly) ────────────────────────────────
        if metrics_list:
            st.markdown("<p class='section-title'>📊 Performance Comparison</p>",
                        unsafe_allow_html=True)
            metric_keys = ["accuracy", "precision", "recall", "f1"]
            metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
            colours = ["#6C63FF", "#FF6584", "#43D9AD"]

            fig = go.Figure()
            for m, colour in zip(metrics_list, colours):
                fig.add_trace(go.Bar(
                    name=m["model"],
                    x=metric_labels,
                    y=[m.get(k, 0) for k in metric_keys],
                    marker_color=colour,
                    marker_line_color="white",
                    marker_line_width=0.5,
                    opacity=0.85,
                    text=[f"{m.get(k,0):.3f}" for k in metric_keys],
                    textposition="outside",
                    textfont=dict(color="white", size=11),
                ))
            fig.update_layout(
                barmode="group",
                plot_bgcolor="#1a1d2e",
                paper_bgcolor="#0e1117",
                font=dict(color="white", family="Inter"),
                legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3561",
                            borderwidth=1, font=dict(color="white")),
                xaxis=dict(gridcolor="#2d3561", tickfont=dict(color="white", size=12)),
                yaxis=dict(gridcolor="#2d3561", tickfont=dict(color="white"),
                           range=[0, 1.15]),
                height=420,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Pipeline diagram ─────────────────────────────────────────────
        st.markdown("<p class='section-title'>🔄 System Architecture</p>",
                    unsafe_allow_html=True)
        cols = st.columns(7)
        steps = [
            ("📂", "Raw Data", "creditcard.csv"),
            ("→", "", ""),
            ("⚙️", "Preprocess", "Scale · Balance"),
            ("→", "", ""),
            ("🧠", "Train Models", "LR · RF · QSVC"),
            ("→", "", ""),
            ("🎯", "Predict", "Fraud / Normal"),
        ]
        for col, (icon, label, sub) in zip(cols, steps):
            with col:
                if icon == "→":
                    st.markdown("<div style='text-align:center;color:#6C63FF;font-size:24px;padding-top:20px;'>→</div>",
                                unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='pipeline-step'>
                        <div style='font-size:24px;margin-bottom:6px;'>{icon}</div>
                        <div style='color:#e2e8f0;font-weight:600;'>{label}</div>
                        <div style='font-size:11px;color:#64748b;margin-top:4px;'>{sub}</div>
                    </div>
                    """, unsafe_allow_html=True)


# ===========================================================================
# Page: Transaction Analysis
# ===========================================================================

elif "Transaction" in page:
    st.markdown("<p class='hero-title' style='font-size:32px;'>🔍 Transaction Analysis</p>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#8892b0;'>Enter transaction features to detect potential fraud.</p>",
                unsafe_allow_html=True)

    if not models_trained():
        st.error("❌ Models not trained. Run `python train.py` first.")
        st.stop()

    models = load_classical_models()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p class='section-title'>🧮 Transaction Features</p>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    ℹ️ Features V1–V28 are PCA-transformed components of the original transaction data
    (values are anonymised). <strong>Amount</strong> is the transaction value in EUR.
    Typical fraud transactions often show anomalous values in V1, V4, V11, V14, V17.
    </div>
    """, unsafe_allow_html=True)

    # Build form
    with st.form("transaction_form"):
        st.markdown("**PCA Components (V1 – V17)**")

        # Row 1: V1–V6
        row1 = st.columns(6)
        v_vals = {}
        feat_names = [f"V{i}" for i in range(1, 18)] + ["scaled_Amount", "scaled_Time"]

        v_defaults = {
            "V1": -1.36, "V2": -0.07, "V3": 2.53, "V4": 1.38,
            "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
            "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
            "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
            "V17": 0.21,
        }

        col_groups = [st.columns(6), st.columns(6), st.columns(5)]
        feat_list = [f"V{i}" for i in range(1, 18)]
        idx = 0
        for cols in col_groups:
            for col in cols:
                if idx < len(feat_list):
                    fn = feat_list[idx]
                    with col:
                        v_vals[fn] = st.number_input(
                            fn, value=float(v_defaults.get(fn, 0.0)),
                            format="%.4f", step=0.0001,
                            key=f"feat_{fn}"
                        )
                    idx += 1

        st.markdown("<br>**Transaction Details**", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([2, 2, 4])
        with c1:
            amount = st.number_input("Amount (EUR)", min_value=0.0,
                                     value=149.62, format="%.2f")
        with c2:
            time_s = st.number_input("Time (seconds elapsed)", min_value=0.0,
                                     value=0.0, format="%.0f")

        submitted = st.form_submit_button("🔎 Analyse Transaction", type="primary",
                                          use_container_width=True)

    if submitted:
        from sklearn.preprocessing import StandardScaler

        # Build feature vector for classical models
        feature_dict = dict(v_vals)
        feature_dict["scaled_Amount"] = (amount - 88.35) / 250.12   # approximate scaler
        feature_dict["scaled_Time"]   = (time_s - 94813.86) / 47488.15

        # Fill remaining V18–V28 with 0 (form only shows V1–V17 for clarity)
        for i in range(18, 29):
            feature_dict[f"V{i}"] = 0.0

        # The model was trained on these columns
        train_cols = [f"V{i}" for i in range(1, 29)] + ["scaled_Amount", "scaled_Time"]
        X_single = np.array([[feature_dict.get(c, 0.0) for c in train_cols]])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p class='section-title'>🎯 Prediction Results</p>",
                    unsafe_allow_html=True)

        res_cols = st.columns(2)

        for col, (model_key, model_name) in zip(res_cols, [("lr", "Logistic Regression"),
                                                             ("rf", "Random Forest")]):
            if model_key not in models:
                continue
            model = models[model_key]
            pred = model.predict(X_single)[0]
            proba = model.predict_proba(X_single)[0]
            fraud_prob = proba[1]
            normal_prob = proba[0]

            is_fraud = (pred == 1)
            box_class = "result-fraud" if is_fraud else "result-normal"
            label     = "⚠️ POTENTIAL FRAUD" if is_fraud else "✅ NORMAL"
            color     = "#FF6584" if is_fraud else "#43D9AD"

            with col:
                st.markdown(f"""
                <div class='{box_class}'>
                    <div style='font-size:13px;color:#8892b0;font-weight:600;
                                text-transform:uppercase;letter-spacing:1px;'>
                        {model_name}
                    </div>
                    <div style='font-size:28px;font-weight:800;color:{color};
                                margin:12px 0;'>{label}</div>
                    <div style='font-size:14px;color:#e2e8f0;'>
                        Fraud confidence: <strong style='color:{color};'>
                        {fraud_prob:.1%}</strong>
                    </div>
                    <div style='font-size:14px;color:#e2e8f0;'>
                        Normal confidence: <strong style='color:#43D9AD;'>
                        {normal_prob:.1%}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Gauge chart
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fraud_prob * 100,
                    title=dict(text="Fraud Risk %", font=dict(color="white", size=14)),
                    number=dict(suffix="%", font=dict(color="white", size=20)),
                    gauge=dict(
                        axis=dict(range=[0, 100],
                                  tickcolor="white", tickfont=dict(color="white")),
                        bar=dict(color=color),
                        bgcolor="#1a1d2e",
                        bordercolor="#2d3561",
                        steps=[
                            dict(range=[0, 30],  color="#1a2e1a"),
                            dict(range=[30, 70], color="#2e2a1a"),
                            dict(range=[70, 100], color="#2e1a1a"),
                        ],
                        threshold=dict(
                            line=dict(color="white", width=2),
                            thickness=0.75, value=50
                        ),
                    ),
                ))
                gauge.update_layout(
                    paper_bgcolor="#0e1117",
                    font=dict(color="white"),
                    height=220,
                    margin=dict(t=30, b=10, l=10, r=10),
                )
                st.plotly_chart(gauge, use_container_width=True)

        # Quantum note
        qsvc = load_quantum_model()
        if qsvc is not None:
            metadata = load_metadata()
            q_features = metadata.get("quantum_features", ["V1","V4","V11","V14","V17"])

            X_q = np.array([[
                np.clip(feature_dict.get(f, 0.0), 0, np.pi)
                for f in q_features
            ]])
            try:
                q_pred = qsvc.predict(X_q)[0]
                q_label = "⚠️ POTENTIAL FRAUD" if q_pred == 1 else "✅ NORMAL"
                q_color = "#FF6584" if q_pred == 1 else "#43D9AD"
                st.markdown(f"""
                <div class='result-{"fraud" if q_pred == 1 else "normal"}' style='margin-top:16px;'>
                    <div style='font-size:13px;color:#8892b0;font-weight:600;
                                text-transform:uppercase;letter-spacing:1px;'>
                        ⚛️ QSVC (Quantum Model)
                    </div>
                    <div style='font-size:24px;font-weight:800;color:{q_color};margin:10px 0;'>
                        {q_label}
                    </div>
                    <div style='font-size:13px;color:#8892b0;'>
                        Based on features: {', '.join(q_features)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.info(f"ℹ️ Quantum prediction unavailable: {e}")
        else:
            st.markdown("""
            <div class='info-box'>
            ⚛️ Quantum (QSVC) model not found.
            Train with: <code>python train.py</code>
            </div>
            """, unsafe_allow_html=True)


# ===========================================================================
# Page: Model Comparison
# ===========================================================================

elif "Comparison" in page:
    st.markdown("<p class='hero-title' style='font-size:32px;'>📊 Model Comparison</p>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#8892b0;'>Side-by-side evaluation of classical and quantum models.</p>",
                unsafe_allow_html=True)

    if not models_trained():
        st.error("❌ Models not trained. Run `python train.py` first.")
        st.stop()

    metadata = load_metadata()
    if not metadata:
        st.warning("⚠️ Metadata not found. Please re-run training.")
        st.stop()

    metrics_list = metadata.get("metrics", [])

    # Metrics table
    st.markdown("<p class='section-title'>📋 Metrics Table</p>", unsafe_allow_html=True)
    rows = []
    for m in metrics_list:
        rows.append({
            "Model": m["model"],
            "Accuracy":  f"{m.get('accuracy',0):.4f}",
            "Precision": f"{m.get('precision',0):.4f}",
            "Recall":    f"{m.get('recall',0):.4f}",
            "F1-Score":  f"{m.get('f1',0):.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Load saved figures
    st.markdown("<br><p class='section-title'>📊 Bar Chart Comparison</p>",
                unsafe_allow_html=True)
    bar_path = os.path.join(MODELS_DIR, "metric_comparison.png")
    if os.path.exists(bar_path):
        st.image(bar_path, use_column_width=True)
    else:
        st.info("Run training to generate comparison charts.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<p class='section-title'>🕸️ Radar Chart</p>", unsafe_allow_html=True)
        rad_path = os.path.join(MODELS_DIR, "radar_chart.png")
        if os.path.exists(rad_path):
            st.image(rad_path, use_column_width=True)

    with col2:
        st.markdown("<p class='section-title'>🔲 Confusion Matrices</p>",
                    unsafe_allow_html=True)
        cm_path = os.path.join(MODELS_DIR, "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_column_width=True)

    # Quantum advantage disclaimer
    st.markdown("""
    <div class='warning-box'>
    <strong>⚠️ Important — Quantum Advantage Disclaimer</strong><br>
    On this small, real-world dataset with a simulator, we do <em>not</em> claim quantum
    advantage. QSVC runs on <strong>only 5 features</strong> (limited by qubit count) while
    classical models use all 30 features. The purpose of the quantum model here is
    <strong>educational demonstration</strong> of QML feasibility, not a performance claim.
    Genuine quantum advantage in ML is still an active research area.
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# Page: Dataset Statistics
# ===========================================================================

elif "Dataset" in page:
    st.markdown("<p class='hero-title' style='font-size:32px;'>📈 Dataset Statistics</p>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#8892b0;'>Credit Card Fraud Detection — Kaggle ULB Dataset</p>",
                unsafe_allow_html=True)

    df = load_dataset_cached()
    if df is None:
        st.error("❌ Dataset not found. Place `creditcard.csv` in the `data/` directory.")
        st.stop()

    # Basic stats
    n_total = len(df)
    n_fraud = df["Class"].sum()
    n_normal = n_total - n_fraud
    fraud_pct = n_fraud / n_total * 100

    cols = st.columns(4)
    kpis = [
        ("Total Transactions", f"{n_total:,}", ""),
        ("Normal",             f"{n_normal:,}", f"{100-fraud_pct:.2f}%"),
        ("Fraud",              f"{n_fraud:,}",  f"{fraud_pct:.4f}%"),
        ("Features",           "30",            "V1–V28 + Time + Amount"),
    ]
    for col, (label, value, sub) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{value}</div>
                <div class='metric-sub'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Class imbalance donut chart
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("<p class='section-title'>🍩 Class Distribution</p>",
                    unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            labels=["Normal", "Fraud"],
            values=[n_normal, n_fraud],
            hole=0.65,
            marker_colors=["#43D9AD", "#FF6584"],
            textinfo="label+percent",
            textfont=dict(color="white", size=13),
        ))
        fig_donut.add_annotation(
            text=f"{fraud_pct:.3f}%<br><span style='font-size:12px;'>fraud rate</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="white", size=16),
        )
        fig_donut.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="white"),
            legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3561",
                        borderwidth=1, font=dict(color="white")),
            height=350, margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        st.markdown("<p class='section-title'>💰 Amount Distribution by Class</p>",
                    unsafe_allow_html=True)
        fig_amount = go.Figure()
        fig_amount.add_trace(go.Histogram(
            x=df[df["Class"] == 0]["Amount"],
            name="Normal",
            marker_color="#43D9AD",
            opacity=0.7,
            xbins=dict(size=20),
        ))
        fig_amount.add_trace(go.Histogram(
            x=df[df["Class"] == 1]["Amount"],
            name="Fraud",
            marker_color="#FF6584",
            opacity=0.7,
            xbins=dict(size=20),
        ))
        fig_amount.update_layout(
            barmode="overlay",
            paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
            font=dict(color="white"),
            xaxis_title="Amount (EUR)",
            yaxis_title="Count",
            legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3561",
                        borderwidth=1, font=dict(color="white")),
            height=350, margin=dict(t=10, b=10),
            xaxis=dict(range=[0, 2000], gridcolor="#2d3561"),
            yaxis=dict(gridcolor="#2d3561"),
        )
        st.plotly_chart(fig_amount, use_container_width=True)

    # Feature correlation heatmap (sample)
    st.markdown("<p class='section-title'>🌡️ Feature Correlation (Sample)</p>",
                unsafe_allow_html=True)
    sample_cols = ["V1", "V4", "V11", "V14", "V17", "Amount", "Class"]
    corr = df[sample_cols].corr()
    fig_heat = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        title="",
    )
    fig_heat.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
        font=dict(color="white"),
        coloraxis_colorbar=dict(title="r", tickfont=dict(color="white"),
                                titlefont=dict(color="white")),
        height=400, margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Raw stats table
    st.markdown("<p class='section-title'>📊 Summary Statistics</p>",
                unsafe_allow_html=True)
    st.dataframe(df[sample_cols].describe().round(4), use_container_width=True)


# ===========================================================================
# Page: Quantum Circuit
# ===========================================================================

elif "Quantum Circuit" in page:
    st.markdown("<p class='hero-title' style='font-size:32px;'>⚛️ Quantum Circuit</p>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#8892b0;'>ZZFeatureMap circuit used in QuantumShield.</p>",
                unsafe_allow_html=True)

    # QML pipeline explanation
    st.markdown("<p class='section-title'>🔄 QML Pipeline</p>", unsafe_allow_html=True)
    steps_q = st.columns(7)
    qsteps = [
        ("📊", "5 Features", "V1,V4,V11\nV14,V17"),
        ("→", "", ""),
        ("🔄", "ZZFeatureMap", "Angle Encode\nto Qubits"),
        ("→", "", ""),
        ("⚛️", "Quantum Kernel", "|⟨φ(x)|φ(y)⟩|²"),
        ("→", "", ""),
        ("🎯", "QSVC", "Fraud/Normal"),
    ]
    for col, (icon, label, sub) in zip(steps_q, qsteps):
        with col:
            if icon == "→":
                st.markdown(
                    "<div style='text-align:center;color:#6C63FF;"
                    "font-size:24px;padding-top:25px;'>→</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"""
                <div class='pipeline-step'>
                    <div style='font-size:24px;margin-bottom:6px;'>{icon}</div>
                    <div style='color:#e2e8f0;font-weight:600;'>{label}</div>
                    <div style='font-size:10px;color:#64748b;margin-top:4px;white-space:pre;'>{sub}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Circuit diagram
    st.markdown("<p class='section-title'>📐 ZZFeatureMap Circuit</p>", unsafe_allow_html=True)

    n_qubits = st.slider("Number of qubits (features)", min_value=2, max_value=5,
                         value=5, step=1, key="n_qubits_slider")
    reps = st.slider("Circuit repetitions (reps)", min_value=1, max_value=3,
                     value=2, step=1, key="reps_slider")

    try:
        from qiskit.circuit.library import ZZFeatureMap
        from src.quantum_model import get_circuit_figure, get_circuit_diagram

        fig_circuit = get_circuit_figure(num_features=n_qubits, reps=reps)
        st.pyplot(fig_circuit, use_container_width=True)
        plt.close("all")

        with st.expander("📄 ASCII Circuit Diagram"):
            try:
                diagram = get_circuit_diagram(num_features=n_qubits, reps=reps)
                st.code(diagram, language="text")
            except Exception:
                fm = ZZFeatureMap(feature_dimension=n_qubits, reps=reps)
                st.code(str(fm.decompose().draw()), language="text")

    except ImportError as e:
        st.error(f"Qiskit not installed or import error: {e}")
        st.info("Install with: `pip install qiskit qiskit-machine-learning qiskit-algorithms`")

    # Quantum math explanation
    st.markdown("<p class='section-title'>📐 Mathematical Foundation</p>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='info-box'>
        <strong>ZZFeatureMap Encoding</strong><br><br>
        For input vector <strong>x = [x₁, x₂, …, xₙ]</strong>:<br><br>
        <strong>Step 1</strong> — Hadamard layer:<br>
        &nbsp;&nbsp;H⊗ⁿ |0⟩ = |+⟩⊗ⁿ<br><br>
        <strong>Step 2</strong> — Single-qubit rotations:<br>
        &nbsp;&nbsp;Rz(2xᵢ) on qubit i<br><br>
        <strong>Step 3</strong> — Entanglement (ZZ interactions):<br>
        &nbsp;&nbsp;CX · Rz(2(π-xᵢ)(π-xⱼ)) · CX<br>
        &nbsp;&nbsp;for all pairs (i,j)<br><br>
        Repeated <em>reps</em> times → deep entanglement.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='info-box'>
        <strong>Quantum Kernel Function</strong><br><br>
        The kernel value between two data points <strong>x</strong> and <strong>y</strong>:<br><br>
        &nbsp;&nbsp;<strong>K(x, y) = |⟨φ(x)|φ(y)⟩|²</strong><br><br>
        where |φ(x)⟩ is the quantum state produced by the ZZFeatureMap for input x.<br><br>
        <strong>Why this matters:</strong><br>
        Classical kernels (RBF, polynomial) compute similarity in fixed function spaces.
        A quantum kernel implicitly computes similarity in an exponentially large
        Hilbert space — a space that cannot be efficiently represented classically
        for large qubit counts.<br><br>
        For <em>n</em> qubits → 2ⁿ-dimensional Hilbert space.
        </div>
        """, unsafe_allow_html=True)

    # Qubit resource table
    st.markdown("<p class='section-title'>💻 Simulation Resources</p>",
                unsafe_allow_html=True)
    resource_data = {
        "Qubits (n)": [2, 3, 4, 5, 6, 7, 10],
        "Hilbert Space (2ⁿ)": [4, 8, 16, 32, 64, 128, 1024],
        "Laptop-feasible": ["✅", "✅", "✅", "✅", "⚠️ Slow", "⚠️ Very Slow", "❌"],
    }
    st.dataframe(pd.DataFrame(resource_data), use_container_width=True, hide_index=True)


# ===========================================================================
# Page: About & QML Guide
# ===========================================================================

elif "About" in page:
    st.markdown("<p class='hero-title' style='font-size:32px;'>ℹ️ About QuantumShield</p>",
                unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class='info-box'>
        <strong>Problem Statement</strong><br>
        Credit-card fraud causes billions in annual losses. Real-time detection
        requires ML models that generalise well despite extreme class imbalance
        (fraud is only ~0.17% of transactions). Classical models are effective but
        cannot easily explore certain high-dimensional feature spaces.
        </div>

        <div class='info-box' style='margin-top:12px;'>
        <strong>Our Approach</strong><br>
        QuantumShield compares classical ML (Logistic Regression, Random Forest)
        with a Quantum Machine Learning model (ZZFeatureMap + Quantum Kernel + QSVC).
        We use the real Kaggle credit-card fraud dataset and honestly report results —
        including the limitations of quantum simulation on a laptop.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='info-box'>
        <strong>Technology Stack</strong><br>
        🐍 Python 3.11+<br>
        ⚛️ Qiskit ≥ 1.0<br>
        🤖 Qiskit Machine Learning ≥ 0.8<br>
        🧮 Scikit-learn<br>
        📊 Streamlit<br>
        📈 Plotly / Matplotlib<br>
        🐼 Pandas / NumPy<br>
        💾 Joblib
        </div>

        <div class='info-box' style='margin-top:12px;'>
        <strong>Dataset</strong><br>
        Kaggle: ULB Credit Card Fraud Detection<br>
        284,807 transactions (Sep 2013)<br>
        492 fraud cases (0.172%)<br>
        30 features (V1–V28 PCA + Time + Amount)
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<p class='section-title'>🎓 How QML Works</p>",
                unsafe_allow_html=True)

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1d2e,#1e2540);
                border-radius:16px; padding:28px; border:1px solid #2d3561;'>

    <div style='display:grid; grid-template-columns:1fr 60px 1fr; gap:16px; align-items:center;'>

        <div style='background:#0d1529; border-radius:12px; padding:20px; border:1px solid #2d3561;'>
            <div style='color:#43D9AD; font-weight:700; font-size:16px; margin-bottom:12px;'>
                🖥️ Classical ML Pipeline
            </div>
            <div style='color:#e2e8f0; font-size:14px; line-height:2;'>
                📊 Raw Data<br>
                &nbsp;&nbsp;&nbsp;↓<br>
                ⚙️ Feature Engineering<br>
                &nbsp;&nbsp;&nbsp;↓<br>
                🧮 Classical Model<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(LR / RF / SVM)<br>
                &nbsp;&nbsp;&nbsp;↓<br>
                🎯 Prediction
            </div>
        </div>

        <div style='text-align:center; font-size:28px; color:#6C63FF;'>VS</div>

        <div style='background:#0d1529; border-radius:12px; padding:20px; border:1px solid #6C63FF;'>
            <div style='color:#6C63FF; font-weight:700; font-size:16px; margin-bottom:12px;'>
                ⚛️ Quantum ML Pipeline
            </div>
            <div style='color:#e2e8f0; font-size:14px; line-height:2;'>
                📊 Raw Data<br>
                &nbsp;&nbsp;&nbsp;↓<br>
                🔄 Feature Map (ZZFeatureMap)<br>
                &nbsp;&nbsp;&nbsp;↓<br>
                ⚛️ Quantum Circuit<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|φ(x)⟩ in Hilbert space<br>
                &nbsp;&nbsp;&nbsp;↓<br>
                🔢 Quantum Kernel K(x,y)<br>
                &nbsp;&nbsp;&nbsp;↓<br>
                🎯 Prediction (QSVC)
            </div>
        </div>

    </div>

    <div style='margin-top:20px; padding:16px; background:#0d1529; border-radius:8px;
                border-left:4px solid #6C63FF;'>
        <div style='color:#8892b0; font-size:13px;'>
        <strong style='color:#e2e8f0;'>Key Insight:</strong>
        The quantum kernel K(x,y) = |⟨φ(x)|φ(y)⟩|² measures similarity in a
        2ⁿ-dimensional Hilbert space. For n=5 qubits this is a 32-dimensional space —
        small here, but for n=50 qubits it would be 2⁵⁰ ≈ 10¹⁵ dimensions,
        completely intractable to simulate classically.
        </div>
    </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='section-title'>⚠️ Limitations</p>",
                unsafe_allow_html=True)
    limitations = [
        "Quantum simulation on a classical computer does not demonstrate speedup — it's for educational purposes.",
        "QSVC uses only 5 features vs 30 for classical models — a fair comparison requires equal features.",
        "The quantum kernel matrix computation scales as O(n²) in time — impractical for large datasets.",
        "Noise from real quantum hardware (not simulated here) would degrade accuracy.",
        "Genuine quantum advantage in ML remains unproven for near-term devices (NISQ era).",
    ]
    for lim in limitations:
        st.markdown(f"""
        <div class='warning-box'>⚠️ {lim}</div>
        """, unsafe_allow_html=True)

    st.markdown("<p class='section-title'>🚀 Future Improvements</p>",
                unsafe_allow_html=True)
    future = [
        "Run on real IBM Quantum hardware via Qiskit Runtime",
        "Implement QGAN (Quantum GAN) for synthetic fraud sample generation",
        "Explore variational quantum circuits (VQC) with trainable parameters",
        "Use quantum amplitude encoding for larger feature vectors",
        "Benchmark against quantum annealing approaches (D-Wave)",
        "Implement quantum error mitigation for noisy device results",
    ]
    for item in future:
        st.markdown(f"""
        <div class='success-box'>🚀 {item}</div>
        """, unsafe_allow_html=True)
