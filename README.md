# ⚛️ QuantumShield

**Quantum Machine Learning Fraud Detection System**

QuantumShield is a hackathon-ready web application that detects potentially fraudulent financial transactions. It provides a unique comparison between classical machine-learning models (Logistic Regression, Random Forest) and a Quantum Machine Learning (QML) model implemented using Qiskit.

---

## 🎯 Problem Statement
Credit-card fraud causes billions of dollars in losses annually. Real-time detection requires machine learning models that can generalize well despite extreme class imbalance (in typical datasets, fraud accounts for < 0.2% of transactions). While classical models are highly effective, they may struggle to efficiently map complex, high-dimensional correlations. 

## 💡 Proposed Solution
QuantumShield explores the potential of **Quantum Machine Learning**. By encoding transaction features into quantum states, we can implicitly evaluate similarity in an exponentially large Hilbert space. This project demonstrates a pipeline that compares standard classical classifiers against a Quantum Support Vector Classifier (QSVC).

## 📐 Architecture
The application follows a modular pipeline:
1. **Preprocessing:** Handles missing values, scales features (StandardScaler), and applies undersampling to balance the dataset.
2. **Classical ML:** Trains Logistic Regression and Random Forest models on the full feature set.
3. **Quantum ML:** Uses a subset of features (to keep quantum simulation tractable) encoded via a `ZZFeatureMap`, processes them through a `FidelityQuantumKernel`, and trains a `QSVC`.
4. **Evaluation:** Computes metrics (Accuracy, Precision, Recall, F1-Score) and generates visualizations.
5. **Dashboard:** A Streamlit web app for interactive analysis and real-time inference.

## ✨ Features
* **Interactive Dashboard:** Beautiful, modern Streamlit interface.
* **Real-time Inference:** Enter transaction details to instantly get fraud probabilities from both classical and quantum models.
* **Automated Pipeline:** End-to-end script (`train.py`) that handles data prep, training, and evaluation.
* **Quantum Circuit Visualization:** View the generated `ZZFeatureMap` circuit dynamically in the app.
* **Comprehensive Metrics:** Radar charts, grouped bar charts, and confusion matrices comparing all models.

## 🛠️ Technologies
* **Python 3.11+**
* **Qiskit (≥1.0) & Qiskit Machine Learning:** Quantum circuit building and QSVC implementation.
* **Scikit-learn:** Classical machine learning and data preprocessing.
* **Pandas & NumPy:** Data manipulation.
* **Streamlit:** Frontend dashboard.
* **Plotly & Matplotlib:** Data visualization.

---

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/QuantumShield.git
   cd QuantumShield
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset Instructions

This project uses the Kaggle **Credit Card Fraud Detection** dataset.
1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Create a `data` folder in the project root if it doesn't exist.
3. Place `creditcard.csv` inside the `data/` folder.

*(Note: If you use the provided Kagglehub download script in the environment, this is handled automatically).*

## 🏃‍♂️ How to Run

**1. Train the models:**
Before running the app, you need to train the models.
```bash
# Run full training (includes Quantum model - takes ~5-10 mins)
python train.py --max-q 300

# OR skip the quantum training for a fast test:
python train.py --skip-quantum
```

**2. Launch the Streamlit dashboard:**
```bash
streamlit run app.py
```

---

## 🧠 Quantum Computing Explanation

In classical ML, we rely on kernels (like RBF or Polynomial) to compute the similarity between data points in a fixed function space. 

In Quantum ML:
1. **Data Encoding:** We use a **`ZZFeatureMap`** to map our classical transaction features into quantum states (qubits).
2. **Entanglement:** The circuit uses entanglement (ZZ interactions), capturing complex correlations between features.
3. **Quantum Kernel:** We compute the similarity (inner product) between two data points by measuring the overlap of their quantum states: $K(x, y) = |\langle \phi(x)|\phi(y)\rangle|^2$.
4. **Classification:** This quantum kernel matrix is then fed into a standard Support Vector Machine (QSVC).

This allows the model to find separating hyperplanes in a $2^n$-dimensional Hilbert space (where $n$ is the number of qubits/features).

## 📈 Results Section
*(Note: Actual results may vary slightly due to the random sampling in the balancing phase).*
* **Classical Models** (Logistic Regression, Random Forest) generally achieve high accuracy (>94%) and excellent F1-scores using the full 30-feature dataset.
* **Quantum Model** (QSVC) is restricted to 5 features to allow for practical simulation times on a standard laptop. Consequently, its performance serves as an educational proof-of-concept rather than a state-of-the-art detector.

## ⚠️ Limitations
* **Simulation Constraints:** Running quantum circuits on a classical simulator requires exponential memory. We limited the QSVC to 5 features (5 qubits) and a small subset of training data.
* **No Quantum Advantage (Yet):** Because we are simulating a tiny quantum system classically, there is no speedup. True quantum advantage would require executing larger models on fault-tolerant quantum hardware.

## 🔮 Future Improvements
* **Real Quantum Hardware:** Execute the kernel estimation on an actual IBM Quantum backend via Qiskit Runtime.
* **Advanced Feature Maps:** Experiment with custom parameterized quantum circuits (PQCs) tailored for financial data.
* **Amplitude Encoding:** Implement amplitude encoding to compress more classical features into fewer qubits.
* **Quantum GANs:** Explore generating synthetic minority-class (fraud) samples using a Quantum Generative Adversarial Network.
