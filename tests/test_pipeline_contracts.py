import unittest

import numpy as np
import pandas as pd

from src.preprocessing import scale_features
from src.evaluation import compute_all_metrics


class PipelineContractTests(unittest.TestCase):
    def test_scale_features_returns_reusable_fit_parameters(self):
        raw = pd.DataFrame({
            "V1": [-1.0, 1.0, 3.0],
            "Time": [0.0, 10.0, 20.0],
            "Amount": [10.0, 20.0, 30.0],
            "Class": [0, 1, 0],
        })

        scaled, params = scale_features(raw, return_scalers=True)

        self.assertAlmostEqual(params["Amount"]["mean"], 20.0)
        self.assertAlmostEqual(params["Time"]["mean"], 10.0)
        amount = (40.0 - params["Amount"]["mean"]) / params["Amount"]["scale"]
        time_value = (30.0 - params["Time"]["mean"]) / params["Time"]["scale"]
        self.assertAlmostEqual(amount, 2 * np.sqrt(1.5))
        self.assertAlmostEqual(time_value, 2 * np.sqrt(1.5))
        self.assertIn("scaled_Amount", scaled.columns)
        self.assertIn("scaled_Time", scaled.columns)

    def test_skip_quantum_omits_qsvc_metric_instead_of_faking_one(self):
        y_true = np.array([0, 1, 0, 1])
        metrics = compute_all_metrics(
            y_true,
            np.array([0, 1, 1, 0]),
            np.array([0, 1, 0, 1]),
            None,
        )
        self.assertEqual([item["model"] for item in metrics], [
            "Logistic Regression", "Random Forest"
        ])


if __name__ == "__main__":
    unittest.main()
