"""
Unit and integration test suite for Credit Card Customer Segmentation platform.
Uses Python's standard library unittest so it runs everywhere.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import CreditCardPreprocessor, Log1pTransformer
from src.inference import score_customer
from src.monitoring import calculate_psi, evaluate_portfolio_drift


class TestCreditCardPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_customer_dict = {
            "CUST_ID": "TEST_CUST_1001",
            "BALANCE": 150.0,
            "BALANCE_FREQUENCY": 0.9,
            "PURCHASES": 1200.0,
            "ONEOFF_PURCHASES": 600.0,
            "INSTALLMENTS_PURCHASES": 600.0,
            "CASH_ADVANCE": 0.0,
            "PURCHASES_FREQUENCY": 0.8,
            "ONEOFF_PURCHASES_FREQUENCY": 0.5,
            "PURCHASES_INSTALLMENTS_FREQUENCY": 0.5,
            "CASH_ADVANCE_FREQUENCY": 0.0,
            "CASH_ADVANCE_TRX": 0,
            "PURCHASES_TRX": 15,
            "CREDIT_LIMIT": 5000.0,
            "PAYMENTS": 1200.0,
            "MINIMUM_PAYMENTS": 120.0,
            "PRC_FULL_PAYMENT": 1.0,
            "TENURE": 12
        }

        self.sample_cash_customer_dict = {
            "CUST_ID": "TEST_CUST_2002",
            "BALANCE": 3000.0,
            "BALANCE_FREQUENCY": 1.0,
            "PURCHASES": 0.0,
            "ONEOFF_PURCHASES": 0.0,
            "INSTALLMENTS_PURCHASES": 0.0,
            "CASH_ADVANCE": 3500.0,
            "PURCHASES_FREQUENCY": 0.0,
            "ONEOFF_PURCHASES_FREQUENCY": 0.0,
            "PURCHASES_INSTALLMENTS_FREQUENCY": 0.0,
            "CASH_ADVANCE_FREQUENCY": 0.6,
            "CASH_ADVANCE_TRX": 8,
            "PURCHASES_TRX": 0,
            "CREDIT_LIMIT": 4000.0,
            "PAYMENTS": 500.0,
            "MINIMUM_PAYMENTS": 300.0,
            "PRC_FULL_PAYMENT": 0.0,
            "TENURE": 12
        }

    def test_preprocessor_transformation(self):
        df = pd.DataFrame([self.sample_customer_dict])
        preprocessor = CreditCardPreprocessor()
        preprocessor.fit(df)
        transformed = preprocessor.transform(df)

        self.assertNotIn("CUST_ID", transformed.columns)
        self.assertIn("CREDIT_UTILIZATION", transformed.columns)
        self.assertIn("PAYMENT_RATIO", transformed.columns)
        self.assertIn("CASH_SHARE", transformed.columns)
        self.assertIn("INSTALLMENTS_SHARE", transformed.columns)
        self.assertEqual(transformed.isnull().sum().sum(), 0)

    def test_log1p_transformer(self):
        transformer = Log1pTransformer(cols_to_log=["VAL"])
        df = pd.DataFrame({"VAL": [0.0, 10.0, 100.0]})
        transformed = transformer.transform(df)
        self.assertTrue(np.isclose(transformed["VAL"].iloc[0], 0.0))
        self.assertTrue(np.isclose(transformed["VAL"].iloc[1], np.log1p(10.0)))

    def test_inference_single_customer(self):
        result = score_customer(self.sample_customer_dict)
        self.assertIsInstance(result, dict)
        self.assertIn(result["cluster_id"], [0, 1, 2, 3])
        self.assertGreater(len(result["persona_name"]), 0)
        self.assertGreaterEqual(result["confidence_score"], 0.0)
        self.assertLessEqual(result["confidence_score"], 1.0)
        self.assertGreater(len(result["recommended_actions"]), 0)
        self.assertGreater(len(result["do_not"]), 0)

    def test_inference_cash_customer(self):
        result = score_customer(self.sample_cash_customer_dict)
        self.assertIn("Cash-Advance", result["persona_name"])
        self.assertIn(result["cluster_id"], [0, 1, 2, 3])

    def test_inference_handles_missing_values(self):
        corrupt = self.sample_customer_dict.copy()
        corrupt["MINIMUM_PAYMENTS"] = None
        corrupt["CREDIT_LIMIT"] = None
        result = score_customer(corrupt)
        self.assertIn(result["cluster_id"], [0, 1, 2, 3])

    def test_psi_identical_distributions(self):
        np.random.seed(42)
        baseline = np.random.normal(100, 15, 1000)
        current = np.random.normal(100, 15, 1000)
        psi_val, audit_table = calculate_psi(baseline, current)
        self.assertLess(psi_val, 0.10)
        self.assertGreater(len(audit_table), 0)

    def test_psi_shifted_distributions(self):
        np.random.seed(42)
        baseline = np.random.normal(100, 15, 1000)
        shifted = np.random.normal(250, 40, 1000)
        psi_val, _ = calculate_psi(baseline, shifted)
        self.assertGreaterEqual(psi_val, 0.25)


if __name__ == "__main__":
    unittest.main()
