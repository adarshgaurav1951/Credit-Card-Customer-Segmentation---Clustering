"""
Real-time customer scoring and persona assignment inference engine.
"""

import json
import os
import sys

# Ensure workspace root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from typing import Any, Dict, List, Union
import joblib
import numpy as np
import pandas as pd


class CustomerSegmentClassifier:
    """
    Production inference engine for scoring credit card customer records.
    """
    def __init__(
        self,
        pipeline_path: str = "models/cluster_pipeline.joblib",
        metadata_path: str = "models/segment_metadata.json"
    ):
        self.pipeline_path = pipeline_path
        self.metadata_path = metadata_path
        self.pipeline = None
        self.metadata = None
        self._load_artifacts()

    def _load_artifacts(self):
        if not os.path.exists(self.pipeline_path):
            raise FileNotFoundError(f"Pipeline not found at {self.pipeline_path}. Run train_pipeline.py first.")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata not found at {self.metadata_path}. Run train_pipeline.py first.")

        self.pipeline = joblib.load(self.pipeline_path)
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

    def score_single_customer(self, customer_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scores a single customer record and outputs cluster ID, persona details,
        confidence probability, and recommended business actions.
        """
        df = pd.DataFrame([customer_record])
        return self.score_batch(df)[0]

    def score_batch(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Scores a batch of customers in a pandas DataFrame.
        """
        # Step 1: Preprocess, log transform, and scale through pipeline steps
        prep = self.pipeline.named_steps["prep"]
        log = self.pipeline.named_steps["log"]
        scale = self.pipeline.named_steps["scale"]
        kmeans = self.pipeline.named_steps["cluster"]

        X_prep = prep.transform(df)
        X_log = log.transform(X_prep)
        X_scaled = scale.transform(X_log)

        # Distance to cluster centers
        distances = kmeans.transform(X_scaled)  # shape (N, k)
        
        # Approximate soft membership probabilities via negative distance softmax
        # Temperature parameter tau scales sharpness
        tau = 1.0
        exp_neg_dist = np.exp(-distances / tau)
        probs = exp_neg_dist / np.sum(exp_neg_dist, axis=1, keepdims=True)

        assigned_clusters = np.argmin(distances, axis=1)

        results = []
        for i in range(len(df)):
            c_id = int(assigned_clusters[i])
            c_meta = self.metadata["clusters"].get(str(c_id), self.metadata["clusters"].get(c_id, {}))
            confidence = float(probs[i, c_id])

            res = {
                "customer_id": str(df.iloc[i].get("CUST_ID", f"Account_{i+1}")),
                "cluster_id": c_id,
                "persona_name": c_meta.get("name", f"Cluster {c_id}"),
                "tagline": c_meta.get("tagline", ""),
                "confidence_score": round(confidence, 4),
                "cluster_membership_probabilities": {
                    self.metadata["clusters"][str(k)]["name"]: round(float(probs[i, k]), 4)
                    for k in range(kmeans.n_clusters)
                },
                "recommended_actions": c_meta.get("recommended_actions", []),
                "do_not": c_meta.get("do_not", []),
                "primary_kpi": c_meta.get("primary_kpi", ""),
                "risk_profile": c_meta.get("risk_profile", "")
            }
            results.append(res)

        return results


def score_customer(customer_data: Union[Dict[str, Any], pd.DataFrame]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    classifier = CustomerSegmentClassifier()
    if isinstance(customer_data, dict):
        return classifier.score_single_customer(customer_data)
    elif isinstance(customer_data, pd.DataFrame):
        return classifier.score_batch(customer_data)
    else:
        raise ValueError("customer_data must be a dict or a pandas DataFrame")


if __name__ == "__main__":
    # Test with a mock customer
    sample_customer = {
        "CUST_ID": "TEST_001",
        "BALANCE": 45.0,
        "BALANCE_FREQUENCY": 0.8,
        "PURCHASES": 1250.0,
        "ONEOFF_PURCHASES": 650.0,
        "INSTALLMENTS_PURCHASES": 600.0,
        "CASH_ADVANCE": 0.0,
        "PURCHASES_FREQUENCY": 0.9,
        "ONEOFF_PURCHASES_FREQUENCY": 0.5,
        "PURCHASES_INSTALLMENTS_FREQUENCY": 0.7,
        "CASH_ADVANCE_FREQUENCY": 0.0,
        "CASH_ADVANCE_TRX": 0,
        "PURCHASES_TRX": 25,
        "CREDIT_LIMIT": 6000.0,
        "PAYMENTS": 1250.0,
        "MINIMUM_PAYMENTS": 100.0,
        "PRC_FULL_PAYMENT": 1.0,
        "TENURE": 12
    }
    result = score_customer(sample_customer)
    print(json.dumps(result, indent=2))
