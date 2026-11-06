"""
Batch Customer Scoring CLI for Credit Card Customer Segmentation.
Usage:
    python score_batch.py --input "data/CC GENERAL.csv" --output scored_customers.csv
"""

import argparse
import os
import sys
import pandas as pd

# Ensure workspace root in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.inference import CustomerSegmentClassifier


def run_batch_scoring(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)

    print(f"Loading customer batch from '{input_path}'...")
    df_in = pd.read_csv(input_path)
    total_customers = len(df_in)
    print(f"Loaded {total_customers:,} accounts.")

    print("Initializing Customer Segment Classifier...")
    classifier = CustomerSegmentClassifier()

    print("Executing batch scoring and persona assignment...")
    results = classifier.score_batch(df_in)

    # Convert results into structured DataFrame
    scored_records = []
    for r in results:
        record = {
            "CUSTOMER_ID": r["customer_id"],
            "CLUSTER_ID": r["cluster_id"],
            "PERSONA_NAME": r["persona_name"],
            "CONFIDENCE_SCORE": r["confidence_score"],
            "RECOMMENDED_ACTION_1": r["recommended_actions"][0] if r["recommended_actions"] else "",
            "RECOMMENDED_ACTION_2": r["recommended_actions"][1] if len(r["recommended_actions"]) > 1 else "",
            "PRIMARY_RISK_GUARDRAIL": r["do_not"][0] if r["do_not"] else "",
            "PRIMARY_KPI": r["primary_kpi"],
            "RISK_PROFILE": r["risk_profile"]
        }
        # Add probability columns for all clusters
        for persona_name, prob in r["cluster_membership_probabilities"].items():
            col_name = f"PROB_{persona_name.upper().replace(' ', '_').replace('-', '_')}"
            record[col_name] = prob

        scored_records.append(record)

    df_scored = pd.DataFrame(scored_records)

    # Join back with original features
    df_final = pd.concat([df_in.reset_index(drop=True), df_scored.drop(columns=["CUSTOMER_ID"])], axis=1)

    print(f"Exporting scored batch to '{output_path}'...")
    df_final.to_csv(output_path, index=False)
    print(f"Batch scoring completed successfully! {len(df_final):,} rows written to '{output_path}'.")

    # Print summary distribution
    print("\nBatch Persona Distribution:")
    summary = df_final["PERSONA_NAME"].value_counts()
    for persona, count in summary.items():
        print(f" - {persona}: {count:,} ({count / total_customers * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Credit Card Customer Batch Scoring CLI")
    parser.add_argument("--input", "-i", type=str, default="data/CC GENERAL.csv", help="Path to raw customer CSV")
    parser.add_argument("--output", "-o", type=str, default="scored_customers.csv", help="Path to output scored CSV")
    args = parser.parse_args()

    run_batch_scoring(args.input, args.output)


if __name__ == "__main__":
    main()
