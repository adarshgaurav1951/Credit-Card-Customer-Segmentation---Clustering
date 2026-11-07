"""
End-to-end training and serialization script for Credit Card Segmentation Pipeline.
Generates models/cluster_pipeline.joblib and models/segment_metadata.json based on 80% Train.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text

from src.features import CreditCardPreprocessor, Log1pTransformer

PERSONAS_BY_NAME = {
    "Disciplined Transactors": {
        "name": "Disciplined Transactors",
        "tagline": "High Purchases, Negligible Balance, Full-Statement Payers",
        "description": "Cardholders who utilize their credit line strictly as an everyday payment mechanism. They exhibit very low revolving balances ($113 avg), high purchase frequency, near-zero cash advances, and pay in full at 3x the portfolio rate.",
        "financial_value": "High interchange fees, near-zero credit write-off risk.",
        "risk_profile": "Very Low Risk. Unlikely to generate net interest margin (NIM), but zero default volatility.",
        "recommended_actions": [
            "Enroll in merchant rewards partnerships (travel, dining, grocery points accelerators)",
            "Promote premium credit tier upgrades with concierge and lounge benefits",
            "Offer spend-streak incentives (e.g. spend $1,500/month for bonus cashback)"
        ],
        "do_not": [
            "Do not market high-APR balance transfer or installment loan products",
            "Do not aggressively cross-sell cash advance or emergency liquidity lines"
        ],
        "primary_kpi": "Active spend per account & 90-day retention rate"
    },
    "Active Lifestyle Revolvers": {
        "name": "Active Lifestyle Revolvers",
        "tagline": "Frequent Retail Spenders Carrying Balances (High Interest Margin)",
        "description": "Active, engaged consumers with the highest purchase volume ($1,584 avg) and strong frequency. They rarely take cash advances but carry ongoing balances ($1,508 avg) with low full-payment rates (3%), generating prime interest income.",
        "financial_value": "Highly profitable through steady net interest margin and interchange fees.",
        "risk_profile": "Moderate Risk. Solid payment behavior, but susceptible to macroeconomic strain if debt accumulates.",
        "recommended_actions": [
            "Offer promotional low-rate balance consolidation offers for competitive cards",
            "Introduce split-pay installment options for purchases exceeding $200",
            "Provide proactive credit limit reviews tied to continuous on-time payments"
        ],
        "do_not": [
            "Do not abruptly restrict limits without early delinquency indicators",
            "Do not overwhelm with cash advance promotions"
        ],
        "primary_kpi": "Revolving balance growth, delinquency rate (<30 DPD), and interest revenue"
    },
    "Cash-Advance Borrowers": {
        "name": "Cash-Advance Borrowers",
        "tagline": "Emergency Liquidity Users, Negligible Purchase Activity",
        "description": "Customers who rarely use the card for retail shopping (Purchases avg $1.22, frequency ~0.0), but utilize the line as a short-term cash source ($1,971 avg cash advance, 27% cash frequency).",
        "financial_value": "High cash-advance fee income, but substantial credit and charge-off exposure.",
        "risk_profile": "High Risk. High cash reliance often correlates with paycheck-to-paycheck cash flow or liquidity distress.",
        "recommended_actions": [
            "Offer structured debt relief or personal installment loans at fixed interest rates",
            "Provide automated budget tracking and financial wellness advisory resources",
            "Establish early-warning monitoring for rapid cash-advance acceleration"
        ],
        "do_not": [
            "Do not offer unsolicited credit limit increases",
            "Do not market luxury spending or retail rewards bonuses"
        ],
        "primary_kpi": "Cash advance share reduction and 90+ DPD default rate"
    },
    "Heavy Hybrid Borrowers": {
        "name": "Heavy Hybrid Borrowers",
        "tagline": "High-Limit, High-Balance, Dual-Usage Portfolio Drivers",
        "description": "Top-tier credit line holders ($5,242 avg limit) who heavily utilize both purchase facilities ($1,221 avg) and cash advances ($2,229 avg). They carry the largest average balance ($2,845) while also making the largest monthly payments ($2,478).",
        "financial_value": "Significant revenue drivers across interest, interchange, and fees, but high credit commitment.",
        "risk_profile": "High / Critical Exposure. Large nominal dollar exposure requires active utilization and debt-service capacity monitoring.",
        "recommended_actions": [
            "Engage through dedicated relationship manager support and flexible repayment schedules",
            "Offer automated debt consolidation plans to lower their interest burden",
            "Conduct dynamic risk-based pricing and monthly capacity reviews"
        ],
        "do_not": [
            "Do not treat as standard prime cardholders without collateral or income re-verification",
            "Do not allow utilization to surge past 85% unchecked"
        ],
        "primary_kpi": "Loss rate / write-offs, payment-to-balance ratio, and net portfolio revenue"
    }
}


def identify_persona(row):
    """Dynamically matches centroid financial metrics to standardized persona name."""
    if row['PRC_FULL_PAYMENT'] > 0.25 and row['BALANCE'] < 600:
        return "Disciplined Transactors"
    elif row['PURCHASES'] < 100 and row['CASH_ADVANCE'] > 1000:
        return "Cash-Advance Borrowers"
    elif row['PURCHASES'] > 1200 and row['CASH_ADVANCE'] < 300:
        return "Active Lifestyle Revolvers"
    else:
        return "Heavy Hybrid Borrowers"


def train_and_export():
    data_path = os.path.join("data", "CC GENERAL.csv")
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # 80% Train / 20% Test Partition
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    print(f"Partitioned into Train ({len(train_df):,} accounts, 80%) and Test ({len(test_df):,} accounts, 20%)")

    # 1. Pipeline construction
    preprocessor = CreditCardPreprocessor(clip_cash_freq=True)
    log_transformer = Log1pTransformer()
    scaler = StandardScaler()
    clusterer = KMeans(n_clusters=4, n_init=20, random_state=42)

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("log", log_transformer),
        ("scale", scaler),
        ("cluster", clusterer)
    ])

    print("Fitting production pipeline strictly on 80% Train set...")
    pipeline.fit(train_df)
    labels = pipeline.predict(train_df)
    train_df["CLUSTER"] = labels

    # 2. Extract cluster profiles
    numeric_cols = [
        "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES",
        "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY",
        "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
        "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX",
        "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE"
    ]

    cluster_means = train_df.groupby("CLUSTER")[numeric_cols].mean()
    portfolio_mean = train_df[numeric_cols].mean()
    index_df = (cluster_means / portfolio_mean * 100).round(1)

    cluster_counts = train_df["CLUSTER"].value_counts().to_dict()
    total_records = len(train_df)

    metadata = {
        "n_clusters": 4,
        "total_records": total_records,
        "feature_names_raw": numeric_cols,
        "clusters": {}
    }

    for c_id in range(4):
        size = cluster_counts.get(c_id, 0)
        pct = round(size / total_records * 100, 2)
        row = cluster_means.loc[c_id]
        persona_name = identify_persona(row)
        p_info = PERSONAS_BY_NAME.get(persona_name, {})

        metadata["clusters"][c_id] = {
            "cluster_id": c_id,
            "name": p_info.get("name", persona_name),
            "tagline": p_info.get("tagline", ""),
            "description": p_info.get("description", ""),
            "financial_value": p_info.get("financial_value", ""),
            "risk_profile": p_info.get("risk_profile", ""),
            "recommended_actions": p_info.get("recommended_actions", []),
            "do_not": p_info.get("do_not", []),
            "primary_kpi": p_info.get("primary_kpi", ""),
            "size": size,
            "percentage": pct,
            "means": cluster_means.loc[c_id].round(2).to_dict(),
            "index_vs_portfolio": index_df.loc[c_id].round(1).to_dict()
        }

    # 3. Fit a surrogate decision tree for explainability
    print("Fitting surrogate decision tree for rule extraction...")
    X_eng = preprocessor.transform(train_df)
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_eng, labels)
    tree_rules = export_text(dt, feature_names=list(X_eng.columns))
    metadata["surrogate_decision_tree_accuracy"] = round(float(dt.score(X_eng, labels)), 4)
    metadata["surrogate_decision_tree_rules"] = tree_rules

    # 4. Save artifacts
    os.makedirs("models", exist_ok=True)
    pipeline_file = os.path.join("models", "cluster_pipeline.joblib")
    metadata_file = os.path.join("models", "segment_metadata.json")
    baseline_file = os.path.join("models", "baseline_sample.csv")

    joblib.dump(pipeline, pipeline_file)
    print(f"Saved pipeline to {pipeline_file}")

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")

    # Save holdout test sample for drift simulation
    test_df.drop(columns=["CLUSTER"], errors="ignore").sample(n=min(1000, len(test_df)), random_state=42).to_csv(baseline_file, index=False)
    print(f"Saved baseline sample from 20% test holdout to {baseline_file}")

    print("Training and serialization completed successfully!")


if __name__ == "__main__":
    train_and_export()
