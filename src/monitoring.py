"""
Production monitoring and drift detection for Credit Card Customer Segmentation.
Implements Population Stability Index (PSI) and segment distribution tracking.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd


def calculate_psi(
    expected: Union[np.ndarray, pd.Series],
    actual: Union[np.ndarray, pd.Series],
    num_buckets: int = 10,
    epsilon: float = 1e-4
) -> Tuple[float, pd.DataFrame]:
    """
    Computes the Population Stability Index (PSI) between a baseline (expected)
    and production (actual) continuous feature distribution.

    Interpretation:
    - PSI < 0.10: No significant shift; distribution is stable.
    - 0.10 <= PSI < 0.25: Moderate drift; investigate behavioral shifts.
    - PSI >= 0.25: Severe drift; trigger automated model retraining.
    """
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    # Determine quantile bins from expected baseline
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected, quantiles)
    bins = np.unique(bins)  # Handle tied percentiles

    if len(bins) < 2:
        bins = np.array([expected.min() - 1e-5, expected.max() + 1e-5])

    bins[0] = -np.inf
    bins[-1] = np.inf

    # Bucket counts
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    # Convert to proportions with epsilon smoothing to prevent div by zero
    expected_pct = (expected_counts / len(expected)).clip(min=epsilon)
    actual_pct = (actual_counts / len(actual)).clip(min=epsilon)

    # Re-normalize to sum to 1.0 after smoothing
    expected_pct /= expected_pct.sum()
    actual_pct /= actual_pct.sum()

    # PSI calculation
    psi_terms = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    total_psi = float(np.sum(psi_terms))

    audit_table = pd.DataFrame({
        "bucket_min": bins[:-1],
        "bucket_max": bins[1:],
        "expected_count": expected_counts,
        "actual_count": actual_counts,
        "expected_pct": expected_pct,
        "actual_pct": actual_pct,
        "psi_contribution": psi_terms
    })

    return total_psi, audit_table


def calculate_categorical_psi(
    expected_labels: Union[np.ndarray, pd.Series],
    actual_labels: Union[np.ndarray, pd.Series],
    epsilon: float = 1e-4
) -> Tuple[float, pd.DataFrame]:
    """
    Computes PSI for discrete category distributions (e.g. cluster assignment shifts).
    """
    expected = pd.Series(expected_labels).value_counts(normalize=True)
    actual = pd.Series(actual_labels).value_counts(normalize=True)

    all_categories = sorted(list(set(expected.index).union(set(actual.index))))
    
    exp_pcts = np.array([expected.get(cat, epsilon) for cat in all_categories])
    act_pcts = np.array([actual.get(cat, epsilon) for cat in all_categories])

    # Re-normalize
    exp_pcts /= exp_pcts.sum()
    act_pcts /= act_pcts.sum()

    psi_terms = (act_pcts - exp_pcts) * np.log(act_pcts / exp_pcts)
    total_psi = float(np.sum(psi_terms))

    audit_table = pd.DataFrame({
        "category": all_categories,
        "expected_pct": exp_pcts,
        "actual_pct": act_pcts,
        "psi_contribution": psi_terms
    })

    return total_psi, audit_table


def evaluate_portfolio_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: List[str]
) -> pd.DataFrame:
    """
    Evaluates drift across all key financial features and determines overall risk status.
    """
    results = []
    for feat in features:
        if feat in baseline_df.columns and feat in current_df.columns:
            psi_val, _ = calculate_psi(baseline_df[feat], current_df[feat])
            
            if psi_val < 0.10:
                status = "Stable (Green)"
                action = "None"
            elif psi_val < 0.25:
                status = "Moderate Drift (Yellow)"
                action = "Monitor & Investigate"
            else:
                status = "Critical Drift (Red)"
                action = "Trigger Model Retraining"

            results.append({
                "feature": feat,
                "psi": round(psi_val, 4),
                "status": status,
                "recommended_action": action
            })

    return pd.DataFrame(results).sort_values("psi", ascending=False)

