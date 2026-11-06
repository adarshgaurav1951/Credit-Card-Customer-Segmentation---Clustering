"""
Feature engineering and transformation utilities for Credit Card Customer Segmentation.
Designed for seamless integration into scikit-learn Pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MONEY_COUNT_COLS = [
    'BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
    'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS',
    'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'MONTHLY_PURCHASES'
]


class CreditCardPreprocessor(BaseEstimator, TransformerMixin):
    """
    Cleans raw credit card data, imputes missing values, clips anomaly frequencies,
    and engineers banking domain ratios.
    """
    def __init__(self, clip_cash_freq: bool = True):
        self.clip_cash_freq = clip_cash_freq
        self.credit_limit_median_ = None
        self.min_payments_median_ = None
        self.feature_names_out_ = None

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        if 'CUST_ID' in df.columns:
            df = df.drop(columns=['CUST_ID'])
        
        self.credit_limit_median_ = float(df['CREDIT_LIMIT'].median()) if 'CREDIT_LIMIT' in df else 4500.0
        self.min_payments_median_ = float(df['MINIMUM_PAYMENTS'].median()) if 'MINIMUM_PAYMENTS' in df else 864.0
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()
        if 'CUST_ID' in df.columns:
            df = df.drop(columns=['CUST_ID'])

        # 1. Flag missingness before imputation
        if 'MINIMUM_PAYMENTS' in df.columns:
            df['MINIMUM_PAYMENTS'] = pd.to_numeric(df['MINIMUM_PAYMENTS'], errors='coerce')
            df['MIN_PAYMENT_MISSING'] = df['MINIMUM_PAYMENTS'].isnull().astype(float)
            df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(self.min_payments_median_)
        else:
            df['MIN_PAYMENT_MISSING'] = 0.0

        if 'CREDIT_LIMIT' in df.columns:
            df['CREDIT_LIMIT'] = pd.to_numeric(df['CREDIT_LIMIT'], errors='coerce')
            df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(self.credit_limit_median_)

        # 2. Clip frequency anomalies
        if self.clip_cash_freq and 'CASH_ADVANCE_FREQUENCY' in df.columns:
            df['CASH_ADVANCE_FREQUENCY'] = df['CASH_ADVANCE_FREQUENCY'].clip(0.0, 1.0)

        # 3. Domain feature engineering (with numerical guards)
        eps = 1e-5
        
        # Credit Utilization
        df['CREDIT_UTILIZATION'] = (df['BALANCE'] / (df['CREDIT_LIMIT'] + eps)).clip(0.0, 3.0)
        
        # Payment Ratio (Payment vs Minimum Payment Due)
        df['PAYMENT_RATIO'] = (df['PAYMENTS'] / (df['MINIMUM_PAYMENTS'] + eps)).clip(0.0, 25.0)
        
        # Balance Payment Ratio
        df['PAYMENT_TO_BALANCE'] = (df['PAYMENTS'] / (df['BALANCE'] + eps)).clip(0.0, 10.0)
        
        # Cash Advance Share
        df['CASH_SHARE'] = df['CASH_ADVANCE'] / (df['PURCHASES'] + df['CASH_ADVANCE'] + eps)
        
        # Installment Share
        df['INSTALLMENTS_SHARE'] = df['INSTALLMENTS_PURCHASES'] / (df['PURCHASES'] + eps)
        
        # Monthly Purchase Velocity
        tenure = df['TENURE'].clip(lower=1) if 'TENURE' in df.columns else 12
        df['MONTHLY_PURCHASES'] = df['PURCHASES'] / tenure

        # Binary Behavioral Indicators
        df['HAS_PURCHASES'] = (df['PURCHASES'] > 0).astype(float)
        df['HAS_CASH_ADVANCE'] = (df['CASH_ADVANCE'] > 0).astype(float)
        df['IS_FULL_PAYER'] = (df['PRC_FULL_PAYMENT'] >= 0.99).astype(float)

        self.feature_names_out_ = list(df.columns)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """
    Applies np.log1p strictly to non-negative monetary and count features,
    preserving zero values and bounded ratio features.
    """
    def __init__(self, cols_to_log=None):
        self.cols_to_log = cols_to_log or MONEY_COUNT_COLS
        self.feature_names_out_ = None

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()
        for col in self.cols_to_log:
            if col in df.columns:
                # Ensure float dtype and non-negative before log1p
                series = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                df[col] = np.log1p(series.clip(lower=0.0))
        self.feature_names_out_ = list(df.columns)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

