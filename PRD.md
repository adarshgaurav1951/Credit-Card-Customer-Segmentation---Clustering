# Product Requirements Document (PRD)
# Credit Card Customer Behavioral Segmentation & Targeted Activation Platform

**Document Version:** 1.0.0  
**Status:** Approved for Implementation  
**Target Delivery Date:** Q4 2026  
**Primary Stakeholders:** Retail Banking Marketing, Credit Risk Management, Product Strategy, Analytics & MLOps  

---

## 1. Executive Summary & Product Vision

### 1.1 Vision Statement
To transform raw credit card usage records into an automated, explainable, and compliant behavioral segmentation platform that empowers bank marketing, credit risk, and product teams to deliver hyper-targeted financial services, optimize credit lines, and mitigate default exposure.

### 1.2 The Business Problem
Credit card portfolios exhibit extreme behavioral heterogeneity. Cardholders utilize identical credit products in fundamentally divergent ways:
1. **Transactors** use cards primarily as transactional payment tools, paying full statement balances monthly and maximizing reward points.
2. **Revolvers** carry month-over-month balances, incurring interest charges and generating net interest margin (NIM), while presenting higher loss volatility.
3. **Cash-Advance Users** draw cash liquidity against lines, indicating liquidity stress or emergency loan behavior.
4. **Inactive / Low-Engagement Users** represent dormant acquisition costs and churn risks.

One-size-fits-all communications, flat marketing promotions, and uniform credit limit adjustments fail to drive portfolio profitability and elevate credit risk. This product provides a reliable, data-driven machine learning segmentation pipeline that discovers natural behavioral clusters, characterizes them with interpretable personas, and integrates into production decisioning and drift-monitoring systems.

---

## 2. Personas & Stakeholder Objectives

| Persona | Primary Goal | Key Features Needed |
|---|---|---|
| **VP of Portfolio Marketing** | Maximize interchange and fee revenue while reducing attrition | Actionable persona cards, campaign strategy tables, A/B testing playbooks |
| **Head of Credit Risk** | Identify liquidity-stressed accounts and monitor high-utilization revolvers | Cash-advance velocity metrics, utilization profiling, regulatory guardrails |
| **Credit Product Manager** | Tailor card tiers, installment options, and rewards programs | Segment-level indexing against portfolio averages, installment behavior analysis |
| **MLOps / Risk Governance Lead** | Ensure SR 11-7 model compliance, reproducibility, and prevent proxy discrimination | End-to-end scikit-learn Pipeline, PSI drift alerts, model governance documentation |
| **Frontline Relationship Banker** | Instant customer persona lookup during service calls or web banking | Interactive Streamlit scoring dashboard and next-best-action recommendation |

---

## 3. Measurable Success Criteria & OKRs

The segmentation engine must satisfy both statistical and operational criteria:

### 3.1 Statistical & Machine Learning Criteria
- **Distinctness:** Silhouette score $\ge 0.20$ on dense behavioral space; Davies-Bouldin index $\le 1.65$; statistically significant separation across centroids on ANOVA/Kruskal-Wallis ($p < 0.001$).
- **Sizability:** No cluster shall represent fewer than $5.0\%$ of the portfolio (preventing micro-outlier clusters from driving marketing strategy).
- **Stability:** Adjusted Rand Index (ARI) $\ge 0.70$ across 20-fold bootstrap subsamples (80% sampling) and cross-seed initializations.
- **Explainability:** A shallow surrogate decision tree (depth $\le 4$) must achieve $\ge 85\%$ multi-class accuracy predicting cluster assignments from original features.

### 3.2 Business & Financial OKRs
- **Targeting Efficiency:** Achieve $\ge 25\%$ uplift in campaign response rates versus portfolio-wide blanket messaging.
- **Risk Mitigation:** Proactively identify $100\%$ of chronic cash-advance users for tailored liquidity assistance rather than hazardous limit increases.
- **Operational Latency:** Real-time customer scoring endpoint execution time $\le 50\text{ ms}$ per customer record.

---

## 4. Dataset Specification & Ingestion Contract

### 4.1 Ingestion Contract
- **Source:** Kaggle `CC GENERAL.csv` (8,950 accounts, 18 columns).
- **Primary Key:** `CUST_ID` (Anonymized customer identifier, excluded from clustering).
- **Observation Horizon:** 6 to 12 months active portfolio window.

### 4.2 Raw Feature Catalog & Anomaly Boundaries

| Feature Group | Column Name | Type | Expected Range | Known Anomaly / Validation Check |
|---|---|---|---|---|
| **Account Identity** | `CUST_ID` | String | Format `C10001`+ | Drop before scaling; maintain join key |
| **Balance** | `BALANCE` | Float | $\ge 0$ | Right-skewed ($0$ to $\$19,043$) |
| | `BALANCE_FREQUENCY` | Float | $[0.0, 1.0]$ | Frequency of balance updates |
| **Purchases** | `PURCHASES` | Float | $\ge 0$ | Max $\$49,039$; heavily skewed |
| | `ONEOFF_PURCHASES` | Float | $\ge 0$ | Single-transaction purchases |
| | `INSTALLMENTS_PURCHASES` | Float | $\ge 0$ | Installment-based transactions |
| **Habits & Frequencies** | `PURCHASES_FREQUENCY` | Float | $[0.0, 1.0]$ | Frequency of purchases |
| | `ONEOFF_PURCHASES_FREQUENCY` | Float | $[0.0, 1.0]$ | Frequency of one-off purchases |
| | `PURCHASES_INSTALLMENTS_FREQUENCY` | Float | $[0.0, 1.0]$ | Frequency of installment purchases |
| | `PURCHASES_TRX` | Integer | $\ge 0$ | Purchase transaction count |
| **Cash Advance** | `CASH_ADVANCE` | Float | $\ge 0$ | Cash withdrawal against credit line |
| | `CASH_ADVANCE_FREQUENCY` | Float | $[0.0, 1.0]$ | **Bug in raw data:** 8 rows contain values up to 1.5. Must clip to 1.0 |
| | `CASH_ADVANCE_TRX` | Integer | $\ge 0$ | Count of cash advance transactions |
| **Credit & Payments** | `CREDIT_LIMIT` | Float | $> 0$ | 1 missing value ($0.01\%$). Median impute |
| | `PAYMENTS` | Float | $\ge 0$ | Cumulative payment amount |
| | `MINIMUM_PAYMENTS` | Float | $\ge 0$ | 313 missing values ($3.50\%$). Impute + indicator flag |
| | `PRC_FULL_PAYMENT` | Float | $[0.0, 1.0]$ | Percent of statements paid in full |
| **Tenure** | `TENURE` | Integer | $[6, 12]$ | Account tenure in months |

---

## 5. End-to-End System Architecture

```
[Raw Data: CC GENERAL.csv]
       │
       ▼
[Frame 5: Data Validation & Audit Engine]
  - Null check (MINIMUM_PAYMENTS: 313, CREDIT_LIMIT: 1)
  - Anomaly validation (CASH_ADVANCE_FREQUENCY clipping)
  - Skewness & boundary assertions
       │
       ▼
[Frame 6: Cleaning & Imputation Pipeline]
  - Missingness flag: MIN_PAYMENTS_MISSING
  - SimpleImputer(strategy='median')
  - Outlier preservation via log-transformation
       │
       ▼
[Frame 8: Feature Engineering Layer]
  - CREDIT_UTILIZATION = BALANCE / CREDIT_LIMIT
  - PAYMENT_RATIO = PAYMENTS / (MINIMUM_PAYMENTS + eps)
  - CASH_SHARE = CASH_ADVANCE / (PURCHASES + CASH_ADVANCE + eps)
  - INSTALLMENTS_SHARE = INSTALLMENTS_PURCHASES / (PURCHASES + eps)
  - MONTHLY_PURCHASES = PURCHASES / TENURE
  - Flag Indicators: HAS_PURCHASES, HAS_CASH_ADVANCE, IS_FULL_PAYER
       │
       ▼
[Frame 9: Transformation & Scaling Pipeline]
  - Non-negative log1p on monetary and count features
  - StandardScaler() on normalized feature vector
  - Dimensionality reduction benchmark: PCA (85% explained variance)
       │
       ▼
[Frames 10-12: Multi-Model Evaluation & Stability Engine]
  - Baseline K-Means (k=2..10) with Elbow, Silhouette, Davies-Bouldin, Calinski-Harabasz
  - Comparative Models: Gaussian Mixture Models (BIC), Agglomerative (Ward), HDBSCAN
  - 20-fold Bootstrap Stability Testing (Adjusted Rand Index)
       │
       ▼
[Frames 13-15: Profiling, Personas & Surrogate Explainability]
  - Population Indexing: (Cluster_Mean / Portfolio_Mean) * 100
  - Persona Generation: 4 Distinct Behavioral Archetypes
  - Surrogate Decision Tree Rules & SHAP Feature Attributions
  - Dimensionality Visualization: PCA 2D & UMAP Embeddings
       │
       ▼
[Frames 16-18: Activation, Governance, Deployment & Monitoring]
  - Strategy Playbooks & A/B Test Power Analysis
  - Compliance Audit: SR 11-7, GDPR Art 22, ECOA & Proxy Bias Analysis
  - Model Serialization: joblib Pipeline (models/cluster_pipeline.joblib)
  - Real-Time Inference Module (src/inference.py)
  - Population Stability Index (PSI) Drift Monitor (src/monitoring.py)
  - Interactive Web Application (app.py)
```

---

## 6. Functional Frame-by-Frame Specifications (18 Frames)

### Frame 1: Business Problem Definition
- Map retail banking challenges to machine learning clustering.
- Establish business justification: why blanket marketing destroys value.

### Frame 2: Success Criteria & Constraints
- Quantify statistical thresholds (Silhouette, DB, Calinski-Harabasz).
- Enforce business boundaries: $k \in [3, 6]$, minimum cluster size $\ge 5\%$.

### Frame 3: Data Acquisition & Sourcing
- Maintain raw data isolation in `data/CC GENERAL.csv`.
- Construct comprehensive data dictionary with domain definitions.

### Frame 4: Workspace & Environment Setup
- Seed management (`random_state=42`) for 100% reproducibility across all stochastic operations.
- Verify environment libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `umap-learn`, `shap`, `streamlit`, `joblib`.

### Frame 5: Load, Inspect, and Validate
- Execute automated sanity checks: duplicate verification, negative value checks, frequency ceiling checks.
- Compile and log the Data Anomaly Audit Table.

### Frame 6: Clean and Impute
- Engineer `MIN_PAYMENT_MISSING` boolean indicator.
- Apply median imputation inside scikit-learn compatible transformers.
- Clip `CASH_ADVANCE_FREQUENCY` to $[0.0, 1.0]$.
- Preserve authentic high-value accounts without truncation.

### Frame 7: Exploratory Data Analysis (EDA)
- Bimodal and right-skewed distribution analysis (raw vs. $\log_{1p}$).
- Multicollinearity inspection (`PURCHASES` vs. `ONEOFF_PURCHASES` vs. `INSTALLMENTS_PURCHASES`).
- Payment-to-balance ratio distribution and transaction-free customer identification.

### Frame 8: Feature Engineering
- Construct domain ratios with zero-division safety:
  - $\text{Credit Utilization} = \text{BALANCE} / \text{CREDIT\_LIMIT}$
  - $\text{Payment Ratio} = \text{PAYMENTS} / (\text{MINIMUM\_PAYMENTS} + \epsilon)$
  - $\text{Cash Advance Share} = \text{CASH\_ADVANCE} / (\text{PURCHASES} + \text{CASH\_ADVANCE} + \epsilon)$
  - $\text{Monthly Purchase Rate} = \text{PURCHASES} / \text{TENURE}$
  - $\text{Installment Share} = \text{INSTALLMENTS\_PURCHASES} / (\text{PURCHASES} + \epsilon)$

### Frame 9: Transformation and Pipeline Construction
- Encapsulate preprocessing inside `sklearn.pipeline.Pipeline`.
- Apply $\log(1 + x)$ to positive monetary columns.
- Scale to zero mean and unit variance via `StandardScaler`.
- Benchmark PCA variance retention curve.

### Frame 10: Multi-Metric Model Selection ($K$-Means)
- Compute inertia (Elbow), Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz across $k \in [2, 10]$ with $n\_init=20$.
- Select optimal $k$ balancing mathematical clustering tightness with marketing operability.

### Frame 11: Alternative Algorithm Benchmarking
- Benchmark against:
  1. **Gaussian Mixture Models (GMM):** Soft membership probabilities, Bayesian Information Criterion (BIC) minimization.
  2. **Agglomerative Hierarchical Clustering:** Ward's linkage dendrogram examination.
  3. **HDBSCAN:** Density-based noise detection and non-spherical cluster discovery.
- Formulate standardized model comparison matrix.

### Frame 12: Cluster Stability Verification
- Implement 20-iteration bootstrap resampling ($80\%$ random subsets).
- Calculate Adjusted Rand Index (ARI) distribution against full-sample assignments.
- Analyze cluster assignment confidence and boundary margin samples.

### Frame 13: Segment Profiling & Indexing
- Join labels back to raw unscaled feature values.
- Calculate segment medians, means, and standard deviations.
- Compute Index to Portfolio Baseline:
  $$\text{Index} = \left(\frac{\bar{X}_{\text{cluster}}}{\bar{X}_{\text{portfolio}}}\right) \times 100$$
- Visualize via index heatmap and comparative boxplots.

### Frame 14: Actionable Persona Synthesis
- Create 4 production persona profiles:
  1. **Persona 1: Transacting VIP Spenders** (High purchases, high frequency, full payment, low cash advance).
  2. **Persona 2: Revolving Balance Carriers** (High balance, moderate utilization, low full-payment rate).
  3. **Persona 3: Cash-Advance Borrowers** (High cash advance, low purchases, liquidity seeking).
  4. **Persona 4: Low-Engagement / Inactive Accounts** (Low balance, low purchases, low activity).
- Specify behavioral triggers, value proposition, recommended financial products, and explicit "Do Not" guardrails.

### Frame 15: Explainability & Low-Dimensional Projection
- Fit surrogate Decision Tree classifier ($\text{depth} \le 4$) to extract human-readable `IF-THEN` business rules.
- Compute SHAP values for global feature importance ranking.
- Project clusters onto 2D PCA and non-linear UMAP manifolds.

### Frame 16: Activation Plan & A/B Testing Design
- Construct full campaign activation matrix: product offer, preferred channel, value proposition, primary KPI, and stop-loss criteria.
- Design randomized controlled trial (A/B test): sample sizing, power calculation ($\beta = 0.80, \alpha = 0.05$), minimum detectable effect (MDE), and duration.

### Frame 17: Model Governance, Fairness & Ethics
- Formulate Model Risk Management documentation aligned with Federal Reserve SR 11-7 guidelines.
- Evaluate GDPR Article 22 compliance for profiling and automated decisioning.
- Address Equal Credit Opportunity Act (ECOA) disparate impact risks and hidden behavioral proxies.
- Formalize customer data minimization and privacy retention mandates.

### Frame 18: Deployment, Monitoring & Dashboard
- Persist trained end-to-end scoring pipeline (`models/cluster_pipeline.joblib`).
- Implement production scoring function: `score_customer(customer_dict) -> (cluster_id, persona_name, confidence, actions)`.
- Develop Population Stability Index (PSI) drift monitoring engine with automated alerting:
  - $\text{PSI} < 0.10$: Stable; no action required.
  - $0.10 \le \text{PSI} < 0.25$: Moderate drift; warning flag and exploratory review.
  - $\text{PSI} \ge 0.25$: Severe drift; automatic model retraining trigger.
- Launch interactive Streamlit application (`app.py`).

---

## 7. Non-Functional & Technical Requirements

- **Runtime Environment:** Python 3.12+, Linux/POSIX compliant.
- **Portability:** Fully reproducible notebook (`credit_card_segmentation.ipynb`) executed top-to-bottom without manual interventions.
- **Serialization Standard:** Python `joblib` binary format with schema validation metadata.
- **Code Standards:** PEP 8 compliance, explicit type hints, deterministic random seeds, zero hardcoded directory paths.

---

## 8. Rollout Plan & Milestones

1. **Sprint 1 (Frames 1-7):** Data validation, exploratory analysis, cleaning and imputation engine.
2. **Sprint 2 (Frames 8-12):** Feature engineering, model evaluation ($K$-Means, GMM, Ward, HDBSCAN), and bootstrap stability testing.
3. **Sprint 3 (Frames 13-15):** Profiling, index heatmap, persona cards, surrogate decision trees, and SHAP explainability.
4. **Sprint 4 (Frames 16-18):** Campaign activation playbooks, SR 11-7 governance framework, production pipeline export, PSI drift monitoring, and Streamlit dashboard.

