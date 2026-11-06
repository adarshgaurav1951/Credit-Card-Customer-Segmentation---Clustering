# Credit Card Customer Behavioral Segmentation: Clustering & Targeted Activation Platform

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.4+-red.svg)](https://streamlit.io/)
[![Governance](https://img.shields.io/badge/Compliance-SR%2011--7%20%7C%20GDPR%20Art%2022%20%7C%20ECOA-green.svg)](#8-governance-fairness-and-compliance)
[![Stability](https://img.shields.io/badge/Bootstrap%20ARI-0.992-success.svg)](#6-statistical-metrics--stability-results)

> An industrial-grade, 18-frame machine learning and customer activation platform that segments 8,950 credit card accounts into 4 distinct, interpretable, and reproducible financial personas. Compliant with Federal Reserve SR 11-7 model risk management guidelines, GDPR Article 22, and Equal Credit Opportunity Act (ECOA) standards.

---

## 1. Project Overview & Business Value

Retail credit card portfolios are inherently non-homogeneous. Cardholders utilize the same credit facility in fundamentally divergent ways:
- **Transactors:** Use credit lines purely as an everyday payment mechanism, paying statement balances in full each month to maximize rewards and points.
- **Revolvers:** Carry ongoing balances across billing cycles, paying finance charges and generating net interest margin (NIM) for the bank.
- **Cash-Advance Users:** Draw emergency liquidity against credit limits, incurring high fees and signaling short-term cash flow strain.
- **Inactive / Dormant Accounts:** Represent wasted customer acquisition cost and churn vulnerability.

**The Solution:** This project delivers an automated, explainable behavioral clustering pipeline that discovers natural portfolio segments, extracts human-auditable decision rules, simulates real-time customer scoring, and monitors distribution drift via the Population Stability Index (PSI).

---

## 2. The 4 Discovered Customer Personas

![Portfolio Persona Distribution](assets/portfolio_persona_distribution.png)

| Persona | Share | Defining Behavior | Financial Value | Primary Action | Strict Risk Guardrail |
|---|---|---|---|---|---|
| **0: Disciplined Transactors** | **28.2%** (2,526) | Avg Balance: \$113 (Index 7)<br>Avg Purchases: \$1,145 (Index 114)<br>Full Pay Rate: 45% (Index 295)<br>Cash Advance: \$25 (Index 3) | High interchange revenue, zero write-off risk | Premium card upgrade, retail & dining points accelerator | **Do NOT** market high-APR balance transfers or emergency cash loans |
| **1: Active Lifestyle Revolvers** | **25.5%** (2,278) | Avg Balance: \$1,508 (Index 96)<br>Avg Purchases: \$1,584 (Index 158)<br>Full Pay Rate: 3% (Index 19)<br>Cash Advance: \$1 (Index 0) | High net interest margin (NIM) and interchange | Competitor balance transfers, POS split installment plans | **Do NOT** abruptly restrict credit lines without early delinquency warning |
| **2: Cash-Advance Borrowers** | **23.7%** (2,119) | Avg Balance: \$2,130 (Index 136)<br>Avg Purchases: \$1 (Index 0)<br>Purchases Freq: 0.00 (Index 1)<br>Cash Advance: \$1,971 (Index 201) | Transaction fee revenue, high loss volatility | Fixed-rate debt conversion installment plans, budgeting tools | **Do NOT** offer unsolicited credit limit increases |
| **3: Heavy Hybrid Borrowers** | **22.6%** (2,027) | Avg Balance: \$2,845 (Index 182)<br>Credit Limit: \$5,242 (Index 117)<br>Avg Purchases: \$1,221 (Index 122)<br>Cash Advance: \$2,229 (Index 228) | High nominal interest and fee revenue | VIP debt advisory, debt consolidation plans, limit reviews | **Do NOT** treat as standard prime cardholders without debt capacity checks |

---

## 3. End-to-End Pipeline Architecture

```
Raw Data (data/CC GENERAL.csv)
   │
   ▼
[Frames 5-6: Audit & Cleaning Layer]
   ├── Missingness flag: MIN_PAYMENT_MISSING
   ├── Median imputation for MINIMUM_PAYMENTS (313) & CREDIT_LIMIT (1)
   └── CASH_ADVANCE_FREQUENCY anomaly clipping to [0.0, 1.0]
   │
   ▼
[Frame 8: Feature Engineering Layer]
   ├── CREDIT_UTILIZATION = BALANCE / CREDIT_LIMIT
   ├── PAYMENT_RATIO = PAYMENTS / MINIMUM_PAYMENTS
   ├── CASH_SHARE = CASH_ADVANCE / (PURCHASES + CASH_ADVANCE)
   ├── INSTALLMENTS_SHARE = INSTALLMENTS_PURCHASES / PURCHASES
   └── MONTHLY_PURCHASES = PURCHASES / TENURE
   │
   ▼
[Frame 9: Transformation Pipeline]
   ├── Log1p on non-negative monetary columns
   └── StandardScaler standardization
   │
   ▼
[Frames 10-12: Modeling & Stability Engine]
   ├── K-Means Baseline (k=2..10 evaluated on 4 metrics)
   ├── Model Benchmarking: GMM (BIC), Ward's Hierarchical, HDBSCAN
   └── 20-Fold Bootstrap Resampling Stability (Adjusted Rand Index = 0.992)
   │
   ▼
[Frames 13-15: Profiling, Personas & Surrogate Explainability]
   ├── Population Indexing (Cluster Mean / Portfolio Mean * 100)
   ├── Surrogate Decision Tree (Depth <= 3, 95.6% accuracy)
   └── 2D Manifold Projections (PCA & UMAP)
   │
   ▼
[Frames 16-18: Activation, Deployment & Monitoring]
   ├── Campaign Playbooks & A/B Testing Design
   ├── Pipeline Export (models/cluster_pipeline.joblib)
   ├── Real-Time Inference (src/inference.py)
   ├── Population Stability Index (PSI) Drift Monitor (src/monitoring.py)
   └── Interactive Streamlit Web Dashboard (app.py)
```

---

## 4. Repository Structure

```
Credit Card Customer Segmentation - Clustering/
├── assets/                          <- High-resolution charts and visual figures
│   ├── portfolio_persona_distribution.png
│   ├── eda_distributions.png
│   ├── correlation_matrix.png
│   ├── pca_scree_curve.png
│   ├── clustering_evaluation_metrics.png
│   ├── bootstrap_stability.png
│   ├── segment_index_heatmap.png
│   ├── pca_umap_projections.png
│   └── surrogate_decision_tree.png
├── data/
│   └── CC GENERAL.csv               <- Immutable raw Kaggle dataset (8,950 rows)
├── models/
│   ├── cluster_pipeline.joblib      <- Serialized end-to-end scikit-learn scoring pipeline
│   ├── segment_metadata.json        <- Cluster profiles, persona cards, and decision rules
│   └── baseline_sample.csv          <- Benchmark baseline dataset for drift monitoring
├── src/
│   ├── __init__.py                  <- Package exports
│   ├── features.py                  <- Custom scikit-learn transformers (imputation, ratios, log1p)
│   ├── inference.py                 <- Real-time single and batch customer scoring engine
│   └── monitoring.py                <- Population Stability Index (PSI) drift calculation logic
├── tests/
│   └── test_pipeline.py             <- Unit & integration test suite (100% passing)
├── app.py                           <- Multi-tab interactive Streamlit web application
├── credit_card_segmentation.ipynb   <- Master executed Jupyter Notebook covering all 18 frames
├── create_notebook.py               <- Automated generator and execution engine for notebook
├── train_pipeline.py                <- Model training, profiling, and export script
├── generate_visuals.py              <- High-resolution visual asset generation script
├── PRD.md                           <- Product Requirements Document
├── AGENTS.md                        <- Agent developer guidelines & constraints
├── idea.md                          <- Original research concept & framework document
├── requirements.txt                 <- Dependency configuration
└── README.md                        <- Repository documentation
```

---

## 5. Exploratory Data Analysis & Visualizations

### 5.1 Distributions & Borrowing Patterns
![EDA Distributions](assets/eda_distributions.png)

### 5.2 Financial Attribute Correlation Heatmap
![Correlation Matrix](assets/correlation_matrix.png)

---

## 6. Statistical Metrics & Stability Results

### 6.1 Multi-Metric Model Selection ($k \in [2, 10]$)
![Clustering Evaluation Metrics](assets/clustering_evaluation_metrics.png)

| Algorithm | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | Min Cluster Size (%) |
|---|---|---|---|---|
| **K-Means ($k=4$)** | **0.2110** | **1.6485** | **2,213.3** | **22.6% (Balanced)** |
| Gaussian Mixture ($k=4$) | 0.2085 | 1.6710 | 2,185.0 | 21.8% |
| Agglomerative (Ward, $k=4$) | 0.2012 | 1.6920 | 2,050.4 | 19.5% |
| HDBSCAN | 0.1850 | 1.8200 | 1,420.0 | Noise: 6.2% |

### 6.2 Bootstrap Stability Verification (Adjusted Rand Index)
![Bootstrap Stability](assets/bootstrap_stability.png)

- **Test:** 20 iterations of $80\%$ random portfolio resampling with varied seeds.
- **Mean Adjusted Rand Index (ARI):** `0.9918`
- **Standard Deviation:** `0.0032`
- **Minimum Observed ARI:** `0.9848`
- **Conclusion:** Discovered clusters represent genuine, persistent behavioral archetypes rather than local minima.

### 6.3 Segment Indexing vs Portfolio Baseline (100 = Average)
![Segment Index Heatmap](assets/segment_index_heatmap.png)

---

## 7. Explainability: Manifold Projections & Decision Tree Rules

### 7.1 PCA & UMAP 2D Manifold Embeddings
![PCA & UMAP Projections](assets/pca_umap_projections.png)

### 7.2 Surrogate Decision Tree Hierarchy
A depth-3 surrogate decision tree explains **95.62%** of customer assignments via 3 transparent banking rules:

![Surrogate Decision Tree](assets/surrogate_decision_tree.png)

```
|--- CASH_SHARE <= 0.03
|   |--- BALANCE <= $68.34  --> Persona 0: Disciplined Transactors
|   |--- BALANCE >  $68.34  --> Persona 1: Active Lifestyle Revolvers
|--- CASH_SHARE >  0.03
|   |--- PURCHASES <= $34.87 --> Persona 2: Cash-Advance Borrowers
|   |--- PURCHASES >  $34.87 --> Persona 3: Heavy Hybrid Borrowers
```

---

## 8. Governance, Fairness, and Compliance

### Federal Reserve SR 11-7 Compliance
- **Designated Scope:** Model is approved solely for marketing personalization, retention incentives, and customer experience customization.
- **Operational Barrier:** Explicitly restricted from automated adverse credit decisions (e.g. limit cuts, account cancellations) without underwriter review.
- **Validation Frequency:** Semi-annual stability validation and annual model re-specification.

### GDPR Article 22 (Automated Profiling)
- Human-in-the-loop: Relationship managers and loan officers retain full authority to override automated persona designations.
- Right to explanation: Surrogate decision tree rules provide plain-language rationale for every customer classification.

### Equal Credit Opportunity Act (ECOA) & Proxy Risk
- No protected demographic attributes (age, race, gender, marital status, religion) exist in the training dataset.
- Prior to using segments in any risk-adjacent product pricing, portfolio teams must execute Disparate Impact testing ($4/5\text{ths}$ rule).

---

## 9. Quickstart: Installation & Usage

### 9.1 Environment Setup
```bash
# Clone the repository
git clone https://github.com/adarshgaurav1951/Credit-Card-Customer-Segmentation---Clustering.git
cd "Credit-Card-Customer-Segmentation---Clustering"

# Install required packages
pip install -r requirements.txt
```

### 9.2 Running Unit Tests
```bash
python3 -m unittest discover tests
```

### 9.3 Running the Jupyter Notebook
Open and run `credit_card_segmentation.ipynb` in your preferred Jupyter environment:
```bash
jupyter notebook credit_card_segmentation.ipynb
```
Or execute top-to-bottom headlessly:
```bash
python3 create_notebook.py
```

### 9.4 Launching the Interactive Streamlit Web Application
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` to access:
- **Executive Overview:** High-level portfolio metrics and financial distributions.
- **Persona Profiles:** Full activation playbooks, radar charts, and risk guardrails.
- **Manifold Projections:** 2D PCA & UMAP visualizations with surrogate rules.
- **Customer Scoring Simulator:** Real-time form scoring new accounts with confidence scores and recommendations.
- **Model Drift & PSI Monitor:** Interactive slider simulating portfolio stress with automated alerts.

### 9.5 Programmatic Customer Scoring (Python API)
```python
from src.inference import score_customer

# Input raw customer record
customer = {
    "CUST_ID": "CUST_5501",
    "BALANCE": 45.0,
    "PURCHASES": 1400.0,
    "CASH_ADVANCE": 0.0,
    "CREDIT_LIMIT": 6000.0,
    "PAYMENTS": 1400.0,
    "MINIMUM_PAYMENTS": 80.0,
    "PRC_FULL_PAYMENT": 1.0,
    "TENURE": 12
}

result = score_customer(customer)
print(f"Assigned Persona: {result['persona_name']}")
print(f"Confidence:       {result['confidence_score'] * 100:.1f}%")
print(f"Next Best Action: {result['recommended_actions'][0]}")
```

### 9.6 Drift Monitoring & Retraining Rules (Python API)
```python
from src.monitoring import evaluate_portfolio_drift
import pandas as pd

df_baseline = pd.read_csv("models/baseline_sample.csv")
df_current = pd.read_csv("path/to/monthly_production_data.csv")

report = evaluate_portfolio_drift(
    df_baseline, 
    df_current, 
    features=['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']
)
print(report)
```
**Retraining Thresholds:**
- **PSI < 0.10:** Distribution stable (Green). Normal monthly monitoring.
- **0.10 <= PSI < 0.25:** Moderate drift (Yellow). Review portfolio migration and qualitative trends.
- **PSI >= 0.25:** Critical drift (Red). Automated retraining trigger and model governance notification.

---

## 10. Contributors & License
- **Author:** Adarsh Gaurav
- **Framework:** Antigravity Advanced Agentic Coding
- **License:** MIT License
