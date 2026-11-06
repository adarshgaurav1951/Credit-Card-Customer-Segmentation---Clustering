# Credit Card Customer Segmentation - Clustering: Project Idea (Researched, Pipeline, Frame by Frame)

## 1. One-Line Idea
Group credit card customers based on spending behavior to create actionable customer personas for targeted financial services and marketing.

## 2. What This Document Is
1. A research summary: what the data looks like, how banks really use segmentation, which methods and tools people use, and what the regulatory and ethical limits are
2. An end-to-end pipeline for the project make prd file and implement it
3. 18 frames (steps) to build it, each with a goal, tasks, a code hint, and a "done when" check

Source notes are in Section 16. Where sources disagreed, I say so, and you should verify against the original page.

---

# Part A: Research Summary

## 3. Why Banks Segment Credit Card Customers
Card customers use the same product in very different ways. Some pay the full balance every month and use the card for rewards. Others carry a balance and pay interest. Some use the card to withdraw cash like a loan, and others barely use it. One marketing message or one credit policy cannot suit all of them.

Segmentation lets a bank:
- Match offers to behavior (rewards for frequent spenders, low-rate offers for balance carriers)
- Decide who may be ready for a credit limit increase
- Spot customers who are inactive and may leave
- Design products, fees, and communication per group
- Focus collections or support on higher-risk behavior patterns

### The classic behavior split
Industry literature has long divided cardholders into **transactors** (pay in full, little or no interest, driven by rewards) and **revolvers** (carry balances, pay interest, respond to low rates), often with extra groups such as inactive users or higher-risk borrowers. A McKinsey article on card segmentation argues that issuers need more granular segments than this basic pair, based on behavior and customer needs. Your clustering project is a data-driven way to discover these groups instead of assuming them.

### What published projects on this dataset usually find
Several case studies of the same Kaggle dataset report segments along these lines (the exact numbers vary by method):
- **Transactors:** low balance, pay a larger share in full, careful spending
- **Revolvers:** high balance, high cash advance activity, low share paid in full (often the most profitable but riskiest group)
- **High-limit or prime customers:** very high credit limit and high spending, a target for premium offers or limit increases
- **Low-tenure or low-activity customers:** newer or less engaged, a target for activation campaigns

Treat these as hypotheses to test, not answers. Your clusters may differ.

## 4. The Dataset

**Name:** Credit Card Dataset for Clustering ("CC GENERAL"), Kaggle, uploaded by arjunbhasin2013.

| Fact | Detail |
|---|---|
| Size | 8,950 customers and 18 columns (one is `CUST_ID`) |
| Level | One row per customer |
| Period | Described on Kaggle as behavior over the last 6 months. Some write-ups say 12 months, so check the Kaggle page |
| Missing values | `MINIMUM_PAYMENTS`: 313 missing. `CREDIT_LIMIT`: 1 missing |
| Tenure | Values from 6 to 12 months, mostly 12 (mean about 11.5) |
| Distributions | Money columns (balance, purchases, cash advance, payments) are strongly right-skewed with extreme values |
| Full payers | About 15 percent of the average `PRC_FULL_PAYMENT` in one write-up, with a spread of customers at 0 and at 1 |

**Columns:**

| Group | Columns | Meaning |
|---|---|---|
| ID | CUST_ID | Customer identifier (drop before clustering) |
| Balance | BALANCE, BALANCE_FREQUENCY | Balance left on the card, and how often the balance updates (0 to 1) |
| Purchases | PURCHASES, ONEOFF_PURCHASES, INSTALLMENTS_PURCHASES | Total purchase amount and its one-off and installment parts |
| Purchase habits | PURCHASES_FREQUENCY, ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY, PURCHASES_TRX | How often and how many times purchases were made |
| Cash advance | CASH_ADVANCE, CASH_ADVANCE_FREQUENCY, CASH_ADVANCE_TRX | Cash withdrawn against the credit limit, and how often |
| Limit and payments | CREDIT_LIMIT, PAYMENTS, MINIMUM_PAYMENTS, PRC_FULL_PAYMENT | Limit, amounts paid, minimum payment due, and share of statements paid in full |
| Relationship | TENURE | Months as a customer |

**Data quality warnings found in the research:**
- Some columns are described differently across write-ups. For example, `ONEOFF_PURCHASES` is called a "maximum purchase amount" in one place and a total in others. Read the Kaggle description and check the numbers against `PURCHASES` before you rely on one interpretation.
- `CASH_ADVANCE_FREQUENCY` is documented as a 0 to 1 value, but at least one summary shows a maximum of 1.5. Treat values above 1 as a data quality issue to flag.
- The data is anonymized and has no demographics, income, or product details. That limits personas to behavior only.

**Alternatives:**
- UCI "Default of Credit Card Clients" (about 30,000 rows with demographics, bill and payment history) is better suited to classification, but you can also cluster it
- For advanced work: transaction-level data from a bank, or open banking data (not freely available)

## 5. Methods Found in the Literature and in Practice

| Step | Common approach |
|---|---|
| Missing values | Median or KNN imputation for `MINIMUM_PAYMENTS`, drop or fill the single `CREDIT_LIMIT` gap, plus a "was missing" flag |
| Skew and outliers | Log transform (`log1p`) of money columns, winsorizing or robust scaling |
| Features | Ratios such as credit utilization, payment ratio, and cash advance share, plus RFM-style variables (recency, frequency, monetary) and tenure |
| Dimensionality | PCA to reduce correlated columns and for visualization |
| Algorithms | K-means (most common), hierarchical clustering, Gaussian Mixture Models, DBSCAN or HDBSCAN, and K-prototypes for mixed data |
| Choosing the number of clusters | Elbow plot, silhouette score, Davies-Bouldin, Calinski-Harabasz, BIC for mixture models, plus business usefulness |
| Interpretation | Compare each segment's averages with the overall average, name the segment, and describe its needs and risk |
| Advanced | Autoencoder embeddings with clustering, self-organizing maps, "Customer2Vec"-style customer vectors, and multi-view clustering that combines usage and risk views |

One research paper found that segmentation combined with a predictive risk score can separate a high-value group from a higher-risk group. A thesis on open banking data compared autoencoder and self-organizing map hybrids for behavioral segmentation. These show where to go beyond plain K-means.

## 6. Advanced AI Layer (What Makes It Stand Out)

| Technique | What it adds | Caution |
|---|---|---|
| Autoencoder or deep embeddings, then clustering | Learns a compact customer representation from many behaviors | Harder to explain. Compare against simple K-means |
| Gaussian Mixture Models (soft clustering) | Each customer gets a probability of belonging to each segment, which suits customers who sit between groups | Needs the right number of components (use BIC) |
| HDBSCAN on a UMAP embedding | Finds groups of different shapes and marks outliers as noise | Settings change the result. Check stability |
| Stability testing (bootstrap and seed repeats) | Shows whether segments are real or an accident of one run | Often skipped, and it is a strong signal of quality |
| Surrogate decision tree or SHAP on segment labels | Explains in plain rules what defines each segment | Explains the clustering, not causes |
| LLM-assisted persona drafts | Turns cluster statistics into readable persona descriptions | The LLM must only use the numbers you give it, and a human must review every claim |
| Segment-assignment pipeline | Scores new customers into existing segments | Needs preprocessing saved with the model |
| Drift monitoring (PSI) | Detects when behavior changes and segments go stale | Needs new data over time (simulate it) |
| Next-best-action or uplift thinking | Moves from "who they are" to "what offer works" | Requires experiment data. Discuss as a design, do not claim results |

## 7. How Banks Use the Output (Industrial Level)
A segmentation is only valuable when someone acts on it. Typical actions per segment:

| Segment type (example) | Possible action | KPI to track |
|---|---|---|
| Transactors | Rewards offers, merchant partnerships, spend-based incentives | Spend per customer, activation, retention |
| Revolvers | Balance-transfer or low-rate offers, repayment support, early-warning checks | Balance, delinquency rate, revenue per account |
| High-limit and high-spend | Premium products, limit review, concierge benefits | Spend, premium uptake, attrition |
| Cash advance heavy | Review fees and risk, offer alternatives such as installment plans | Cash advance share, delinquency |
| Low-activity or new | Activation campaigns, first-purchase bonuses, onboarding messages | Share with first purchase, usage after 90 days |

**Testing before scaling:** a good rollout runs an A/B test for each action on a sample of the segment before scaling. Your project can include the test design (sample, metric, duration) even without real results.

## 8. Governance, Privacy, and Fairness (Important at Industry Level)
- **Model governance:** banks manage models under model risk frameworks (in the US, SR 11-7 is the foundational guidance). These expect documented purpose and assumptions, validation, and ongoing monitoring. A segmentation used only for marketing is lower risk than one that feeds credit decisions, but you should still document the purpose, data, limits, and monitoring plan.
- **Privacy and automated decisions:** in the EU, GDPR covers profiling and, under Article 22, certain automated decisions with significant effects. If segments are ever used in credit decisions, involve compliance and legal review. Do not assume marketing segmentation is automatically exempt.
- **Fair lending:** US fair lending law (ECOA) prohibits discrimination in credit decisions. Segments used in credit or pricing decisions must be checked for disparate impact.
- **Hidden proxies:** a research paper on bank transaction data warns that clustering can implicitly capture sensitive or protected characteristics even when they are not in the data. If you have any demographic columns in another dataset, test how segments differ across protected groups.
- **Personas are not people:** a persona is a summary of behavior, not a verdict on a customer. Avoid labels that stigmatize customers.

---

# Part B: Pipeline and Architecture

## 10. End-to-End Pipeline

```
Raw data (CC GENERAL.csv)
   |
   v
1 Ingestion --> 2 Validation --> 3 Cleaning and imputation
   |
   v
4 Feature engineering --> 5 Transformation (log, scaling, PCA)
   |
   v
6 Clustering and model selection (K-means, GMM, HDBSCAN, autoencoder option)
   |
   v
7 Validation (metrics, stability) --> 8 Profiling and personas
   |
   v
9 Activation plan (offers, channels, KPIs, A/B test design)
   |
   v
10 Deployment (segment-assignment pipeline, dashboard)
   |
   v
11 Monitoring (drift, segment sizes, retraining) --+
   ^                                               |
   +------------- feedback and retraining ---------+
```

| Stage | Output | Frames |
|---|---|---|
| 1 Ingestion | Raw CSV stored unchanged | 3, 4, 5 |
| 2 Validation | List of data problems and rule checks | 5 |
| 3 Cleaning and imputation | Clean DataFrame | 6 |
| 4 Feature engineering | Ratio and behavior features | 8 |
| 5 Transformation | Scaled (and optionally PCA) matrix | 9 |
| 6 Clustering and selection | Chosen algorithm and k | 10, 11 |
| 7 Validation | Metrics and stability results | 10, 11, 12 |
| 8 Profiling and personas | Segment profiles and persona cards | 13, 14, 15 |
| 9 Activation plan | Offer and KPI table, test design | 16 |
| 10 Deployment | Saved pipeline and dashboard | 18 |
| 11 Monitoring | Drift check and retraining rules | 18 |

---

# Part C: Frame by Frame

18 frames from Frame 1 (Understand Business Problem) to Frame 18 (Deploy, Monitor, and Document).
See `PRD.md` and `credit_card_segmentation.ipynb` for complete specifications and full implementation.

