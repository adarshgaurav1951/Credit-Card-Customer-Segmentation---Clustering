# Agent Rules & Development Guidelines

- **Project:** Credit Card Customer Segmentation - Clustering
- **Primary Deliverables:** 
  - `credit_card_segmentation.ipynb`: Single notebook executed top-to-bottom across all 18 frames.
  - `PRD.md`: Formal Product Requirements Document.
  - `models/cluster_pipeline.joblib`: Serialized end-to-end scikit-learn pipeline for scoring new customers.
  - `models/segment_metadata.json`: Segment personas, feature benchmarks, and rule sets.
  - `src/inference.py`: Production scoring module for individual and batch scoring.
  - `src/monitoring.py`: Population Stability Index (PSI) drift monitoring engine.
  - `app.py`: Interactive Streamlit dashboard for business users and bankers.
  - `README.md`: Comprehensive project documentation.

## Strict Engineering Guidelines
1. **Never modify the original CSV:** Keep `data/CC GENERAL.csv` completely immutable.
2. **Key Separation:** Drop `CUST_ID` before feature transformation and clustering; retain an index mapping to reconnect predictions back to accounts.
3. **Pipeline Integrity:** Wrap all imputation, non-negative log transformations, feature engineering, and standard scaling into scikit-learn compatible transformers or a single `Pipeline` so that new customer records can be scored seamlessly.
4. **Reproducibility:** Always fix `random_state=42` and `n_init=20` across all stochastic algorithms (K-Means, GMM, PCA, Decision Tree, UMAP).
5. **Rigorous Model Selection:** Never choose $k$ based on inertia alone. Evaluate the Elbow method, Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, and business operational viability (minimum cluster size $\ge 5\%$).
6. **Stability Testing:** Test cluster stability using bootstrap resampling (Adjusted Rand Index across 20 resamples).
7. **Actionable Personas:** Every segment must have:
   - Cluster size and percentage
   - Profiling table with averages and index to portfolio baseline
   - Plain-language persona name (respectful, non-stigmatizing)
   - Value and risk profile
   - Recommended actions & "Do Not" guardrails
8. **Explainability:** Provide surrogate decision tree rules and SHAP feature importances.
9. **Governance Compliance:** Document SR 11-7 model risk, GDPR Article 22, and ECOA proxy risk guardrails.
10. **Notebook Format:** Every frame must have an explicit Markdown heading (`# Frame N: ...`), an introductory goal summary, concise and well-commented code, and a post-execution summary.

