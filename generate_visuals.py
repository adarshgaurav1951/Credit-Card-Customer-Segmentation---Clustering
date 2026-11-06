"""
Generates high-resolution visualization assets for GitHub documentation and reports.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Ensure root directory in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.features import CreditCardPreprocessor, Log1pTransformer
import joblib

os.makedirs("assets", exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
RANDOM_STATE = 42

print("Loading data...")
df = pd.read_csv("data/CC GENERAL.csv")
pipeline = joblib.load("models/cluster_pipeline.joblib")

# Preprocessed & scaled matrices
prep = pipeline.named_steps["prep"]
log = pipeline.named_steps["log"]
scale = pipeline.named_steps["scale"]
km = pipeline.named_steps["cluster"]

df_prep = prep.transform(df)
df_log = log.transform(df_prep)
Z = scale.transform(df_log)
labels = km.predict(Z)
df["CLUSTER"] = labels

cluster_names = {
    0: "0: Disciplined Transactors",
    1: "1: Active Lifestyle Revolvers",
    2: "2: Cash-Advance Borrowers",
    3: "3: Heavy Hybrid Borrowers"
}
df["PERSONA"] = df["CLUSTER"].map(cluster_names)

# -------------------------------------------------------------
# 1. EDA Distributions
# -------------------------------------------------------------
print("Generating 1. assets/eda_distributions.png...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

sns.histplot(df['BALANCE'], kde=True, ax=axes[0, 0], color='#1f77b4', bins=40)
axes[0, 0].set_title("Distribution of Raw BALANCE (Heavy Positive Skew)")
axes[0, 0].set_xlabel("Balance ($)")

sns.histplot(np.log1p(df['BALANCE']), kde=True, ax=axes[0, 1], color='#2ca02c', bins=40)
axes[0, 1].set_title("Distribution of log1p(BALANCE) (Normalized Distribution)")
axes[0, 1].set_xlabel("log1p(Balance)")

sc = axes[1, 0].scatter(
    np.log1p(df['PURCHASES']), np.log1p(df['PAYMENTS']),
    alpha=0.3, c=df['PRC_FULL_PAYMENT'], cmap='viridis', s=15
)
axes[1, 0].set_title("log1p(Purchases) vs log1p(Payments) by Full Payment %")
axes[1, 0].set_xlabel("log1p(Purchases)")
axes[1, 0].set_ylabel("log1p(Payments)")
cbar = fig.colorbar(sc, ax=axes[1, 0])
cbar.set_label("Full Payment Ratio")

axes[1, 1].scatter(
    np.log1p(df['PURCHASES']), np.log1p(df['CASH_ADVANCE']),
    alpha=0.25, color='#d62728', s=15
)
axes[1, 1].set_title("Borrowing Trade-Off: log1p(Purchases) vs log1p(Cash Advance)")
axes[1, 1].set_xlabel("log1p(Purchases)")
axes[1, 1].set_ylabel("log1p(Cash Advance)")

plt.tight_layout()
plt.savefig("assets/eda_distributions.png", bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 2. Correlation Matrix
# -------------------------------------------------------------
print("Generating 2. assets/correlation_matrix.png...")
plt.figure(figsize=(11, 8), dpi=300)
key_cols = [
    'BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
    'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT'
]
corr = df[key_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5)
plt.title("Correlation Matrix of Core Financial Attributes", fontsize=14, pad=12)
plt.tight_layout()
plt.savefig("assets/correlation_matrix.png", bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 3. PCA Scree Curve
# -------------------------------------------------------------
print("Generating 3. assets/pca_scree_curve.png...")
pca_full = PCA(random_state=RANDOM_STATE).fit(Z)
cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100

plt.figure(figsize=(10, 5), dpi=300)
plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o', linestyle='--', color='#2b5c8f', linewidth=2)
plt.axhline(y=80, color='r', linestyle=':', label='80% Explained Variance')
plt.axhline(y=85, color='orange', linestyle=':', label='85% Explained Variance')
plt.xlabel("Number of Principal Components", fontsize=12)
plt.ylabel("Cumulative Explained Variance (%)", fontsize=12)
plt.title("PCA Explained Variance Retention Curve", fontsize=14, pad=12)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig("assets/pca_scree_curve.png", bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 4. Clustering Evaluation Metrics
# -------------------------------------------------------------
print("Generating 4. assets/clustering_evaluation_metrics.png...")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

k_range = list(range(2, 11))
inertias, sils, dbs, min_sizes = [], [], [], []

for k in k_range:
    km_test = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z)
    inertias.append(km_test.inertia_)
    sils.append(silhouette_score(Z, km_test.labels_))
    dbs.append(davies_bouldin_score(Z, km_test.labels_))
    sizes = pd.Series(km_test.labels_).value_counts(normalize=True)
    min_sizes.append(sizes.min() * 100)

fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300)

axes[0, 0].plot(k_range, inertias, marker='o', color='#1f77b4', linewidth=2)
axes[0, 0].set_title("Elbow Method: Inertia vs k")
axes[0, 0].set_xlabel("Number of Clusters (k)")
axes[0, 0].set_ylabel("Inertia")

axes[0, 1].plot(k_range, sils, marker='s', color='#2ca02c', linewidth=2)
axes[0, 1].axvline(x=4, color='black', linestyle='--', alpha=0.7, label='Optimal k=4')
axes[0, 1].set_title("Silhouette Score vs k (Higher is Better)")
axes[0, 1].set_xlabel("Number of Clusters (k)")
axes[0, 1].set_ylabel("Silhouette Score")
axes[0, 1].legend()

axes[1, 0].plot(k_range, dbs, marker='^', color='#d62728', linewidth=2)
axes[1, 0].axvline(x=4, color='black', linestyle='--', alpha=0.7, label='Optimal k=4')
axes[1, 0].set_title("Davies-Bouldin Index vs k (Lower is Better)")
axes[1, 0].set_xlabel("Number of Clusters (k)")
axes[1, 0].set_ylabel("Davies-Bouldin Index")
axes[1, 0].legend()

axes[1, 1].plot(k_range, min_sizes, marker='d', color='#9467bd', linewidth=2)
axes[1, 1].axhline(y=5.0, color='r', linestyle='--', label='5% Minimum Viability Threshold')
axes[1, 1].axvline(x=4, color='black', linestyle='--', alpha=0.7, label='Optimal k=4 (22.6% min)')
axes[1, 1].set_title("Minimum Cluster Size % vs k")
axes[1, 1].set_xlabel("Number of Clusters (k)")
axes[1, 1].set_ylabel("Smallest Cluster Share (%)")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("assets/clustering_evaluation_metrics.png", bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 5. Bootstrap Stability
# -------------------------------------------------------------
print("Generating 5. assets/bootstrap_stability.png...")
from sklearn.metrics import adjusted_rand_score
rng = np.random.default_rng(RANDOM_STATE)
N = len(Z)
sample_size = int(0.80 * N)
ari_scores = []

for i in range(20):
    sub_idx = rng.choice(N, size=sample_size, replace=False)
    km_sub = KMeans(n_clusters=4, n_init=10, random_state=i).fit(Z[sub_idx])
    ari = adjusted_rand_score(labels[sub_idx], km_sub.labels_)
    ari_scores.append(ari)

mean_ari = np.mean(ari_scores)
plt.figure(figsize=(9, 4), dpi=300)
sns.boxplot(x=ari_scores, color='#85c1e9', width=0.35)
sns.stripplot(x=ari_scores, color='#1b4f72', size=9, jitter=0.2)
plt.title(f"20-Iteration Bootstrap Resampling Stability (Mean ARI = {mean_ari:.4f})", fontsize=13, pad=10)
plt.xlabel("Adjusted Rand Index (ARI)", fontsize=11)
plt.tight_layout()
plt.savefig("assets/bootstrap_stability.png", bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 6. Segment Index Heatmap
# -------------------------------------------------------------
print("Generating 6. assets/segment_index_heatmap.png...")
profile_cols = [
    'BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
    'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS',
    'PRC_FULL_PAYMENT', 'PURCHASES_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'TENURE'
]
cluster_means = df.groupby('PERSONA')[profile_cols].mean()
overall_means = df[profile_cols].mean()
index_table = (cluster_means / overall_means * 100).round(1)

plt.figure(figsize=(15, 6), dpi=300)
sns.heatmap(index_table, annot=True, fmt=".0f", cmap="vlag", center=100, linewidths=0.5, cbar_kws={'label': 'Index vs Portfolio Average (100 = Baseline)'})
plt.title("Customer Segment Index Heatmap (100 = Portfolio Average)", fontsize=14, pad=12)
plt.ylabel("Customer Persona", fontsize=12)
plt.tight_layout()
plt.savefig("assets/segment_index_heatmap.png", bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 7. PCA & UMAP Projections
# -------------------------------------------------------------
print("Generating 7. assets/pca_umap_projections.png...")
pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
Z_pca = pca_2d.fit_transform(Z)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, name in cluster_names.items():
    mask = (labels == i)
    axes[0].scatter(
        Z_pca[mask, 0], Z_pca[mask, 1],
        alpha=0.45, s=20, label=name, color=palette[i]
    )

axes[0].set_title(f"PCA 2D Projection ({pca_2d.explained_variance_ratio_.sum()*100:.1f}% Variance Explained)", fontsize=13)
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")
axes[0].legend(loc='upper right', frameon=True, fontsize=9)

try:
    import umap
    reducer = umap.UMAP(n_neighbors=25, min_dist=0.3, random_state=RANDOM_STATE)
    Z_umap = reducer.fit_transform(Z)
    for i, name in cluster_names.items():
        mask = (labels == i)
        axes[1].scatter(
            Z_umap[mask, 0], Z_umap[mask, 1],
            alpha=0.45, s=20, label=name, color=palette[i]
        )
    axes[1].set_title("UMAP 2D Non-Linear Manifold Projection", fontsize=13)
    axes[1].set_xlabel("UMAP Dimension 1")
    axes[1].set_ylabel("UMAP Dimension 2")
    axes[1].legend(loc='upper right', frameon=True, fontsize=9)
except Exception as e:
    axes[1].text(0.5, 0.5, f"UMAP: {e}", ha='center')

plt.tight_layout()
plt.savefig("assets/pca_umap_projections.png", bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 8. Surrogate Decision Tree Rules
# -------------------------------------------------------------
print("Generating 8. assets/surrogate_decision_tree.png...")
X_explain = prep.transform(df.drop(columns=['CLUSTER', 'PERSONA'], errors='ignore'))
dt = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
dt.fit(X_explain, labels)

plt.figure(figsize=(20, 9), dpi=300)
plot_tree(
    dt,
    feature_names=list(X_explain.columns),
    class_names=[
        "0: Transactors",
        "1: Revolvers",
        "2: Cash Borrowers",
        "3: Heavy Hybrid"
    ],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title(f"Surrogate Decision Tree (Fidelity Accuracy: {dt.score(X_explain, labels)*100:.2f}%)", fontsize=16, pad=15)
plt.tight_layout()
plt.savefig("assets/surrogate_decision_tree.png", bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 9. Portfolio Bar Summary
# -------------------------------------------------------------
print("Generating 9. assets/portfolio_persona_distribution.png...")
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
counts = df['PERSONA'].value_counts()
bars = ax.barh(counts.index, counts.values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], height=0.55)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 30, bar.get_y() + bar.get_height()/2, f"{w:,} ({w/len(df)*100:.1f}%)", va='center', fontsize=11, fontweight='bold')
ax.set_xlim(0, max(counts.values) * 1.2)
ax.set_title("Customer Distribution Across the 4 Discovered Personas", fontsize=14, pad=12)
ax.set_xlabel("Number of Cardholders", fontsize=12)
plt.tight_layout()
plt.savefig("assets/portfolio_persona_distribution.png", bbox_inches='tight')
plt.close()

print("All visual assets successfully generated in assets/!")

