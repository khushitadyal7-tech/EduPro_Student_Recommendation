
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==========================================
# 1. LOAD PROCESSED FEATURES
# ==========================================

file_path = "data/processed/clustering_features.csv"

X = pd.read_csv(file_path)

print("CLUSTERING DATA LOADED!")
print("Dataset Shape:", X.shape)

# ==========================================
# 2. TEST DIFFERENT CLUSTER COUNTS
# ==========================================

inertia_values = []
silhouette_values = []

cluster_range = range(2, 11)

print("\n========== CLUSTER EVALUATION ==========")

for k in cluster_range:

    print(f"\nTesting {k} clusters...")

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X)

    inertia_values.append(kmeans.inertia_)

    score = silhouette_score(
        X,
        labels,
        sample_size=1000,
        random_state=42
    )

    silhouette_values.append(score)

    print("Inertia:", round(kmeans.inertia_, 2))
    print("Silhouette Score:", round(score, 4))

# ==========================================
# 3. CREATE RESULTS TABLE
# ==========================================

results = pd.DataFrame({
    "Number of Clusters": list(cluster_range),
    "Inertia": inertia_values,
    "Silhouette Score": silhouette_values
})

print("\n========== FINAL RESULTS ==========")
print(results.to_string(index=False))

# ==========================================
# 4. SAVE RESULTS
# ==========================================

os.makedirs("reports/clustering", exist_ok=True)

results.to_csv(
    "reports/clustering/cluster_evaluation.csv",
    index=False
)

# ==========================================
# 5. ELBOW METHOD GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(cluster_range),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(list(cluster_range))
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "reports/clustering/elbow_method.png"
)

plt.show()

# ==========================================
# 6. SILHOUETTE SCORE GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(cluster_range),
    silhouette_values,
    marker="o"
)

plt.title("Silhouette Score Analysis")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.xticks(list(cluster_range))
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "reports/clustering/silhouette_scores.png"
)

plt.show()

print("\n========================================")
print("CLUSTER EVALUATION COMPLETED!")
print("========================================")
