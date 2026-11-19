
import pandas as pd
import joblib

from pathlib import Path
from sklearn.cluster import KMeans

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

PROFILE_FILE = BASE_DIR / "data" / "processed" / "student_profiles.csv"
FEATURE_FILE = BASE_DIR / "data" / "processed" / "clustering_features.csv"

OUTPUT_FILE = BASE_DIR / "data" / "processed" / "student_clusters.csv"
MODEL_FILE = BASE_DIR / "models" / "kmeans_model.pkl"

# ==========================================
# 2. LOAD DATA
# ==========================================

profiles = pd.read_csv(PROFILE_FILE)
features = pd.read_csv(FEATURE_FILE)

print("DATA LOADED SUCCESSFULLY!")
print("Student Profiles:", profiles.shape)
print("Clustering Features:", features.shape)

# ==========================================
# 3. TRAIN FINAL WORKING MODEL
# ==========================================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(features)

# ==========================================
# 4. ASSIGN CLUSTERS
# ==========================================

profiles["Cluster"] = labels

# ==========================================
# 5. SAVE RESULTS
# ==========================================

profiles.to_csv(OUTPUT_FILE, index=False)

joblib.dump(kmeans, MODEL_FILE)

# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

print("\n========== FINAL CLUSTER DISTRIBUTION ==========")

print(
    profiles["Cluster"]
    .value_counts()
    .sort_index()
)

print("\nStudent assignments saved!")
print("K-Means model saved!")

print("\n========================================")
print("FOUR-CLUSTER MODEL COMPLETED!")
print("========================================")
