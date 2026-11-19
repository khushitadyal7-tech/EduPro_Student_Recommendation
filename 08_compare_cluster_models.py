
import pandas as pd
import joblib

from pathlib import Path
from sklearn.cluster import KMeans

# ==========================================
# 1. PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

PROFILE_FILE = BASE_DIR / "data" / "processed" / "student_profiles.csv"
FEATURE_FILE = BASE_DIR / "data" / "processed" / "clustering_features.csv"

OUTPUT_DIR = BASE_DIR / "reports" / "clustering" / "model_comparison"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD DATA
# ==========================================

profiles = pd.read_csv(PROFILE_FILE)
features = pd.read_csv(FEATURE_FILE)

print("DATA LOADED SUCCESSFULLY!")
print("Student Profiles:", profiles.shape)
print("Clustering Features:", features.shape)

# ==========================================
# 3. SELECT PROFILE FEATURES
# ==========================================

numeric_columns = [
    "Age",
    "TotalCourses",
    "TotalSpending",
    "AverageSpending",
    "AverageCourseRating",
    "CategoryDiversity",
    "BeginnerCourseRatio",
    "IntermediateCourses",
    "AdvancedCourses",
    "AdvancedCourseRatio"
]

# ==========================================
# 4. TRAIN AND COMPARE MODELS
# ==========================================

for k in [2, 4]:

    print("\n" + "=" * 45)
    print(f"TRAINING MODEL WITH {k} CLUSTERS")
    print("=" * 45)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(features)

    # Copy student profiles
    result = profiles.copy()
    result["Cluster"] = labels

    # Save assignments
    result.to_csv(
        OUTPUT_DIR / f"student_clusters_k{k}.csv",
        index=False
    )

    # Save trained model
    joblib.dump(
        model,
        MODEL_DIR / f"kmeans_model_k{k}.pkl"
    )

    # --------------------------------------
    # CLUSTER SIZES
    # --------------------------------------

    print("\nCLUSTER SIZES:")

    print(
        result["Cluster"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------
    # NUMERICAL PROFILES
    # --------------------------------------

    print("\nCLUSTER PROFILE SUMMARY:")

    summary = result.groupby("Cluster")[numeric_columns].mean()

    print(summary.round(2).to_string())

    summary.to_csv(
        OUTPUT_DIR / f"cluster_profile_k{k}.csv"
    )

    # --------------------------------------
    # PREFERRED COURSE LEVEL
    # --------------------------------------

    print("\nPREFERRED COURSE LEVEL:")

    level_table = pd.crosstab(
        result["Cluster"],
        result["PreferredLevel"]
    )

    print(level_table.to_string())

    level_table.to_csv(
        OUTPUT_DIR / f"course_level_k{k}.csv"
    )

    # --------------------------------------
    # PREFERRED CATEGORY
    # --------------------------------------

    print("\nPREFERRED CATEGORY:")

    category_table = pd.crosstab(
        result["Cluster"],
        result["PreferredCategory"]
    )

    print(category_table.to_string())

    category_table.to_csv(
        OUTPUT_DIR / f"category_k{k}.csv"
    )

print("\n" + "=" * 45)
print("MODEL COMPARISON COMPLETED!")
print("=" * 45)
