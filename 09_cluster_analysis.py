
import pandas as pd
from pathlib import Path

# ==========================================
# 1. SET PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "student_clusters.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "clustering"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD CLUSTERED DATA
# ==========================================

df = pd.read_csv(INPUT_FILE)

print("CLUSTERED STUDENT DATA LOADED!")
print("Dataset Shape:", df.shape)

# ==========================================
# 3. NUMERICAL CLUSTER ANALYSIS
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

cluster_summary = df.groupby("Cluster")[numeric_columns].mean()

print("\n========== CLUSTER PROFILE SUMMARY ==========")
print(cluster_summary.round(2).to_string())

# Save numerical summary
cluster_summary.to_csv(
    OUTPUT_DIR / "cluster_profile_summary.csv"
)

# ==========================================
# 4. STUDENT COUNT IN EACH CLUSTER
# ==========================================

cluster_sizes = df["Cluster"].value_counts().sort_index()

print("\n========== CLUSTER SIZES ==========")
print(cluster_sizes)

cluster_sizes.to_csv(
    OUTPUT_DIR / "cluster_sizes.csv",
    header=["StudentCount"]
)

# ==========================================
# 5. PREFERRED CATEGORY DISTRIBUTION
# ==========================================

category_distribution = pd.crosstab(
    df["Cluster"],
    df["PreferredCategory"]
)

print("\n========== PREFERRED CATEGORY ==========")
print(category_distribution)

category_distribution.to_csv(
    OUTPUT_DIR / "preferred_category_distribution.csv"
)

# ==========================================
# 6. PREFERRED LEVEL DISTRIBUTION
# ==========================================

level_distribution = pd.crosstab(
    df["Cluster"],
    df["PreferredLevel"]
)

print("\n========== PREFERRED COURSE LEVEL ==========")
print(level_distribution)

level_distribution.to_csv(
    OUTPUT_DIR / "preferred_level_distribution.csv"
)

# ==========================================
# 7. GENDER DISTRIBUTION
# ==========================================

gender_distribution = pd.crosstab(
    df["Cluster"],
    df["Gender"]
)

print("\n========== GENDER DISTRIBUTION ==========")
print(gender_distribution)

gender_distribution.to_csv(
    OUTPUT_DIR / "gender_distribution.csv"
)

# ==========================================
# 8. MOST COMMON PREFERENCES
# ==========================================

print("\n========== MOST COMMON PREFERENCES ==========")

for cluster in sorted(df["Cluster"].unique()):

    cluster_data = df[df["Cluster"] == cluster]

    common_category = cluster_data["PreferredCategory"].mode()[0]
    common_level = cluster_data["PreferredLevel"].mode()[0]

    print(f"\nCluster {cluster}")
    print("Most Preferred Category:", common_category)
    print("Most Preferred Level:", common_level)

# ==========================================
# 9. COMPLETION
# ==========================================

print("\n========================================")
print("CLUSTER ANALYSIS COMPLETED!")
print("========================================")
print("All cluster reports saved in:", OUTPUT_DIR)
