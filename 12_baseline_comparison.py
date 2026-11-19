
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans


# ============================================================
# 1. SETTINGS AND FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "raw", "EduPro Online Platform.xlsx"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "reports", "baseline_comparison"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_K = 5
NUMBER_OF_CLUSTERS = 4

print("\n" + "=" * 65)
print("EDUPRO - BASELINE RECOMMENDATION COMPARISON")
print("=" * 65)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLOADING DATA...")

users = pd.read_excel(DATA_PATH, sheet_name="Users")
courses = pd.read_excel(DATA_PATH, sheet_name="Courses")
transactions = pd.read_excel(DATA_PATH, sheet_name="Transactions")

print("Users:", users.shape)
print("Courses:", courses.shape)
print("Transactions:", transactions.shape)


# ============================================================
# 3. NORMALIZE IDS AND DATA TYPES
# ============================================================

def normalize_id(value):
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


users["UserID"] = users["UserID"].apply(normalize_id)
courses["CourseID"] = courses["CourseID"].apply(normalize_id)

transactions["UserID"] = transactions["UserID"].apply(normalize_id)
transactions["CourseID"] = transactions["CourseID"].apply(normalize_id)

transactions["TransactionDate"] = pd.to_datetime(
    transactions["TransactionDate"],
    errors="coerce"
)

transactions["Amount"] = pd.to_numeric(
    transactions["Amount"], errors="coerce"
)

courses["CourseRating"] = pd.to_numeric(
    courses["CourseRating"], errors="coerce"
)

transactions = transactions.dropna(
    subset=["TransactionDate", "UserID", "CourseID"]
)

courses["CourseCategory"] = (
    courses["CourseCategory"].fillna("Unknown")
)

courses["CourseLevel"] = (
    courses["CourseLevel"].fillna("Unknown")
)

print("\nDATA PREPARATION COMPLETED!")


# ============================================================
# 4. TEMPORAL TRAIN-TEST SPLIT
# ============================================================

print("\nCREATING TRAIN-TEST SPLIT...")

cutoff_date = transactions["TransactionDate"].quantile(0.80)

train_transactions = transactions[
    transactions["TransactionDate"] <= cutoff_date
].copy()

test_transactions = transactions[
    transactions["TransactionDate"] > cutoff_date
].copy()

print("Training cutoff:", cutoff_date)
print("Training transactions:", len(train_transactions))
print("Testing transactions:", len(test_transactions))


# ============================================================
# 5. BUILD STUDENT PROFILES FROM TRAINING DATA ONLY
# ============================================================

print("\nBUILDING TRAINING STUDENT PROFILES...")

train_data = train_transactions.merge(
    courses,
    on="CourseID",
    how="inner"
)


def get_mode(series):
    modes = series.mode()

    if len(modes) > 0:
        return modes.iloc[0]

    return "Unknown"


student_profiles = train_data.groupby("UserID").agg(
    TotalCourses=("CourseID", "nunique"),
    TotalSpending=("Amount", "sum"),
    AverageSpending=("Amount", "mean"),
    PreferredCategory=("CourseCategory", get_mode),
    PreferredLevel=("CourseLevel", get_mode),
    AverageCourseRating=("CourseRating", "mean"),
    CategoryDiversity=("CourseCategory", "nunique"),
    AdvancedCourses=(
        "CourseLevel",
        lambda x: (x == "Advanced").sum()
    ),
    BeginnerCourses=(
        "CourseLevel",
        lambda x: (x == "Beginner").sum()
    )
).reset_index()

student_profiles = student_profiles.merge(
    users[["UserID", "Age", "Gender"]],
    on="UserID",
    how="left"
)

student_profiles["BeginnerCourseRatio"] = (
    student_profiles["BeginnerCourses"]
    / student_profiles["TotalCourses"]
)

student_profiles["AdvancedCourseRatio"] = (
    student_profiles["AdvancedCourses"]
    / student_profiles["TotalCourses"]
)

print("Student profiles:", student_profiles.shape)


# ============================================================
# 6. PREPARE FEATURES AND TRAIN K-MEANS
# ============================================================

numeric_features = [
    "Age",
    "TotalCourses",
    "AverageSpending",
    "AverageCourseRating",
    "CategoryDiversity",
    "BeginnerCourseRatio",
    "AdvancedCourseRatio"
]

categorical_features = [
    "Gender",
    "PreferredCategory",
    "PreferredLevel"
]

for column in numeric_features:
    student_profiles[column] = pd.to_numeric(
        student_profiles[column],
        errors="coerce"
    )

    student_profiles[column] = (
        student_profiles[column].fillna(
            student_profiles[column].median()
        )
    )

for column in categorical_features:
    student_profiles[column] = (
        student_profiles[column]
        .fillna("Unknown")
        .astype(str)
    )

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)

X_train = preprocessor.fit_transform(student_profiles)

kmeans = KMeans(
    n_clusters=NUMBER_OF_CLUSTERS,
    random_state=42,
    n_init=10
)

student_profiles["Cluster"] = kmeans.fit_predict(X_train)

print("\nK-MEANS MODEL TRAINED!")
print(student_profiles["Cluster"].value_counts().sort_index())


# ============================================================
# 7. CALCULATE GLOBAL COURSE POPULARITY
# ============================================================

print("\nCALCULATING GLOBAL COURSE POPULARITY...")

global_popularity = (
    train_transactions
    .groupby("CourseID")["UserID"]
    .nunique()
    .reset_index(name="GlobalPopularity")
)


# ============================================================
# 8. CALCULATE CLUSTER COURSE POPULARITY
# ============================================================

print("CALCULATING CLUSTER COURSE POPULARITY...")

cluster_sizes = (
    student_profiles.groupby("Cluster")["UserID"]
    .nunique()
)

cluster_transactions = train_transactions.merge(
    student_profiles[["UserID", "Cluster"]],
    on="UserID",
    how="inner"
)

cluster_popularity = (
    cluster_transactions
    .groupby(["Cluster", "CourseID"])["UserID"]
    .nunique()
    .reset_index(name="StudentsTakingCourse")
)

cluster_popularity["ClusterPopularity"] = (
    cluster_popularity.apply(
        lambda row: (
            row["StudentsTakingCourse"]
            / cluster_sizes.loc[row["Cluster"]]
        ),
        axis=1
    )
)


# ============================================================
# 9. PREPARE COURSE RATINGS
# ============================================================

rating_max = courses["CourseRating"].max()

if pd.isna(rating_max) or rating_max <= 0:
    rating_max = 5

courses["RatingScore"] = (
    courses["CourseRating"].fillna(0) / rating_max
)


# ============================================================
# 10. RECOMMENDATION METHODS
# ============================================================

def get_recommendations(user_id, method):

    student = student_profiles[
        student_profiles["UserID"] == user_id
    ]

    if student.empty:
        return []

    student = student.iloc[0]

    # Exclude courses taken during training
    taken_courses = set(
        train_transactions.loc[
            train_transactions["UserID"] == user_id,
            "CourseID"
        ]
    )

    candidates = courses[
        ~courses["CourseID"].isin(taken_courses)
    ].copy()

    if candidates.empty:
        return []

    # --------------------------------------------------------
    # METHOD 1: GLOBAL POPULARITY
    # --------------------------------------------------------

    if method == "Global Popularity":

        candidates = candidates.merge(
            global_popularity,
            on="CourseID",
            how="left"
        )

        candidates["GlobalPopularity"] = (
            candidates["GlobalPopularity"].fillna(0)
        )

        candidates = candidates.sort_values(
            by=["GlobalPopularity", "CourseRating"],
            ascending=[False, False],
            na_position="last"
        )

    # --------------------------------------------------------
    # METHOD 2: CLUSTER POPULARITY
    # --------------------------------------------------------

    elif method == "Cluster Popularity":

        cluster = student["Cluster"]

        popularity = cluster_popularity[
            cluster_popularity["Cluster"] == cluster
        ][["CourseID", "ClusterPopularity"]]

        candidates = candidates.merge(
            popularity,
            on="CourseID",
            how="left"
        )

        candidates["ClusterPopularity"] = (
            candidates["ClusterPopularity"].fillna(0)
        )

        candidates = candidates.sort_values(
            by=["ClusterPopularity", "CourseRating"],
            ascending=[False, False],
            na_position="last"
        )

    # --------------------------------------------------------
    # METHOD 3: PERSONALIZED HYBRID
    # --------------------------------------------------------

    elif method == "Hybrid Recommendation":

        cluster = student["Cluster"]

        popularity = cluster_popularity[
            cluster_popularity["Cluster"] == cluster
        ][["CourseID", "ClusterPopularity"]]

        candidates = candidates.merge(
            popularity,
            on="CourseID",
            how="left"
        )

        candidates["ClusterPopularity"] = (
            candidates["ClusterPopularity"].fillna(0)
        )

        candidates["CategoryMatch"] = (
            candidates["CourseCategory"]
            == student["PreferredCategory"]
        ).astype(int)

        candidates["LevelMatch"] = (
            candidates["CourseLevel"]
            == student["PreferredLevel"]
        ).astype(int)

        candidates["RecommendationScore"] = (
            0.50 * candidates["ClusterPopularity"]
            + 0.20 * candidates["CategoryMatch"]
            + 0.20 * candidates["LevelMatch"]
            + 0.10 * candidates["RatingScore"]
        )

        candidates = candidates.sort_values(
            by=["RecommendationScore", "CourseRating"],
            ascending=[False, False],
            na_position="last"
        )

    return candidates.head(TOP_K)["CourseID"].tolist()


# ============================================================
# 11. EVALUATE ALL THREE METHODS
# ============================================================

print("\nEVALUATING ALL THREE METHODS...")

methods = [
    "Global Popularity",
    "Cluster Popularity",
    "Hybrid Recommendation"
]

test_users = set(test_transactions["UserID"])
training_users = set(student_profiles["UserID"])

eligible_users = sorted(
    test_users.intersection(training_users)
)

print("Eligible students:", len(eligible_users))

results = []
coverage_sets = {
    method: set() for method in methods
}

for user_id in eligible_users:

    actual_courses = set(
        test_transactions.loc[
            test_transactions["UserID"] == user_id,
            "CourseID"
        ]
    )

    for method in methods:

        recommended = get_recommendations(
            user_id,
            method
        )

        recommended_set = set(recommended)

        hits = len(
            recommended_set.intersection(actual_courses)
        )

        precision = hits / TOP_K

        recall = (
            hits / len(actual_courses)
            if len(actual_courses) > 0
            else 0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )

        coverage_sets[method].update(recommended_set)

        results.append({
            "UserID": user_id,
            "Method": method,
            "ActualTestCourses": len(actual_courses),
            "RelevantRecommendations": hits,
            "Precision@5": precision,
            "Recall@5": recall,
            "F1@5": f1
        })


# ============================================================
# 12. CALCULATE COMPARISON RESULTS
# ============================================================

results_df = pd.DataFrame(results)

summary_rows = []

for method in methods:

    method_results = results_df[
        results_df["Method"] == method
    ]

    coverage = (
        len(coverage_sets[method]) / len(courses)
    )

    summary_rows.append({
        "Method": method,
        "Precision@5": method_results["Precision@5"].mean(),
        "Recall@5": method_results["Recall@5"].mean(),
        "F1@5": method_results["F1@5"].mean(),
        "CatalogCoverage": coverage
    })

summary_df = pd.DataFrame(summary_rows)

print("\n" + "=" * 75)
print("BASELINE COMPARISON RESULTS")
print("=" * 75)

print(
    summary_df.round(4).to_string(index=False)
)


# ============================================================
# 13. IDENTIFY THE BEST METHOD
# ============================================================

best_method = summary_df.loc[
    summary_df["F1@5"].idxmax(),
    "Method"
]

print("\nBEST METHOD BASED ON F1@5:")
print(best_method)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "baseline_comparison_summary.csv"
)

details_path = os.path.join(
    OUTPUT_DIR,
    "baseline_comparison_by_student.csv"
)

summary_df.to_csv(summary_path, index=False)
results_df.to_csv(details_path, index=False)

print("\nFILES SAVED SUCCESSFULLY!")

print("Summary:", summary_path)
print("Student details:", details_path)

print("\nBASELINE COMPARISON COMPLETED!")
