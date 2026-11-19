
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "raw", "EduPro Online Platform.xlsx"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "reports", "recommendation_evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_K = 5
NUMBER_OF_CLUSTERS = 4

print("\n" + "=" * 65)
print("EDUPRO RECOMMENDATION SYSTEM - EVALUATION")
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
# 3. NORMALIZE IDS
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

courses["CoursePrice"] = pd.to_numeric(
    courses["CoursePrice"], errors="coerce"
)

transactions = transactions.dropna(
    subset=["TransactionDate", "UserID", "CourseID"]
)

print("\nDATA LOADED AND IDS NORMALIZED!")


# ============================================================
# 4. CREATE A TEMPORAL TRAIN-TEST SPLIT
# ============================================================

print("\nCREATING TEMPORAL TRAIN-TEST SPLIT...")

cutoff_date = transactions["TransactionDate"].quantile(0.80)

train_transactions = transactions[
    transactions["TransactionDate"] <= cutoff_date
].copy()

test_transactions = transactions[
    transactions["TransactionDate"] > cutoff_date
].copy()

print("Training cutoff date:", cutoff_date)
print("Training transactions:", len(train_transactions))
print("Testing transactions:", len(test_transactions))


# ============================================================
# 5. PREPARE TRAINING DATA
# ============================================================

train_data = train_transactions.merge(
    courses,
    on="CourseID",
    how="inner"
)

print("\nTRAINING DATA:", train_data.shape)


# ============================================================
# 6. BUILD STUDENT PROFILES USING TRAINING DATA ONLY
# ============================================================

print("\nBUILDING TRAINING STUDENT PROFILES...")


def get_mode(series):
    modes = series.mode()

    if len(modes) > 0:
        return modes.iloc[0]

    return "Unknown"


student_profiles = train_data.groupby("UserID").agg(
    TotalCourses=("CourseID", "nunique"),
    TotalTransactions=("CourseID", "count"),
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
    ),
    IntermediateCourses=(
        "CourseLevel",
        lambda x: (x == "Intermediate").sum()
    )
).reset_index()

# Add student demographic information
student_profiles = student_profiles.merge(
    users[["UserID", "Age", "Gender"]],
    on="UserID",
    how="left"
)

# Calculate course-level ratios
student_profiles["BeginnerCourseRatio"] = (
    student_profiles["BeginnerCourses"]
    / student_profiles["TotalCourses"]
)

student_profiles["AdvancedCourseRatio"] = (
    student_profiles["AdvancedCourses"]
    / student_profiles["TotalCourses"]
)

student_profiles = student_profiles.replace(
    [np.inf, -np.inf], np.nan
)

print("Training student profiles:", student_profiles.shape)


# ============================================================
# 7. PREPARE CLUSTERING FEATURES
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

# Fill missing values using training data
for column in numeric_features:
    student_profiles[column] = pd.to_numeric(
        student_profiles[column], errors="coerce"
    )

    median_value = student_profiles[column].median()
    student_profiles[column] = (
        student_profiles[column].fillna(median_value)
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

print("Clustering feature shape:", X_train.shape)


# ============================================================
# 8. TRAIN K-MEANS ON TRAINING PROFILES ONLY
# ============================================================

print("\nTRAINING K-MEANS MODEL...")

kmeans = KMeans(
    n_clusters=NUMBER_OF_CLUSTERS,
    random_state=42,
    n_init=10
)

student_profiles["Cluster"] = kmeans.fit_predict(X_train)

print("\nTRAINING CLUSTER DISTRIBUTION:")
print(student_profiles["Cluster"].value_counts().sort_index())


# ============================================================
# 9. CALCULATE COURSE POPULARITY FROM TRAINING DATA ONLY
# ============================================================

print("\nCALCULATING TRAINING COURSE POPULARITY...")

cluster_sizes = (
    student_profiles.groupby("Cluster")["UserID"]
    .nunique()
)

cluster_transactions = train_transactions.merge(
    student_profiles[["UserID", "Cluster"]],
    on="UserID",
    how="inner"
)

course_popularity = (
    cluster_transactions
    .groupby(["Cluster", "CourseID"])["UserID"]
    .nunique()
    .reset_index(name="StudentsTakingCourse")
)

course_popularity["ClusterPopularity"] = (
    course_popularity.apply(
        lambda row: (
            row["StudentsTakingCourse"]
            / cluster_sizes.loc[row["Cluster"]]
        ),
        axis=1
    )
)

print("Course popularity calculated!")


# ============================================================
# 10. PREPARE COURSE CATALOGUE
# ============================================================

courses["CourseCategory"] = (
    courses["CourseCategory"].fillna("Unknown")
)

courses["CourseLevel"] = (
    courses["CourseLevel"].fillna("Unknown")
)

rating_max = courses["CourseRating"].max()

if pd.isna(rating_max) or rating_max <= 0:
    rating_max = 5

courses["RatingScore"] = (
    courses["CourseRating"].fillna(0) / rating_max
)


# ============================================================
# 11. RECOMMEND COURSES FOR A STUDENT
# ============================================================

def recommend_courses(user_id, top_k=5):

    student = student_profiles[
        student_profiles["UserID"] == user_id
    ]

    if student.empty:
        return []

    student = student.iloc[0]

    cluster = student["Cluster"]

    preferred_category = student["PreferredCategory"]
    preferred_level = student["PreferredLevel"]

    # Courses already taken in training data
    taken_courses = set(
        train_transactions.loc[
            train_transactions["UserID"] == user_id,
            "CourseID"
        ]
    )

    # Remove previously taken courses
    candidates = courses[
        ~courses["CourseID"].isin(taken_courses)
    ].copy()

    if candidates.empty:
        return []

    # Add cluster popularity
    popularity_for_cluster = course_popularity[
        course_popularity["Cluster"] == cluster
    ][["CourseID", "ClusterPopularity"]]

    candidates = candidates.merge(
        popularity_for_cluster,
        on="CourseID",
        how="left"
    )

    candidates["ClusterPopularity"] = (
        candidates["ClusterPopularity"].fillna(0)
    )

    # Preference matching
    candidates["CategoryMatch"] = (
        candidates["CourseCategory"] == preferred_category
    ).astype(int)

    candidates["LevelMatch"] = (
        candidates["CourseLevel"] == preferred_level
    ).astype(int)

    # Weighted recommendation score
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

    return candidates.head(top_k)["CourseID"].tolist()


# ============================================================
# 12. EVALUATE USING LATER TRANSACTIONS
# ============================================================

print("\nEVALUATING RECOMMENDATIONS...")

test_users = set(test_transactions["UserID"])
training_users = set(student_profiles["UserID"])

eligible_users = sorted(test_users.intersection(training_users))

print("Students with training and testing history:", len(eligible_users))

results = []

all_recommended_courses = set()

for user_id in eligible_users:

    # Actual later courses selected by this student
    actual_courses = set(
        test_transactions.loc[
            test_transactions["UserID"] == user_id,
            "CourseID"
        ]
    )

    # Generate recommendations using training data only
    recommended_courses = recommend_courses(
        user_id,
        TOP_K
    )

    recommended_set = set(recommended_courses)

    # Count relevant recommendations
    hits = len(recommended_set.intersection(actual_courses))

    precision = hits / TOP_K

    if len(actual_courses) > 0:
        recall = hits / len(actual_courses)
    else:
        recall = 0

    if precision + recall > 0:
        f1_score = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1_score = 0

    all_recommended_courses.update(recommended_set)

    results.append({
        "UserID": user_id,
        "ActualTestCourses": len(actual_courses),
        "RecommendedCourses": len(recommended_courses),
        "RelevantRecommendations": hits,
        "Precision@5": precision,
        "Recall@5": recall,
        "F1@5": f1_score
    })


# ============================================================
# 13. CALCULATE OVERALL METRICS
# ============================================================

results_df = pd.DataFrame(results)

if results_df.empty:

    print("\nNo eligible students were available for evaluation.")
    print("Check the training and testing date split.")

else:

    precision_at_5 = results_df["Precision@5"].mean()
    recall_at_5 = results_df["Recall@5"].mean()
    f1_at_5 = results_df["F1@5"].mean()

    catalog_coverage = (
        len(all_recommended_courses) / len(courses)
    )

    print("\n" + "=" * 65)
    print("RECOMMENDATION EVALUATION RESULTS")
    print("=" * 65)

    print("Eligible students:", len(results_df))
    print("Precision@5:", round(precision_at_5, 4))
    print("Recall@5:", round(recall_at_5, 4))
    print("F1@5:", round(f1_at_5, 4))
    print("Catalog Coverage:", round(catalog_coverage, 4))

    print("\nINTERPRETATION:")
    print(
        "Precision@5 measures the proportion of the top five "
        "recommendations that appeared in later transactions."
    )
    print(
        "Recall@5 measures the proportion of later courses "
        "recovered by the top five recommendations."
    )
    print(
        "Catalog Coverage measures the fraction of the catalogue "
        "recommended at least once."
    )

    # Save detailed student results
    results_path = os.path.join(
        OUTPUT_DIR,
        "recommendation_evaluation_by_student.csv"
    )

    results_df.to_csv(results_path, index=False)

    # Save overall metrics
    summary = pd.DataFrame({
        "Metric": [
            "Eligible Students",
            "Precision@5",
            "Recall@5",
            "F1@5",
            "Catalog Coverage"
        ],
        "Value": [
            len(results_df),
            precision_at_5,
            recall_at_5,
            f1_at_5,
            catalog_coverage
        ]
    })

    summary_path = os.path.join(
        OUTPUT_DIR,
        "recommendation_evaluation_summary.csv"
    )

    summary.to_csv(summary_path, index=False)

    print("\nFILES SAVED SUCCESSFULLY!")
    print("Student results:", results_path)
    print("Overall summary:", summary_path)

print("\nEVALUATION COMPLETED!")
