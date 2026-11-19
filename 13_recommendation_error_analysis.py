
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans


# ============================================================
# 1. SETTINGS AND PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "raw", "EduPro Online Platform.xlsx"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "reports", "recommendation_error_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_K = 5
NUMBER_OF_CLUSTERS = 4

print("\n" + "=" * 70)
print("EDUPRO - RECOMMENDATION ERROR ANALYSIS")
print("=" * 70)


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
# 3. CLEAN AND NORMALIZE DATA
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
    transactions["Amount"],
    errors="coerce"
)

courses["CourseRating"] = pd.to_numeric(
    courses["CourseRating"],
    errors="coerce"
)

courses["CourseCategory"] = (
    courses["CourseCategory"].fillna("Unknown")
)

courses["CourseLevel"] = (
    courses["CourseLevel"].fillna("Unknown")
)

transactions = transactions.dropna(
    subset=["TransactionDate", "UserID", "CourseID"]
)

print("\nDATA PREPARATION COMPLETED!")


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

print("\nCREATING TEMPORAL TRAIN-TEST SPLIT...")

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
# 5. BUILD TRAINING STUDENT PROFILES
# ============================================================

print("\nBUILDING STUDENT PROFILES...")

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

print("\nK-MEANS TRAINING COMPLETED!")

print(
    student_profiles["Cluster"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 7. CALCULATE CLUSTER COURSE POPULARITY
# ============================================================

print("\nCALCULATING COURSE POPULARITY...")

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
# 8. PREPARE COURSE RATINGS
# ============================================================

rating_max = courses["CourseRating"].max()

if pd.isna(rating_max) or rating_max <= 0:
    rating_max = 5

courses["RatingScore"] = (
    courses["CourseRating"].fillna(0) / rating_max
)


# ============================================================
# 9. GENERATE HYBRID RECOMMENDATIONS
# ============================================================

def recommend_courses(user_id):

    student = student_profiles[
        student_profiles["UserID"] == user_id
    ]

    if student.empty:
        return pd.DataFrame()

    student = student.iloc[0]

    cluster = student["Cluster"]

    # Courses already taken during training
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
        return pd.DataFrame()

    # Get popularity for this student's cluster
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

    # Match student preferences
    candidates["CategoryMatch"] = (
        candidates["CourseCategory"]
        == student["PreferredCategory"]
    ).astype(int)

    candidates["LevelMatch"] = (
        candidates["CourseLevel"]
        == student["PreferredLevel"]
    ).astype(int)

    # Hybrid recommendation score
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

    return candidates.head(TOP_K).copy()


# ============================================================
# 10. EVALUATE EACH ELIGIBLE STUDENT
# ============================================================

print("\nANALYZING RECOMMENDATIONS...")

test_users = set(test_transactions["UserID"])
training_users = set(student_profiles["UserID"])

eligible_users = sorted(
    test_users.intersection(training_users)
)

print("Eligible students:", len(eligible_users))

student_results = []
course_results = []
category_results = []

all_recommended_courses = set()

for user_id in eligible_users:

    student = student_profiles[
        student_profiles["UserID"] == user_id
    ].iloc[0]

    cluster = int(student["Cluster"])

    # Actual courses selected in the testing period
    actual_courses = set(
        test_transactions.loc[
            test_transactions["UserID"] == user_id,
            "CourseID"
        ]
    )

    # Generate top-five recommendations
    recommendations = recommend_courses(user_id)

    recommended_courses = set(
        recommendations["CourseID"].tolist()
    )

    all_recommended_courses.update(recommended_courses)

    # Correct recommendations
    hits = recommended_courses.intersection(actual_courses)

    # Recommended but not selected later
    false_positives = recommended_courses - actual_courses

    # Selected later but not recommended
    false_negatives = actual_courses - recommended_courses

    precision = len(hits) / TOP_K

    recall = (
        len(hits) / len(actual_courses)
        if len(actual_courses) > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    student_results.append({
        "UserID": user_id,
        "Cluster": cluster,
        "PreferredCategory": student["PreferredCategory"],
        "PreferredLevel": student["PreferredLevel"],
        "ActualTestCourses": len(actual_courses),
        "RecommendedCourses": len(recommended_courses),
        "RelevantRecommendations": len(hits),
        "FalsePositives": len(false_positives),
        "MissedCourses": len(false_negatives),
        "Precision@5": precision,
        "Recall@5": recall,
        "F1@5": f1
    })

    # Record each recommended course
    for _, row in recommendations.iterrows():

        course_id = row["CourseID"]

        is_relevant = course_id in actual_courses

        course_results.append({
            "UserID": user_id,
            "Cluster": cluster,
            "CourseID": course_id,
            "CourseName": row["CourseName"],
            "CourseCategory": row["CourseCategory"],
            "CourseLevel": row["CourseLevel"],
            "RecommendationScore": row["RecommendationScore"],
            "WasSelectedLater": int(is_relevant)
        })

    # Record later course selections for category analysis
    for course_id in actual_courses:

        course_info = courses[
            courses["CourseID"] == course_id
        ]

        if course_info.empty:
            continue

        course_info = course_info.iloc[0]

        category_results.append({
            "UserID": user_id,
            "Cluster": cluster,
            "CourseID": course_id,
            "CourseCategory": course_info["CourseCategory"],
            "WasRecommended": int(
                course_id in recommended_courses
            )
        })


# ============================================================
# 11. CREATE ANALYSIS DATAFRAMES
# ============================================================

student_df = pd.DataFrame(student_results)
course_df = pd.DataFrame(course_results)
category_df = pd.DataFrame(category_results)

print("\nERROR ANALYSIS DATA CREATED!")


# ============================================================
# 12. CLUSTER-LEVEL PERFORMANCE
# ============================================================

cluster_summary = (
    student_df.groupby("Cluster")
    .agg(
        Students=("UserID", "count"),
        AveragePrecision=("Precision@5", "mean"),
        AverageRecall=("Recall@5", "mean"),
        AverageF1=("F1@5", "mean"),
        TotalFalsePositives=("FalsePositives", "sum"),
        TotalMissedCourses=("MissedCourses", "sum")
    )
    .reset_index()
)

print("\n" + "=" * 70)
print("CLUSTER-LEVEL PERFORMANCE")
print("=" * 70)

print(cluster_summary.round(4).to_string(index=False))


# ============================================================
# 13. COURSE-LEVEL ERROR ANALYSIS
# ============================================================

course_summary = (
    course_df.groupby(
        ["CourseID", "CourseName", "CourseCategory", "CourseLevel"]
    )
    .agg(
        TimesRecommended=("WasSelectedLater", "count"),
        TimesSelectedLater=("WasSelectedLater", "sum")
    )
    .reset_index()
)

course_summary["RecommendationHitRate"] = (
    course_summary["TimesSelectedLater"]
    / course_summary["TimesRecommended"]
)

course_summary = course_summary.sort_values(
    by=["TimesRecommended", "RecommendationHitRate"],
    ascending=[False, True]
)

print("\n" + "=" * 70)
print("MOST FREQUENTLY RECOMMENDED COURSES")
print("=" * 70)

print(
    course_summary.head(10).round(4).to_string(index=False)
)


# ============================================================
# 14. CATEGORY-LEVEL ANALYSIS
# ============================================================

category_summary = (
    category_df.groupby("CourseCategory")
    .agg(
        ActualLaterSelections=("CourseID", "count"),
        RecommendedLaterSelections=("WasRecommended", "sum")
    )
    .reset_index()
)

category_summary["CategoryRecall"] = (
    category_summary["RecommendedLaterSelections"]
    / category_summary["ActualLaterSelections"]
)

category_summary = category_summary.sort_values(
    by="CategoryRecall",
    ascending=False
)

print("\n" + "=" * 70)
print("CATEGORY-LEVEL RECALL")
print("=" * 70)

print(category_summary.round(4).to_string(index=False))


# ============================================================
# 15. OVERALL SUMMARY
# ============================================================

overall_precision = student_df["Precision@5"].mean()
overall_recall = student_df["Recall@5"].mean()
overall_f1 = student_df["F1@5"].mean()

total_false_positives = student_df["FalsePositives"].sum()
total_missed_courses = student_df["MissedCourses"].sum()

catalog_coverage = (
    len(all_recommended_courses) / len(courses)
)

print("\n" + "=" * 70)
print("OVERALL ERROR ANALYSIS")
print("=" * 70)

print("Students evaluated:", len(student_df))
print("Precision@5:", round(overall_precision, 4))
print("Recall@5:", round(overall_recall, 4))
print("F1@5:", round(overall_f1, 4))
print("Total false positives:", total_false_positives)
print("Total missed courses:", total_missed_courses)
print("Catalog coverage:", round(catalog_coverage, 4))


# ============================================================
# 16. SAVE ALL REPORTS
# ============================================================

student_path = os.path.join(
    OUTPUT_DIR,
    "student_error_analysis.csv"
)

course_path = os.path.join(
    OUTPUT_DIR,
    "course_error_analysis.csv"
)

cluster_path = os.path.join(
    OUTPUT_DIR,
    "cluster_error_analysis.csv"
)

category_path = os.path.join(
    OUTPUT_DIR,
    "category_error_analysis.csv"
)

student_df.to_csv(student_path, index=False)
course_summary.to_csv(course_path, index=False)
cluster_summary.to_csv(cluster_path, index=False)
category_summary.to_csv(category_path, index=False)

print("\n" + "=" * 70)
print("ALL ERROR ANALYSIS REPORTS SAVED!")
print("=" * 70)

print("Student analysis:", student_path)
print("Course analysis:", course_path)
print("Cluster analysis:", cluster_path)
print("Category analysis:", category_path)

print("\nRECOMMENDATION ERROR ANALYSIS COMPLETED!")
