
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans


# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "raw" / "EduPro Online Platform.xlsx"

OUTPUT_DIR = BASE_DIR / "reports" / "recommendation_optimization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5
NUMBER_OF_CLUSTERS = 4
RANDOM_STATE = 42


# Each weight combination adds up to 1.00.
WEIGHT_CONFIGURATIONS = {
    "Original": (0.50, 0.20, 0.20, 0.10),
    "Popularity_Heavy": (0.60, 0.15, 0.15, 0.10),
    "Balanced_Preferences": (0.40, 0.30, 0.20, 0.10),
    "Level_Focused": (0.40, 0.20, 0.30, 0.10),
    "Category_Focused": (0.40, 0.30, 0.20, 0.10),
    "Rating_Focused": (0.30, 0.25, 0.25, 0.20),
    "Equal_Components": (0.25, 0.25, 0.25, 0.25),
    "Popularity_Preference": (0.50, 0.25, 0.15, 0.10),
    "Popularity_Level": (0.50, 0.15, 0.25, 0.10),
    "Lower_Popularity": (0.30, 0.30, 0.30, 0.10),
}


# ============================================================
# 2. LOAD AND CLEAN DATA
# ============================================================

print("\n" + "=" * 70)
print("EDUPRO - RECOMMENDATION MODEL OPTIMIZATION")
print("=" * 70)

print("\nLOADING DATA...")

users = pd.read_excel(DATA_PATH, sheet_name="Users")
courses = pd.read_excel(DATA_PATH, sheet_name="Courses")
transactions = pd.read_excel(DATA_PATH, sheet_name="Transactions")


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

transactions = transactions.dropna(
    subset=["TransactionDate", "UserID", "CourseID"]
)

courses["CourseCategory"] = courses["CourseCategory"].fillna("Unknown")
courses["CourseLevel"] = courses["CourseLevel"].fillna("Unknown")

rating_max = courses["CourseRating"].max()

if pd.isna(rating_max) or rating_max <= 0:
    rating_max = 5

courses["RatingScore"] = (
    courses["CourseRating"].fillna(0) / rating_max
)

print("Users:", users.shape)
print("Courses:", courses.shape)
print("Transactions:", transactions.shape)


# ============================================================
# 3. CREATE OUTER TRAINING AND FINAL TEST PERIODS
# ============================================================

print("\nCREATING FINAL TRAIN-TEST SPLIT...")

final_cutoff = transactions["TransactionDate"].quantile(0.80)

outer_train = transactions[
    transactions["TransactionDate"] <= final_cutoff
].copy()

final_test = transactions[
    transactions["TransactionDate"] > final_cutoff
].copy()

print("Final training cutoff:", final_cutoff)
print("Final training transactions:", len(outer_train))
print("Final testing transactions:", len(final_test))


# ============================================================
# 4. CREATE AN INTERNAL TRAINING AND VALIDATION SPLIT
# ============================================================

print("\nCREATING INTERNAL VALIDATION SPLIT...")

validation_cutoff = outer_train["TransactionDate"].quantile(0.80)

internal_train = outer_train[
    outer_train["TransactionDate"] <= validation_cutoff
].copy()

validation = outer_train[
    outer_train["TransactionDate"] > validation_cutoff
].copy()

print("Validation cutoff:", validation_cutoff)
print("Internal training transactions:", len(internal_train))
print("Validation transactions:", len(validation))

print("\nThe final test period will not be used for weight selection.")


# ============================================================
# 5. BUILD THE RECOMMENDATION MODEL
# ============================================================

def build_recommendation_model(training_transactions):

    training_data = training_transactions.merge(
        courses,
        on="CourseID",
        how="inner"
    )

    # Build student profiles using training interactions only.
    student_profiles = training_data.groupby("UserID").agg(
        TotalCourses=("CourseID", "nunique"),
        TotalTransactions=("CourseID", "count"),
        TotalSpending=("Amount", "sum"),
        AverageSpending=("Amount", "mean"),
        PreferredCategory=("CourseCategory", lambda x: x.mode().iloc[0]),
        PreferredLevel=("CourseLevel", lambda x: x.mode().iloc[0]),
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

    kmeans = KMeans(
        n_clusters=NUMBER_OF_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10
    )

    student_profiles["Cluster"] = kmeans.fit_predict(X_train)

    # Calculate popularity using training interactions only.
    cluster_sizes = (
        student_profiles.groupby("Cluster")["UserID"]
        .nunique()
    )

    cluster_transactions = training_transactions.merge(
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

    return {
        "profiles": student_profiles,
        "popularity": course_popularity,
        "training_transactions": training_transactions
    }


# ============================================================
# 6. GENERATE RECOMMENDATIONS USING SELECTED WEIGHTS
# ============================================================

def recommend_courses(
    user_id,
    model,
    weights,
    top_k=TOP_K
):

    popularity_weight, category_weight, level_weight, rating_weight = weights

    profiles = model["profiles"]
    popularity = model["popularity"]
    model_training = model["training_transactions"]

    student = profiles[
        profiles["UserID"] == user_id
    ]

    if student.empty:
        return []

    student = student.iloc[0]

    cluster = student["Cluster"]

    preferred_category = student["PreferredCategory"]
    preferred_level = student["PreferredLevel"]

    taken_courses = set(
        model_training.loc[
            model_training["UserID"] == user_id,
            "CourseID"
        ]
    )

    candidates = courses[
        ~courses["CourseID"].isin(taken_courses)
    ].copy()

    if candidates.empty:
        return []

    cluster_popularity = popularity[
        popularity["Cluster"] == cluster
    ][["CourseID", "ClusterPopularity"]]

    candidates = candidates.merge(
        cluster_popularity,
        on="CourseID",
        how="left"
    )

    candidates["ClusterPopularity"] = (
        candidates["ClusterPopularity"].fillna(0)
    )

    candidates["CategoryMatch"] = (
        candidates["CourseCategory"] == preferred_category
    ).astype(int)

    candidates["LevelMatch"] = (
        candidates["CourseLevel"] == preferred_level
    ).astype(int)

    candidates["RecommendationScore"] = (
        popularity_weight * candidates["ClusterPopularity"]
        + category_weight * candidates["CategoryMatch"]
        + level_weight * candidates["LevelMatch"]
        + rating_weight * candidates["RatingScore"]
    )

    candidates = candidates.sort_values(
        by=["RecommendationScore", "CourseRating"],
        ascending=[False, False],
        na_position="last"
    )

    return candidates.head(top_k)["CourseID"].tolist()


# ============================================================
# 7. EVALUATE A WEIGHT CONFIGURATION
# ============================================================

def evaluate_model(model, evaluation_transactions, weights):

    eligible_users = sorted(
        set(evaluation_transactions["UserID"])
        .intersection(set(model["profiles"]["UserID"]))
    )

    results = []
    recommended_courses_all = set()

    for user_id in eligible_users:

        actual_courses = set(
            evaluation_transactions.loc[
                evaluation_transactions["UserID"] == user_id,
                "CourseID"
            ]
        )

        recommended = recommend_courses(
            user_id,
            model,
            weights,
            TOP_K
        )

        recommended_set = set(recommended)

        hits = len(recommended_set.intersection(actual_courses))

        precision = hits / TOP_K

        recall = (
            hits / len(actual_courses)
            if len(actual_courses) > 0
            else 0
        )

        if precision + recall > 0:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )
        else:
            f1 = 0

        recommended_courses_all.update(recommended_set)

        results.append({
            "UserID": user_id,
            "Precision@5": precision,
            "Recall@5": recall,
            "F1@5": f1
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:

        return {
            "EligibleStudents": 0,
            "Precision@5": 0,
            "Recall@5": 0,
            "F1@5": 0,
            "CatalogCoverage": 0
        }

    return {
        "EligibleStudents": len(results_df),
        "Precision@5": results_df["Precision@5"].mean(),
        "Recall@5": results_df["Recall@5"].mean(),
        "F1@5": results_df["F1@5"].mean(),
        "CatalogCoverage": (
            len(recommended_courses_all) / len(courses)
        )
    }


# ============================================================
# 8. TRAIN ON INTERNAL TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("TRAINING MODEL FOR WEIGHT OPTIMIZATION")
print("=" * 70)

validation_model = build_recommendation_model(internal_train)

print("\nInternal training cluster distribution:")

print(
    validation_model["profiles"]["Cluster"]
    .value_counts()
    .sort_index()
)

validation_eligible = (
    set(validation["UserID"])
    .intersection(set(validation_model["profiles"]["UserID"]))
)

print("\nValidation students with earlier training history:", len(validation_eligible))


# ============================================================
# 9. TEST DIFFERENT WEIGHT CONFIGURATIONS
# ============================================================

print("\n" + "=" * 70)
print("TESTING RECOMMENDATION WEIGHTS")
print("=" * 70)

weight_results = []

for name, weights in WEIGHT_CONFIGURATIONS.items():

    print("\nTesting:", name)
    print("Weights:", weights)

    metrics = evaluate_model(
        validation_model,
        validation,
        weights
    )

    result = {
        "Configuration": name,
        "PopularityWeight": weights[0],
        "CategoryWeight": weights[1],
        "LevelWeight": weights[2],
        "RatingWeight": weights[3],
        **metrics
    }

    weight_results.append(result)

    print("Validation Precision@5:", round(metrics["Precision@5"], 4))
    print("Validation Recall@5:", round(metrics["Recall@5"], 4))
    print("Validation F1@5:", round(metrics["F1@5"], 4))


weight_results_df = pd.DataFrame(weight_results)

weight_results_df = weight_results_df.sort_values(
    by=["F1@5", "Precision@5", "CatalogCoverage"],
    ascending=False
)

weight_results_path = OUTPUT_DIR / "weight_optimization_results.csv"

weight_results_df.to_csv(
    weight_results_path,
    index=False
)


# ============================================================
# 10. SELECT THE BEST VALIDATION CONFIGURATION
# ============================================================

if weight_results_df.empty or weight_results_df["EligibleStudents"].iloc[0] == 0:

    raise ValueError(
        "No eligible validation students found. "
        "Check the date split and transaction data."
    )

best_result = weight_results_df.iloc[0]

best_configuration = best_result["Configuration"]

best_weights = (
    best_result["PopularityWeight"],
    best_result["CategoryWeight"],
    best_result["LevelWeight"],
    best_result["RatingWeight"]
)

print("\n" + "=" * 70)
print("BEST VALIDATION CONFIGURATION")
print("=" * 70)

print("Configuration:", best_configuration)
print("Selected weights:", best_weights)
print("Validation F1@5:", round(best_result["F1@5"], 4))


# ============================================================
# 11. RETRAIN USING ALL OUTER TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("RETRAINING MODEL ON FULL TRAINING PERIOD")
print("=" * 70)

final_model = build_recommendation_model(outer_train)

print("\nFinal training cluster distribution:")

print(
    final_model["profiles"]["Cluster"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 12. FINAL EVALUATION ON UNTOUCHED TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION ON UNTOUCHED TEST DATA")
print("=" * 70)

# Evaluate the original weights as a reference.
original_weights = WEIGHT_CONFIGURATIONS["Original"]

original_test_metrics = evaluate_model(
    final_model,
    final_test,
    original_weights
)

# Evaluate the best weights selected using validation data.
optimized_test_metrics = evaluate_model(
    final_model,
    final_test,
    best_weights
)

final_comparison = pd.DataFrame([
    {
        "Configuration": "Original",
        "PopularityWeight": original_weights[0],
        "CategoryWeight": original_weights[1],
        "LevelWeight": original_weights[2],
        "RatingWeight": original_weights[3],
        **original_test_metrics
    },
    {
        "Configuration": best_configuration,
        "PopularityWeight": best_weights[0],
        "CategoryWeight": best_weights[1],
        "LevelWeight": best_weights[2],
        "RatingWeight": best_weights[3],
        **optimized_test_metrics
    }
])

print("\nFINAL TEST COMPARISON:")

print(final_comparison.to_string(index=False))

final_comparison_path = OUTPUT_DIR / "final_test_comparison.csv"

final_comparison.to_csv(
    final_comparison_path,
    index=False
)


# ============================================================
# 13. SAVE OPTIMIZATION SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Item": [
        "Final Training Cutoff",
        "Validation Cutoff",
        "Internal Training Transactions",
        "Validation Transactions",
        "Final Training Transactions",
        "Final Testing Transactions",
        "Best Configuration",
        "Best Popularity Weight",
        "Best Category Weight",
        "Best Level Weight",
        "Best Rating Weight",
        "Best Validation F1@5",
        "Original Final Test F1@5",
        "Optimized Final Test F1@5"
    ],
    "Value": [
        str(final_cutoff),
        str(validation_cutoff),
        len(internal_train),
        len(validation),
        len(outer_train),
        len(final_test),
        best_configuration,
        best_weights[0],
        best_weights[1],
        best_weights[2],
        best_weights[3],
        best_result["F1@5"],
        original_test_metrics["F1@5"],
        optimized_test_metrics["F1@5"]
    ]
})

summary_path = OUTPUT_DIR / "optimization_summary.csv"

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 14. DISPLAY FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("OPTIMIZATION COMPLETED!")
print("=" * 70)

print("\nBest configuration:", best_configuration)

print("\nOriginal test F1@5:",
      round(original_test_metrics["F1@5"], 4))

print("Optimized test F1@5:",
      round(optimized_test_metrics["F1@5"], 4))

difference = (
    optimized_test_metrics["F1@5"]
    - original_test_metrics["F1@5"]
)

print("F1@5 difference:", round(difference, 4))

if difference > 0:
    print("\nThe optimized weights performed better on the final test period.")

elif difference < 0:
    print("\nThe original weights performed better on the final test period.")

else:
    print("\nBoth configurations achieved the same final test F1@5.")

print("\nFILES SAVED:")
print("Weight results:", weight_results_path)
print("Final comparison:", final_comparison_path)
print("Optimization summary:", summary_path)

print("\nSTEP 14 COMPLETED!")
