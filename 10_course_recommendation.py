
import pandas as pd
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

EXCEL_FILE = BASE_DIR / "data" / "raw" / "EduPro Online Platform.xlsx"

CLUSTER_FILE = BASE_DIR / "data" / "processed" / "student_clusters.csv"

OUTPUT_DIR = BASE_DIR / "reports" / "recommendations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

print("\nLOADING DATA...")

students = pd.read_csv(
    CLUSTER_FILE,
    dtype={"UserID": str}
)

courses = pd.read_excel(
    EXCEL_FILE,
    sheet_name="Courses"
)

transactions = pd.read_excel(
    EXCEL_FILE,
    sheet_name="Transactions"
)

# Standardize IDs for matching
students["UserID"] = students["UserID"].str.strip()

transactions["UserID"] = (
    transactions["UserID"].astype(str).str.strip()
)

students["CourseID"] = None

transactions["CourseID"] = (
    transactions["CourseID"].astype(str).str.strip()
)

courses["CourseID"] = (
    courses["CourseID"].astype(str).str.strip()
)

print("Student Data:", students.shape)
print("Course Data:", courses.shape)
print("Transaction Data:", transactions.shape)


# ==========================================
# 3. DISPLAY VALID STUDENT IDS
# ==========================================

print("\nFIRST 10 VALID STUDENT IDs:")

print(students["UserID"].head(10).to_list())

print("\nTOTAL STUDENTS:", students["UserID"].nunique())


# ==========================================
# 4. CALCULATE CLUSTER POPULARITY
# ==========================================

cluster_sizes = (
    students.groupby("Cluster")["UserID"].nunique()
)

cluster_transactions = transactions.merge(
    students[["UserID", "Cluster"]],
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

print("\nCOURSE POPULARITY CALCULATED!")


# ==========================================
# 5. RECOMMENDATION FUNCTION
# ==========================================

def recommend_courses(user_id, top_n=5):

    user_id = str(user_id).strip()

    # Find student
    student = students[
        students["UserID"] == user_id
    ]

    if student.empty:

        print("\nSTUDENT ID NOT FOUND!")

        print("Please enter an ID from the valid IDs displayed above.")

        return None

    student = student.iloc[0]

    cluster = student["Cluster"]

    preferred_category = student["PreferredCategory"]

    preferred_level = student["PreferredLevel"]

    # --------------------------------------
    # COURSES ALREADY TAKEN
    # --------------------------------------

    taken_courses = transactions.loc[
        transactions["UserID"] == user_id,
        "CourseID"
    ].unique()

    # Remove already completed/purchased courses
    candidates = courses[
        ~courses["CourseID"].isin(taken_courses)
    ].copy()

    if candidates.empty:

        print("\nNo new courses available for recommendation.")

        return None

    # --------------------------------------
    # CLUSTER POPULARITY
    # --------------------------------------

    cluster_courses = course_popularity[
        course_popularity["Cluster"] == cluster
    ][["CourseID", "ClusterPopularity"]]

    candidates = candidates.merge(
        cluster_courses,
        on="CourseID",
        how="left"
    )

    candidates["ClusterPopularity"] = (
        candidates["ClusterPopularity"].fillna(0)
    )

    # --------------------------------------
    # CATEGORY MATCH
    # --------------------------------------

    candidates["CategoryMatch"] = (
        candidates["CourseCategory"] == preferred_category
    ).astype(int)

    # --------------------------------------
    # COURSE LEVEL MATCH
    # --------------------------------------

    candidates["LevelMatch"] = (
        candidates["CourseLevel"] == preferred_level
    ).astype(int)

    # --------------------------------------
    # COURSE RATING SCORE
    # --------------------------------------

    candidates["CourseRating"] = pd.to_numeric(
        candidates["CourseRating"],
        errors="coerce"
    )

    max_rating = candidates["CourseRating"].max()

    if pd.isna(max_rating) or max_rating <= 0:

        candidates["RatingScore"] = 0

    else:

        candidates["RatingScore"] = (
            candidates["CourseRating"] / max_rating
        ).fillna(0)

    # --------------------------------------
    # RECOMMENDATION SCORE
    # --------------------------------------

    candidates["RecommendationScore"] = (
        0.50 * candidates["ClusterPopularity"]
        + 0.20 * candidates["CategoryMatch"]
        + 0.20 * candidates["LevelMatch"]
        + 0.10 * candidates["RatingScore"]
    )

    # --------------------------------------
    # RECOMMENDATION REASONS
    # --------------------------------------

    def get_reason(row):

        reasons = []

        if row["CategoryMatch"] == 1:
            reasons.append("Matches preferred category")

        if row["LevelMatch"] == 1:
            reasons.append("Matches preferred level")

        if row["ClusterPopularity"] > 0:
            reasons.append("Popular among similar learners")

        if row["RatingScore"] >= 0.8:
            reasons.append("Highly rated course")

        if not reasons:
            reasons.append("Alternative course suggestion")

        return ", ".join(reasons)

    candidates["RecommendationReason"] = (
        candidates.apply(get_reason, axis=1)
    )

    # --------------------------------------
    # SORT RECOMMENDATIONS
    # --------------------------------------

    recommendations = candidates.sort_values(
        by=["RecommendationScore", "CourseRating"],
        ascending=False,
        na_position="last"
    )

    recommendations = recommendations.head(top_n)

    # --------------------------------------
    # SELECT DISPLAY COLUMNS
    # --------------------------------------

    output_columns = [
        "CourseID",
        "CourseName",
        "CourseCategory",
        "CourseLevel",
        "CoursePrice",
        "CourseRating",
        "RecommendationScore",
        "RecommendationReason"
    ]

    recommendations = recommendations[
        output_columns
    ]

    # --------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------

    print("\n" + "=" * 60)

    print("PERSONALIZED COURSE RECOMMENDATIONS")

    print("=" * 60)

    print("Student ID:", user_id)

    print("Student Cluster:", cluster)

    print("Preferred Category:", preferred_category)

    print("Preferred Level:", preferred_level)

    print("Previously Taken Courses:", len(taken_courses))

    print("\nRECOMMENDED COURSES:\n")

    print(recommendations.to_string(index=False))

    # --------------------------------------
    # SAVE RECOMMENDATIONS
    # --------------------------------------

    output_file = OUTPUT_DIR / (
        f"recommendations_user_{user_id}.csv"
    )

    recommendations.to_csv(
        output_file,
        index=False
    )

    print("\nRecommendations saved successfully!")

    print("Saved File:", output_file)

    return recommendations


# ==========================================
# 6. USER INPUT
# ==========================================

print("\n" + "=" * 60)

print("EDUPRO COURSE RECOMMENDATION SYSTEM")

print("=" * 60)

user_id = input(
    "\nEnter Student UserID from the list above: "
).strip()

try:

    top_n = int(
        input("How many courses to recommend? ")
    )

    if top_n <= 0:

        print("Enter a number greater than zero.")

    else:

        recommend_courses(user_id, top_n)

except ValueError:

    print("Please enter a valid number.")
