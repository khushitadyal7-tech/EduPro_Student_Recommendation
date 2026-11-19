
import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EduPro - Personalized Course Recommendations",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DEPLOYMENT_DATA_DIR = BASE_DIR / "deployment_data"
USERS_PATH = DEPLOYMENT_DATA_DIR / "users.csv"
COURSES_PATH = DEPLOYMENT_DATA_DIR / "courses.csv"
TRANSACTIONS_PATH = DEPLOYMENT_DATA_DIR / "transactions.csv"

NUMBER_OF_CLUSTERS = 4

# Final selected recommendation weights
POPULARITY_WEIGHT = 0.50
CATEGORY_WEIGHT = 0.20
LEVEL_WEIGHT = 0.20
RATING_WEIGHT = 0.10


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def normalize_id(value):

    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


def get_mode(series):

    modes = series.mode()

    if len(modes) > 0:
        return modes.iloc[0]

    return "Unknown"


# ============================================================
# 3. LOAD DATA AND TRAIN FINAL MODEL
# ============================================================

@st.cache_resource
def load_and_train_model():

    # Check that all required deployment files are available.
    required_files = [
        USERS_PATH,
        COURSES_PATH,
        TRANSACTIONS_PATH
    ]

    missing_files = [
        path.name for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing deployment data file(s): "
            + ", ".join(missing_files)
            + ". Make sure the deployment_data folder contains "
              "users.csv, courses.csv, and transactions.csv."
        )

    # Load the synthetic deployment datasets.
    users = pd.read_csv(USERS_PATH)
    courses = pd.read_csv(COURSES_PATH)
    transactions = pd.read_csv(TRANSACTIONS_PATH)

    # Validate the columns required by the application.
    required_columns = {
        "users.csv": ["UserID", "Age", "Gender"],
        "courses.csv": [
            "CourseID",
            "CourseName",
            "CourseCategory",
            "CourseLevel",
            "CoursePrice",
            "CourseRating"
        ],
        "transactions.csv": [
            "UserID",
            "CourseID",
            "Amount"
        ]
    }

    loaded_data = {
        "users.csv": users,
        "courses.csv": courses,
        "transactions.csv": transactions
    }

    for file_name, columns in required_columns.items():

        missing_columns = [
            column for column in columns
            if column not in loaded_data[file_name].columns
        ]

        if missing_columns:
            raise ValueError(
                f"{file_name} is missing required column(s): "
                + ", ".join(missing_columns)
            )

    if users.empty or courses.empty or transactions.empty:
        raise ValueError(
            "One or more deployment CSV files are empty. "
            "Please check users.csv, courses.csv, and transactions.csv."
        )

    # Normalize identifiers
    users["UserID"] = users["UserID"].apply(normalize_id)
    courses["CourseID"] = courses["CourseID"].apply(normalize_id)

    transactions["UserID"] = (
        transactions["UserID"].apply(normalize_id)
    )

    transactions["CourseID"] = (
        transactions["CourseID"].apply(normalize_id)
    )

    # Clean numeric columns
    transactions["Amount"] = pd.to_numeric(
        transactions["Amount"],
        errors="coerce"
    ).fillna(0)

    courses["CourseRating"] = pd.to_numeric(
        courses["CourseRating"],
        errors="coerce"
    )

    courses["CoursePrice"] = pd.to_numeric(
        courses["CoursePrice"],
        errors="coerce"
    )

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

    # Merge transactions with course information
    training_data = transactions.merge(
        courses,
        on="CourseID",
        how="inner"
    )

    # Build student profiles
    student_profiles = training_data.groupby("UserID").agg(

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

    # Add demographics
    student_profiles = student_profiles.merge(
        users[["UserID", "Age", "Gender"]],
        on="UserID",
        how="left"
    )

    # Course-level ratios
    student_profiles["BeginnerCourseRatio"] = (
        student_profiles["BeginnerCourses"]
        / student_profiles["TotalCourses"]
    )

    student_profiles["AdvancedCourseRatio"] = (
        student_profiles["AdvancedCourses"]
        / student_profiles["TotalCourses"]
    )

    # Clustering features
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

    # Transform features
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

    X = preprocessor.fit_transform(student_profiles)

    # Train final K-Means model
    kmeans = KMeans(
        n_clusters=NUMBER_OF_CLUSTERS,
        random_state=42,
        n_init=10
    )

    student_profiles["Cluster"] = kmeans.fit_predict(X)

    # Calculate cluster popularity
    cluster_sizes = (
        student_profiles.groupby("Cluster")["UserID"]
        .nunique()
    )

    cluster_transactions = transactions.merge(
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

    return (
        student_profiles,
        courses,
        transactions,
        course_popularity
    )


# ============================================================
# 4. GENERATE PERSONALIZED RECOMMENDATIONS
# ============================================================

def recommend_courses(
    user_id,
    students,
    courses,
    transactions,
    popularity,
    top_n
):

    student = students[
        students["UserID"] == user_id
    ]

    if student.empty:
        return None, None

    student = student.iloc[0]

    cluster = student["Cluster"]

    preferred_category = student["PreferredCategory"]

    preferred_level = student["PreferredLevel"]

    # Find previously taken courses
    taken_courses = transactions.loc[
        transactions["UserID"] == user_id,
        "CourseID"
    ].unique()

    # Exclude already taken courses
    candidates = courses[
        ~courses["CourseID"].isin(taken_courses)
    ].copy()

    if candidates.empty:
        return student, pd.DataFrame()

    # Add cluster popularity
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

    # Category preference
    candidates["CategoryMatch"] = (
        candidates["CourseCategory"] == preferred_category
    ).astype(int)

    # Level preference
    candidates["LevelMatch"] = (
        candidates["CourseLevel"] == preferred_level
    ).astype(int)

    # Final recommendation score
    candidates["RecommendationScore"] = (
        POPULARITY_WEIGHT * candidates["ClusterPopularity"]
        + CATEGORY_WEIGHT * candidates["CategoryMatch"]
        + LEVEL_WEIGHT * candidates["LevelMatch"]
        + RATING_WEIGHT * candidates["RatingScore"]
    )

    # Recommendation explanations
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

    # Sort and select top recommendations
    recommendations = candidates.sort_values(
        by=["RecommendationScore", "CourseRating"],
        ascending=False,
        na_position="last"
    ).head(top_n)

    return student, recommendations


# ============================================================
# 5. APPLICATION INTERFACE
# ============================================================

st.title("🎓 EduPro")

st.subheader(
    "Student Segmentation and Personalized Course Recommendation System"
)

st.write(
    "Discover courses suited to each learner's interests, "
    "learning preferences and behaviour."
)

st.divider()

try:

    with st.spinner("Loading data and training the recommendation model..."):

        (
            students,
            courses,
            transactions,
            popularity
        ) = load_and_train_model()

except Exception as error:

    st.error(f"Unable to load the project data: {error}")

    st.stop()


# ============================================================
# 6. DISPLAY PROJECT OVERVIEW
# ============================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Students",
    f"{students['UserID'].nunique():,}"
)

col2.metric(
    "Available Courses",
    f"{courses['CourseID'].nunique():,}"
)

col3.metric(
    "Learner Segments",
    NUMBER_OF_CLUSTERS
)

st.divider()


# ============================================================
# 7. SELECT A STUDENT
# ============================================================

st.header("Find Personalized Courses")

student_ids = sorted(students["UserID"].unique())

selected_user = st.selectbox(
    "Select Student UserID",
    student_ids
)

top_n = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("Generate Recommendations", type="primary"):

    student, recommendations = recommend_courses(
        selected_user,
        students,
        courses,
        transactions,
        popularity,
        top_n
    )

    if student is None:

        st.error("Student not found.")

    elif recommendations.empty:

        st.warning("No new courses are available for this student.")

    else:

        # ----------------------------------------------------
        # 8. STUDENT PROFILE
        # ----------------------------------------------------

        st.divider()

        st.header("Student Learning Profile")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Student ID",
            selected_user
        )

        col2.metric(
            "Assigned Cluster",
            int(student["Cluster"]) + 1
        )

        col3.metric(
            "Courses Taken",
            int(student["TotalCourses"])
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Preferred Category",
            student["PreferredCategory"]
        )

        col2.metric(
            "Preferred Level",
            student["PreferredLevel"]
        )

        col3.metric(
            "Category Diversity",
            int(student["CategoryDiversity"])
        )

        st.write("**Additional Student Information**")

        st.write(f"- Age: {student['Age']}")

        st.write(f"- Gender: {student['Gender']}")

        st.write(
            f"- Average Course Rating: "
            f"{student['AverageCourseRating']:.2f}"
        )

        st.write(
            f"- Average Spending: "
            f"{student['AverageSpending']:.2f}"
        )

        # ----------------------------------------------------
        # 9. RECOMMENDATION RESULTS
        # ----------------------------------------------------

        st.divider()

        st.header("Personalized Course Recommendations")

        st.caption(
            "Previously taken courses are excluded from these results."
        )

        display_columns = [
            "CourseID",
            "CourseName",
            "CourseCategory",
            "CourseLevel",
            "CoursePrice",
            "CourseRating",
            "RecommendationScore",
            "RecommendationReason"
        ]

        display_data = recommendations[
            display_columns
        ].copy()

        display_data["RecommendationScore"] = (
            display_data["RecommendationScore"].round(4)
        )

        display_data["CourseRating"] = (
            display_data["CourseRating"].round(2)
        )

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # 10. DOWNLOAD RECOMMENDATIONS
        # ----------------------------------------------------

        csv_data = recommendations[
            display_columns
        ].to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Recommendations as CSV",
            data=csv_data,
            file_name=f"EduPro_Recommendations_{selected_user}.csv",
            mime="text/csv"
        )

        st.success(
            f"Successfully generated {len(recommendations)} "
            f"recommendations for {selected_user}!"
        )


# ============================================================
# 11. SIDEBAR INFORMATION
# ============================================================

st.sidebar.title("About EduPro")

st.sidebar.write(
    """
    This project uses machine learning and recommendation
    techniques to personalize course suggestions.
    """
)

st.sidebar.subheader("Technologies Used")

st.sidebar.write(
    """
    - Python
    - Pandas
    - Scikit-learn
    - K-Means Clustering
    - Streamlit
    """
)

st.sidebar.subheader("Recommendation Factors")

st.sidebar.write(
    """
    - Cluster Popularity: 50%
    - Category Match: 20%
    - Level Match: 20%
    - Course Rating: 10%
    """
)

st.sidebar.caption(
    "Final demonstration model trained using synthetic demonstration data."
)
