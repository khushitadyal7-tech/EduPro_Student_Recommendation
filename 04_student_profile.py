
import pandas as pd
import os

# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = "data/raw/EduPro Online Platform.xlsx"

data = pd.read_excel(file_path, sheet_name=None)

users = data["Users"]
courses = data["Courses"]
transactions = data["Transactions"]

print("DATA LOADED SUCCESSFULLY!")

# ==========================================
# 2. MERGE TRANSACTIONS WITH COURSE DETAILS
# ==========================================

df = transactions.merge(
    courses,
    on="CourseID",
    how="left"
)

print("\nMERGED DATA SHAPE:", df.shape)

# ==========================================
# 3. CREATE STUDENT ENGAGEMENT FEATURES
# ==========================================

engagement = df.groupby("UserID").agg(
    TotalCourses=("CourseID", "nunique"),
    TotalTransactions=("TransactionID", "count"),
    TotalSpending=("Amount", "sum"),
    AverageSpending=("Amount", "mean")
).reset_index()

# ==========================================
# 4. CREATE COURSE PREFERENCE FEATURES
# ==========================================

preferences = df.groupby("UserID").agg(
    PreferredCategory=(
        "CourseCategory",
        lambda x: x.mode().iloc[0]
    ),
    PreferredLevel=(
        "CourseLevel",
        lambda x: x.mode().iloc[0]
    ),
    AverageCourseRating=("CourseRating", "mean"),
    CategoryDiversity=("CourseCategory", "nunique")
).reset_index()

# ==========================================
# 5. CREATE COURSE LEVEL FEATURES
# ==========================================

level_features = df.pivot_table(
    index="UserID",
    columns="CourseLevel",
    values="CourseID",
    aggfunc="nunique",
    fill_value=0
).reset_index()

level_features.columns.name = None

# Rename columns for clarity
level_features = level_features.rename(
    columns={
        "Beginner": "BeginnerCourses",
        "Intermediate": "IntermediateCourses",
        "Advanced": "AdvancedCourses"
    }
)

# ==========================================
# 6. COMBINE ALL STUDENT FEATURES
# ==========================================

student_profiles = users[
    ["UserID", "Age", "Gender"]
].merge(
    engagement,
    on="UserID",
    how="left"
).merge(
    preferences,
    on="UserID",
    how="left"
).merge(
    level_features,
    on="UserID",
    how="left"
)

# ==========================================
# 7. CALCULATE LEARNING DEPTH
# ==========================================

student_profiles["AdvancedCourseRatio"] = (
    student_profiles["AdvancedCourses"]
    / student_profiles["TotalCourses"]
)

student_profiles["BeginnerCourseRatio"] = (
    student_profiles["BeginnerCourses"]
    / student_profiles["TotalCourses"]
)

# ==========================================
# 8. DISPLAY STUDENT PROFILES
# ==========================================

print("\n========== STUDENT PROFILES ==========")

print("\nProfile Dataset Shape:")
print(student_profiles.shape)

print("\nFirst Five Student Profiles:")
print(student_profiles.head())

print("\nStudent Profile Information:")
student_profiles.info()

print("\nMissing Values:")
print(student_profiles.isnull().sum())

print("\nStatistical Summary:")
print(student_profiles.describe())

# ==========================================
# 9. SAVE STUDENT PROFILES
# ==========================================

os.makedirs("data/processed", exist_ok=True)

output_path = "data/processed/student_profiles.csv"

student_profiles.to_csv(output_path, index=False)

print("\nStudent Profiles Saved Successfully!")
print("File Location:", output_path)

print("\n================================")
print("STUDENT PROFILE CREATION COMPLETED!")
print("================================")
