
import pandas as pd
import matplotlib.pyplot as plt
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
# 2. MERGE TRANSACTIONS WITH COURSES
# ==========================================

df = transactions.merge(
    courses,
    on="CourseID",
    how="left"
)

print("\nMERGED DATASET SHAPE:", df.shape)

# ==========================================
# 3. STUDENT ANALYSIS
# ==========================================

print("\n========== STUDENT ANALYSIS ==========")

print("Total Students:", users["UserID"].nunique())

print("\nGender Distribution:")
print(users["Gender"].value_counts())

print("\nAge Statistics:")
print(users["Age"].describe())

# ==========================================
# 4. ENROLMENT ANALYSIS
# ==========================================

print("\n========== ENROLMENT ANALYSIS ==========")

print("Total Transactions:", len(transactions))

print("Unique Students with Transactions:",
      transactions["UserID"].nunique())

print("\nTop 10 Popular Courses:")
print(df["CourseName"].value_counts().head(10))

print("\nCategory-wise Transactions:")
print(df["CourseCategory"].value_counts())

print("\nCourse Level Distribution:")
print(df["CourseLevel"].value_counts())

print("\nCourse Type Distribution:")
print(df["CourseType"].value_counts())

# ==========================================
# 5. STUDENT BEHAVIOUR ANALYSIS
# ==========================================

print("\n========== STUDENT BEHAVIOUR ==========")

student_enrolments = transactions.groupby("UserID").size()

print("\nTransactions Per Student:")
print(student_enrolments.describe())

student_spending = transactions.groupby("UserID")["Amount"].sum()

print("\nStudent Spending Statistics:")
print(student_spending.describe())

print("\nAverage Spending Per Student:",
      round(student_spending.mean(), 2))

# ==========================================
# 6. COURSE RATING ANALYSIS
# ==========================================

print("\n========== COURSE RATING ANALYSIS ==========")

print("Average Course Rating:",
      round(courses["CourseRating"].mean(), 2))

print("\nTop Rated Courses:")
print(
    courses.sort_values(
        by="CourseRating",
        ascending=False
    )[["CourseName", "CourseRating"]].head(10)
)

# ==========================================
# 7. VISUALIZATIONS
# ==========================================

os.makedirs("reports/eda", exist_ok=True)

# Category-wise transactions
plt.figure(figsize=(10, 5))
df["CourseCategory"].value_counts().plot(kind="bar")
plt.title("Transactions by Course Category")
plt.xlabel("Course Category")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("reports/eda/category_enrolments.png")
plt.show()

# Course level distribution
plt.figure(figsize=(8, 5))
df["CourseLevel"].value_counts().plot(kind="bar")
plt.title("Transactions by Course Level")
plt.xlabel("Course Level")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig("reports/eda/course_levels.png")
plt.show()

# Student transaction distribution
plt.figure(figsize=(8, 5))
student_enrolments.plot(kind="hist", bins=20)
plt.title("Transactions Per Student")
plt.xlabel("Number of Transactions")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("reports/eda/student_enrolments.png")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")
