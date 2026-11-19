
import pandas as pd

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
# 2. CHECK TRANSACTION DATE RANGE
# ==========================================

print("\n========== DATE ANALYSIS ==========")

print("First Transaction:",
      transactions["TransactionDate"].min())

print("Last Transaction:",
      transactions["TransactionDate"].max())

# ==========================================
# 3. CHECK REPEATED STUDENT-COURSE PAIRS
# ==========================================

print("\n========== REPEATED COURSE ANALYSIS ==========")

repeated_pairs = transactions.duplicated(
    subset=["UserID", "CourseID"],
    keep=False
)

print("Repeated Student-Course Records:",
      repeated_pairs.sum())

print("Unique Student-Course Combinations:",
      transactions[["UserID", "CourseID"]].drop_duplicates().shape[0])

print("\nTop Students with Most Transactions:")
print(
    transactions.groupby("UserID")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

# ==========================================
# 4. UNIQUE COURSES PER STUDENT
# ==========================================

print("\n========== UNIQUE COURSE ANALYSIS ==========")

unique_courses_per_student = (
    transactions.groupby("UserID")["CourseID"].nunique()
)

print(unique_courses_per_student.describe())

# ==========================================
# 5. STUDENTS WITHOUT TRANSACTIONS
# ==========================================

print("\n========== INACTIVE STUDENT ANALYSIS ==========")

students_with_transactions = transactions["UserID"].nunique()

students_without_transactions = (
    users["UserID"].nunique() - students_with_transactions
)

print("Total Students:", users["UserID"].nunique())

print("Students with Transactions:", students_with_transactions)

print("Students without Transactions:", students_without_transactions)

# ==========================================
# 6. COMPARE TRANSACTION AMOUNT WITH COURSE PRICE
# ==========================================

print("\n========== PRICE VALIDATION ==========")

merged = transactions.merge(
    courses[["CourseID", "CoursePrice"]],
    on="CourseID",
    how="left"
)

merged["PriceDifference"] = (
    merged["Amount"] - merged["CoursePrice"]
).round(2)

print("Transactions with Amount Different from Course Price:",
      (merged["PriceDifference"].abs() > 0.01).sum())

print("\nPrice Difference Statistics:")
print(merged["PriceDifference"].describe())

# ==========================================
# 7. FINAL SUMMARY
# ==========================================

print("\n================================")
print("DATA VALIDATION COMPLETED!")
print("================================")
