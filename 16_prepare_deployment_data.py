
import pandas as pd
import numpy as np
from pathlib import Path

# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

EXCEL_PATH = (
    BASE_DIR / "data" / "raw" / "EduPro Online Platform.xlsx"
)

OUTPUT_DIR = BASE_DIR / "deployment_data"
OUTPUT_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

# ==================================================
# 2. LOAD COURSE CATALOGUE
# ==================================================

print("Loading original course catalogue...")

courses = pd.read_excel(
    EXCEL_PATH,
    sheet_name="Courses"
)

required_columns = [
    "CourseID",
    "CourseName",
    "CourseCategory",
    "CourseLevel",
    "CoursePrice",
    "CourseRating"
]

courses = courses[required_columns].copy()

courses = courses.drop_duplicates(
    subset=["CourseID"]
).reset_index(drop=True)

courses["CourseID"] = courses["CourseID"].astype(str)

courses["CoursePrice"] = pd.to_numeric(
    courses["CoursePrice"],
    errors="coerce"
).fillna(0)

courses["CourseRating"] = pd.to_numeric(
    courses["CourseRating"],
    errors="coerce"
).fillna(0)

# ==================================================
# 3. GENERATE SYNTHETIC STUDENTS
# ==================================================

print("Generating fictional student records...")

number_of_students = 3000

users = pd.DataFrame({
    "UserID": [
        f"DEMO{i:05d}"
        for i in range(1, number_of_students + 1)
    ],
    "Age": rng.integers(18, 36, number_of_students),
    "Gender": rng.choice(
        ["Male", "Female"],
        number_of_students
    )
})

# ==================================================
# 4. GENERATE SYNTHETIC TRANSACTIONS
# ==================================================

print("Generating fictional transactions...")

number_of_transactions = 10000

course_ids = courses["CourseID"].to_numpy()

pairs = set()

# Give every student at least one course
for user_id in users["UserID"]:
    course_id = rng.choice(course_ids)
    pairs.add((user_id, course_id))

# Add more unique student-course combinations
while len(pairs) < number_of_transactions:

    user_id = rng.choice(users["UserID"])
    course_id = rng.choice(course_ids)

    pairs.add((user_id, course_id))

pairs = list(pairs)

transactions = pd.DataFrame(
    pairs,
    columns=["UserID", "CourseID"]
)

transactions = transactions.merge(
    courses[["CourseID", "CoursePrice"]],
    on="CourseID",
    how="left"
)

transactions = transactions.rename(
    columns={"CoursePrice": "Amount"}
)

transactions["TransactionID"] = [
    f"DEMO_TXN{i:05d}"
    for i in range(1, len(transactions) + 1)
]

transactions["TransactionDate"] = (
    pd.to_datetime("2025-01-01")
    + pd.to_timedelta(
        rng.integers(0, 365, len(transactions)),
        unit="D"
    )
)

transactions = transactions[
    [
        "TransactionID",
        "UserID",
        "CourseID",
        "TransactionDate",
        "Amount"
    ]
]

# ==================================================
# 5. SAVE DEPLOYMENT DATA
# ==================================================

users.to_csv(
    OUTPUT_DIR / "users.csv",
    index=False
)

courses.to_csv(
    OUTPUT_DIR / "courses.csv",
    index=False
)

transactions.to_csv(
    OUTPUT_DIR / "transactions.csv",
    index=False
)

# ==================================================
# 6. VALIDATION
# ==================================================

print("\n======================================")
print("SYNTHETIC DEPLOYMENT DATA CREATED")
print("======================================")

print("Students:", len(users))
print("Courses:", len(courses))
print("Transactions:", len(transactions))

print(
    "Unique student-course pairs:",
    transactions[["UserID", "CourseID"]]
    .drop_duplicates()
    .shape[0]
)

print(
    "Students with transactions:",
    transactions["UserID"].nunique()
)

print("\nFiles saved in:", OUTPUT_DIR)

print("\nIMPORTANT:")
print("These are synthetic demonstration records.")
print("They do not represent real student activity.")
