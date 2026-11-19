
import pandas as pd

# ==========================================
# 1. LOAD THE EXCEL WORKBOOK
# ==========================================

file_path = "data/raw/EduPro Online Platform.xlsx"

data = pd.read_excel(file_path, sheet_name=None)

print("DATASET LOADED SUCCESSFULLY!")

# ==========================================
# 2. INSPECT ALL SHEETS
# ==========================================

for sheet_name, df in data.items():

    print("\n================================")
    print("SHEET:", sheet_name)
    print("================================")

    print("\nDataset Shape:", df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nFirst Five Records:")
    print(df.head())

    print("\nDataset Information:")
    df.info()

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:", df.duplicated().sum())


# ==========================================
# 3. CHECK ID UNIQUENESS
# ==========================================

users = data["Users"]
teachers = data["Teachers"]
courses = data["Courses"]
transactions = data["Transactions"]

print("\n========== ID VALIDATION ==========")

print("Duplicate UserIDs:",
      users["UserID"].duplicated().sum())

print("Duplicate TeacherIDs:",
      teachers["TeacherID"].duplicated().sum())

print("Duplicate CourseIDs:",
      courses["CourseID"].duplicated().sum())

print("Duplicate TransactionIDs:",
      transactions["TransactionID"].duplicated().sum())


# ==========================================
# 4. CHECK TABLE RELATIONSHIPS
# ==========================================

print("\n========== RELATIONSHIP VALIDATION ==========")

print("UserIDs not found in Users:",
      (~transactions["UserID"].isin(users["UserID"])).sum())

print("CourseIDs not found in Courses:",
      (~transactions["CourseID"].isin(courses["CourseID"])).sum())

print("TeacherIDs not found in Teachers:",
      (~transactions["TeacherID"].isin(teachers["TeacherID"])).sum())


# ==========================================
# 5. COMPLETION MESSAGE
# ==========================================

print("\n================================")
print("DATA UNDERSTANDING COMPLETED!")
print("================================")
