
import pandas as pd

# ==========================================
# 1. LOAD STUDENT PROFILES
# ==========================================

file_path = "data/processed/student_profiles.csv"

df = pd.read_csv(file_path)

print("STUDENT PROFILES LOADED!")
print("Dataset Shape:", df.shape)

# ==========================================
# 2. CHECK ZERO SPENDING
# ==========================================

print("\n========== SPENDING ANALYSIS ==========")

zero_spending = df[df["TotalSpending"] == 0]

print("Students with Zero Spending:",
      len(zero_spending))

print("Percentage of Students with Zero Spending:",
      round(len(zero_spending) / len(df) * 100, 2), "%")

print("\nCourse Spending Statistics:")
print(df["TotalSpending"].describe())

# ==========================================
# 3. CHECK COURSE LEVEL COUNTS
# ==========================================

print("\n========== COURSE LEVEL VALIDATION ==========")

level_total = (
    df["BeginnerCourses"]
    + df["IntermediateCourses"]
    + df["AdvancedCourses"]
)

print("Students whose level counts match total courses:",
      (level_total == df["TotalCourses"]).sum())

print("Students with mismatched level counts:",
      (level_total != df["TotalCourses"]).sum())

# ==========================================
# 4. CHECK COURSE LEVEL RATIOS
# ==========================================

print("\n========== COURSE LEVEL RATIOS ==========")

df["IntermediateCourseRatio"] = (
    df["IntermediateCourses"] / df["TotalCourses"]
)

ratio_total = (
    df["BeginnerCourseRatio"]
    + df["IntermediateCourseRatio"]
    + df["AdvancedCourseRatio"]
)

print("Average Beginner Ratio:",
      round(df["BeginnerCourseRatio"].mean(), 3))

print("Average Intermediate Ratio:",
      round(df["IntermediateCourseRatio"].mean(), 3))

print("Average Advanced Ratio:",
      round(df["AdvancedCourseRatio"].mean(), 3))

print("Students whose level ratios sum to 1:",
      ((ratio_total - 1).abs() < 0.000001).sum())

# ==========================================
# 5. CHECK CATEGORY DIVERSITY
# ==========================================

print("\n========== CATEGORY DIVERSITY ==========")

print(df["CategoryDiversity"].value_counts().sort_index())

print("\nAverage Categories Explored:",
      round(df["CategoryDiversity"].mean(), 2))

# ==========================================
# 6. CHECK PREFERRED CATEGORY AND LEVEL
# ==========================================

print("\n========== STUDENT PREFERENCES ==========")

print("Preferred Categories:")
print(df["PreferredCategory"].value_counts())

print("\nPreferred Course Levels:")
print(df["PreferredLevel"].value_counts())

# ==========================================
# 7. CHECK AGE AND RATINGS
# ==========================================

print("\n========== AGE AND RATING ANALYSIS ==========")

print("Age Range:",
      df["Age"].min(), "to", df["Age"].max())

print("Average Course Rating:",
      round(df["AverageCourseRating"].mean(), 2))

print("\nAverage Spending:",
      round(df["AverageSpending"].mean(), 2))

# ==========================================
# 8. CHECK DUPLICATE FEATURES
# ==========================================

print("\n========== FEATURE REDUNDANCY ==========")

print("TotalCourses equals TotalTransactions:",
      (df["TotalCourses"] == df["TotalTransactions"]).all())

print("\nFEATURE REVIEW COMPLETED!")
