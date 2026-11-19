
import pandas as pd
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ==========================================
# 1. PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "student_profiles.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "clustering_features.csv"
MODEL_FILE = BASE_DIR / "models" / "preprocessor.pkl"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_FILE)

print("STUDENT PROFILES LOADED!")
print("Original Dataset Shape:", df.shape)

# ==========================================
# 3. CREATE INTERMEDIATE COURSE RATIO
# ==========================================

df["IntermediateCourseRatio"] = (
    df["IntermediateCourses"] / df["TotalCourses"]
)

print("\nIntermediate Course Ratio Created!")

# ==========================================
# 4. SELECT FEATURES
# ==========================================

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

selected_features = numeric_features + categorical_features

X = df[selected_features].copy()

print("\nSelected Features:")
print(selected_features)

print("\nMissing Values:")
print(X.isnull().sum())

# ==========================================
# 5. PREPROCESSING
# ==========================================

numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X)

# ==========================================
# 6. GET FEATURE NAMES
# ==========================================

feature_names = preprocessor.get_feature_names_out()

processed_df = pd.DataFrame(
    X_processed,
    columns=feature_names
)

# ==========================================
# 7. SAVE RESULTS
# ==========================================

processed_df.to_csv(OUTPUT_FILE, index=False)

joblib.dump(preprocessor, MODEL_FILE)

print("\n========== PREPROCESSING RESULTS ==========")
print("Original Feature Count:", len(selected_features))
print("Transformed Feature Count:", len(feature_names))
print("Transformed Dataset Shape:", processed_df.shape)

print("\nProcessed Features Saved!")
print("Preprocessor Saved!")

print("\n========================================")
print("FEATURE PREPARATION COMPLETED!")
print("========================================")
