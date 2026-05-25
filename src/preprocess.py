"""
Titanic Preprocessing Pipeline — Step 3: Automated Preprocessing & Cleaning
Task type: Classification
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH     = os.path.join(PROJECT_ROOT, "data", "Titanic-Dataset.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ── 2. Feature engineering — BEFORE dropping any columns ─────────────────────
# Extract Title from Name
df = df.assign(
    Title=df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
)

# Map rare titles
rare_mask = ~df["Title"].isin(["Mr", "Mrs", "Miss", "Master"])
df = df.assign(
    Title=df["Title"].where(~rare_mask, other="Rare")
)
print(f"Title distribution:\n{df['Title'].value_counts().to_string()}\n")

# ── 3. Drop non-feature columns ───────────────────────────────────────────────
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
print(f"After dropping columns: {df.shape[1]} columns remaining")

# ── 4. Encode target ──────────────────────────────────────────────────────────
le = LabelEncoder()
y = le.fit_transform(df["Survived"].values)
X = df.drop(columns=["Survived"])
print(f"Target classes: {le.classes_}  →  encoded as {np.unique(y)}")

# ── 5. Train-test split (80/20 stratified) ────────────────────────────────────
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
print(f"\nSplit shapes:")
print(f"  X_train_raw : {X_train_raw.shape}")
print(f"  X_test_raw  : {X_test_raw.shape}")
print(f"  y_train     : {y_train.shape}")
print(f"  y_test      : {y_test.shape}")

# Class distribution
train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
test_dist  = pd.Series(y_test).value_counts(normalize=True).sort_index()
print(f"\nClass distribution in y_train (label → proportion):")
for cls, prop in train_dist.items():
    print(f"  Class {cls} ({le.classes_[cls]}): {prop:.3f}")
print(f"Class distribution in y_test:")
for cls, prop in test_dist.items():
    print(f"  Class {cls} ({le.classes_[cls]}): {prop:.3f}")

# ── 6. Build ColumnTransformer preprocessor ───────────────────────────────────
NUMERIC_FEATURES     = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ]
)

# ── 7. Fit on X_train_raw only; transform both splits ─────────────────────────
X_train = preprocessor.fit_transform(X_train_raw)
X_test  = preprocessor.transform(X_test_raw)
print(f"\nPreprocessed array shapes:")
print(f"  X_train : {X_train.shape}")
print(f"  X_test  : {X_test.shape}")

# ── 8. Save all artifacts ─────────────────────────────────────────────────────
artifacts = {
    os.path.join(MODELS_DIR, "X_train_raw.pkl") : ("joblib", X_train_raw),
    os.path.join(MODELS_DIR, "X_test_raw.pkl")  : ("joblib", X_test_raw),
    os.path.join(MODELS_DIR, "label_encoder.pkl"): ("joblib", le),
    os.path.join(MODELS_DIR, "preprocessor.pkl") : ("joblib", preprocessor),
}
numpy_artifacts = {
    os.path.join(MODELS_DIR, "y_train.npy")  : y_train,
    os.path.join(MODELS_DIR, "y_test.npy")   : y_test,
    os.path.join(MODELS_DIR, "X_train.npy")  : X_train,
    os.path.join(MODELS_DIR, "X_test.npy")   : X_test,
}

for path, (fmt, obj) in artifacts.items():
    joblib.dump(obj, path)

for path, arr in numpy_artifacts.items():
    np.save(path, arr)

print("\nSaved artifacts:")
all_paths = list(artifacts.keys()) + list(numpy_artifacts.keys())
for p in all_paths:
    size_kb = os.path.getsize(p) / 1024
    print(f"  {os.path.relpath(p, PROJECT_ROOT)}  ({size_kb:.1f} KB)")

print("\nPreprocessing complete — all artifacts written to models/")
