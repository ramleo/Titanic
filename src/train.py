"""
train.py — Optimization Agent: Steps 4–6
Classification pipeline: GridSearchCV across 4 candidate models.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = "/Users/wrks/Downloads/Claude-documentation/ML-Titanic/Titanic_20260525_084931"

# ── 1. Load artifacts ──────────────────────────────────────────────────────────
X_train_raw   = joblib.load(f"{PROJECT_ROOT}/models/X_train_raw.pkl")
X_test_raw    = joblib.load(f"{PROJECT_ROOT}/models/X_test_raw.pkl")
y_train       = np.load(f"{PROJECT_ROOT}/models/y_train.npy")
y_test        = np.load(f"{PROJECT_ROOT}/models/y_test.npy")
preprocessor  = joblib.load(f"{PROJECT_ROOT}/models/preprocessor.pkl")
label_encoder = joblib.load(f"{PROJECT_ROOT}/models/label_encoder.pkl")

print(f"X_train_raw shape : {X_train_raw.shape}")
print(f"X_test_raw  shape : {X_test_raw.shape}")
print(f"y_train     shape : {y_train.shape}")
print(f"y_test      shape : {y_test.shape}")
print(f"Classes           : {label_encoder.classes_}\n")

# ── 2. Candidate models & grids ────────────────────────────────────────────────
candidates = [
    (
        "LogisticRegression",
        LogisticRegression(solver="saga", max_iter=1000, random_state=42),
        {"model__C": [0.1, 1, 10]},
    ),
    (
        "RandomForest",
        RandomForestClassifier(random_state=42),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth":    [None, 5, 10],
        },
    ),
    (
        "SVC",
        SVC(probability=True, random_state=42),
        {"model__C": [1, 10], "model__kernel": ["rbf", "linear"]},
    ),
    (
        "GradientBoosting",
        GradientBoostingClassifier(random_state=42),
        {
            "model__n_estimators":  [100, 200],
            "model__max_depth":     [3, 5],
            "model__learning_rate": [0.05, 0.1],
        },
    ),
]

# ── 3. Grid search over each candidate ────────────────────────────────────────
results = []

for name, model, param_grid in candidates:
    print(f"  Training {name} ...")
    pipe = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", model),
    ])
    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X_train_raw, y_train)
    results.append({
        "name":       name,
        "cv_score":   gs.best_score_,
        "best_params": gs.best_params_,
        "pipeline":   gs.best_estimator_,
    })
    print(f"    CV accuracy: {gs.best_score_:.4f}  |  {gs.best_params_}")

# ── 4. Results table ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'Model':<22}  {'CV Accuracy':>12}  Best Params")
print("-" * 70)
for r in sorted(results, key=lambda x: x["cv_score"], reverse=True):
    print(f"{r['name']:<22}  {r['cv_score']:>12.4f}  {r['best_params']}")
print("=" * 70)

# ── 5. Select best model ───────────────────────────────────────────────────────
best = max(results, key=lambda x: x["cv_score"])
print(f"\nBest model : {best['name']}")
print(f"Best params: {best['best_params']}")
print(f"CV accuracy: {best['cv_score']:.4f}")

# ── 6. Evaluate on test set ───────────────────────────────────────────────────
best_pipeline = best["pipeline"]
y_pred = best_pipeline.predict(X_test_raw)

test_acc = accuracy_score(y_test, y_pred)
class_names = [str(c) for c in label_encoder.classes_]
report   = classification_report(y_test, y_pred, target_names=class_names)

print(f"\nTest Accuracy : {test_acc:.4f}")
print("\nClassification Report:")
print(report)

# ── 7. Save final pipeline ─────────────────────────────────────────────────────
out_path = f"{PROJECT_ROOT}/models/final_pipeline.pkl"
joblib.dump(best_pipeline, out_path)
print(f"final_pipeline.pkl saved → {out_path}")
