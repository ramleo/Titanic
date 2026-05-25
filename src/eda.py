"""
EDA Agent Script — Step 2: Data Inspection & EDA
Dataset: Titanic-Dataset.csv | Target: Survived | Task: Classification
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = "/Users/wrks/Downloads/Claude-documentation/ML-Titanic/Titanic_20260525_084931"
CSV_PATH = os.path.join(ROOT, "data", "Titanic-Dataset.csv")
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

TARGET = "Survived"

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

print("=" * 60)
print("DATASET SHAPE & COLUMNS")
print("=" * 60)
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# ── Column dtypes ──────────────────────────────────────────────────────────────
print("\n--- Column dtypes and unique counts ---")
for col in df.columns:
    print(f"  {col:<15} dtype={str(df[col].dtype):<10} unique={df[col].nunique():<6} nulls={df[col].isnull().sum()}")

# ── Missing values ─────────────────────────────────────────────────────────────
print("\n--- Missing Values ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"count": missing, "pct": missing_pct})
missing_df = missing_df[missing_df["count"] > 0].sort_values("count", ascending=False)
print(missing_df.to_string())

# ── Class Balance ──────────────────────────────────────────────────────────────
print("\n--- Class Balance (Survived) ---")
vc = df[TARGET].value_counts()
for label, cnt in vc.items():
    print(f"  Survived={label}: {cnt} ({cnt/len(df)*100:.1f}%)")

# ── Numeric Stats ──────────────────────────────────────────────────────────────
print("\n--- Numeric Summary (key columns) ---")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(df[num_cols].describe().round(2).to_string())

# ── Correlation with target ────────────────────────────────────────────────────
print("\n--- Correlation with Target (Survived) ---")
corr_with_target = df[num_cols].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print(corr_with_target.round(4).to_string())

# ── Outlier Detection (IQR) ────────────────────────────────────────────────────
print("\n--- Outlier Detection (IQR method) ---")
for col in ["Age", "Fare", "SibSp", "Parch"]:
    if col not in df.columns:
        continue
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    print(f"  {col}: {n_out} outliers (Q1={Q1:.1f}, Q3={Q3:.1f}, IQR={IQR:.1f})")

# ── Survival rates by category ─────────────────────────────────────────────────
print("\n--- Survival rate by Sex ---")
print(df.groupby("Sex")[TARGET].agg(["mean", "count"]).round(3).to_string())

print("\n--- Survival rate by Pclass ---")
print(df.groupby("Pclass")[TARGET].agg(["mean", "count"]).round(3).to_string())

print("\n--- Survival rate by Embarked ---")
print(df.groupby("Embarked")[TARGET].agg(["mean", "count"]).round(3).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

sns.set_theme(style="whitegrid", palette="muted")

# 1. Class balance bar chart
fig, ax = plt.subplots(figsize=(6, 4))
vc_plot = df[TARGET].value_counts().rename({0: "Not Survived (0)", 1: "Survived (1)"})
bars = ax.bar(vc_plot.index, vc_plot.values, color=["#d9534f", "#5cb85c"], edgecolor="white", width=0.5)
for bar, val in zip(bars, vc_plot.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), ha="center", va="bottom", fontsize=11)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Not Survived (0)", "Survived (1)"])
ax.set_ylabel("Count")
ax.set_title("Survival Class Balance")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_class_balance.png"), dpi=120)
plt.close()
print("Saved: 01_class_balance.png")

# 2. Age distribution
fig, ax = plt.subplots(figsize=(8, 4))
df["Age"].dropna().hist(bins=30, ax=ax, color="#5b9bd5", edgecolor="white")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.set_title("Age Distribution")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_age_distribution.png"), dpi=120)
plt.close()
print("Saved: 02_age_distribution.png")

# 3. Fare distribution (log-scaled for readability)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["Fare"].dropna().hist(bins=40, ax=axes[0], color="#f0ad4e", edgecolor="white")
axes[0].set_title("Fare Distribution (raw)")
axes[0].set_xlabel("Fare")
axes[0].set_ylabel("Count")
np.log1p(df["Fare"].dropna()).hist(bins=40, ax=axes[1], color="#f0ad4e", edgecolor="white")
axes[1].set_title("Fare Distribution (log1p)")
axes[1].set_xlabel("log1p(Fare)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_fare_distribution.png"), dpi=120)
plt.close()
print("Saved: 03_fare_distribution.png")

# 4. Survival rate by Sex
fig, ax = plt.subplots(figsize=(6, 4))
sex_surv = df.groupby("Sex")[TARGET].mean().reset_index()
sns.barplot(data=sex_surv, x="Sex", y=TARGET, ax=ax, palette=["#5cb85c", "#d9534f"])
ax.set_ylabel("Survival Rate")
ax.set_title("Survival Rate by Sex")
ax.set_ylim(0, 1)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width()/2., p.get_height()),
                ha="center", va="bottom", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_survival_by_sex.png"), dpi=120)
plt.close()
print("Saved: 04_survival_by_sex.png")

# 5. Survival rate by Pclass
fig, ax = plt.subplots(figsize=(6, 4))
pclass_surv = df.groupby("Pclass")[TARGET].mean().reset_index()
sns.barplot(data=pclass_surv, x="Pclass", y=TARGET, ax=ax, palette="Blues_r")
ax.set_ylabel("Survival Rate")
ax.set_title("Survival Rate by Passenger Class")
ax.set_ylim(0, 1)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width()/2., p.get_height()),
                ha="center", va="bottom", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_survival_by_pclass.png"), dpi=120)
plt.close()
print("Saved: 05_survival_by_pclass.png")

# 6. Correlation heatmap (numeric features)
fig, ax = plt.subplots(figsize=(9, 7))
corr_matrix = df[num_cols].corr().round(2)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 9})
ax.set_title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_correlation_heatmap.png"), dpi=120)
plt.close()
print("Saved: 06_correlation_heatmap.png")

# 7. Missing values bar chart
fig, ax = plt.subplots(figsize=(8, 4))
if len(missing_df) > 0:
    missing_df["count"].plot(kind="bar", ax=ax, color="#d9534f", edgecolor="white")
    ax.set_title("Missing Values per Column")
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    for i, v in enumerate(missing_df["count"]):
        ax.text(i, v + 1, f"{v}\n({missing_df['pct'].iloc[i]}%)", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
else:
    ax.text(0.5, 0.5, "No Missing Values", ha="center", va="center", fontsize=14)
    ax.set_title("Missing Values per Column")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_missing_values.png"), dpi=120)
plt.close()
print("Saved: 07_missing_values.png")

# 8. Age by Survived (KDE / box)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for label, grp in df.groupby(TARGET):
    grp["Age"].dropna().plot.kde(ax=axes[0], label=f"Survived={label}")
axes[0].set_title("Age KDE by Survival")
axes[0].set_xlabel("Age")
axes[0].legend()
sns.boxplot(data=df, x=TARGET, y="Age", ax=axes[1], palette=["#d9534f", "#5cb85c"])
axes[1].set_title("Age Boxplot by Survival")
axes[1].set_xlabel("Survived")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "08_age_by_survival.png"), dpi=120)
plt.close()
print("Saved: 08_age_by_survival.png")

# 9. Fare by Survived (boxplot)
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=df, x=TARGET, y="Fare", ax=ax, palette=["#d9534f", "#5cb85c"])
ax.set_title("Fare by Survival Status")
ax.set_xlabel("Survived")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "09_fare_by_survival.png"), dpi=120)
plt.close()
print("Saved: 09_fare_by_survival.png")

# 10. Pairplot (key numeric features coloured by Survived)
pair_cols = [c for c in ["Age", "Fare", "Pclass", "SibSp", "Parch", TARGET] if c in df.columns]
pair_df = df[pair_cols].dropna()
g = sns.pairplot(pair_df, hue=TARGET, palette={0: "#d9534f", 1: "#5cb85c"},
                 plot_kws={"alpha": 0.5}, diag_kind="kde", corner=True)
g.figure.suptitle("Pairplot of Key Numeric Features by Survival", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "10_pairplot.png"), dpi=100)
plt.close()
print("Saved: 10_pairplot.png")

print("\n=== EDA COMPLETE — all plots saved to plots/ ===")
