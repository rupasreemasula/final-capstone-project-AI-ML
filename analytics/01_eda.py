"""
Module 2 - Analytics Pipeline: Part A (Tasks 1-6)
Profiles, cleans, and tells the data story for the Titanic dataset.

This is the ONE AND ONLY load of the raw dataset for the whole module
(sns.load_dataset('titanic')). It is saved immediately as titanic.csv, the
committed offline fallback used both by the rest of this script and by
02_modeling.py (which reads titanic.csv and never calls sns.load_dataset
again).

Run:
    python 01_eda.py

Outputs (written next to this script):
    titanic.csv           - raw offline-fallback dataset (Task 1)
    eda_output.txt         - full text log of every printed result (Tasks 1-6)
    charts/*.png           - all Task 3-5 charts
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering, no display server needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "titanic.csv"
CHARTS_DIR = HERE / "charts"
OUTPUT_LOG = HERE / "eda_output.txt"
CHARTS_DIR.mkdir(exist_ok=True)

LOG_LINES: list[str] = []


def log(*args) -> None:
    text = " ".join(str(a) for a in args)
    print(text)
    LOG_LINES.append(text)


# ---------------------------------------------------------------------------
# Task 1: load once, profile, save offline fallback
# ---------------------------------------------------------------------------

def task1_load_and_profile() -> pd.DataFrame:
    log("=" * 78)
    log("TASK 1 - Load & profile")
    log("=" * 78)

    df = sns.load_dataset("titanic")  # the ONE network/cache load for the whole module

    # Immediately save the raw offline fallback, before any cleaning.
    df.to_csv(CSV_PATH, index=False)
    log(f"Saved raw offline fallback -> {CSV_PATH} ({len(df)} rows)")

    log("\n--- df.shape ---")
    log(df.shape)

    log("\n--- df.info() ---")
    import io
    buf = io.StringIO()
    df.info(buf=buf)
    log(buf.getvalue())

    log("--- df.describe() ---")
    log(df.describe(include="all").to_string())

    log("\n--- % missing values per column (only columns with any missing) ---")
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    log(missing_pct.round(4).to_string())

    return df


# ---------------------------------------------------------------------------
# Task 2: missing-value handling per the percentage threshold rule
# ---------------------------------------------------------------------------

def task2_clean(df: pd.DataFrame) -> pd.DataFrame:
    log("\n" + "=" * 78)
    log("TASK 2 - Missing-value handling (threshold rule: <5% drop rows, "
        "5-30% impute, very high -> drop column / 'missing' category)")
    log("=" * 78)

    df = df.copy()
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    for col, pct in missing_pct.items():
        log(f"\nColumn '{col}': {pct:.4f}% missing")
        if col == "deck":
            log("  -> 77.22% missing: far too high for reliable imputation "
                "(would be fabricating ~3/4 of the column). deck is also "
                "largely redundant with pclass/fare (higher classes have "
                "better-recorded cabins), so it is DROPPED entirely rather "
                "than encoded as its own 'missing' category, since a column "
                "that is 3/4 placeholder adds noise rather than signal.")
        elif col == "age":
            log("  -> 19.87% missing: within the 5%-30% 'impute' band. "
                "Imputed with the MEDIAN age within each (pclass, sex) "
                "group (falling back to the overall median if a group is "
                "empty), which is more accurate than a single global "
                "median since age correlates strongly with class and sex "
                "in this dataset.")
        elif col in ("embarked", "embark_town"):
            log(f"  -> {pct:.4f}% missing: below the 5% threshold, so the "
                "affected ROWS are dropped rather than imputed (only 2 "
                "rows total, shared between embarked/embark_town).")

    # deck: drop column (>30% missing, unreliable to impute)
    df = df.drop(columns=["deck"])

    # embarked/embark_town: <5% missing -> drop the affected rows
    before = len(df)
    df = df.dropna(subset=["embarked", "embark_town"])
    log(f"\nDropped {before - len(df)} row(s) with missing embarked/embark_town.")

    # age: 5-30% missing -> median-impute within (pclass, sex) groups
    n_age_missing = int(df["age"].isna().sum())
    df["age"] = df.groupby(["pclass", "sex"])["age"].transform(
        lambda s: s.fillna(s.median())
    )
    df["age"] = df["age"].fillna(df["age"].median())  # safety net, no empty groups expected
    log(f"Median-imputed {n_age_missing} missing 'age' value(s) using (pclass, sex) group medians.")

    log(f"\nRemaining missing values after cleaning:\n{df.isna().sum()[df.isna().sum() > 0]}")
    log(f"\nCleaned shape: {df.shape}")

    return df


# ---------------------------------------------------------------------------
# Task 3: univariate analysis (age, fare)
# ---------------------------------------------------------------------------

def iqr_outliers(series: pd.Series) -> tuple[int, float, float]:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = int(((series < lower) | (series > upper)).sum())
    return n_outliers, lower, upper


def task3_univariate(df: pd.DataFrame) -> None:
    log("\n" + "=" * 78)
    log("TASK 3 - Univariate analysis: age & fare")
    log("=" * 78)

    for col in ["age", "fare"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(df[col], bins=30, color="#4C72B0", edgecolor="white")
        axes[0].set_title(f"{col} - histogram")
        axes[0].set_xlabel(col)
        axes[1].boxplot(df[col], vert=True)
        axes[1].set_title(f"{col} - box plot")
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / f"{col}_hist_box.png", dpi=120)
        plt.close(fig)

        n_out, lower, upper = iqr_outliers(df[col])
        log(f"\n{col}: IQR outlier bounds = [{lower:.2f}, {upper:.2f}]  "
            f"-> {n_out} outlier(s) (of {len(df)} rows)")

    fare = df["fare"]
    mean_, median_, mode_ = fare.mean(), fare.median(), fare.mode().iloc[0]
    log(f"\nfare: mean={mean_:.2f}, median={median_:.2f}, mode={mode_:.2f}")
    if mean_ > median_ > mode_:
        skew_desc = "right-skewed (mean > median > mode)"
    elif mean_ < median_ < mode_:
        skew_desc = "left-skewed (mean < median < mode)"
    else:
        skew_desc = "approximately symmetric (mean ~= median ~= mode)"
    log(f"fare distribution is {skew_desc}.")


# ---------------------------------------------------------------------------
# Task 4: bivariate analysis + correlation heatmap
# ---------------------------------------------------------------------------

def task4_bivariate(df: pd.DataFrame) -> None:
    log("\n" + "=" * 78)
    log("TASK 4 - Bivariate analysis")
    log("=" * 78)

    # boolean masking with & / |
    log("\n--- Survival rate by sex ---")
    for s in df["sex"].unique():
        mask = df["sex"] == s
        rate = df.loc[mask, "survived"].mean()
        log(f"  sex={s}: survival rate = {rate:.4f} (n={mask.sum()})")

    log("\n--- Survival rate by pclass ---")
    for p in sorted(df["pclass"].unique()):
        mask = df["pclass"] == p
        rate = df.loc[mask, "survived"].mean()
        log(f"  pclass={p}: survival rate = {rate:.4f} (n={mask.sum()})")

    log("\n--- Survival rate by sex & pclass ---")
    for s in sorted(df["sex"].unique()):
        for p in sorted(df["pclass"].unique()):
            mask = (df["sex"] == s) & (df["pclass"] == p)
            rate = df.loc[mask, "survived"].mean()
            log(f"  sex={s} & pclass={p}: survival rate = {rate:.4f} (n={mask.sum()})")

    # Extra boolean-OR ('|') example, alongside the AND ('&') masks above, per
    # the task's "&/| combinations" wording:
    top_or_mid_class = (df["pclass"] == 1) | (df["pclass"] == 2)
    rate = df.loc[top_or_mid_class, "survived"].mean()
    log(f"\n  pclass==1 OR pclass==2 (boolean OR mask): survival rate = "
        f"{rate:.4f} (n={top_or_mid_class.sum()})")

    # correlation matrix on exactly these 6 numeric columns
    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = df[corr_cols].corr()
    log("\n--- Correlation matrix (6 specified columns only) ---")
    log(corr.round(3).to_string())

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap (survived, pclass, age, sibsp, parch, fare)")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    # top-2 strongest off-diagonal absolute correlations
    pairs = []
    for i, a in enumerate(corr_cols):
        for b in corr_cols[i + 1:]:
            pairs.append((a, b, corr.loc[a, b]))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    log("\n--- Top 2 strongest |correlation| off-diagonal pairs ---")
    for a, b, v in pairs[:2]:
        log(f"  {a} <-> {b}: r = {v:.3f}")


# ---------------------------------------------------------------------------
# Task 5: multivariate "data story" (>= 4 charts)
# ---------------------------------------------------------------------------

def task5_multivariate(df: pd.DataFrame) -> None:
    log("\n" + "=" * 78)
    log("TASK 5 - Multivariate data story (charts saved to charts/; "
        "written interpretation lives in analytics/README.md)")
    log("=" * 78)

    # Chart 1: survival rate by sex (bar)
    fig, ax = plt.subplots(figsize=(5, 4))
    df.groupby("sex")["survived"].mean().plot(kind="bar", ax=ax, color=["#DD8452", "#4C72B0"])
    ax.set_ylabel("Survival rate")
    ax.set_title("Survival rate by sex")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "story_1_survival_by_sex.png", dpi=120)
    plt.close(fig)

    # Chart 2: survival rate by pclass (bar)
    fig, ax = plt.subplots(figsize=(5, 4))
    df.groupby("pclass")["survived"].mean().plot(kind="bar", ax=ax, color="#55A868")
    ax.set_ylabel("Survival rate")
    ax.set_title("Survival rate by passenger class")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "story_2_survival_by_pclass.png", dpi=120)
    plt.close(fig)

    # Chart 3: survival rate by sex & pclass (grouped bar)
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped = df.groupby(["pclass", "sex"])["survived"].mean().unstack()
    grouped.plot(kind="bar", ax=ax)
    ax.set_ylabel("Survival rate")
    ax.set_title("Survival rate by class and sex")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "story_3_survival_by_pclass_sex.png", dpi=120)
    plt.close(fig)

    # Chart 4: age distribution by survival, split by pclass (box)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="pclass", y="age", hue="survived", ax=ax)
    ax.set_title("Age distribution by class and survival")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "story_4_age_by_pclass_survival.png", dpi=120)
    plt.close(fig)

    # Chart 5: fare vs age scatter, colored by survival
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=df, x="age", y="fare", hue="survived", alpha=0.6, ax=ax)
    ax.set_title("Fare vs age, colored by survival")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "story_5_fare_vs_age_by_survival.png", dpi=120)
    plt.close(fig)

    log("Saved 5 story charts to charts/story_1..5_*.png (>= 4 required).")


# ---------------------------------------------------------------------------
# Task 6: EDA-stage standardization sanity check (age, fare) - NOT used by modeling
# ---------------------------------------------------------------------------

def task6_standardization_check(df: pd.DataFrame) -> None:
    log("\n" + "=" * 78)
    log("TASK 6 - EDA-stage standardization sanity check (age, fare) - "
        "exploratory only, does NOT feed the modeling pipeline in 02_modeling.py")
    log("=" * 78)

    cols = ["age", "fare"]
    log("\nBefore standardization:")
    log(df[cols].agg(["mean", "std"]).round(4).to_string())

    scaler = StandardScaler()
    standardized = pd.DataFrame(
        scaler.fit_transform(df[cols]), columns=[f"{c}_z" for c in cols], index=df.index
    )
    log("\nAfter standardization (z = (x - mean) / std):")
    log(standardized.agg(["mean", "std"]).round(4).to_string())
    log("\n(mean ~= 0 and std ~= 1 for both columns, confirming correct standardization.)")


def main() -> None:
    df_raw = task1_load_and_profile()
    df_clean = task2_clean(df_raw)
    task3_univariate(df_clean)
    task4_bivariate(df_clean)
    task5_multivariate(df_clean)
    task6_standardization_check(df_clean)

    OUTPUT_LOG.write_text("\n".join(LOG_LINES), encoding="utf-8")
    print(f"\nFull EDA text log saved -> {OUTPUT_LOG}")


if __name__ == "__main__":
    main()
