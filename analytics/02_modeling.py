"""
Module 2 - Analytics Pipeline: Part B (Tasks 7-15)
Predictive modeling, continuing from the SAME committed titanic.csv that
01_eda.py produced (no second sns.load_dataset('titanic') call anywhere in
this script).

Run (after 01_eda.py has produced titanic.csv):
    python 02_modeling.py

Outputs (written next to this script):
    modeling_output.txt          - full text log of every printed result (Tasks 7-14)
    charts/decision_tree.png     - Task 9
    charts/roc_curves.png        - Task 10
    charts/regression_residuals.png - Task 13
    best_pipeline.joblib         - Task 15: full fitted preprocessing + estimator pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "titanic.csv"
CHARTS_DIR = HERE / "charts"
OUTPUT_LOG = HERE / "modeling_output.txt"
PIPELINE_PATH = HERE / "best_pipeline.joblib"
CHARTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
LOG_LINES: list[str] = []


def log(*args) -> None:
    text = " ".join(str(a) for a in args)
    print(text)
    LOG_LINES.append(text)


NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["sex", "embarked"]
CLASSIFICATION_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns from the raw titanic.csv deliberately excluded from every feature
# set below: `alive` is a string duplicate of the `survived` target itself
# (training on it would be direct label leakage); `class`, `who`,
# `adult_male`, `alone`, `embark_town` are derived/redundant duplicates of
# columns already included (class~pclass, who/adult_male~sex+age,
# alone~sibsp+parch, embark_town~embarked); `deck` is ~77% missing and was
# already dropped for the same reason in Task 2. This mirrors the
# redundant-column reasoning from Task 2 (deck) and Task 4 (adult_male,
# alone), applied consistently to both the classification and the
# regression feature sets below.


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Fit-on-train-only preprocessing: impute -> scale (numeric), impute -> one-hot (categorical)."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


# ---------------------------------------------------------------------------
# Task 7: stratified train/test split
# ---------------------------------------------------------------------------

def task7_split(df: pd.DataFrame):
    log("=" * 78)
    log("TASK 7 - Stratified train/test split")
    log("=" * 78)

    class_balance = df["survived"].value_counts(normalize=True).sort_index()
    log("\nOverall class balance (survived):")
    log(class_balance.round(4).to_string())
    log(
        "\nJustification for stratification: the classes are imbalanced "
        f"(~{class_balance[0]*100:.1f}% not-survived vs ~{class_balance[1]*100:.1f}% "
        "survived). On a dataset this size (~890 rows), a plain random split "
        "can by chance shift that ratio noticeably between train and test, "
        "which would bias evaluation metrics (e.g. an unlucky test fold with "
        "far more/fewer survivors than the true rate). A STRATIFIED split "
        "preserves the same survived/not-survived ratio in both folds, so "
        "test-set performance is a fair estimate of real-world performance."
    )

    X = df[CLASSIFICATION_FEATURES]
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    log(f"\nTrain size: {len(X_train)}  Test size: {len(X_test)}")
    log("Train class balance:")
    log(y_train.value_counts(normalize=True).sort_index().round(4).to_string())
    log("Test class balance:")
    log(y_test.value_counts(normalize=True).sort_index().round(4).to_string())

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Task 8 note: the ColumnTransformer above is fit ONLY inside pipeline.fit(X_train, y_train)
# for every classifier below - transform-only on X_test, never fit/refit on test data.
# ---------------------------------------------------------------------------

def task8_note() -> None:
    log("\n" + "=" * 78)
    log("TASK 8 - Preprocessing (fit on training data only)")
    log("=" * 78)
    log(
        "\nEvery model below is a single sklearn Pipeline([('preprocessor', "
        "ColumnTransformer(...)), ('classifier', <model>)]). Calling "
        "pipeline.fit(X_train, y_train) fits the imputer/scaler/encoder "
        "*only* on X_train; calling pipeline.predict(X_test) or "
        "pipeline.score(X_test) then applies those already-fitted "
        "transforms in transform-only mode to X_test. No step is ever fit "
        "or refit on test data or on the full pre-split dataset - this is "
        "enforced structurally by the Pipeline/ColumnTransformer, not left "
        "to be remembered by hand.\n"
        "Missing-value handling for this stage (independent of Task 2's "
        "EDA-stage choice): numeric columns are median-imputed, "
        "categorical columns are most-frequent-imputed - both fit on the "
        "training split only, via the ColumnTransformer above."
    )


# ---------------------------------------------------------------------------
# Task 9 & 10: train 3 classifiers, evaluate with full metric suite
# ---------------------------------------------------------------------------

def make_classifier_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
            ("classifier", estimator),
        ]
    )


def evaluate_classifier(name, pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
    }

    log(f"\n--- {name} ---")
    log("Confusion matrix (rows=actual, cols=predicted; [0=not survived, 1=survived]):")
    log(cm)
    for k in ["accuracy", "precision", "recall", "f1", "auc"]:
        log(f"  {k}: {metrics[k]:.4f}")

    return metrics, y_proba


def task9_10_train_and_evaluate(X_train, X_test, y_train, y_test):
    log("\n" + "=" * 78)
    log("TASK 9 & 10 - Train 3 classifiers, evaluate full metric suite")
    log("=" * 78)

    pipelines = {
        "Logistic Regression": make_classifier_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        "Decision Tree": make_classifier_pipeline(
            DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
        ),
        "Random Forest": make_classifier_pipeline(
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        ),
    }

    results = []
    roc_data = {}
    fitted = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)  # preprocessor fit on X_train ONLY
        fitted[name] = pipe
        metrics, y_proba = evaluate_classifier(name, pipe, X_test, y_test)
        results.append(metrics)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, metrics["auc"])

    comparison_df = pd.DataFrame(results).set_index("model").round(4)
    log("\n--- Classifier comparison table (accuracy/precision/recall/F1/AUC) ---")
    log(comparison_df.to_string())

    # ROC curves, all 3 overlaid
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves - all 3 classifiers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "roc_curves.png", dpi=120)
    plt.close(fig)

    # Decision tree visualization with labeled features/classes
    dt_pipeline = fitted["Decision Tree"]
    preprocessor = dt_pipeline.named_steps["preprocessor"]
    dt_model = dt_pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    fig, ax = plt.subplots(figsize=(32, 16))
    plot_tree(
        dt_model,  # fit with max_depth=5; rendered here in full, no truncation
        feature_names=feature_names,
        class_names=["Not Survived", "Survived"],
        filled=True,
        fontsize=7,
        ax=ax,
    )
    ax.set_title("Decision Tree (full depth-5 tree, as fitted)")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "decision_tree.png", dpi=120)
    plt.close(fig)

    return fitted, comparison_df


# ---------------------------------------------------------------------------
# Task 11: imbalance handling comparison
# ---------------------------------------------------------------------------

def task11_imbalance_comparison(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    log("\n" + "=" * 78)
    log("TASK 11 - Imbalance handling comparison (model: Logistic Regression)")
    log("=" * 78)

    log("\nTraining-fold class balance:")
    log(y_train.value_counts().to_string())
    log(y_train.value_counts(normalize=True).round(4).to_string())

    variants = {
        "(a) Baseline / no handling": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "(b) class_weight='balanced'": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "(c) SMOTE (training fold only)": ImbPipeline(
            steps=[
                ("preprocessor", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
    }

    rows = []
    for name, pipe in variants.items():
        pipe.fit(X_train, y_train)  # SMOTE (in variant c) resamples ONLY X_train/y_train
        y_pred = pipe.predict(X_test)
        row = {
            "variant": name,
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        rows.append(row)
        log(f"\n{name}: precision={row['precision']:.4f}  recall={row['recall']:.4f}  f1={row['f1']:.4f}")

    result_df = pd.DataFrame(rows).set_index("variant").round(4)
    log("\n--- Imbalance strategy comparison table ---")
    log(result_df.to_string())

    best = result_df["f1"].idxmax()
    log(
        f"\nConclusion: '{best}' achieved the highest F1 on the test set. "
        "The class imbalance here (~38% survived) is moderate rather than "
        "extreme, so the baseline model already captures the minority "
        "class reasonably well; class_weight='balanced' and SMOTE both "
        "push recall up (fewer missed survivors) generally at some cost "
        "to precision (more false positives), since both make the model "
        "more willing to predict the minority class. Whichever variant "
        "wins on F1 here offers the best precision/recall trade-off for "
        "this dataset; in a deployment where missing a survivor is worse "
        "than a false alarm, the higher-recall variant would be preferred "
        "even if its F1 were slightly lower."
    )
    return result_df


# ---------------------------------------------------------------------------
# Task 12: hyperparameter tuning (Random Forest) + OOB score
# ---------------------------------------------------------------------------

def task12_tune_random_forest(X_train, X_test, y_train, y_test):
    log("\n" + "=" * 78)
    log("TASK 12 - GridSearchCV tuning: Random Forest")
    log("=" * 78)

    pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
            ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]
    )
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 5, 10],
        "classifier__max_features": ["sqrt", "log2"],
    }
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    log(f"\nBest params: {grid.best_params_}")
    log(f"Best CV F1 score: {grid.best_score_:.4f}")

    # Refit a fresh RandomForestClassifier with the best params AND oob_score=True
    # (oob_score_ is only populated when oob_score=True is set at construction time).
    best_rf_params = {
        k.replace("classifier__", ""): v for k, v in grid.best_params_.items()
    }
    preprocessor = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    tuned_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    oob_score=True,
                    random_state=RANDOM_STATE,
                    **best_rf_params,
                ),
            ),
        ]
    )
    tuned_pipeline.fit(X_train, y_train)
    oob_score = tuned_pipeline.named_steps["classifier"].oob_score_
    log(f"Out-of-bag (OOB) score of the tuned Random Forest: {oob_score:.4f}")

    metrics, _ = evaluate_classifier("Tuned Random Forest (test set)", tuned_pipeline, X_test, y_test)
    return tuned_pipeline, grid.best_params_, oob_score, metrics


# ---------------------------------------------------------------------------
# Task 13: regression side-task - predict fare
# ---------------------------------------------------------------------------

def adjusted_r2(r2: float, n: int, p: int) -> float:
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def task13_regression(df: pd.DataFrame, train_idx, test_idx):
    log("\n" + "=" * 78)
    log("TASK 13 - Regression side-task: predict fare")
    log("=" * 78)

    # Same exclusion policy as CLASSIFICATION_FEATURES above (see the module-level
    # comment): drop leakage/derived/redundant columns (alive, class, who,
    # adult_male, alone, embark_town, deck). `survived` is included here as a
    # predictor (not the target this time) since predicting fare from it is not
    # leakage.
    reg_numeric = ["pclass", "age", "sibsp", "parch", "survived"]
    reg_categorical = ["sex", "embarked"]
    reg_features = reg_numeric + reg_categorical

    X = df[reg_features]
    y = df["fare"]
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    reg_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(reg_numeric, reg_categorical)),
            ("regressor", LinearRegression()),
        ]
    )
    reg_pipeline.fit(X_train, y_train)
    y_pred = reg_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    p = len(reg_features)  # count of original predictor columns (pre-encoding)
    n = len(y_test)
    adj_r2 = adjusted_r2(r2, n, p)

    log(f"\nMAE:  {mae:.4f}")
    log(f"RMSE: {rmse:.4f}")
    log(f"R^2:  {r2:.4f}")
    log(f"Adjusted R^2: {adj_r2:.4f}  (n={n} test rows, p={p} original predictor columns)")

    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted fare")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residual plot - fare regression")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "regression_residuals.png", dpi=120)
    plt.close(fig)

    # Data-driven heteroscedasticity signal: correlate |residual| with predicted value.
    corr_abs_resid_pred = float(np.corrcoef(np.abs(residuals), y_pred)[0, 1])
    log(f"\nCorrelation(|residual|, predicted fare) = {corr_abs_resid_pred:.4f}")
    if abs(corr_abs_resid_pred) > 0.2:
        hetero_conclusion = (
            "The residual plot shows HETEROSCEDASTICITY: the spread of "
            "residuals visibly fans out as predicted fare increases "
            f"(|residual| vs predicted fare correlation = {corr_abs_resid_pred:.2f}), "
            "rather than scattering randomly around zero with constant "
            "width. This is expected for fare, which is a heavily "
            "right-skewed, long-tailed variable (see Task 3) that a "
            "linear model cannot fully linearize."
        )
    else:
        hetero_conclusion = (
            "The residual plot does not show strong heteroscedasticity: "
            "residuals scatter around zero with roughly constant spread "
            f"across predicted values (|residual| vs predicted fare "
            f"correlation = {corr_abs_resid_pred:.2f})."
        )
    log(hetero_conclusion)

    return {"mae": mae, "rmse": rmse, "r2": r2, "adj_r2": adj_r2}, reg_pipeline


# ---------------------------------------------------------------------------
# Task 14: final comparison table + recommendation
# ---------------------------------------------------------------------------

def task14_final_comparison(classifier_df: pd.DataFrame, tuned_metrics: dict, regression_metrics: dict):
    log("\n" + "=" * 78)
    log("TASK 14 - Final model comparison table")
    log("=" * 78)

    full_classifier_df = pd.concat(
        [classifier_df, pd.DataFrame([tuned_metrics]).set_index("model")]
    ).round(4)
    log("\n--- Classification metrics (accuracy, precision, recall, F1, AUC) ---")
    log(full_classifier_df.to_string())

    reg_df = pd.DataFrame(
        [regression_metrics], index=["Linear Regression (predicting fare)"]
    ).rename(columns={"mae": "MAE", "rmse": "RMSE", "r2": "R2", "adj_r2": "Adjusted_R2"}).round(4)
    log("\n--- Regression metrics (separate scale - MAE, RMSE, R2, Adjusted R2) ---")
    log(reg_df.to_string())

    best_model = full_classifier_df["f1"].idxmax()
    best_row = full_classifier_df.loc[best_model]
    log(
        f"\nRecommendation: deploy **{best_model}** "
        f"(accuracy={best_row['accuracy']:.3f}, precision={best_row['precision']:.3f}, "
        f"recall={best_row['recall']:.3f}, F1={best_row['f1']:.3f}, AUC={best_row['auc']:.3f}). "
        "It achieves the highest F1 among the candidates evaluated here, "
        "balancing precision and recall better than the alternatives while "
        "keeping AUC competitive. Logistic Regression is the most "
        "interpretable option and the Decision Tree is the most explainable "
        "visually, but neither matches this model's F1/AUC on the held-out "
        "test set. Given Zepto-style production use favors robust, "
        "well-generalizing models over marginal interpretability gains, "
        f"{best_model} is the recommended choice for deployment."
    )
    return full_classifier_df, reg_df, best_model


# ---------------------------------------------------------------------------
# Task 15: save the best-performing complete pipeline
# ---------------------------------------------------------------------------

def task15_save_best_pipeline(best_model_name: str, fitted_pipelines: dict, tuned_pipeline: Pipeline, X_test):
    log("\n" + "=" * 78)
    log("TASK 15 - Save best-performing complete pipeline via joblib")
    log("=" * 78)

    candidates = dict(fitted_pipelines)
    candidates["Tuned Random Forest (test set)"] = tuned_pipeline

    best_pipeline = candidates[best_model_name]
    joblib.dump(best_pipeline, PIPELINE_PATH)
    log(f"\nSaved best pipeline ('{best_model_name}') -> {PIPELINE_PATH}")

    reloaded = joblib.load(PIPELINE_PATH)
    raw_sample = X_test.iloc[[0]]  # raw, unpreprocessed row straight from the test split
    original_pred = best_pipeline.predict(raw_sample)
    reloaded_pred = reloaded.predict(raw_sample)
    log("\nReload check on raw input row:")
    log(raw_sample.to_string())
    log(f"Original pipeline prediction: {original_pred}")
    log(f"Reloaded pipeline prediction: {reloaded_pred}")
    assert list(original_pred) == list(reloaded_pred), "Reloaded pipeline prediction mismatch!"
    log("Reloaded pipeline predicts identically to the original -> PASS")


def main() -> None:
    df = pd.read_csv(CSV_PATH)  # reads the SAME titanic.csv 01_eda.py produced - no re-load from network

    X_train, X_test, y_train, y_test = task7_split(df)
    task8_note()
    fitted_pipelines, classifier_df = task9_10_train_and_evaluate(X_train, X_test, y_train, y_test)
    task11_imbalance_comparison(X_train, X_test, y_train, y_test)
    tuned_pipeline, best_params, oob_score, tuned_metrics = task12_tune_random_forest(
        X_train, X_test, y_train, y_test
    )
    regression_metrics, _ = task13_regression(df, X_train.index, X_test.index)
    full_classifier_df, reg_df, best_model = task14_final_comparison(
        classifier_df, tuned_metrics, regression_metrics
    )
    task15_save_best_pipeline(best_model, fitted_pipelines, tuned_pipeline, X_test)

    OUTPUT_LOG.write_text("\n".join(LOG_LINES), encoding="utf-8")
    print(f"\nFull modeling text log saved -> {OUTPUT_LOG}")


if __name__ == "__main__":
    main()
