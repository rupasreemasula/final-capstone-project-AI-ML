# Module 2 — Analytics Pipeline (`/analytics`)

Profiles, cleans, and tells the data story for the Titanic dataset (Part A), then builds and
rigorously evaluates a full predictive-modeling pipeline on top of the same data (Part B).

## How to run

```bash
cd analytics
python 01_eda.py        # Part A (Tasks 1-6): load once, profile, clean, EDA, charts
python 02_modeling.py   # Part B (Tasks 7-15): reads the same titanic.csv, models, tunes, saves pipeline
```

`sns.load_dataset('titanic')` is called **exactly once**, inside `01_eda.py`, which immediately
saves the raw result to `titanic.csv`. `02_modeling.py` reads that same committed CSV
(`pd.read_csv("titanic.csv")`) and never touches the network — so the whole module is fully
regenerable offline after the first run.

### Outputs produced

| File | Produced by | Contents |
|---|---|---|
| `titanic.csv` | `01_eda.py` | raw offline-fallback dataset (Task 1) — the one and only load |
| `eda_output.txt` | `01_eda.py` | full text log: profiling, missing %, IQR outliers, survival rates, correlations, standardization check |
| `charts/age_hist_box.png`, `fare_hist_box.png` | `01_eda.py` | Task 3 univariate charts |
| `charts/correlation_heatmap.png` | `01_eda.py` | Task 4 correlation heatmap |
| `charts/story_1..5_*.png` | `01_eda.py` | Task 5 multivariate data-story charts |
| `modeling_output.txt` | `02_modeling.py` | full text log: split, metrics, imbalance comparison, tuning, regression, final table |
| `charts/decision_tree.png` | `02_modeling.py` | Task 9 decision tree visualization |
| `charts/roc_curves.png` | `02_modeling.py` | Task 10 ROC curves, all 3 classifiers |
| `charts/regression_residuals.png` | `02_modeling.py` | Task 13 residual plot |
| `best_pipeline.joblib` | `02_modeling.py` | Task 15 full fitted pipeline (preprocessing + estimator) |

---

## Part A — Profiling, cleaning, and the data story

### Task 1 — Profile

Raw dataset: **891 rows × 15 columns** (`sns.load_dataset('titanic')`).

Missing values (% of 891 rows), for every column that has any:

| Column | % missing |
|---|---|
| `deck` | 77.2166% |
| `age` | 19.8653% |
| `embarked` | 0.2245% |
| `embark_town` | 0.2245% |

### Task 2 — Missing-value handling (threshold rule: <5% drop rows, 5–30% impute, very-high → drop column / "missing" category)

| Column | Missing % | Decision | Justification |
|---|---|---|---|
| `deck` | 77.2166% | Drop column | The missing rate is far above the 30% range, so I decided to drop the column rather than encode "missing" as a separate category. With so much data missing, imputing the values would not be reliable. |
| `age` | 19.8653% | Impute | The missing rate is between 5% and 30%, so I kept the column and imputed the missing values. I used the median age within each (`pclass`, `sex`) group. This is better than using one overall median because age patterns differ between passenger classes and between males and females. |
| `embarked` | 0.2245% | Drop affected rows | The missing rate is below the 5% threshold, so I dropped the affected rows. There were only 2 rows with missing values. |
| `embark_town` | 0.2245% | Drop affected rows | The missing rate is also below the 5% threshold. These are the same 2 rows affected in `embarked`, so they were dropped together. |

In total, 177 `age` values were imputed using the (`pclass`, `sex`) group medians, and 2 rows
were dropped for the missing `embarked`/`embark_town` values.

Cleaned shape after Task 2: **889 rows × 14 columns**, 0 remaining missing values.

### Task 3 — Univariate analysis: `age` and `fare`

IQR outlier rule `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]` (charts: `charts/age_hist_box.png`,
`charts/fare_hist_box.png`):

| Column | IQR bounds | Outliers |
|---|---|---|
| `age` | [-0.25, 57.75] | **32** (of 889) |
| `fare` | [-26.76, 65.66] | **114** (of 889) |

`fare`: **mean = 32.10, median = 14.45, mode = 8.05**.

`fare` is **right-skewed**. The mean is much higher than the median because a few passengers
paid very high fares, which pulls the mean upward. So the ordering is
**mode (8.05) < median (14.45) < mean (32.10)**, which is a typical sign of a right-skewed
distribution. A likely real-world explanation is that passengers had different ticket classes
and ticket prices, so first-class passengers generally paid more while some passengers paid
much lower fares.

**On the 114 `fare` outliers:** I would keep them rather than delete them automatically. The
114 values are statistical outliers according to the IQR rule, but that does not mean they are
mistakes. Very high fares can be genuine values, especially because passengers travelled in
different classes. So unless there is evidence that those values are data-entry errors, I
would keep them because they represent real differences in the dataset.

### Task 4 — Bivariate analysis

Survival rate by **sex**:

| sex | survival rate | n |
|---|---|---|
| male | 0.1889 | 577 |
| female | 0.7404 | 312 |

Survival rate by **pclass**:

| pclass | survival rate | n |
|---|---|---|
| 1 | 0.6262 | 214 |
| 2 | 0.4728 | 184 |
| 3 | 0.2424 | 491 |

Survival rate by **sex & pclass**:

| sex | pclass | survival rate | n |
|---|---|---|---|
| female | 1 | 0.9674 | 92 |
| female | 2 | 0.9211 | 76 |
| female | 3 | 0.5000 | 144 |
| male | 1 | 0.3689 | 122 |
| male | 2 | 0.1574 | 108 |
| male | 3 | 0.1354 | 347 |

Boolean masking used `&` throughout the three breakdowns above (e.g.
`(df["sex"] == s) & (df["pclass"] == p)`); a `|` (OR) example is also included for completeness:
**`pclass==1 OR pclass==2` → survival rate = 0.5553 (n=398)**, i.e. combined 1st/2nd-class
passengers survived at roughly 56%, versus 24% for 3rd class alone.

**Correlation matrix** (exactly `survived, pclass, age, sibsp, parch, fare` — `adult_male` and
`alone` excluded as derived/redundant flags): see `charts/correlation_heatmap.png`.

**Two strongest |correlation| off-diagonal pairs:**
1. **`pclass` ↔ `fare`: r = −0.548.** The correlation is negative because `pclass` is numbered
   backwards (1 = first class, 2 = second, 3 = third). As the `pclass` number increases, the
   passenger is moving to a lower class, and the fare generally decreases — so a higher
   `pclass` number means a lower fare, which produces a negative correlation. It does not mean
   first-class passengers paid less; it is because of how the classes are numerically coded.
2. **`sibsp` ↔ `parch`: r = +0.415.** The positive correlation means passengers with more
   siblings/spouses travelling with them also tended to have more parents/children travelling
   with them. So it suggests that some passengers were travelling as part of larger family
   groups, rather than travelling completely alone.

### Task 5 — Multivariate "data story" (5 charts, ≥ 4 required)

1. **`charts/story_1_survival_by_sex.png`** — Survival rate by sex. The biggest thing I
   noticed is the very large difference between women and men. About 74% of women survived,
   compared with only 18.9% of men. This suggests that sex was a very strong factor in
   survival in this dataset. One possible reason is that women may have been given priority
   during evacuation, but the chart itself only shows the relationship; it does not prove
   the reason.
2. **`charts/story_2_survival_by_pclass.png`** — Survival rate by passenger class. Survival
   dropped quite a lot as passenger class went down, from 62.6% in 1st class to 47.3% in 2nd
   class and 24.2% in 3rd class. I think class was an important factor, although the chart
   doesn't tell us the exact reason for the difference.
3. **`charts/story_3_survival_by_pclass_sex.png`** — Survival rate by class and sex together.
   This chart was more interesting to me because it shows that sex and passenger class both
   mattered, but sex seems to have had a very strong effect. For example, 50% of 3rd-class
   women survived, while only 36.9% of 1st-class men survived. So being in a higher class did
   not automatically mean a person had a higher survival rate. The 3rd-class female survival
   rate of 50% surprised me the most, because I expected passenger class to have a stronger
   effect.
4. **`charts/story_4_age_by_pclass_survival.png`** — Age distribution by class and survival
   (box plot). Age seems to have had some relationship with survival, but the difference does
   not look as strong as the difference by sex or class. Survivors appear to be slightly
   younger across the classes. At the same time, 1st-class passengers were generally older
   than 3rd-class passengers, so some of the age pattern may be connected with passenger
   class.
5. **`charts/story_5_fare_vs_age_by_survival.png`** — Fare vs. age scatter, coloured by
   survival. The fare chart mostly supports what we saw with passenger class. Survivors are
   more common among passengers who paid higher fares, while the low-fare area has many more
   non-survivors. So fare gives us some extra detail, but it is also related to class (the
   correlation between `pclass` and `fare` is -0.548).

**Putting these five charts together:** sex appears to have been the strongest factor in
survival, passenger class also mattered but did not override it, and age had only a weaker
effect that may partly be connected with class rather than standing on its own.

### Task 6 — EDA-stage standardization sanity check (`age`, `fare`)

| | age (before) | fare (before) | age_z (after) | fare_z (after) |
|---|---|---|---|---|
| mean | 29.0654 | 32.0967 | 0.0000 | 0.0000 |
| std | 13.2702 | 49.6975 | 1.0006 | 1.0006 |

Both z-scored columns land at mean ≈ 0 and std ≈ 1, confirming the standardization is correct.

**Why this is kept separate from the modeling pipeline:** the standardization here is only an
EDA check, to see what happens when `age` and `fare` are converted to a common scale. It is not
part of the actual modeling pipeline. For modeling, the scaler must be fitted only on the
training data and then used to transform the test data. If I standardized all 889 rows before
splitting, information from the test set would be used when calculating the scaling values.
This would cause data leakage and could make the model's test results look better than they
really are. Part B therefore performs its own train-only scaling inside the `Pipeline`.

---

## Part B — Predictive modeling

### Task 7 — Stratified train/test split

Overall class balance: **61.6% not-survived / 38.4% survived** — moderately imbalanced.

**Justification for stratification:** I used a stratified split because the dataset has 61.6%
non-survivors and 38.4% survivors. With only 178 rows in the test set, a normal random split
could accidentally give us a very different class balance. For example, if the test set had
around 50% survivors, it would not represent the original dataset properly. The model could
then appear to perform better or worse simply because the test set has an unusual mix of
survivors and non-survivors, which could make the accuracy and F1 score misleading.
Stratification keeps the same approximate 62/38 class ratio in both the training and test sets,
making the evaluation more representative of the original data.

The resulting split confirms this: train = 61.66%/38.34%, test = 61.45%/38.55%.

### Task 8 — Preprocessing (fit on training data only)

Implemented as a single `sklearn.pipeline.Pipeline([('preprocessor', ColumnTransformer(...)),
('classifier', <model>)])` for every model. `ColumnTransformer`:

- **Numeric** (`pclass, age, sibsp, parch, fare`) → `SimpleImputer(strategy="median")` →
  `StandardScaler()`
- **Categorical** (`sex, embarked`) → `SimpleImputer(strategy="most_frequent")` →
  `OneHotEncoder(handle_unknown="ignore")`

**Missing-value choice for modeling:** numeric columns are filled with the **median** and
categorical columns with the **most frequent** value. I chose the median because it is less
affected by extreme values. In Task 3, `fare` was found to be right-skewed, with some
passengers paying very high fares. Those large values can pull the mean upward, while the
median is much less affected. So when there are outliers or a skewed distribution, the median
gives a more stable estimate for filling missing numeric values.

**Feature selection:** the model uses `pclass, age, sibsp, parch, fare, sex, embarked` and
excludes `alive`, `class`, `who`, `adult_male`, `alone`, `embark_town`, and `deck`.

`alive` is excluded because it contains the same information as the target `survived`
(`alive = yes` means `survived = 1`, `alive = no` means `survived = 0`). If I included `alive`
as an input feature, the model would essentially be given the answer it is supposed to predict.
This is a serious case of data leakage, and the accuracy would be expected to be almost 100%
because the model could use `alive` to directly determine the target.

The other excluded columns contain information that is already represented by other features:
`class` gives almost the same information as `pclass`, `alone` can be derived from `sibsp` and
`parch`, and `embark_town` gives the same underlying information as `embarked`. Keeping all of
them would add redundant information without giving the model much that is new, and it makes
the preprocessing and the model less simple to understand. This is the same reasoning the task
itself applies when it identifies `adult_male` and `alone` as derived/redundant flags and
excludes them from the correlation matrix in Task 4. `deck` is excluded because it is ~77%
missing, as already decided in Task 2.

**Why the preprocessing is fit on training data only:** the scaler calculates values such as
the mean and standard deviation from the data it is fitted on. If I fitted it using all 889
rows before splitting, those calculations would include information from the test set, and the
model would indirectly get information about the test data before evaluation. That is data
leakage. The correct process is:

- Training data → fit the scaler → transform the training data
- Test data → use the already-fitted scaler → transform the test data

Calling `pipeline.fit(X_train, y_train)` fits the imputer/scaler/encoder **only** on the
training split; `pipeline.predict(X_test)` then applies those already-fitted transforms in
transform-only mode. This is enforced structurally by the `Pipeline`/`ColumnTransformer`, so no
step is ever fit or refit on test data or on the full pre-split dataset.

### Task 9 & 10 — Three classifiers, full metric suite

Decision tree (fit to `max_depth=5`, rendered in full — no truncation) with labeled
features/classes: `charts/decision_tree.png`. ROC curves for all three: `charts/roc_curves.png`.

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8437 |
| Decision Tree | 0.7654 | 0.7547 | 0.5797 | 0.6557 | 0.7971 |
| Random Forest | 0.8156 | 0.8000 | 0.6957 | 0.7442 | 0.8300 |

Confusion matrices (rows = actual, cols = predicted, `[0=not survived, 1=survived]`):

- Logistic Regression: `[[98, 12], [23, 46]]`
- Decision Tree: `[[97, 13], [29, 40]]`
- Random Forest: `[[98, 12], [21, 48]]`

### Task 11 — Imbalance handling comparison (Logistic Regression)

Training-fold class balance: **439 not-survived (61.66%) / 273 survived (38.34%)**.

| Variant | Precision | Recall | F1 |
|---|---|---|---|
| (a) Baseline / no handling | 0.7931 | 0.6667 | 0.7244 |
| (b) `class_weight='balanced'` | 0.7297 | 0.7826 | 0.7552 |
| (c) SMOTE (training fold only) | 0.7397 | 0.7826 | 0.7606 |

**Conclusion:** I would choose **SMOTE** for this project, because it gives the highest F1
score (0.7606, compared with 0.7244 for the baseline and 0.7552 for `class_weight='balanced'`).
It also has the same recall as the balanced version (0.7826) while having slightly better
precision (0.7397 vs 0.7297).

The trade-off across all three variants is the same: after rebalancing, recall increases from
0.6667 to 0.7826 while precision falls from 0.7931 to around 0.73. In simple words, the
rebalanced models became better at finding passengers who actually survived, but they also
made more incorrect survival predictions — the model catches more survivors, but some of the
people it predicts as survivors did not actually survive.

It is also worth noting that this dataset is not extremely imbalanced (about 62% non-survivors
and 38% survivors). Because the imbalance is moderate rather than extreme, rebalancing does not
completely change the model. It mainly shifts the model toward finding more survivors, so the
benefit is noticeable but it comes with a trade-off.

### Task 12 — Hyperparameter tuning (Random Forest)

`GridSearchCV` over `n_estimators ∈ {100, 200}`, `max_depth ∈ {None, 5, 10}`,
`max_features ∈ {"sqrt", "log2"}`, 5-fold CV, scored on F1.

- **Best params:** `max_depth=5, max_features='sqrt', n_estimators=100`
- **Best CV F1:** 0.7459
- **Out-of-bag (OOB) score** (refit with `RandomForestClassifier(oob_score=True, ...)` using
  the best params, trained on the full training split): **0.8272**
- Tuned model's held-out test performance: accuracy 0.8156, precision **0.8750**, recall
  0.6087, F1 0.7179, AUC 0.8431 — confusion matrix `[[104, 6], [27, 42]]`

**Observation on the tuning result:** tuning does not guarantee that every metric will improve.
The tuned Random Forest became much more selective when predicting survival — precision
increased from 0.8000 to 0.8750, but recall decreased from 0.6957 to 0.6087, and F1 decreased
from 0.7442 to 0.7179. So the tuned model makes fewer incorrect positive predictions, which
gives it higher precision, but it also misses more actual survivors, which lowers recall.

This happens because `GridSearchCV` selects the parameters that score best *on average across
the 5 cross-validation folds of the training data*, which does not guarantee that the single
held-out test split will favour the same trade-off. Both results are reported in Task 14's
comparison table below.

### Task 13 — Regression side-task: predicting `fare`

Multivariate linear regression predicting `fare` from `pclass, age, sibsp, parch, survived, sex,
embarked` (same train/test row split as classification, reused for consistency), preprocessed
with the same fit-on-train-only `ColumnTransformer` pattern. `survived` is included here as a
*predictor* (not leakage — the target this time is `fare`), while the same excluded-column
list from Task 8 (`alive, class, who, adult_male, alone, embark_town, deck`) is applied for the
same redundancy reasons.

| Metric | Value |
|---|---|
| MAE | 19.6063 |
| RMSE | 43.7043 |
| R² | 0.3841 |
| Adjusted R² | 0.3589 (n=179 test rows, p=7 original predictor columns) |

Residual plot: `charts/regression_residuals.png`.

**Heteroscedasticity conclusion:** yes, heteroscedasticity is present. The correlation between
the absolute residuals and predicted fare is **0.4210**, which indicates that the size of the
errors tends to increase as predicted fare increases. The residual plot shows a fan/cone-shaped
pattern, where the residuals are more tightly spread at lower predicted fares and become more
spread out at higher predicted fares.

This connects back to Task 3, where `fare` was found to be right-skewed with a long tail of
expensive fares. A straight-line regression model can struggle with these expensive tickets
because the high values are much farther away from most of the data, which leads to larger
prediction errors for higher fares — consistent with the heteroscedasticity seen here.

**On the R² value:** an R² of 0.3841 means the model explains about 38% of the variation in
`fare` using the variables in the regression. The model captures some useful relationship, but
a large amount of the variation in fare is still unexplained. So this linear model has limited
predictive power for fare — it can explain part of the pattern, but it does not predict
individual fares very accurately, especially considering the high RMSE of 43.70.

### Task 14 — Final model comparison table & recommendation

**Classification metrics** (one scale):

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8437 |
| Decision Tree | 0.7654 | 0.7547 | 0.5797 | 0.6557 | 0.7971 |
| Random Forest | 0.8156 | 0.8000 | 0.6957 | **0.7442** | 0.8300 |
| Tuned Random Forest | 0.8156 | **0.8750** | 0.6087 | 0.7179 | 0.8431 |

**Regression metrics** (separate scale — not comparable to the table above):

| Model | MAE | RMSE | R² | Adjusted R² |
|---|---|---|---|---|
| Linear Regression (predicting fare) | 19.6063 | 43.7043 | 0.3841 | 0.3589 |

**Recommendation:** there isn't one objectively correct choice here, because the table shows a
trade-off between performance and interpretability. For this project I would deploy the
**Random Forest**. The main reason is that it has the highest F1 score (**0.7442**) and the
highest accuracy (**0.8156**) of all the models in the table, which indicates it gives a good
balance between precision (0.8000) and recall (0.6957) while also achieving the best overall
accuracy. Logistic Regression has a slightly higher AUC (0.8437) and is easier to explain, so
it would also be a reasonable choice if interpretability were the main priority.

The Decision Tree is the weakest option: it has the lowest F1 (0.6557) and the lowest recall
(0.5797), meaning it misses more of the passengers who actually survived than the other models,
and its accuracy (0.7654) and AUC (0.7971) are lower as well.

### Task 15 — Saved pipeline

`best_pipeline.joblib` contains the complete **fitted `sklearn.Pipeline`** for the chosen model
(Random Forest) — the `ColumnTransformer` (imputers, scaler, one-hot encoder) *and* the
`RandomForestClassifier` bundled together, saved via `joblib.dump(full_pipeline, ...)`. It is
**not** the bare estimator: `02_modeling.py` reloads it with `joblib.load(...)` and confirms it
predicts identically to the original on a **raw, unpreprocessed** test row (e.g.
`pclass=3, age=24.0, sibsp=2, parch=0, fare=24.15, sex=male, embarked=S` → same prediction
before and after reload), proving the saved artifact is usable end-to-end on new raw input.

---

## Acceptance criteria checklist

- [x] Missing-value % reported per affected column, strategy cites the percentage threshold rule
- [x] `titanic.csv` committed (raw fallback, one load only, loadable via `pd.read_csv`)
- [x] IQR outlier counts for `age` (32) and `fare` (114); skew conclusion compares mean/median/mode
- [x] All 3 bivariate breakdowns reported numerically; 6×6 correlation matrix (exact columns, `adult_male`/`alone` excluded); top-2 correlations named and interpreted
- [x] ≥ 4 multivariate charts, each with written interpretation (5 provided); before/after standardization check shown for both columns
- [x] Stratified split before any preprocessing, justified via class balance
- [x] All preprocessing fit on train only, transform-only on test (structurally enforced via Pipeline/ColumnTransformer)
- [x] All 3 classifiers on identical split; decision tree visualized with labeled features/classes; full metric suite per classifier
- [x] 3-way imbalance comparison with written conclusion; SMOTE applied to training fold only
- [x] GridSearchCV best params + OOB score reported for `RandomForestClassifier(oob_score=True, ...)`
- [x] Regression sub-task reports MAE/RMSE/R²/Adjusted R² + explicit heteroscedasticity conclusion
- [x] Final comparison table (classifier metrics and regression metrics as separate groups) + 3–5 sentence recommendation
- [x] Complete fitted pipeline (preprocessing + estimator) saved via `joblib.dump`, reloadable, usable on raw new data
