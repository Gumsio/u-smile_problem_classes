# Work Report — Anka Śmietanka

This report describes the environment setup, documentation updates, pipeline changes, and checks completed during this work session for the U-Smile project. The work focused on cleaning data before analysis or modelling.

## 1. Python environment

Created a local `.venv` environment using Python 3.14.4 and installed the dependencies listed in `requirements.txt`: pandas, openpyxl, and Jupyter. Checked that the packages imported correctly and that pip reported no broken requirements.

Later, added NumPy and scikit-learn to the project requirements for random-forest imputation.

Ran the original cleaning script and confirmed that it saved the CSV to `data/processed/cardiac_patient_data_clean.csv`. The original cleaned output contained 393 rows and 491 columns. The CSV is saved in the processed-data directory, rather than beside the script.

## 2. README repairs and team workflow

Fixed broken Markdown in the README, including mismatched bold and code markers, list indentation, and code-block indentation. Replaced a Google search link with a relative link to `src/clean_data.py`.

Updated the instructions for adding cleaning steps. The README now explains that:

- `src/clean_data.py` runs the pipeline, loads data, and saves the result.
- Cleaning functions belong in `src/cleaning_steps/`.
- Similar transformations should stay together in a file.
- New files, such as `type_casting.py` or `imputation.py`, can be added when needed. A separate file is not required for every function.
- Team members should work through issues, branches, and reviewed pull requests, and coordinate when editing the same file.

Simplified the instructions after review so they use shorter steps and clearer language.

## 3. Modular cleaning pipeline

Moved the original cleaning functions out of `clean_data.py` into three modules:

| Module | Original function | Purpose |
| --- | --- | --- |
| `text_normalization.py` | `clean_column_names()` | Strip column-name whitespace and remove the known duplicate admission-date column |
| `date_formatting.py` | `clean_dates()` | Parse dates, correct a known invalid date, and calculate observation time |
| `missing_values.py` | `clean_missing_values()` | Initially a placeholder; later replaced with imputation logic |

Kept the original cleaning behavior during this refactor and verified that the output matched the earlier pipeline exactly. Imports support both command-line execution and notebook use.

A narrower seven-module layout was tried and then reverted after review. The final approach keeps related transformations together and explains in the README how to add modules when needed.

## 4. Column pruning

Implemented `column_pruning.py` from GitHub issue #4. Split the logic into separate functions so each rule can be understood and changed independently:

- `drop_identifier_columns()`: Remove explicitly named identifiers, initially `cid` and `cid.1`.
- `drop_missing_columns()`: Remove columns whose missing-value fraction exceeds the threshold.
- `drop_low_variance_columns()`: Remove columns where one value dominates the nonmissing entries.
- `drop_high_cardinality_columns()`: Remove text columns with too many distinct values.
- `drop_correlated_columns()`: Remove later numeric columns highly correlated with an already retained column.

`prune_columns()` runs these functions in order. Settings are defined there and passed to the helper functions, which have no separate defaults.

The final defaults used during verification were:

| Setting | Value | Rule |
| --- | --- | --- |
| Missing values | 25% | Drop when the missing fraction is greater than 25% |
| Dominant value | 95% | Drop when one value accounts for at least 95% of nonmissing entries |
| Unique text values | 30% | Drop when distinct values exceed 30% of nonmissing entries |
| Correlation | 0.90 | Drop when absolute Pearson correlation is at least 0.90 |

The missing-value setting was changed from the issue's original 30% rule to the current configurable 25% default. Correlation uses at least three shared observations and keeps the first retained column in input order. The unique-value rule applies only to text columns, including string-valued categories; numeric and date columns are excluded from that rule.

Pruning preserves rows and does not modify the input DataFrame. With these settings, it reduced the dataset from 491 to 293 columns before date conversion. Target-based feature importance was not added because the prediction target has not been selected.

## 5. Date-derived columns

Added `calculate_days_to_end()` in `date_formatting.py` to calculate:

- Admission-to-end days: end date minus admission date.
- Implantation-to-end days: end date minus implantation date.

The source column is `data zgonu lub konca obserwacji`. It can represent death or the end of follow-up, so these durations are not necessarily time to death.

Compared the new admission duration with the existing `Time of observation` column. They matched exactly across all 393 rows. The function therefore reuses `Time of observation` when values and missing entries match, rather than adding a duplicate `days_admission_to_end` column. If they differ, it retains the calculated column.

The function places the duration columns first and removes the three source date columns. Missing or invalid dates produce missing durations. Negative durations are preserved so date-order problems remain visible.

## 6. Missing-value imputation

Replaced the placeholder in `missing_values.py` with MissForest-style iterative imputation using scikit-learn random forests.

The function starts with mean or mode guesses for missing predictors. It then trains a forest for each incomplete column using only rows where that target value was observed, updates the missing entries, and repeats until the stopping rule or iteration limit is reached.

- Numeric columns use regression.
- Text, categorical, boolean, and binary numeric columns use classification.
- Other numeric category codes can be listed explicitly in `categorical_columns`.
- Identifiers are excluded by default.
- Existing values are preserved, and the input DataFrame is not changed.
- A fixed random seed makes runs repeatable.

Defaults are five iterations, 50 trees per forest, and random seed 42. The function warns and continues if pruning has removed a configured categorical column.

The Python MissForest package was investigated, but the implementation uses scikit-learn directly because of dependency-declaration and model-fitting concerns found in that package.

### Limitations identified

Dates, unsupported objects, entirely missing columns, and columns containing infinity cannot be imputed by this implementation and are left unchanged with a warning when they contain missing values. Insufficient predictors or observed values can also prevent imputation.

`ALDOpg/ml` contained mostly numeric text plus three other text entries. It was initially skipped rather than treated as a categorical measurement. The current pruning rules remove that column before imputation.

Strings such as `CHECK` are not automatically treated as missing. Numeric predictions may be fractional, so counts may need separate rules. Numeric category codes with more than two values also need explicit classification settings.

For future predictive modelling, imputation must be fitted using training data only. This cleaning function fits on all rows supplied to it; it is not a reusable train/test imputer.

## 7. Final pipeline order and results

Integrated the date and imputation changes so the pipeline now runs:

1. Clean column names.
2. Clean and standardize dates.
3. Prune columns.
4. Calculate day counts and remove source dates.
5. Impute missing values.
6. Convert pandas data types.
7. Save the processed CSV when the script is run.

Moving day-count calculation before imputation was necessary: otherwise, the newly created duration columns would still contain missing values after imputation had finished. Keeping it after pruning preserves the requested duration columns from pruning.

A full run with the default imputation settings verified:

| Check | Result |
| --- | --- |
| Rows | 393 |
| Final columns | 291 |
| Missing values before imputation | 7,950 |
| Missing values after imputation | 0 |
| Existing observed values | Preserved |
| Imputation warnings | None |
| Runtime in the local environment | About 95 seconds |

These results confirm that all missing entries in the current pipeline output can be filled. They do not establish the accuracy of the estimated values.

## 8. Verification and Git work

Used automated checks and full-data runs to verify threshold boundaries, input preservation, repeatability, category handling, missing dates, date arithmetic, and the interaction between pruning and imputation. Ten imputation-related tests passed after the final pipeline-order change. Tests were subsequently removed from the feature branches' final pull-request changes at the team's request; they remain in earlier commit history.

Worked on separate branches for column pruning, dates, imputation, and documentation. Renamed remote branches to `feat-weird-highlighting-across-the-spreadsheet` and `feat-imputation`, preserving their commits. Merged updated `main` changes into the imputation branch and pushed the requested commits. Some README and initial refactoring updates were pushed directly to `main` when explicitly requested.

Created the `documentation` branch and this report at `documentation/ankasmietanka_report.md`.

## 9. Initial feature-selection results (historical run)

Compared logistic regression and a decision tree to select **20 original
attributes per method** associated with recorded death (`zgon`). This section
summarizes the run `20261028T133247_937065Z` displayed in
`notebooks/feature_selection_comparison.ipynb`.

### Evaluation method

The dataset contained 393 patients. A stratified split assigned 314 patients to
training and 79 to the final test set. Five-fold cross-validation was performed
within the training set. Pruning, imputation, encoding, scaling, and feature
selection were fitted separately on each fold's training rows. Each method
then selected its final 20 attributes using the full training portion.

**ROC AUC is the primary comparison metric.** The values below are means and
sample standard deviations across the five validation folds, not final test
scores or confidence intervals.

| Method | Selected attributes | Mean validation ROC AUC | Standard deviation |
| --- | --- | --- | --- |
| Decision tree | 20 | 0.657 (approximately 0.66) | 0.081 |
| Logistic regression | 20 | 0.589 (approximately 0.59) | 0.087 |

The decision tree had the higher mean validation ROC AUC in this run, by about
0.068. Both mean scores were above the random-ranking reference of 0.5, but
performance was limited and varied across folds. These results alone do not
establish a statistically significant difference between the methods or
clinical usefulness. Final test results are reported separately in the notebook
and are not used to choose features or tune the models.

### Shared attributes

The final selections shared only **three attributes**:

- `Ant_DBP`
- `B_NT - proBNP`
- `B_PLT`

The overlap is 3 out of 20 attributes in either list (15%). Across both lists,
there are 37 distinct attributes. The small overlap shows that the two methods
produced substantially different selections. Logistic regression uses absolute
coefficients, whereas the decision tree uses impurity reduction, so they need
not rank the same attributes highly. The overlap alone does not measure
selection stability; that requires checking selections across folds, available
in `fold_selected_features.csv` and the notebook.

The shared attributes are candidates for further investigation. Their presence
in both lists does not prove that they cause death, and the other selected
attributes should not be dismissed solely because they appear in only one list.

### Interpretation and next steps

This is an exploratory analysis of associations with recorded death. Target
coding assumes `1` means death and `0` means no recorded death; the coding and
measurement timing still need confirmation from the data documentation.
Known outcome-derived dates and durations were excluded, but further outcome
proxies may remain. Follow-up length varies, so this is not a prediction of
death within a fixed period.

For this experiment, both methods retain exactly 20 attributes. Some tree fits
inside validation folds assigned zero importance to enough attributes that
alphabetical tie-breaking was needed to reach 20. Neither final training-set
selection contained a zero-importance attribute. A selected name should still
be assessed together with its score and frequency of selection across folds.

Next, review the data definitions and timing, inspect selection frequency, and
investigate why the two methods choose different attributes. Keep ROC AUC as
the main comparison measure and retain the other metrics as supporting results.
The present results do not identify the main cause of death.

## 10. Expanded feature-selection comparison — 1 November 2026

### Methods and code organization

Expanded the comparison from the initial baselines to ten approaches:
logistic regression, decision tree, ANOVA, mutual information, random-forest
importance, RFE, Boruta, Lasso, Elastic Net, and XGBoost.

The method modules are grouped under `src/feature_selection/`:

- `baselines/`: logistic regression and decision tree.
- `statistics/`: ANOVA and mutual information.
- `model_based/`: random forest, RFE, Boruta, Lasso, Elastic Net, and XGBoost.

Shared preprocessing, capped ranking, reporting and evaluation helpers remain
in the package root. `src/select_features.py` runs the unified comparison;
`src/run_pipeline.py` runs cleaning and then feature selection. The latter
was renamed from `run_data_science.py` for clarity.

Merged the numbered feature-selection branches into local `feature-selection`.
The baseline branch had no unique commits left to merge. The Boruta branch
supplied the baseline runner and its comparison notebook. Pending work on
Lasso, Elastic Net and XGBoost was preserved while resolving dependency and
ignore-file conflicts. The original ANOVA/MI/RF/RFE exploratory notebook,
`test_feature_selection_techniques.ipynb`, was restored byte-for-byte to its
original version. All validated evaluations now appear together in
`feature_selection_comparison.ipynb`.

### New selection methods

**Boruta** uses random forests and shuffled shadow features. Added BorutaPy
and retained its confirmed, tentative and rejected decisions alongside the
capped comparison ranking. Initial settings are 128 trees and at most 50
iterations. An original attribute is labelled confirmed when any encoded
category is confirmed; this grouping is a reporting convention, not a separate
joint test. In the latest final fit, Boruta confirmed **17 original attributes**,
left **8 tentative**, and rejected **262**. The comparison uses the 17 confirmed
attributes plus the three highest-ranked tentative attributes. The notebook
explicitly distinguishes confirmed relevance from inclusion in the top 20.

**Lasso and Elastic Net** use penalized logistic regression because death is a
binary outcome: L1 regularization for Lasso-style selection and mixed L1/L2 for
Elastic Net. Initial settings are `C=0.1`, with `l1_ratio=0.5` for Elastic Net.
An early exploratory whole-cleaned-data fit produced 48 and 101 nonzero
attributes respectively. These were candidate counts, not the final chosen
sets; selection is now capped at 20 positive-score original attributes.

**XGBoost** ranks encoded inputs by gain and groups their importance under
original attribute names. Added the official CPU-only XGBoost dependency.
Initial settings are 200 trees, depth 3 and learning rate 0.05.

**ANOVA, mutual information, random-forest importance and RFE** were brought
into the shared evaluation rather than scoring the original notebook's
whole-dataset rankings. RFE eliminates 10% of encoded inputs per iteration;
its best encoded rank is used for each original attribute.

### Common evaluation procedure

The latest run is `20261101T134938_572565Z`. It uses the same 393 patients,
314 training patients, 79 held-out test patients and five stratified validation
folds within the training portion for every approach. In every fold, pruning,
imputation, encoding, scaling, selection and classifier fitting use only the
fold's training rows. Selection starts from raw data instead of the CSV already
imputed on the full dataset. Numeric predictors use training-fitted median
imputation, and categorical predictors use the most frequent training value.

Each method chooses **at most 20 original attributes**. Encoded categories are
grouped under their original columns. Zero-score attributes are not added to
fill a list. If no attributes survive, a constant training-prevalence model
provides a defined ROC AUC baseline. The latest final selections all contain 20
attributes.

ROC AUC is the primary metric. Methods are ordered by mean validation ROC AUC,
with fold standard deviations shown separately. Test ROC AUC is reported but
not used to choose features or settings. ANOVA, mutual information and RFE use
logistic regression as their evaluation classifier. Other methods use their
corresponding classifier; Boruta uses a random forest. The comparison therefore
measures complete selection/classifier approaches, not selector effects alone.

### Latest results

| Approach | Mean validation ROC AUC | Fold std. deviation | Held-out test ROC AUC |
| --- | --- | --- | --- |
| anova | 0.777 | 0.041 | 0.742 |
| lasso | 0.772 | 0.061 | 0.762 |
| boruta | 0.768 | 0.027 | 0.782 |
| random_forest | 0.764 | 0.022 | 0.765 |
| mutual_information | 0.758 | 0.048 | 0.712 |
| elastic_net | 0.748 | 0.051 | 0.752 |
| xgboost | 0.736 | 0.051 | 0.701 |
| rfe | 0.692 | 0.057 | 0.806 |
| decision_tree | 0.655 | 0.069 | 0.631 |
| logistic_regression | 0.589 | 0.087 | 0.737 |

ANOVA has the highest mean validation ROC AUC in this fixed-setting run, followed
by Lasso and Boruta. RFE has a higher held-out test ROC AUC than its validation
mean; these values come from different patient subsets and fitted selections.
A high test score on one split does not establish reliable superiority. Fold
standard deviations are not confidence intervals, and this comparison does not
establish statistically significant differences or causes of death.

The earlier standalone Lasso/Elastic Net/XGBoost scores used five folds over
all patients. They are not directly interchangeable with the unified run,
which validates inside the 80% training portion. The decision-tree validation
score also changed slightly when zero-score padding was removed. Section 9
records the earlier two-method experiment and is retained as history.

### Readable attribute summaries and saved files

Simplified the comparison notebook to show only the **five highest-ranked
chosen attributes per method**, rather than printing every complete selection.
The full selections remain available in one `top20_attributes_by_method.csv`:
rows are ranks 1–20, columns are method names, and empty cells represent smaller
sets. Added `attribute_selection_frequency.csv` to count how many methods choose
each attribute across their complete final selections, not merely the five
shown. Cross-validation frequency is saved separately in
`fold_selection_frequency.csv`.

There are **91 distinct attributes** across the ten final selected lists.
The most frequently chosen are:

| Attribute | Methods choosing it | Share of methods |
| --- | --- | --- |
| `B_NT - proBNP` | 10 / 10 | 100% |
| `B_Kreatynina` | 8 / 10 | 80% |
| `6mwt_,Dystans` | 7 / 10 | 70% |
| `E_diastarea` | 7 / 10 | 70% |
| `LVEF` | 7 / 10 | 70% |

Agreement across methods is descriptive and does not prove clinical relevance.
Each UTC-named run folder also contains full rankings, selected names in JSON,
per-fold metrics, validation summaries, held-out metrics, fold-selected names,
pairwise overlap counts and configuration metadata. Metadata records data hash,
software versions, settings, random seed and Git state. Generated run folders
are ignored by Git and contain aggregate results rather than patient records.

### Checks and remaining work

Verified all 50 validation results (ten approaches times five folds), the
at-most-20 rule, exclusion of known outcome-derived attributes, notebook
execution and schema, package imports and grouped method paths. Verified the
combined 20-by-10 attribute CSV against the selected-name JSON and independently
checked every cross-method frequency count. Tested the no-feature fallback,
input preservation and repeatability. The exploratory notebook was checked
against its original committed bytes.

Hyperparameter tuning, including possible Optuna studies, was discussed but
**has not been implemented**. Current scores use fixed initial settings. Any
future tuning must keep preprocessing and selection within training folds and
clearly separate tuning scores from evaluation scores. Numeric category-code
meanings, target coding and predictor measurement timing still require review.

A shared TODO file was discussed but has not yet been created. No feature
branches were deleted during consolidation.

## References

- [MissForest algorithm paper](https://arxiv.org/abs/1105.0828)
