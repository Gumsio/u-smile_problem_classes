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

## References

- [Column pruning issue #4](https://github.com/Gumsio/u-smile_problem_classes/issues/4)
- [MissForest algorithm paper](https://arxiv.org/abs/1105.0828)
