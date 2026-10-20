"""MissForest-style iterative random-forest imputation for mixed data.

Algorithm: https://arxiv.org/abs/1105.0828
Models are fitted only on observed target values. Initial mean/mode guesses
are used as predictors, then replaced with forest predictions iteratively.
"""

from collections.abc import Iterable
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def clean_missing_values(
    df: pd.DataFrame,
    *,
    categorical_columns: Iterable[str] = (),
    exclude_columns: Iterable[str] = ("cid", "cid.1"),
    max_iter: int = 5,
    n_estimators: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """Impute supported columns and return a copy, preserving observed values.

    Run after column/date cleaning and column pruning, before final dtype conversion. Numeric
    columns use regression; strings, categories, booleans and binary numeric
    columns use classification. List other numeric category codes explicitly
    in categorical_columns. Configured categories absent after pruning are
    ignored with a warning. Dates, identifiers, unsupported objects, entirely
    missing columns and columns containing infinity are excluded from models.
    Mostly numeric text columns need normalization first and are also skipped.
    Excluded missing values remain missing and produce a warning.

    At least two observed target values and another usable column are needed
    to fit a forest. A column with one distinct observed value is filled with
    that value. Iteration stops when numeric and categorical changes both stop
    improving; the previous iteration is retained in that case. Defaults are
    deterministic. Numeric imputations may be fractional, including counts.

    This function fits on the supplied dataset. For predictive modelling, fit
    imputation on training data only; do not run this over a train/test mixture.
    No CHECK strings or other placeholders are silently treated as missing.
    """
    if not df.columns.is_unique:
        raise ValueError("Column names must be unique before imputation.")
    for name, value in (("max_iter", max_iter), ("n_estimators", n_estimators)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if isinstance(categorical_columns, str) or isinstance(exclude_columns, str):
        raise TypeError("Column settings must be collections of names, not strings.")
    explicit_categories = set(categorical_columns)
    absent = explicit_categories.difference(df.columns)
    if absent:
        warnings.warn(
            f"Ignoring categorical columns absent from input (possibly pruned): {sorted(absent)}",
            UserWarning,
            stacklevel=2,
        )
        explicit_categories.intersection_update(df.columns)
    excluded = set(exclude_columns)
    result = df.copy(deep=True)
    if df.empty or not df.isna().any().any():
        return result

    encoded = {}
    levels = {}
    skipped = {}
    constant_fills = {}
    for column in df.columns:
        series = df[column]
        observed = series.dropna()
        reason = None
        if column in excluded:
            reason = "excluded column"
        elif observed.empty:
            reason = "no observed values"
        elif is_numeric_dtype(series) and not np.isfinite(observed.to_numpy(dtype=float)).all():
            reason = "contains infinity"
        elif (
            column not in explicit_categories
            and not is_numeric_dtype(series)
            and observed.map(lambda value: isinstance(value, str)).all()
            and pd.to_numeric(observed, errors="coerce").notna().mean() >= 0.5
        ):
            reason = "mostly numeric text; normalize values and dtype first"
        else:
            categorical = (
                column in explicit_categories
                or is_bool_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
                or (is_numeric_dtype(series) and observed.nunique() <= 2)
                or observed.map(lambda value: isinstance(value, str)).all()
            )
            if categorical:
                codes, categories = pd.factorize(series, sort=False)
                values = codes.astype(float)
                values[codes < 0] = np.nan
                encoded[column] = values
                levels[column] = categories
            elif is_numeric_dtype(series):
                encoded[column] = series.to_numpy(dtype=float, na_value=np.nan)
            else:
                reason = "unsupported type (for example, a date)"
        if reason:
            if series.isna().any():
                skipped[column] = reason
            continue
        if series.isna().any() and observed.nunique() == 1:
            constant_fills[column] = observed.iloc[0]

    # Use a positional index internally, including when input row labels repeat.
    matrix = pd.DataFrame(encoded, index=pd.RangeIndex(len(df)), dtype=float)
    missing = matrix.isna()
    targets = [c for c in matrix if missing[c].any() and c not in constant_fills]
    usable_targets = []
    for column in targets:
        if matrix.shape[1] < 2 or (~missing[column]).sum() < 2:
            skipped[column] = "insufficient observed values or predictors"
        else:
            usable_targets.append(column)
    targets = sorted(usable_targets, key=lambda column: int(missing[column].sum()))
    for column in matrix:
        observed = matrix.loc[~missing[column], column]
        initial = observed.mode().iloc[0] if column in levels else observed.mean()
        matrix[column] = matrix[column].fillna(initial)

    previous_numeric_change = previous_category_change = np.inf
    for _ in range(max_iter if targets else 0):
        previous = matrix.copy()
        for column in targets:
            target_missing = missing[column].to_numpy()
            predictors = matrix.drop(columns=column).to_numpy()
            categorical = column in levels
            forest_type = RandomForestClassifier if categorical else RandomForestRegressor
            forest = forest_type(
                n_estimators=n_estimators,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=1,
            )
            forest.fit(predictors[~target_missing], matrix.loc[~target_missing, column])
            matrix.loc[target_missing, column] = forest.predict(predictors[target_missing])
        numeric = [c for c in targets if c not in levels]
        categorical = [c for c in targets if c in levels]
        numeric_change = 0.0
        if numeric:
            delta = (matrix[numeric] - previous[numeric]).to_numpy()
            scale = np.square(matrix[numeric].to_numpy()).sum()
            numeric_change = np.square(delta).sum() / max(scale, np.finfo(float).eps)
        category_change = 0.0
        if categorical:
            changed = (matrix[categorical] != previous[categorical]) & missing[categorical]
            category_change = changed.to_numpy().sum() / missing[categorical].to_numpy().sum()
        if numeric_change >= previous_numeric_change and category_change >= previous_category_change:
            matrix = previous
            break
        previous_numeric_change = numeric_change
        previous_category_change = category_change
        if numeric_change == 0 and category_change == 0:
            break

    for column, value in constant_fills.items():
        result[column] = result[column].fillna(value)
    for column in targets:
        positions = np.flatnonzero(missing[column].to_numpy())
        if column in levels:
            predictions = levels[column].take(matrix.loc[positions, column].to_numpy(dtype=int))
            updated = result[column].copy()
        else:
            predictions = matrix.loc[positions, column].to_numpy()
            # Regression may produce fractions even when observed values are integers.
            updated = result[column].astype(float)
        updated.iloc[positions] = predictions
        result[column] = updated
    if skipped:
        details = "; ".join(f"{column}: {reason}" for column, reason in skipped.items())
        warnings.warn(f"MissForest left missing values unchanged in {details}", UserWarning, stacklevel=2)
    return result
