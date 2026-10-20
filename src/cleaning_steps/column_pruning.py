"""Target-independent column pruning for GitHub issue #4."""

from collections.abc import Iterable

import pandas as pd


def prune_columns(
    df: pd.DataFrame,
    *,
    identifier_columns: Iterable[str] = ("cid", "cid.1"),
    missing_threshold: float = 0.25,
    correlation_threshold: float = 0.90,
    dominant_value_threshold: float = 0.95,
    unique_threshold: float = 0.30,
) -> pd.DataFrame:
    """Return a pruned copy without changing rows or the input DataFrame.

    Set defaults here or pass overrides when calling this function. Helper
    functions require explicit settings and have no separate defaults.

    Run after column-name and date cleaning, before future imputation rules
    hide missing values. Identifier names are explicit: unique measurements
    are not automatically treated as identifiers. Absent identifiers are ignored.

    Drop identifiers, then columns with strictly more than ``missing_threshold``
    missing values. For low variation, drop columns whose most common value
    represents at least ``dominant_value_threshold`` of nonmissing entries.
    This scale-independent rule applies to both numeric and categorical data.

    Drop columns with a distinct/nonmissing ratio above ``unique_threshold``.
    This applies only to columns containing text values.
    A threshold of 1 disables this rule.

    Finally, compare numeric columns using absolute pairwise Pearson correlation
    (at least three shared observations). Keep the first column in input order
    and drop later columns correlated with a retained column at or above the
    threshold. Undefined correlations do not cause removal. No target-based
    feature importance is calculated.

    Empty inputs retain their schema except for explicitly listed identifiers.
    """
    # Validate all options even when an earlier step leaves an empty frame.
    _validate_columns(df)
    _validate_threshold("missing_threshold", missing_threshold)
    _validate_threshold("correlation_threshold", correlation_threshold, positive=True)
    _validate_threshold(
        "dominant_value_threshold", dominant_value_threshold, positive=True
    )
    _validate_threshold("unique_threshold", unique_threshold)
    result = drop_identifier_columns(df, identifier_columns=identifier_columns)
    result = drop_missing_columns(result, missing_threshold=missing_threshold)
    result = drop_low_variance_columns(
        result, dominant_value_threshold=dominant_value_threshold
    )
    result = drop_high_cardinality_columns(
        result, unique_threshold=unique_threshold
    )
    return drop_correlated_columns(result, correlation_threshold=correlation_threshold)


def _validate_columns(df: pd.DataFrame) -> None:
    if not df.columns.is_unique:
        raise ValueError("Column names must be unique before pruning.")


def _validate_threshold(name: str, value: float, *, positive: bool = False) -> None:
    if not 0 <= value <= 1 or (positive and value == 0):
        interval = "greater than 0 and at most 1" if positive else "between 0 and 1"
        raise ValueError(f"{name} must be {interval}.")


def drop_identifier_columns(
    df: pd.DataFrame, *, identifier_columns: Iterable[str]
) -> pd.DataFrame:
    """Return a copy without explicitly named identifiers; ignore absent names."""
    _validate_columns(df)
    if isinstance(identifier_columns, str):
        raise TypeError("Pass identifier_columns as a collection, not a string.")
    return df.drop(columns=list(identifier_columns), errors="ignore").copy()


def drop_missing_columns(
    df: pd.DataFrame, *, missing_threshold: float
) -> pd.DataFrame:
    """Drop columns with strictly more than the allowed fraction of missing values."""
    _validate_columns(df)
    _validate_threshold("missing_threshold", missing_threshold)
    if df.empty:
        return df.copy()
    return df.loc[:, df.isna().mean() <= missing_threshold].copy()


def drop_low_variance_columns(
    df: pd.DataFrame, *, dominant_value_threshold: float
) -> pd.DataFrame:
    """Drop all-missing columns or those dominated by one nonmissing value.

    Dominance is measured among nonmissing entries, for numeric and categorical
    columns alike. Zero-row frames retain their schema.
    """
    _validate_columns(df)
    _validate_threshold(
        "dominant_value_threshold", dominant_value_threshold, positive=True
    )
    if df.empty:
        return df.copy()
    low_variation = []
    for column in df.columns:
        frequencies = df[column].value_counts(normalize=True, dropna=True)
        if frequencies.empty or frequencies.iloc[0] >= dominant_value_threshold:
            low_variation.append(column)
    return df.drop(columns=low_variation).copy()


def drop_correlated_columns(
    df: pd.DataFrame, *, correlation_threshold: float
) -> pd.DataFrame:
    """Drop later numeric columns correlated with a retained numeric column.

    Use absolute Pearson correlation with at least three shared observations.
    Keep input order; undefined correlations do not cause removal.
    """
    _validate_columns(df)
    _validate_threshold("correlation_threshold", correlation_threshold, positive=True)
    correlations = (
        df.select_dtypes(include="number").corr(method="pearson", min_periods=3).abs()
    )
    retained = []
    correlated = []
    for column in correlations.columns:
        if any(
            correlations.loc[previous, column] >= correlation_threshold
            for previous in retained
        ):
            correlated.append(column)
        else:
            retained.append(column)
    return df.drop(columns=correlated).copy()


def drop_high_cardinality_columns(
    df: pd.DataFrame, *, unique_threshold: float
) -> pd.DataFrame:
    """Drop columns whose distinct/nonmissing ratio is above the threshold.

    Missing values are excluded from both counts. Empty and all-missing
    columns are retained by this step. Inspect only text columns (including
    string-valued categories). Numeric and date columns are preserved.
    Return a copy and preserve rows.
    """
    _validate_columns(df)
    _validate_threshold("unique_threshold", unique_threshold)
    text_columns = [
        column for column in df.select_dtypes(include=["object", "string", "category"])
        if not df[column].dropna().empty
        and df[column].dropna().map(lambda value: isinstance(value, str)).all()
    ]
    candidates = df[text_columns]
    nonmissing = candidates.count()
    ratios = candidates.nunique(dropna=True).div(nonmissing.where(nonmissing > 0))
    return df.drop(columns=ratios.index[ratios > unique_threshold]).copy()
