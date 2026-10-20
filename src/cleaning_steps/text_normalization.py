"""Cleaning step: text normalization."""

import pandas as pd


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip column names and drop the known duplicate admission-date column.

    Accepts any columns; run before steps that reference normalized names.
    """
    df = df.copy()
    # Strip accidental whitespace around column names
    df.columns = [str(col).strip() for col in df.columns]

    # Drop duplicated column 'Data przyjęcia.1' if present
    if "Data przyjęcia.1" in df.columns:
        df = df.drop(columns=["Data przyjęcia.1"])

    return df
