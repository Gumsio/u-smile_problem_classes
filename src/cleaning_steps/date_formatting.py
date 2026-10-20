"""Cleaning step: date formatting."""

import pandas as pd


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize available date columns and recalculate observation days.

    Run after column-name normalization. Missing date columns are skipped;
    observation duration requires both admission and observation-end dates
    plus an existing day-count column.
    """
    df = df.copy()

    # Fix invalid date '31.04.2020' (April has only 30 days -> 2020-04-30)
    if "data zgonu lub konca obserwacji" in df.columns:
        df["data zgonu lub konca obserwacji"] = (
            df["data zgonu lub konca obserwacji"]
            .astype(str)
            .str.replace("31.04.2020", "2020-04-30")
        )

    # Standardize date columns to pure date format (YYYY-MM-DD)
    date_columns = ["Data przyjęcia", "data implantacji", "data zgonu lub konca obserwacji"]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce").dt.date

    # Recalculate observation time (days count) to fix #VALUE! errors caused by invalid dates
    target_col = "day_count_begin_to_end" if "day_count_begin_to_end" in df.columns else "Unnamed: 5"
    if target_col in df.columns and "data zgonu lub konca obserwacji" in df.columns and "Data przyjęcia" in df.columns:
        d_end = pd.to_datetime(df["data zgonu lub konca obserwacji"])
        d_start = pd.to_datetime(df["Data przyjęcia"])
        df[target_col] = (d_end - d_start).dt.days
        df = df.rename(columns={target_col: "Time of observation"})

    return df
