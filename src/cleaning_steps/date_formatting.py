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


def calculate_days_to_end(df: pd.DataFrame) -> pd.DataFrame:
    """Replace source dates with day-count columns at the front.

    Run after date cleaning. The dataset's end date is not necessarily a death
    date: it also records the end of observation for surviving patients.
    Missing columns, missing dates and invalid dates produce nullable values.
    Negative durations are preserved so date-order problems remain visible.
    Other columns keep their relative order. The input is not modified.
    Reuse an existing observation-time column if it exactly matches admission
    days, including missing values, instead of adding a duplicate column.
    """
    df = df.copy()
    end_column = "data zgonu lub konca obserwacji"
    end = pd.to_datetime(
        df[end_column], format="mixed", errors="coerce"
    ) if end_column in df.columns else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for start_column, output_column in (
        ("Data przyjęcia", "days_admission_to_end"),
        ("data implantacji", "days_implantation_to_end"),
    ):
        if start_column in df.columns:
            start = pd.to_datetime(df[start_column], format="mixed", errors="coerce")
            df[output_column] = (end - start).dt.days.astype("Int64")
        else:
            df[output_column] = pd.array([pd.NA] * len(df), dtype="Int64")
    df = df.drop(
        columns=["Data przyjęcia", "data implantacji", end_column], errors="ignore"
    )
    day_columns = ["days_admission_to_end", "days_implantation_to_end"]
    for observation_column in ("Time of observation", "time_of_observation"):
        if observation_column not in df.columns:
            continue
        observation = df[observation_column]
        admission_days = df["days_admission_to_end"]
        matches = observation.eq(admission_days).fillna(False)
        matches |= observation.isna() & admission_days.isna()
        if matches.all():
            df = df.drop(columns="days_admission_to_end")
            day_columns = [observation_column, "days_implantation_to_end"]
            break
    return df[day_columns + [column for column in df.columns if column not in day_columns]]
