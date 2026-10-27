"""Data cleaning pipeline for U-Smile cardiac patient data."""

from pathlib import Path
import pandas as pd

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cardiac_patient_data.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cardiac_patient_data_clean.csv"


def load_data(filepath=None) -> pd.DataFrame:
    """Load raw patient data directly from CSV."""
    target_path = Path(filepath) if filepath else RAW_DATA_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"File not found: {target_path}")

    return pd.read_csv(target_path, low_memory=False)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize column names and remove duplicates."""
    df = df.copy()
    # Strip accidental whitespace around column names
    df.columns = [str(col).strip() for col in df.columns]

    # Drop duplicated column 'Data przyjęcia.1' if present
    if "Data przyjęcia.1" in df.columns:
        df = df.drop(columns=["Data przyjęcia.1"])

    return df


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize date formats to date only (YYYY-MM-DD) and fix corrupted observation days."""
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


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing and placeholder values (placeholder for further tasks)."""
    df = df.copy()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps in order and convert data types."""
    df = clean_column_names(df)
    df = clean_dates(df)
    df = clean_missing_values(df)
    df = df.convert_dtypes()
    df.drop(columns=['tryb'], inplace=True)     # dropping the column as suggested
    return df


def save_data(df: pd.DataFrame, output_path=None) -> Path:
    """Save cleaned data to the processed data folder."""
    target_path = Path(output_path) if output_path else PROCESSED_DATA_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    return target_path


def main():
    print(f"Loading raw data from: {RAW_DATA_PATH}")
    df = load_data()
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

    print("Cleaning data...")
    cleaned_df = clean_data(df)
    print(f"Cleaned dataset: {len(cleaned_df)} rows and {len(cleaned_df.columns)} columns.")

    out = save_data(cleaned_df)
    print(f"Successfully saved clean data to: {out}")


if __name__ == "__main__":
    main()
