"""Data cleaning pipeline for U-Smile cardiac patient data."""

from pathlib import Path
import pandas as pd

# Support both `python src/clean_data.py` and `from src.clean_data import ...`.
if __package__:
    from .cleaning_steps.text_normalization import clean_column_names
    from .cleaning_steps.date_formatting import clean_dates
    from .cleaning_steps.missing_values import clean_missing_values
else:
    from cleaning_steps.text_normalization import clean_column_names
    from cleaning_steps.date_formatting import clean_dates
    from cleaning_steps.missing_values import clean_missing_values

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


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps in order and convert data types."""
    df = clean_column_names(df)
    df = clean_dates(df)
    df = clean_missing_values(df)
    df = df.convert_dtypes()
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
