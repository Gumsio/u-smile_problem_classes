"""Generate the cleaned CSV, then run leakage-aware baseline feature selection."""

if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clean_data import main as run_cleaning
from src.select_features import run_feature_selection


def main():
    run_cleaning()
    # Selection starts from raw data so full-data MissForest estimates and
    # pruning decisions do not enter the held-out model evaluation.
    run_feature_selection()


if __name__ == '__main__':
    main()
