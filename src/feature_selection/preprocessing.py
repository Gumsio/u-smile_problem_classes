"""Training-only pruning, imputation, scaling and category encoding."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..cleaning_steps.column_pruning import prune_columns
from ..cleaning_steps.text_normalization import clean_column_names

# These encode identity, recorded outcome timing, or an explicitly removed field.
EXCLUDED_COLUMNS = (
    'cid', 'cid.1', 'tryb', 'Data przyjęcia', 'data przyjęcia',
    'Data przyjęcia.1', 'data implantacji', 'data zgonu',
    'data zgonu lub konca obserwacji', 'day_count_begin_to_end', 'Unnamed: 5',
    'Time of observation', 'time_of_observation',
    'days_admission_to_end', 'days_implantation_to_end',
)


def prepare_xy(raw, target='zgon', extra_exclusions=()):
    """Apply fixed column cleanup; no statistics are learned before splitting.

    Missing outcome labels are excluded, never imputed. Unknown nonmissing
    labels fail rather than silently changing the task. Coding 1=death and
    0=no recorded death is an explicit assumption, not a fixed-horizon outcome.
    """
    frame = clean_column_names(raw)
    if not frame.columns.is_unique:
        raise ValueError('Column names must be unique.')
    if target not in frame:
        raise ValueError(f'Target column not found: {target}')
    labels = pd.to_numeric(frame[target], errors='coerce')
    invalid = frame[target].notna() & (labels.isna() | ~labels.isin([0, 1]))
    if invalid.any():
        raise ValueError('The death target must contain only 0, 1, or missing values.')
    present = labels.notna()
    y = labels.loc[present].astype(int).reset_index(drop=True)
    if set(y.unique()) != {0, 1}:
        raise ValueError('Both death classes (0 and 1) are required.')
    excluded = set(EXCLUDED_COLUMNS) | set(extra_exclusions) | {target}
    X = frame.loc[present].drop(columns=list(excluded), errors='ignore').reset_index(drop=True)
    for column in X:
        if pd.api.types.is_numeric_dtype(X[column]):
            X[column] = X[column].astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            # Normalize string/category containers for sklearn; do not guess lab codes.
            X[column] = X[column].astype(object).where(X[column].notna(), np.nan)
            X[column] = X[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
    metadata = {
        'excluded_columns': sorted(set(frame.columns) & excluded),
        'missing_target_rows_removed': int((~present).sum()),
        'target': target,
        'target_coding': {'0': 'no recorded death', '1': 'recorded death'},
        'scope': 'Exploratory association with recorded death, not causal attribution or fixed-horizon risk.',
    }
    return X, y, metadata


class FeaturePreprocessor:
    """Fit column choices and preprocessing on training data only."""

    def fit(self, X, *, prune=True):
        frame = prune_columns(X) if prune else X
        # Selected columns are already pruned; constant/empty fold columns can
        # still be represented by SimpleImputer without reading validation data.
        self.columns_ = list(frame.columns)
        if not self.columns_:
            raise ValueError('No usable predictors remain after training-only pruning.')
        self.numeric_ = list(frame.select_dtypes(include='number').columns)
        self.categorical_ = [c for c in self.columns_ if c not in self.numeric_]
        transforms = []
        if self.numeric_:
            transforms.append(('numeric', make_pipeline(
                SimpleImputer(strategy='median', keep_empty_features=True),
                StandardScaler(),
            ), self.numeric_))
        if self.categorical_:
            transforms.append(('categorical', make_pipeline(
                SimpleImputer(strategy='most_frequent', keep_empty_features=True),
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            ), self.categorical_))
        self.transformer_ = ColumnTransformer(transforms, remainder='drop')
        self.transformer_.fit(frame)
        self.source_columns_ = list(self.numeric_)
        if self.categorical_:
            encoder = self.transformer_.named_transformers_['categorical'].steps[-1][1]
            for column, categories in zip(self.categorical_, encoder.categories_):
                self.source_columns_.extend([column] * len(categories))
        return self

    def transform(self, X):
        return self.transformer_.transform(X[self.columns_])

    def aggregate_importance(self, values):
        """Sum encoded-column importance under each original attribute name."""
        return pd.Series(values, index=self.source_columns_).groupby(level=0, sort=False).sum()
