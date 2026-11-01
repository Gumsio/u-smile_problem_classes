"""Cross-validated ROC AUC for capped model-based feature selections."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..cleaning_steps.column_pruning import prune_columns
from ..cleaning_steps.text_normalization import clean_column_names
from .model_based import lasso, elastic_net
from .regularized_selection import rank_attributes

EXCLUDED = (
    'zgon', 'cid', 'cid.1', 'tryb', 'Time of observation', 'time_of_observation',
    'days_admission_to_end', 'days_implantation_to_end', 'Data przyjęcia',
    'Data przyjęcia.1', 'data implantacji', 'data zgonu',
    'data zgonu lub konca obserwacji', 'day_count_begin_to_end', 'Unnamed: 5',
)


def evaluate_methods(raw, *, methods=None, n_splits=5, random_state=42):
    """Re-select at most 20 attributes and refit in every training fold.

    Read raw rather than full-data imputed inputs. All statistical pruning,
    imputation, encoding, scaling and selection use training rows only. The
    final classifier uses the same model settings as the selector, fitted
    again using only its chosen original attributes. No settings are tuned.
    """
    frame = clean_column_names(raw)
    labels = pd.to_numeric(frame['zgon'], errors='coerce')
    invalid = frame['zgon'].notna() & ~labels.isin([0, 1])
    if invalid.any():
        raise ValueError('zgon must contain 0, 1 or missing labels.')
    present = labels.notna()
    y = labels[present].astype(int).reset_index(drop=True)
    if set(y.unique()) != {0, 1} or y.value_counts().min() < n_splits:
        raise ValueError('Each outcome class needs at least n_splits observations.')
    X = frame.loc[present].drop(columns=list(EXCLUDED), errors='ignore').reset_index(drop=True)
    # Fixed type normalization learns no values from held-out rows.
    for column in X:
        if pd.api.types.is_numeric_dtype(X[column]):
            X[column] = X[column].astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            X[column] = X[column].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows, selected_rows = [], []
    methods = {'lasso': lasso, 'elastic_net': elastic_net} if methods is None else methods
    if not methods:
        raise ValueError('At least one selection method is required.')
    for name, method in methods.items():
        for fold, (train, valid) in enumerate(splitter.split(X, y), start=1):
            train_X = prune_columns(X.iloc[train])
            chosen = []
            if train_X.shape[1]:
                _, _, ranking = rank_attributes(train_X, y.iloc[train], method, random_state=random_state)
                chosen = ranking.loc[ranking.selected, 'feature'].tolist()
            if chosen:
                model, prep, _ = rank_attributes(
                    train_X[chosen], y.iloc[train], method, random_state=random_state,
                )
                probability = model.predict_proba(prep.transform(X.iloc[valid][chosen]))[:, 1]
            else:
                # Strong regularization can select no attributes. Report the
                # constant training prevalence, not an error or fabricated set.
                probability = np.full(len(valid), y.iloc[train].mean())
            rows.append({'method': name, 'fold': fold, 'roc_auc': roc_auc_score(y.iloc[valid], probability),
                         'selected_attributes': len(chosen)})
            selected_rows.extend({'method': name, 'fold': fold, 'feature': column} for column in chosen)
    scores = pd.DataFrame(rows)
    summary = scores.groupby('method').agg(
        roc_auc_mean=('roc_auc', 'mean'), roc_auc_std=('roc_auc', 'std'),
        min_attributes=('selected_attributes', 'min'), max_attributes=('selected_attributes', 'max'),
    ).sort_values('roc_auc_mean', ascending=False)
    return scores, summary, pd.DataFrame(selected_rows, columns=['method', 'fold', 'feature'])


def evaluate_regularized(raw, *, n_splits=5, random_state=42):
    """Keep the existing Lasso/Elastic Net notebook interface."""
    return evaluate_methods(raw, n_splits=n_splits, random_state=random_state)
