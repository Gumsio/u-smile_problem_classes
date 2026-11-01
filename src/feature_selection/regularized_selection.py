"""Shared encoding and capped original-attribute rankings for model-based selectors."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def rank_attributes(X, y, method, *, n_features=20, random_state=42, **model_settings):
    """Fit on the supplied rows and return model, preprocessing and rankings.

    Use training rows only when evaluating on held-out patients. Nonzero
    Model importance values determine rankings. Select at most 20 nonzero
    original attributes; never pad with zero-score attributes. Sum encoded-column
    importance values to rank original attributes without parsing encoded names.
    """
    if not X.columns.is_unique:
        raise ValueError('Predictor names must be unique.')
    if len(X) != len(y):
        raise ValueError('Predictors and outcome must have the same number of rows.')
    if isinstance(n_features, bool) or not isinstance(n_features, int) or not 1 <= n_features <= 20:
        raise ValueError('n_features must be an integer between 1 and 20.')
    if X.shape[1] == 0:
        raise ValueError('At least one predictor is required.')
    if pd.isna(y).any() or set(pd.unique(y)) != {0, 1}:
        raise ValueError('Outcome must contain both classes 0 and 1, with no missing labels.')
    frame = X.copy()
    numeric = list(frame.select_dtypes(include='number').columns)
    categorical = [c for c in frame if c not in numeric]
    for column in numeric:
        frame[column] = frame[column].astype(float).replace([np.inf, -np.inf], np.nan)
    for column in categorical:
        frame[column] = frame[column].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
    transforms = []
    if numeric:
        transforms.append(('numeric', make_pipeline(
            SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(),
        ), numeric))
    if categorical:
        transforms.append(('categorical', make_pipeline(
            SimpleImputer(strategy='most_frequent', keep_empty_features=True),
            OneHotEncoder(handle_unknown='ignore', sparse_output=False),
        ), categorical))
    preprocessor = ColumnTransformer(transforms)
    matrix = preprocessor.fit_transform(frame)
    # Scale dummy columns too, so the penalty acts on standardized inputs.
    scaler = StandardScaler()
    matrix = scaler.fit_transform(matrix)
    model = method.build_model(random_state, **model_settings)
    model.fit(matrix, np.asarray(y))
    sources = list(numeric)
    if categorical:
        encoder = preprocessor.named_transformers_['categorical'].steps[-1][1]
        for column, categories in zip(categorical, encoder.categories_):
            sources.extend([column] * len(categories))
    importance = pd.Series(method.feature_importance(model), index=sources)
    ranking = importance.groupby(level=0, sort=False).sum().rename('importance').rename_axis('feature').reset_index()
    ranking = ranking.sort_values(['importance', 'feature'], ascending=[False, True]).reset_index(drop=True)
    ranking.insert(0, 'rank', np.arange(1, len(ranking) + 1))
    ranking['has_nonzero_importance'] = ranking.importance > 1e-8
    ranking['selected'] = (ranking['rank'] <= n_features) & ranking['has_nonzero_importance']
    return model, make_pipeline(preprocessor, scaler), ranking
