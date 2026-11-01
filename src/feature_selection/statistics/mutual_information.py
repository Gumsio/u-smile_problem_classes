"""Mutual-information scores between predictors and a classification outcome."""
from sklearn.feature_selection import mutual_info_classif


def score_features(X, y, *, random_state=42):
    """Preserve the notebook's existing estimator settings."""
    return mutual_info_classif(X, y, random_state=random_state)


from ..baselines.logistic_regression import build_model


def rank_features(preprocessor, X, y, random_state=42):
    """Rank original attributes on the supplied training rows only."""
    import numpy as np
    matrix = preprocessor.transform(X)
    from sklearn.feature_selection import mutual_info_classif
    discrete = np.array([c in preprocessor.categorical_ for c in preprocessor.source_columns_])
    scores = mutual_info_classif(matrix, y, discrete_features=discrete, random_state=random_state)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
    ranking = preprocessor.aggregate_importance(scores).rename('importance').rename_axis('feature').reset_index()
    return ranking.sort_values(['importance', 'feature'], ascending=[False, True]).reset_index(drop=True)
