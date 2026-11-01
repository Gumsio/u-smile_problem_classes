"""Univariate ANOVA F scores and p-values for a classification outcome."""
from sklearn.feature_selection import f_classif


def score_features(X, y):
    """Return one F score and uncorrected p-value per encoded input column."""
    return f_classif(X, y)


from ..baselines.logistic_regression import build_model


def rank_features(preprocessor, X, y, random_state=42):
    """Rank original attributes on the supplied training rows only."""
    import numpy as np
    matrix = preprocessor.transform(X)
    scores, _ = score_features(matrix, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
    ranking = preprocessor.aggregate_importance(scores).rename('importance').rename_axis('feature').reset_index()
    return ranking.sort_values(['importance', 'feature'], ascending=[False, True]).reset_index(drop=True)
