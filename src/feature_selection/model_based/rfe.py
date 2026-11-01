"""Recursive elimination using logistic regression coefficients."""
import pandas as pd
from sklearn.feature_selection import RFE
from ..baselines.logistic_regression import build_model


def rank_features(preprocessor, X, y, random_state=42):
    """Eliminate 10% of encoded inputs per iteration, then group original names."""
    matrix = preprocessor.transform(X)
    selector = RFE(build_model(random_state), n_features_to_select=min(20, matrix.shape[1]), step=0.1)
    selector.fit(matrix, y)
    encoded = pd.DataFrame({'feature': preprocessor.source_columns_,
                            'rfe_rank': selector.ranking_, 'rfe_supported': selector.support_})
    ranking = encoded.groupby('feature', sort=False).agg(
        rfe_rank=('rfe_rank', 'min'), rfe_supported=('rfe_supported', 'any'),
    ).reset_index()
    ranking['importance'] = 1.0 / ranking['rfe_rank']
    return ranking.sort_values(['rfe_rank', 'feature']).reset_index(drop=True)
