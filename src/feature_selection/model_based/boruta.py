"""BorutaPy ranking, evaluated with a random-forest classifier.

Boruta finds all relevant features, not a prescribed number. The runner retains
native status and rank, then takes 20 original columns for the fixed comparison.
"""

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

SELECTOR_SETTINGS = {
    'n_estimators': 128, 'max_iter': 50, 'perc': 100,
    'alpha': 0.05, 'two_step': True,
}


def build_model(random_state=42):
    """Classifier for evaluating the selected attributes, not Boruta itself."""
    return RandomForestClassifier(
        n_estimators=128, max_depth=5, class_weight='balanced',
        random_state=random_state, n_jobs=1,
    )


def rank_features(preprocessor, X, y, random_state=42, *, max_iter=None, n_estimators=None):
    """Run Boruta on training rows and map dummy-column results to originals.

    An original attribute is confirmed if any of its encoded columns is
    confirmed, otherwise tentative if any is tentative. This aggregation is
    a comparison convention, not a separate Boruta test on the whole attribute.
    Sort by best encoded Boruta rank, summed median real-feature importance,
    then name. Status and rank remain available independently of the top 20.
    """
    settings = dict(SELECTOR_SETTINGS)
    if max_iter is not None:
        settings['max_iter'] = max_iter
    if n_estimators is not None:
        settings['n_estimators'] = n_estimators
    selector = BorutaPy(
        estimator=build_model(random_state), random_state=random_state,
        verbose=0, **settings,
    )
    selector.fit(preprocessor.transform(X), np.asarray(y))
    median_importance = np.nanmedian(selector.importance_history_[1:], axis=0)
    encoded = pd.DataFrame({
        'feature': preprocessor.source_columns_,
        'boruta_rank': selector.ranking_,
        'confirmed': selector.support_,
        'tentative': selector.support_weak_,
        'importance': median_importance,
    })
    ranking = encoded.groupby('feature', sort=False).agg(
        boruta_rank=('boruta_rank', 'min'),
        confirmed=('confirmed', 'any'),
        tentative=('tentative', 'any'),
        importance=('importance', 'sum'),
    ).reset_index()
    ranking['boruta_status'] = np.select(
        [ranking['confirmed'], ranking['tentative']], ['confirmed', 'tentative'], default='rejected',
    )
    return ranking.drop(columns=['confirmed', 'tentative']).sort_values(
        ['boruta_rank', 'importance', 'feature'], ascending=[True, False, True],
    ).reset_index(drop=True)
