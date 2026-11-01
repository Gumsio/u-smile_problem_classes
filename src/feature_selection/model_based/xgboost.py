"""XGBoost gain-based feature ranking for the binary death outcome."""

import numpy as np
from xgboost import XGBClassifier


def build_model(random_state=42, **settings):
    """Fixed initial settings; validation patients are never used for stopping."""
    options = dict(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective='binary:logistic', eval_metric='auc',
        importance_type='gain', tree_method='hist',
        random_state=random_state, n_jobs=1,
    )
    options.update(settings)
    return XGBClassifier(**options)


def feature_importance(model):
    """Average split gain, normalized by XGBoost, for each encoded input."""
    return np.asarray(model.feature_importances_, dtype=float)
