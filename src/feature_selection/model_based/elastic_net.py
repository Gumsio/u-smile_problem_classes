"""Elastic Net feature selection for a binary outcome using logistic regression."""

import inspect
import numpy as np
from sklearn.linear_model import LogisticRegression


def build_model(random_state=42, *, C=0.1, l1_ratio=0.5):
    """Mix L1 and L2 penalties; l1_ratio=1 is L1 and 0 is L2."""
    options = dict(solver='saga', C=C, l1_ratio=l1_ratio, class_weight='balanced',
                   max_iter=10000, tol=1e-4, random_state=random_state)
    penalty = inspect.signature(LogisticRegression).parameters.get('penalty')
    if penalty is not None and penalty.default != 'deprecated':
        options['penalty'] = 'elasticnet'
    return LogisticRegression(**options)


def feature_importance(model):
    return np.abs(model.coef_[0])
