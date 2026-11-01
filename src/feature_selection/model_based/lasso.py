"""Lasso-style feature selection for a binary outcome: L1 logistic regression."""

import inspect
import numpy as np
from sklearn.linear_model import LogisticRegression


def build_model(random_state=42, *, C=0.1):
    """C controls inverse regularization strength; smaller values shrink more."""
    options = dict(solver='saga', C=C, class_weight='balanced',
                   max_iter=10000, tol=1e-4, random_state=random_state)
    # sklearn 1.8+ specifies regularization through l1_ratio; older versions
    # require penalty. Keep the project's sklearn>=1.4 compatibility.
    penalty = inspect.signature(LogisticRegression).parameters.get('penalty')
    if penalty is not None and penalty.default != 'deprecated':
        options['penalty'] = 'l1'
    else:
        options['l1_ratio'] = 1.0
    return LogisticRegression(**options)


def feature_importance(model):
    return np.abs(model.coef_[0])
