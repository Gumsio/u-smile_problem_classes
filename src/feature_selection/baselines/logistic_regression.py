"""Logistic regression ranking based on standardized absolute coefficients."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def build_model(random_state=42):
    return LogisticRegression(
        solver='liblinear', C=1.0, class_weight='balanced',
        max_iter=4000, random_state=random_state,
    )


def feature_importance(model):
    return np.abs(model.coef_[0])
