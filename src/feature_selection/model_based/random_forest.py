"""Random-forest impurity importance and classifier."""
from sklearn.ensemble import RandomForestClassifier


def build_model(random_state=42):
    return RandomForestClassifier(n_estimators=128, max_depth=5,
                                  class_weight='balanced', random_state=random_state, n_jobs=1)


def feature_importance(model):
    return model.feature_importances_
