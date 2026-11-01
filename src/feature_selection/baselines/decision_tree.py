"""Decision tree ranking based on impurity reduction."""
from sklearn.tree import DecisionTreeClassifier


def build_model(random_state=42):
    return DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=5,
        class_weight='balanced', random_state=random_state,
    )


def feature_importance(model):
    return model.feature_importances_
