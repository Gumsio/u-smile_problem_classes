"""Run all feature selectors and save at most 20 original attributes per method."""

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path
import subprocess
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.dummy import DummyClassifier

# Use the same package imports when called as a script or python -m src.select_features.
if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.clean_data import RAW_DATA_PATH, PROJECT_ROOT
from src.feature_selection.baselines import logistic_regression, decision_tree
from src.feature_selection.model_based import boruta, lasso, elastic_net, xgboost, random_forest, rfe
from src.feature_selection.statistics import anova, mutual_information
from src.feature_selection.preprocessing import FeaturePreprocessor, prepare_xy
from src.feature_selection.reporting import save_selection_reports, MIN_ROC_AUC
from src.cleaning_steps.column_pruning import prune_columns

PRIMARY_METRIC = "roc_auc"
N_FEATURES = 20
METHODS = {
    'logistic_regression': logistic_regression, 'decision_tree': decision_tree,
    'anova': anova, 'mutual_information': mutual_information,
    'random_forest': random_forest, 'rfe': rfe, 'boruta': boruta,
    'lasso': lasso, 'elastic_net': elastic_net, 'xgboost': xgboost,
}


def fit_selector(method, X, y, random_state):
    """Rank on training patients, then refit using at most 20 nonzero columns."""
    preprocessor = FeaturePreprocessor().fit(X)
    if hasattr(method, 'rank_features'):
        ranking = method.rank_features(preprocessor, X, y, random_state)
    else:
        ranking_model = method.build_model(random_state)
        ranking_model.fit(preprocessor.transform(X), y)
        importance = preprocessor.aggregate_importance(method.feature_importance(ranking_model))
        ranking = importance.rename('importance').rename_axis('feature').reset_index()
        ranking = ranking.sort_values(['importance', 'feature'], ascending=[False, True]).reset_index(drop=True)
    ranking.insert(0, 'rank', np.arange(1, len(ranking) + 1))
    ranking['selected'] = (ranking['rank'] <= N_FEATURES) & (ranking['importance'] > 1e-8)
    selected = ranking.loc[ranking['selected'], 'feature'].tolist()
    if 'boruta_status' in ranking and (ranking.loc[ranking['selected'], 'boruta_status'] != 'confirmed').any():
        warnings.warn('The 20-attribute Boruta comparison includes tentative or rejected attributes; '
                      'consult boruta_status in the saved rankings.', stacklevel=2)
    if not selected:
        model = DummyClassifier(strategy='prior').fit(np.zeros((len(X), 1)), y)
        return None, model, ranking, selected
    final_preprocessor = FeaturePreprocessor().fit(X[selected], prune=False)
    model = method.build_model(random_state)
    model.fit(final_preprocessor.transform(X[selected]), y)
    return final_preprocessor, model, ranking, selected


def evaluate(preprocessor, model, X, y):
    transformed = preprocessor.transform(X) if preprocessor is not None else np.zeros((len(X), 1))
    probabilities = model.predict_proba(transformed)[:, list(model.classes_).index(1)]
    predicted = model.predict(transformed)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        'roc_auc': float(roc_auc_score(y, probabilities)),
        'average_precision': float(average_precision_score(y, probabilities)),
        'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
        'accuracy': float(accuracy_score(y, predicted)),
        'precision': float(precision_score(y, predicted, zero_division=0)),
        'recall': float(recall_score(y, predicted, zero_division=0)),
        'f1': float(f1_score(y, predicted, zero_division=0)),
        'true_negative': int(tn), 'false_positive': int(fp),
        'false_negative': int(fn), 'true_positive': int(tp),
    }


def git_metadata():
    def read(*arguments):
        result = subprocess.run(['git', *arguments], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {'commit': read('rev-parse', 'HEAD'), 'working_tree_changes': read('status', '--porcelain')}


def run_feature_selection(input_path=RAW_DATA_PATH, output_root=None, *, random_state=42):
    """Evaluate selections of at most 20 attributes with a held-out test set.

    Five CV folds use only the training portion. Each fold independently fits
    pruning, imputers, encoders, scaling and ranking, then refits on at most 20 features.
    The final selections are fitted on all training patients, never test labels.
    """
    input_path = Path(input_path)
    raw = pd.read_csv(input_path, low_memory=False)
    X, y, preparation = prepare_xy(raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state,
    )
    if y_train.value_counts().min() < 5:
        raise ValueError('Each training class needs at least five patients for five-fold validation.')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_rows, ranking_tables, test_rows, selected_sets, fold_selections = [], [], [], {}, []
    models_config = {}
    for name, method in METHODS.items():
        print(f'{name}: five-fold validation and final selection of at most 20 attributes...', flush=True)
        for fold, (training, validation) in enumerate(cv.split(X_train, y_train), start=1):
            preprocessor, model, ranking, selected = fit_selector(
                method, X_train.iloc[training], y_train.iloc[training], random_state,
            )
            cv_rows.append({'method': name, 'fold': fold, **evaluate(
                preprocessor, model, X_train.iloc[validation], y_train.iloc[validation],
            )})
            fold_selections.extend({'method': name, 'fold': fold, 'feature': column} for column in selected)
        preprocessor, model, ranking, selected = fit_selector(method, X_train, y_train, random_state)
        selected_sets[name] = selected
        ranking.insert(0, 'method', name)
        ranking_tables.append(ranking)
        models_config[name] = model.get_params()
        test_rows.append({'method': name, 'n_features': len(selected),
                          **evaluate(preprocessor, model, X_test, y_test)})

    output_root = Path(output_root) if output_root else PROJECT_ROOT / 'results' / 'feature_selection'
    output = output_root / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    cv_results = pd.DataFrame(cv_rows)
    cv_results.to_csv(output / 'cv_results.csv', index=False)
    metric_names = ['roc_auc', 'average_precision', 'balanced_accuracy', 'accuracy', 'precision', 'recall', 'f1']
    summary = cv_results.groupby('method')[metric_names].agg(['mean', 'std'])
    summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]
    summary = summary.sort_values(f'{PRIMARY_METRIC}_mean', ascending=False, kind='stable')
    summary.to_csv(output / 'cv_summary.csv')
    final_rankings = pd.concat(ranking_tables, ignore_index=True)
    final_rankings.to_csv(output / 'feature_rankings.csv', index=False)
    eligible = set(summary.index[summary['roc_auc_mean'].ge(MIN_ROC_AUC)])
    final_rankings['eligible_for_final_selection'] = final_rankings['method'].isin(eligible)
    final_rankings['final_selected'] = final_rankings['selected'] & final_rankings['eligible_for_final_selection']
    final_rankings.to_csv(output / 'feature_rankings.csv', index=False)
    save_selection_reports(final_rankings, output, summary)
    selected_sets = {name: names for name, names in selected_sets.items() if name in eligible}
    pd.DataFrame(test_rows).to_csv(output / 'test_results.csv', index=False)
    pd.DataFrame(fold_selections).to_csv(output / 'fold_selected_features.csv', index=False)
    overlap = {
        f'{left}__{right}': sorted(set(selected_sets[left]) & set(selected_sets[right]))
        for left, right in combinations(selected_sets, 2)
    }
    pd.DataFrame([
        {'method_a': left, 'method_b': right, 'shared_count': len(set(selected_sets[left]) & set(selected_sets[right]))}
        for left, right in combinations(selected_sets, 2)
    ], columns=['method_a', 'method_b', 'shared_count']).to_csv(output / 'feature_overlap.csv', index=False)
    (output / 'selected_features.json').write_text(json.dumps(selected_sets, indent=2, ensure_ascii=False) + '\n')
    config = {
        **preparation, 'n_features': N_FEATURES, 'random_state': random_state,
        'primary_metric': PRIMARY_METRIC,
        'min_final_selection_roc_auc': MIN_ROC_AUC,
        'consensus': {'limit': 20, 'primary_rank': 'one vote per eligible method',
                      'tie_break': 'mean (21-rank)/20 across all eligible methods; absent=0; then name',
                      'interpretation': 'descriptive consensus; no independent consensus ROC AUC'},
        'eligible_methods': sorted(eligible),
        'comparison_rule': 'Higher mean cross-validation ROC AUC; test scores are reporting only.',
        'test_fraction': 0.2, 'cv_folds': 5, 'train_rows': len(X_train), 'test_rows': len(X_test),
        'class_counts': {str(k): int(v) for k, v in y.value_counts().items()},
        'source': str(input_path.resolve()), 'source_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
        'versions': {'pandas': pd.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'boruta': version('Boruta'), 'xgboost': version('xgboost-cpu')},
        'models': models_config, 'git': git_metadata(),
        'selector_settings': {'boruta': boruta.SELECTOR_SETTINGS},
        'boruta_evaluator': 'RandomForestClassifier; Boruta is the selector, not a predictive model.',
        'boruta_top20_rule': 'Best encoded Boruta rank, then summed median importance, then name. Status is kept; top 20 is not necessarily the confirmed set.',
        'pruning_settings': {
            name: parameter.default
            for name, parameter in inspect.signature(prune_columns).parameters.items()
            if parameter.default is not inspect.Parameter.empty
        },
        'preprocessing': {
            'numeric': 'median imputation then StandardScaler',
            'categorical': 'most-frequent imputation then one-hot encoding; unseen labels ignored',
            'fit_scope': 'Each training fold independently; final fit on the 80% training portion',
        },
        'imputation': 'Training-only median for numeric and most-frequent for categorical; not full-data MissForest.',
        'ranking': 'Scores aggregated to original attributes: coefficients, statistical scores, impurity/gain, RFE rank or Boruta rank. See method modules.',
        'selection_limit': 'At most 20 original attributes with positive importance; no zero-score padding.',
        'evaluation_classifiers': {name: type(method.build_model()).__name__ for name, method in METHODS.items()},
        'rfe_settings': {'step': 0.1, 'encoded_features_to_select': 20},
        'shared_selected_features': overlap,
        'timing_limitation': 'Predictor collection times are unknown; additional outcome proxies may remain.',
    }
    (output / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    print('Primary metric: ROC AUC. Methods ordered by mean cross-validation ROC AUC.')
    print(summary[[f'{PRIMARY_METRIC}_mean', f'{PRIMARY_METRIC}_std']].to_string())
    print('Held-out test results (reporting only):')
    print(pd.DataFrame(test_rows).set_index('method').loc[summary.index].to_string())
    print(f'Results saved to: {output}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=RAW_DATA_PATH,
                        help='Raw CSV, not the full-data imputed cleaning output.')
    parser.add_argument('--output-root', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_feature_selection(args.input, args.output_root, random_state=args.seed)


if __name__ == '__main__':
    main()
