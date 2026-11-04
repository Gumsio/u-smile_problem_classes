"""Readable summaries of the final training-set attribute selections."""

import json
import warnings
from itertools import combinations

import pandas as pd

MIN_ROC_AUC = 0.70


def save_selection_reports(rankings, output, cv_summary, *, min_roc_auc=MIN_ROC_AUC):
    """Save one rank-by-method CSV and count methods choosing each attribute.

    Only methods with mean validation ROC AUC >= min_roc_auc contribute.
    Exact boundary values qualify; held-out test scores are never used.
    Only selected original attributes are included, with at most 20 per method.
    Empty CSV cells represent smaller selections, not extra chosen attributes.
    Counts refer to final method selections, not cross-validation folds.
    """
    if not 0 <= min_roc_auc <= 1:
        raise ValueError('min_roc_auc must be between 0 and 1.')
    methods = rankings['method'].drop_duplicates().tolist()
    scores = cv_summary.reset_index() if 'method' not in cv_summary.columns else cv_summary.copy()
    if scores['method'].duplicated().any() or set(methods) - set(scores['method']):
        raise ValueError('One mean validation ROC AUC is required for every method.')
    scores['eligible_for_final_selection'] = scores['roc_auc_mean'].ge(min_roc_auc)
    scores[['method', 'roc_auc_mean', 'eligible_for_final_selection']].to_csv(
        output / 'method_eligibility.csv', index=False)
    eligible = set(scores.loc[scores['eligible_for_final_selection'], 'method'])
    all_chosen = rankings.loc[rankings['selected']].sort_values(['method', 'rank'])
    all_sets = {method: all_chosen.loc[all_chosen.method.eq(method), 'feature'].tolist() for method in methods}
    methods = [method for method in methods if method in eligible]
    selected_sets = {method: all_sets[method] for method in methods}
    for name, data in [('all_method_selected_features.json', all_sets), ('selected_features.json', selected_sets)]:
        (output / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    pd.DataFrame([
        {'method_a': a, 'method_b': b, 'shared_count': len(set(selected_sets[a]) & set(selected_sets[b]))}
        for a, b in combinations(methods, 2)
    ], columns=['method_a', 'method_b', 'shared_count']).to_csv(output / 'feature_overlap.csv', index=False)
    chosen = all_chosen.loc[all_chosen.method.isin(methods)]
    columns = {}
    for method in methods:
        names = chosen.loc[chosen['method'] == method, 'feature'].tolist()
        if len(names) > 20 or len(names) != len(set(names)):
            raise ValueError('Each method must choose at most 20 distinct attributes.')
        columns[method] = pd.Series(names, index=range(1, len(names) + 1), dtype=object)
    top20 = pd.DataFrame(columns).reindex(range(1, 21)).rename_axis('rank')
    top20.to_csv(output / 'top20_attributes_by_method.csv')
    frequency = chosen.groupby('feature')['method'].nunique().rename('methods_count').reset_index()
    frequency['methods_percent'] = 100 * frequency['methods_count'] / len(methods) if methods else 0.0
    frequency = frequency.sort_values(['methods_count', 'feature'], ascending=[False, True]).reset_index(drop=True)
    frequency.to_csv(output / 'attribute_selection_frequency.csv', index=False)
    save_consensus_selection(chosen, methods, output)
    return top20, frequency


def save_consensus_selection(chosen, methods, output):
    """Vote on eligible methods' chosen lists, then break ties by mean rank points.

    One method contributes at most one vote per attribute. Rank r receives
    (21-r)/20 points for ranks 1..20; absence receives zero. Average points
    across all eligible methods, including methods not choosing the attribute.
    Final ties are broken alphabetically for deterministic results. This is a
    descriptive consensus, not independently evaluated predictive performance.
    """
    methods = list(methods)
    if len(methods) != len(set(methods)):
        raise ValueError('Eligible method names must be distinct.')
    votes = chosen.loc[chosen['method'].isin(methods)].copy()
    if votes.duplicated(['method', 'feature']).any():
        raise ValueError('An attribute may receive only one vote per method.')
    if votes.groupby('method').size().gt(20).any():
        raise ValueError('Each method may contribute at most 20 attributes.')
    if not votes['rank'].between(1, 20).all() or not votes['rank'].mod(1).eq(0).all():
        raise ValueError('Selected attribute ranks must be integers from 1 to 20.')
    votes['rank_points_integer'] = 21 - votes['rank']
    consensus = votes.groupby('feature').agg(
        votes=('method', 'nunique'),
        total_rank_points=('rank_points_integer', 'sum'),
        selected_by=('method', lambda names: '; '.join(sorted(names))),
    ).reset_index()
    # Integer totals make equal-score tie-breaking independent of float rounding.
    consensus = consensus.sort_values(
        ['votes', 'total_rank_points', 'feature'], ascending=[False, False, True]
    ).reset_index(drop=True)
    consensus['rank_points'] = consensus['total_rank_points'] / (20 * len(methods)) if methods else 0.0
    consensus.insert(0, 'rank', range(1, len(consensus) + 1))
    consensus = consensus[['rank', 'feature', 'votes', 'rank_points', 'selected_by']]
    consensus.to_csv(output / 'consensus_attribute_ranking.csv', index=False)
    final = consensus.head(20).copy()
    final.to_csv(output / 'final_selected_attributes.csv', index=False)
    if len(final) < 20:
        warnings.warn(f'Only {len(final)} distinct attributes are available from eligible methods; '
                      'the consensus is not padded to 20.', stacklevel=2)
    return final
