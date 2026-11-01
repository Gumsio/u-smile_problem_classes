"""Readable summaries of the final training-set attribute selections."""

import pandas as pd


def save_selection_reports(rankings, output):
    """Save one rank-by-method CSV and count methods choosing each attribute.

    Only selected original attributes are included, with at most 20 per method.
    Empty CSV cells represent smaller selections, not extra chosen attributes.
    Counts refer to final method selections, not cross-validation folds.
    """
    methods = rankings['method'].drop_duplicates().tolist()
    chosen = rankings.loc[rankings['selected']].sort_values(['method', 'rank'])
    columns = {}
    for method in methods:
        names = chosen.loc[chosen['method'] == method, 'feature'].tolist()
        if len(names) > 20 or len(names) != len(set(names)):
            raise ValueError('Each method must choose at most 20 distinct attributes.')
        columns[method] = pd.Series(names, index=range(1, len(names) + 1), dtype=object)
    top20 = pd.DataFrame(columns).reindex(range(1, 21)).rename_axis('rank')
    top20.to_csv(output / 'top20_attributes_by_method.csv')
    frequency = chosen.groupby('feature')['method'].nunique().rename('methods_count').reset_index()
    frequency['methods_percent'] = 100 * frequency['methods_count'] / len(methods)
    frequency = frequency.sort_values(['methods_count', 'feature'], ascending=[False, True]).reset_index(drop=True)
    frequency.to_csv(output / 'attribute_selection_frequency.csv', index=False)
    return top20, frequency
