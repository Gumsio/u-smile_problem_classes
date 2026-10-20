"""Behavior checks for the issue #4 pruning rules."""
import unittest

import pandas as pd

from src.cleaning_steps.column_pruning import (
    drop_correlated_columns,
    drop_identifier_columns,
    drop_high_cardinality_columns,
    drop_low_variance_columns,
    drop_missing_columns,
    prune_columns,
)


class ColumnPruningTests(unittest.TestCase):
    def test_wrapper_passes_settings_to_each_step(self):
        from unittest.mock import patch
        import src.cleaning_steps.column_pruning as module

        df = pd.DataFrame({'value': [1, 2, 3]})
        names = ['drop_identifier_columns', 'drop_missing_columns',
                 'drop_low_variance_columns', 'drop_high_cardinality_columns', 'drop_correlated_columns']
        settings = [
            {'identifier_columns': ['custom_id']},
            {'missing_threshold': 0.42},
            {'dominant_value_threshold': 0.87},
            {'unique_threshold': 0.65},
            {'correlation_threshold': 0.76},
        ]
        from contextlib import ExitStack
        with ExitStack() as stack:
            mocks = [stack.enter_context(patch.object(module, name, return_value=df))
                     for name in names]
            prune_columns(df, identifier_columns=['custom_id'], missing_threshold=0.42,
                          dominant_value_threshold=0.87, correlation_threshold=0.76,
                          unique_threshold=0.65)
            for mock, options in zip(mocks, settings):
                mock.assert_called_once_with(df, **options)

    def test_individual_rules_are_independent(self):
        df = pd.DataFrame({
            'cid': range(10),
            'measurement': range(10),
            'constant': ['a'] * 10,
            'missing': [None] * 10,
        })
        original = df.copy(deep=True)
        cases = [
            (drop_identifier_columns, {'identifier_columns': ['cid']}, ['measurement', 'constant', 'missing']),
            (drop_missing_columns, {'missing_threshold': 0.3}, ['cid', 'measurement', 'constant']),
            (drop_low_variance_columns, {'dominant_value_threshold': 0.99}, ['cid', 'measurement']),
            (drop_correlated_columns, {'correlation_threshold': 0.95}, ['cid', 'constant', 'missing']),
        ]
        for function, settings, expected in cases:
            with self.subTest(function=function.__name__):
                result = function(df, **settings)
                self.assertEqual(list(result.columns), expected)
                self.assertEqual(list(result.index), list(df.index))
                pd.testing.assert_frame_equal(df, original)

    def test_rules_and_missing_boundary(self):
        df = pd.DataFrame({
            'cid': range(10),
            'keep': range(10),
            'positive_copy': [2 * x for x in range(10)],
            'negative_copy': [-x for x in range(10)],
            'constant': ['a'] * 10,
            'over_30_percent': ['a', 'b', 'c', 'd', 'e', 'f'] + [None] * 4,
            'exactly_30_percent': ['a', 'b', 'c', 'd', 'e', 'f', 'g'] + [None] * 3,
        })
        original = df.copy(deep=True)
        result = prune_columns(df, missing_threshold=0.30, unique_threshold=1)
        self.assertEqual(list(result.columns), ['keep', 'exactly_30_percent'])
        self.assertEqual(len(result), len(df))
        pd.testing.assert_frame_equal(df, original)

    def test_dominant_value_boundary_and_explicit_identifiers(self):
        df = pd.DataFrame({
            'custom_id': range(100),
            'near_constant': ['a'] * 99 + ['b'],
            'variable': ['a'] * 98 + ['b'] * 2,
        })
        result = prune_columns(df, identifier_columns=['custom_id'], dominant_value_threshold=0.99)
        self.assertEqual(list(result.columns), ['variable'])
        self.assertIn('near_constant', prune_columns(df, dominant_value_threshold=1))

    def test_empty_and_missing_columns(self):
        df = pd.DataFrame(columns=['cid', 'measurement'])
        self.assertEqual(list(prune_columns(df).columns), ['measurement'])
        self.assertEqual(prune_columns(pd.DataFrame({'missing': [None] * 3})).shape, (3, 0))

    def test_undefined_correlation_is_retained(self):
        df = pd.DataFrame({'a': [1, 2, None], 'b': [2, 4, None]})
        self.assertEqual(list(prune_columns(df, missing_threshold=1).columns), ['a', 'b'])

    def test_high_cardinality_boundary_missing_values_and_scope(self):
        df = pd.DataFrame({
            'above': list('abcdabcdab'),
            'boundary': list('abcabcabca'),
            'sparse': ['a', 'b'] + [None] * 8,
            'all_missing': [None] * 10,
            'numeric': range(10),
            'numeric_objects': pd.Series(range(10), dtype=object),
            'numeric_category': pd.Categorical(range(10)),
            'text_category': pd.Categorical(list('abcdabcdab')),
            'dates': pd.date_range('2020-01-01', periods=10),
        })
        original = df.copy(deep=True)
        result = drop_high_cardinality_columns(df, unique_threshold=0.3)
        self.assertEqual(list(result.columns), [
            'boundary', 'all_missing', 'numeric', 'numeric_objects',
            'numeric_category', 'dates',
        ])
        pd.testing.assert_frame_equal(df, original)
        pd.testing.assert_frame_equal(
            drop_high_cardinality_columns(df, unique_threshold=1), df
        )
        empty = df.iloc[:0]
        pd.testing.assert_frame_equal(
            drop_high_cardinality_columns(empty, unique_threshold=0.3), empty
        )
        # The wrapper uses the new default threshold without changing numeric columns.
        result = prune_columns(df[['above', 'numeric']])
        self.assertEqual(list(result.columns), ['numeric'])

    def test_invalid_thresholds(self):
        for argument in ['missing_threshold', 'correlation_threshold', 'dominant_value_threshold', 'unique_threshold']:
            with self.subTest(argument=argument), self.assertRaises(ValueError):
                prune_columns(pd.DataFrame(), **{argument: 1.1})


if __name__ == '__main__':
    unittest.main()
