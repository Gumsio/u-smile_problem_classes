"""Integration checks using real random forests."""
from datetime import date
import unittest
import warnings

import numpy as np
import pandas as pd

from src.cleaning_steps.missing_values import clean_missing_values
from src.cleaning_steps.column_pruning import prune_columns


class MissingValueTests(unittest.TestCase):
    def test_mixed_types_preserve_observed_values_and_index(self):
        df = pd.DataFrame({
            'numeric': [1., 2., 3., None, 5., 6., 7., 8.],
            'text': pd.Series(['a', 'a', None, 'a', 'b', 'b', 'b', 'b'], dtype='string'),
            'category': pd.Categorical(['x', 'x', 'x', 'x', 'y', None, 'y', 'y']),
            'nullable': pd.Series([1, 2, 3, 4, 5, None, 7, 8], dtype='Int64'),
            'bool': pd.Series([True, True, True, None, False, False, False, False], dtype='boolean'),
        })
        df.index = [4, 4, 3, 2, 1, 0, -1, -2]
        original = df.copy(deep=True)
        options = dict(n_estimators=8, max_iter=2)
        result = clean_missing_values(df, **options)
        self.assertFalse(result.isna().any().any())
        self.assertEqual(list(result.index), list(df.index))
        self.assertEqual(list(result.columns), list(df.columns))
        for column in df:
            mask = df[column].notna().to_numpy()
            self.assertEqual(result[column].iloc[mask].tolist(), df[column].iloc[mask].tolist())
        self.assertIn(result['text'].iloc[2], ['a', 'b'])
        self.assertIn(result['category'].iloc[5], ['x', 'y'])
        pd.testing.assert_frame_equal(df, original)
        pd.testing.assert_frame_equal(result, clean_missing_values(df, **options))

    def test_unsupported_columns_are_left_unchanged_and_warn(self):
        df = pd.DataFrame({
            'value': [1., None, 3., 4.], 'predictor': [1., 2., 3., 4.],
            'date': [date(2020, 1, 1), None, date(2020, 1, 3), date(2020, 1, 4)],
            'cid': [1., None, 3., 4.], 'empty': [None] * 4,
            'infinite': [1., np.inf, None, 4.],
        })
        with self.assertWarnsRegex(UserWarning, 'MissForest left missing'):
            result = clean_missing_values(df, n_estimators=5, max_iter=1)
        self.assertFalse(result['value'].isna().any())
        for column in ['date', 'cid', 'empty', 'infinite']:
            pd.testing.assert_series_equal(df[column], result[column])

    def test_single_column_and_constants(self):
        with self.assertWarnsRegex(UserWarning, 'insufficient'):
            df = pd.DataFrame({'value': [1., 2., None]})
            pd.testing.assert_frame_equal(clean_missing_values(df), df)
        df = pd.DataFrame({'constant': ['a', None, 'a']})
        self.assertEqual(clean_missing_values(df)['constant'].tolist(), ['a'] * 3)

    def test_numeric_category_codes_use_observed_classes(self):
        df = pd.DataFrame({'class': [10., 20., 30., None, 10., 20., 30., 10.],
                           'predictor': [1., 2., 3., 4., 5., 6., 7., 8.]})
        result = clean_missing_values(df, categorical_columns=['class'], max_iter=1, n_estimators=5)
        self.assertIn(result.loc[3, 'class'], [10., 20., 30.])

    def test_no_missing_and_empty_inputs(self):
        for df in [pd.DataFrame(), pd.DataFrame({'value': [1, 2]})]:
            pd.testing.assert_frame_equal(clean_missing_values(df), df)

    def test_numeric_text_requires_cleaning_before_imputation(self):
        df = pd.DataFrame({'lab': ['1.2', '2.3', 'CHECK', None],
                           'predictor': [1., 2., 3., 4.]})
        with self.assertWarnsRegex(UserWarning, 'mostly numeric text'):
            result = clean_missing_values(df)
        pd.testing.assert_frame_equal(result, df)

    def test_invalid_settings(self):
        for options in [{'max_iter': 0}, {'n_estimators': 0}, {'categorical_columns': 'value'}]:
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                clean_missing_values(pd.DataFrame({'value': [1., None]}), **options)

    def test_imputation_after_pruning_removed_configured_category(self):
        df = pd.DataFrame({
            'cid': range(10),
            'removed_category': [None] * 8 + ['a', 'b'],
            'measurement': [1., 2., 3., None, 5., 6., 7., 8., 9., 10.],
            'predictor': [3., 8., 2., 7., 1., 10., 4., 9., 5., 6.],
        })
        pruned = prune_columns(df)
        original = pruned.copy(deep=True)
        with self.assertWarnsRegex(UserWarning, 'possibly pruned'):
            result = clean_missing_values(
                pruned, categorical_columns=['removed_category'],
                max_iter=1, n_estimators=5,
            )
        self.assertEqual(list(result.columns), list(pruned.columns))
        self.assertNotIn('cid', result)
        self.assertNotIn('removed_category', result)
        self.assertFalse(result.isna().any().any())
        pd.testing.assert_frame_equal(pruned, original)

    def test_pruning_every_column_is_supported(self):
        pruned = prune_columns(pd.DataFrame({'cid': [1, 2, 3], 'constant': ['a'] * 3}))
        self.assertEqual(pruned.shape, (3, 0))
        pd.testing.assert_frame_equal(clean_missing_values(pruned), pruned)


if __name__ == '__main__':
    unittest.main()
