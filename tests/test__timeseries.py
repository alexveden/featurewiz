import numpy as np
import pandas as pd
import unittest
from featurewiz._timeseries import _percent_rank, pctrank, apply


class TimeSeriesTestCase(unittest.TestCase):
    def test__percent_rank(self):
        self.assertEqual(100, _percent_rank([1, 2, 3, 4, 5, 6, 7, 8]))
        self.assertEqual(0, _percent_rank([1, 2, 3, 4, 5, 6, 7, 0]))
        self.assertEqual(4/7*100, _percent_rank([1, 2, 3, 4, 5, 6, 7, 4]))
        self.assertTrue(np.isclose(np.nan, _percent_rank([1, 2, 3, 4, 5, 6, 7, np.nan]), equal_nan=True))

    def test_pctrank_category(self):
        a =   pd.Series([1, 2, 3, 4, 5, 6, 7, 2, 5])
        cat = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1])

        res = pctrank(a, 4, category=cat)
        self.assertTrue(np.allclose(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 100, 1/3*100, 2/3*100]),
                                    res.values, equal_nan=True))

    def test_pctrank(self):
        res = pctrank(pd.Series([1, 2, 3, 4, 5, 6, 7, 0]), 3)
        self.assertTrue(np.allclose(np.array([np.nan, np.nan, np.nan, 100, 100, 100, 100, 0]), res.values, equal_nan=True))

    def test_pctrank_categorize(self):
        self.assertTrue(np.allclose(np.array([np.nan, np.nan, np.nan, 2]),
                                    pctrank(pd.Series([1, 2, 3, 4]), 3, categorize_by=3),
                                    equal_nan=True))

        self.assertTrue(np.allclose(np.array([np.nan, np.nan, np.nan, 1]),
                                    pctrank(pd.Series([1, 2, 3, 2]), 3, categorize_by=3),
                                    equal_nan=True))

        self.assertTrue(np.allclose(np.array([np.nan, np.nan, np.nan, 0]),
                                    pctrank(pd.Series([1, 2, 3, 0]), 3, categorize_by=3),
                                    equal_nan=True))

        self.assertTrue(np.allclose(np.array([np.nan, np.nan, np.nan, 0]),
                                    pctrank(pd.Series([1, 2, 3, 0]), 3, categorize_by=[0, 33, 100]),
                                    equal_nan=True))
        self.assertTrue(np.allclose(np.array([np.nan, np.nan, np.nan, 1]),
                                    pctrank(pd.Series([1, 2, 3, 2.5]), 3, categorize_by=[0, 33, 100]),
                                    equal_nan=True))

    def test_apply(self):
        a = pd.Series([1, 2, 3, 4, 5, 6, 7])
        res_exp = pd.Series([np.nan, np.nan, 6, 9, 12, 15, 18])
        self.assertTrue(np.allclose(res_exp, apply(a, 3, np.sum), equal_nan=True))

    def test_apply_categorical(self):
        a =   pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
        cat = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1])

        res_exp = pd.Series([np.nan, np.nan, np.nan, np.nan,
                             1+3+5,
                             2+4+6,
                             3+5+7,
                             4+6+8,
                             5+7+9,
                             ])

        self.assertTrue(np.allclose(res_exp, apply(a, 3, np.sum, category=cat), equal_nan=True))

    def test_apply_categorical_return_as_cat(self):
        a =   pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
        cat = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1])

        res_exp = pd.Series([np.nan, np.nan, np.nan, np.nan,
                             np.nan,
                             6+4+2,
                             6+4+2,
                             4+6+8,
                             4+6+8,
                             ])
        result = apply(a, 3, np.sum, category=cat, return_as_cat=0)

        self.assertTrue(np.allclose(res_exp, result, equal_nan=True))

    def test_apply_categorical_return_as_cat_series(self):
        a =   pd.Series(    [1, 2, 3, 4, 5, 6, 7, 8, 9])
        cat = pd.Series(    [1, 0, 1, 0, 1, 0, 1, 0, 1])
        cat_ret = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0])

        res_exp = pd.Series([np.nan, np.nan, np.nan, np.nan,
                             np.nan,
                             5+3+1,
                             6+4+2,
                             7+5+3,
                             4+6+8,
                             ])
        result = apply(a, 3, np.sum, category=cat, return_as_cat=cat_ret)

        self.assertTrue(np.allclose(res_exp, result, equal_nan=True))



if __name__ == '__main__':
    unittest.main()
