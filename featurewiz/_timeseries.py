import numpy as np
import pandas as pd
from scipy import stats
from math import isnan


def apply(series, period, func, category=None, return_as_cat=None) -> pd.Series:
    """
    Apply 'func' to rolling window grouped by 'category'
    :param series: time-series
    :param period: rolling window period
    :param func: function to apply func(x) -> must return a number
    :param category: categorical series (None or pd.Series)
    :param return_as_cat: (only if category not None) return result as different category, i.e. other categories referencing. Int or iterable.
    :return: pd.Series
    """
    nan_ = np.nan
    result = np.full(len(series), np.nan)
    cat_buff = {}

    if category is None:
        _cat = np.full(len(series), 1)
    else:
        _cat = category.values

    for i, (s, c) in enumerate(zip(series, _cat)):
        _cat_values = cat_buff.setdefault(c, np.full(period, np.nan))
        _cat_values = np.roll(_cat_values, -1)
        _cat_values[-1] = s
        cat_buff[c] = _cat_values

        if return_as_cat is None:
            result[i] = func(_cat_values)
        else:
            #
            try:
                ret_category_key = return_as_cat[i]
            except:
                ret_category_key = return_as_cat

            # Return result as other category (i.e. categorical referencing)
            _ret_cat_values = cat_buff.get(ret_category_key, None)

            if _ret_cat_values is None:
                result[i] = nan_
            else:
                result[i] = func(_ret_cat_values)

    return pd.Series(result, index=series.index)


def _percent_rank(a, mincount=0):
    """
    Returns the percent rank of LAST element of a
    :param a: array-like
    :return: percent rank
    """
    gtcount = 0.0
    cnt = 0.0
    k = len(a)-2
    last_a = a[-1]

    if isnan(last_a):
        return np.nan

    while k >= 0:
        if not isnan(a[k]):
            if last_a >= a[k]:
                gtcount += 1.0
            cnt += 1

        k -= 1

    if cnt < mincount-1 or cnt == 0:
        return np.nan
    else:
        return float(gtcount) / float(cnt) * 100.0


def pctrank(series, period, category=None, categorize_by=None) -> pd.Series:
    """
    Calculates rolling percentile rank
    :param series: pd.Series
    :param period: int
    :param category: if not None apply categorical percent rank (None or pd.Series)
    :param categorize_by: category ranked values by decile size. Must be integer or sequence. categorize_by=3 will create
                          3 uniform categories, and categorize_by=[0, 33, 66, 100] will return 3 categories
                          [0-33;33-66;66-100]. It allows to use non-uniform categories width.
    :return: the rank of series values in range 0-100
    """
    if category is not None:
        result_series = apply(series,
                              period,
                              lambda x: _percent_rank(x, mincount=period),
                              category=category)
    else:
        _ser = series.values

        _result = np.full(len(_ser), np.nan)

        for i in range(period, len(_ser)):
            _result[i] = _percent_rank(_ser[i-period:i+1])

        result_series = pd.Series(_result, index=series.index)

    if categorize_by is None:
        return result_series
    else:
        if isinstance(categorize_by, (int, np.int64, np.int32, np.int16, np.int8)):
            bins_range = [100/categorize_by*i for i in range(categorize_by + 1)]
        else:
            bins_range = categorize_by

        assert bins_range[0] == 0.0, 'categorize_by must include zero at the first element'
        assert bins_range[-1] == 100.0, 'categorize_by must include 100 at the last element'

        return pd.cut(result_series, bins=bins_range, labels=False, include_lowest=True)




