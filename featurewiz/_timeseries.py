import numpy as np
import pandas as pd
from scipy import stats
from math import isnan


def apply(series, period, func, category=None) -> pd.Series:
    """
    Apply 'func' to rolling window grouped by 'category'
    :param series: time-series
    :param period: rolling window period
    :param func: function to apply func(x) -> must return a number
    :param category: categorical series (None or pd.Series)
    :return: pd.Series
    """
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

        result[i] = func(_cat_values)

        cat_buff[c] = _cat_values

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


def pctrank(series, period, category=None) -> pd.Series:
    """
    Calculates rolling percentile rank
    :param series: pd.Series
    :param period: int
    :param category: if not None apply categorical percent rank (None or pd.Series)
    :return:
    """
    if category is not None:
        return apply(series, 
                     period, 
                     lambda x: _percent_rank(x, mincount=period), 
                     category=category)
    else:
        _ser = series.values

        result = np.full(len(_ser), np.nan)

        for i in range(period, len(_ser)):
            result[i] = _percent_rank(_ser[i-period:i+1])

        return pd.Series(result, index=series.index)



