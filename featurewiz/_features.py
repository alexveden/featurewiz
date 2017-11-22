import pandas as pd
import numpy as np


def features_info_print(dataset):
    """
    Prints basic features information
    :param dataset:
    :return:
    """
    print(f"{'Feature':<40}{'dtype':<10}{'#missing':>10}{'%missing':>10}   {'FeatureType':<30}\n")
    for feature, dtype in dataset.dtypes.items():
        unique_values = set(dataset[feature].dropna())
        missing_cnt = sum(pd.isnull(dataset[feature]))

        if len(unique_values) < 20:
            ftype = f"Categorical ({len(unique_values)})"
        else:
            ftype = 'Numeric'

        print(f"{feature:<40}{str(dtype):<10}{missing_cnt:>10}{missing_cnt/len(dataset)*100:>10.1f}%   {ftype:<10}")

def features_check_for_bad_values(dataset, y_series=None, raise_error=True):
    def _is_series_has_error(name, series):
        has_error = False

        nan_cnt = pd.isnull(series).sum()
        inf_cnt = np.isinf(series).sum()

        if nan_cnt > 0:
            has_error = True
            print(f"{name}: ERR found {nan_cnt} NaN values")

        if inf_cnt > 0:
            has_error = True
            print(f"{name}: ERR found {inf_cnt} infinity values")

        return has_error

    _dataset_is_valid = True
    for col in dataset:
        if _is_series_has_error(col, dataset[col]):
            _dataset_is_valid = False

    if y_series is not None:
        if _is_series_has_error('y_series', y_series):
            _dataset_is_valid = False

    if not _dataset_is_valid:
        if raise_error:
            raise ValueError("Dataset or y_series is invalid")
    else:
        print("Dataset seems to be valid")


def encode_one_hot(series, series_name, max_categories=30, as_dict=True):
    unique_values = set(series.dropna())
    if len(unique_values) > max_categories:
        raise ValueError(f"Too many categories for {series_name} got {len(unique_values)} max: {max_categories}")

    dummydf = pd.get_dummies(series, prefix=series_name)

    if as_dict:
        return dummydf.to_dict(orient='series')
    else:
        return dummydf


def imput_spikes(series, q=0.01, window=None, ffill=True):
    if q > 0.5:
        raise ValueError(f"q parameter must be <= 0.5, got {q}")
    if window is None:
        q_upper = series.expanding().quantile(1.0-q)
        q_lower = series.expanding().quantile(q)
    else:
        q_upper = series.rolling(window).quantile(1.0 - q)
        q_lower = series.rolling(window).quantile(q)

    _series = series.copy()
    _series[_series > q_upper] = q_upper
    _series[_series < q_lower] = q_lower

    if ffill:
        _series.ffill(inplace=True)

    return _series
