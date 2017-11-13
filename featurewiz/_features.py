import pandas as pd

def features_info_print(dataset):
    """
    Prints basic features information
    :param dataset:
    :return:
    """
    print(f"{'Feature':<30}{'dtype':<10}{'#missing':>10}{'%missing':>10}   {'FeatureType':<30}\n")
    for feature, dtype in dataset.dtypes.items():
        unique_values = set(dataset[feature].dropna())
        missing_cnt = sum(pd.isnull(dataset[feature]))

        if len(unique_values) < 20:
            ftype = f"Categorical ({len(unique_values)})"
        else:
            ftype = 'Numeric'

        print(f"{feature:<30}{str(dtype):<10}{missing_cnt:>10}{missing_cnt/len(dataset)*100:>10.1f}%   {ftype:<10}")