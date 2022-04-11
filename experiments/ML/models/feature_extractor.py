from tsfresh import extract_features
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
import os

def ts_feature_extraction(X_raw, output_path):
    '''
    :param X_raw: raw pandas dataset
    :param output_path: output in which to save the dataset with features extracted
    :return: X dataset with extracted features from tsfresh
    '''
    # features extraction #
    # extract all features with ts fresh and save the resulting dataframe
    X = extract_features(X_raw, column_id="ParameterSet", column_sort="No", column_kind=None, column_value=None)
    X.to_csv(path_or_buf=output_path)
    return X


def ts_features_cleaner(X):
    # retrieve columns with nan, -inf or inf value inside.
    clean_list = X.columns[X.isin([np.nan, np.inf, -np.inf]).any()].tolist()
    X.drop(clean_list, axis = 1, inplace = True)
    return X


# feature selection using sklearn
def select_features(X_train, y_train, X_test):
    # configure to select all features
    fs = SelectKBest(score_func=mutual_info_regression, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
