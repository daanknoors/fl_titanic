"""Data preprocess functions for modeling"""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from src import config


"""Load data files"""


def load_raw_data_a():
    return pd.read_csv(config.PATH_DATA_RAW / config.FILENAME_DATA_A)


def load_raw_data_b():
    return pd.read_csv(config.PATH_DATA_RAW / config.FILENAME_DATA_B)


def load_train_data():
    df_train_a = pd.read_csv(config.PATH_DATA_PREPROCESSED / config.FILENAME_TRAIN_DATA_A)
    df_train_b = pd.read_csv(config.PATH_DATA_PREPROCESSED / config.FILENAME_TRAIN_DATA_B)

    return df_train_a, df_train_b


def load_test_data():
    df_test_a = pd.read_csv(config.PATH_DATA_PREPROCESSED / config.FILENAME_TEST_DATA_A)
    df_test_b = pd.read_csv(config.PATH_DATA_PREPROCESSED / config.FILENAME_TEST_DATA_B)
    return df_test_a, df_test_b


def create_train_test_data(test_size=0.2):
    """Split input datasets in training and testing and save to disk"""
    # load data
    df_a = load_raw_data_a()
    df_b = load_raw_data_b()

    for df, name in zip([df_a, df_b], ['A', 'B']):
        # split train and test
        df_train, df_test = train_test_split(df, test_size=test_size, stratify=df[config.TARGET],
                                             random_state=config.RANDOM_STATE)

        # save to csv
        df_train.to_csv(config.PATH_DATA_PREPROCESSED / f'{name}_train.csv', index=False)
        df_test.to_csv(config.PATH_DATA_PREPROCESSED / f'{name}_test.csv', index=False)


"""Data pre-processing functions"""


def preprocess_data(df):
    """Data pre-processing steps"""
    df = df.copy()
    df = df.drop(config.FEATURES_DROP, errors='ignore')
    return df


def split_x_y(df, y_column=None):
    """Split X and y in dataframe"""
    if not y_column:
        y_column = config.TARGET

    y = df[y_column]
    X = df.drop(columns=[y_column])
    return X, y


class GenericPreprocessor(BaseEstimator, TransformerMixin):
    """Generic preprocessing transformer with mutations for specific data types"""
    def __init__(self, categories=None):
        self.categories = categories


    def fit(self, X, y=None):
        X = X.copy()
        X = X.drop(config.FEATURES_DROP, errors='ignore')

        # define feature categories and check if in input dataframe
        self.features_nominal_ = [x for x in X.columns if x in config.FEATURES_NOMINAL]
        self.features_ordinal_ = [x for x in X.columns if x in config.FEATURES_ORDINAL]
        self.features_continuous_ = [x for x in X.columns if x in config.FEATURES_RANGE + config.FEATURES_INTERVAL]

        self.transformer_ = ColumnTransformer([
            ('ohe_nominal', OneHotEncoder(sparse=False, categories=self.categories, handle_unknown='ignore'), self.features_nominal_),
            ('impute_ordinal', SimpleImputer(missing_values=np.nan, strategy='most_frequent'), self.features_ordinal_),
            ('impute_continuous', SimpleImputer(missing_values=np.nan, strategy='median'), self.features_continuous_)
        ])
        self.transformer_.fit(X, y)

        return self

    def transform(self, X):
        # self._check_input(X)
        Xt = self.transformer_.transform(X)
        return Xt

    # def _check_input(self, X):
    #     features_fit = set(self.features_nominal_ + self.features_ordinal_ + self.features_continuous_)
    #     print(set(X.columns))
    #     print(features_fit)
    #     assert set(X.columns) == features_fit, 'Input dataframe X has columns not seen in fit'