"""Data preprocess functions and classes"""
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer


class GenericPreprocessor(BaseEstimator, TransformerMixin):
    """Generic preprocessing transformer with mutations for specific data types"""
    def __init__(self, features_drop, features_nominal, features_ordinal, features_continuous, categories=None):
        self.features_drop = features_drop
        self.features_nominal = features_nominal
        self.features_ordinal = features_ordinal
        self.features_continuous = features_continuous
        self.categories = categories

    def fit(self, X, y=None):
        X = X.copy()
        X = X.drop(self.features_drop, axis=1, errors='ignore')

        # check if features are in input dataframe
        features_nominal_ = [x for x in X.columns if x in self.features_nominal]
        features_ordinal_ = [x for x in X.columns if x in self.features_ordinal]
        features_continuous_ = [x for x in X.columns if x in self.features_continuous]

        self.transformer_ = ColumnTransformer([
            ('ohe_nominal', OneHotEncoder(sparse=False, categories=self.categories, handle_unknown='ignore'), features_nominal_),
            ('impute_ordinal', SimpleImputer(missing_values=np.nan, strategy='most_frequent'), features_ordinal_),
            ('impute_continuous', IterativeImputer(missing_values=np.nan, min_value=0), features_continuous_)
        ])
        self.transformer_.fit(X, y)

        return self

    def transform(self, X):
        Xt = self.transformer_.transform(X)
        return Xt


if __name__ == '__main__':
    enable_iterative_imputer
