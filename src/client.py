"""Client functions for training models on local data"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from src import config
from src import data


class Client:
    """Client that has pointer to local data"""

    def __init__(self, name, data_object, clf_local=None, transformers=None, statistics=None):
        self.name = name
        self.data_object = data_object
        self.clf_local = clf_local
        self.transformers = transformers
        self.statistics = statistics
        self.local_scores_ = []
        self.global_scores_ = []

    def _check_params(self):
        #todo check whether data inherits from data class (or subclasses it)
        pass

    def fit_classifier(self, classes, batch_size, n_local_iterations):
        X_train, X_test, y_train, y_test = self._run_transformers()

        if not hasattr(self.clf_local, 'coef_'):
            # self.clf_local.fit(X_train, y_train)
            # set first score to 0 for global since the model hasn't been trained yet
            self.global_scores_.append(0)
        else:
            # store global scores after averaging model and prior to re-training the model
            averaged_model_score = self.clf_local.score(X_test, y_test)
            self.global_scores_.append(averaged_model_score)

        # split data in batches
        batch_size = self._check_batch_size(batch_size, X_train.shape[0])
        X_batches, y_batches = self._split_batches(X_train, y_train, batch_size)

        # train model for specified number of epochs
        for _ in range(0, n_local_iterations):
            for X_batch, y_batch in zip(X_batches, y_batches):
                self.clf_local.partial_fit(X_batch, y_batch, classes=classes)

            # store scores
            local_score = self.clf_local.score(X_test, y_test)
            self.local_scores_.append(local_score)

        return self

    def _check_batch_size(self, batch_size, n_records):
        if(batch_size is None) or (batch_size == 0) or (batch_size > n_records):
            return n_records
        return batch_size

    def _split_batches(self, X, y, batch_size):
        """Split features and target in batches according to batch_size"""
        # shuffle X and Y
        X, y = shuffle(X, y)
        n_splits = X.shape[0] / batch_size
        X_batches = np.array_split(X, n_splits)
        y_batches = np.array_split(y, n_splits)
        return X_batches, y_batches



    def _run_transformers(self):
        X_train, y_train = self.data_object.load_train_data(split_xy=True)
        X_test, y_test = self.data_object.load_test_data(split_xy=True)

        X_train = self.transformers.fit_transform(X_train, y_train)
        X_test = self.transformers.transform(X_test)
        return X_train, X_test, y_train, y_test



# class ClientUpdate(BaseEstimator, ClassifierMixin):
#
#     # def __init__(self, clf, coef_init, intercept_init, classes_init, n_local_iterations, batch_size, learning_rate,
#     #              randomize_batches=True):
#     #     self.clf = clf
#     #     self.coef_init = coef_init
#     #     self.intercept_init = intercept_init
#     #     self.classes_init = classes_init
#     #     self.n_local_iterations = n_local_iterations
#     #     self.batch_size = batch_size
#     #     self.learning_rate = learning_rate
#     #     self.randomize_batches = randomize_batches
#
#     def __init__(self, clf, n_epochs, batch_size, randomize_batches=True):
#         self.clf = clf
#         self.n_epochs = n_epochs
#         self.batch_size = batch_size
#         self.randomize_batches = randomize_batches
#
#     def fit(self, X, y):
#         clf = self._check_params()
#         X, y = self._check_input(X, y)
#
#         # set weights of algorithm prior to training
#         # self.set_weights()
#
#         # apply transformers, as Pipeline has no partial_fit()
#         if isinstance(self.clf, Pipeline):
#             X = self._apply_transformers(X, y)
#
#         clf = self._fit(X, y)
#
#         # assign back to pipeline
#         if isinstance(self.clf, Pipeline):
#             name = self.clf.steps[-1][0]
#             self.clf.steps[-1] = (name, clf)
#         else:
#             self.clf = clf
#         return self
#
#
#     def _fit(self, X, y):
#         clf = self.clf
#         if isinstance(clf, Pipeline):
#             clf = clf.steps[-1][1]
#
#         # split data in batches
#         batch_size = self.batch_size if (self.batch_size is not None or self.batch_size == 0) else X.shape[0]
#         X_batches, y_batches = self._split_batches(X, y, batch_size)
#
#         # train model for specified number of epochs
#         for _ in range(0, self.n_epochs):
#             for X_batch, y_batch in zip(X_batches, y_batches):
#                 # if algorithm hasn't been fitted yet
#                 # if not hasattr(clf, 'coef_'):
#                 #     clf.fit(X_batch, y_batch)
#                 # else:
#                 #     clf.partial_fit(X_batch, y_batch)
#                 clf.fit(X_batch, y_batch)
#
#         return clf
#
#     def _apply_transformers(self, X, y):
#         """Apply transformers in pipeline"""
#         if isinstance(self.clf, Pipeline):
#             transformers = Pipeline(self.clf.steps[:-1])
#             if check_is_fitted(transformers):
#                 X = transformers.transform(X)
#             else:
#                 X = transformers.fit_transform(X, y)
#             # add fitted transformers back to pipeline
#             self.clf.steps[:-1] = transformers.steps
#         return X
#
#     def _split_batches(self, X, y, batch_size):
#         """Split features and target in batches according to batch_size"""
#         if self.randomize_batches:
#             X, y = shuffle(X, y)
#         n_splits = X.shape[0] / batch_size
#         X_batches = np.array_split(X, n_splits)
#         y_batches = np.array_split(y, n_splits)
#         return X_batches, y_batches
#
#     # def set_weights(self):
#     #     """Set weights of classifier using coef_, intercept_, and classes_ attributes"""
#     #     self.clf.coef_ = self.coef_init.copy()
#     #     self.clf.intercept_ = self.intercept_init.copy()
#     #     self.clf.classes_ = self.clases_init.copy()
#
#     def _check_params(self):
#         # if hasattr(self.clf, 'best_estimator_'):
#
#         if isinstance(self.clf, Pipeline):
#             clf = self.clf.steps[-1][1]
#         else:
#             clf = self.clf
#         assert isinstance(clf, SGDClassifier), 'ClientUpdate only works with SGDClassifier'
#         return clf
#
#     def _check_input(self, X, y):
#         self.n_records_ = X.shape[1]
#         self.classes_ = np.unique(y)
#         # X = np.array(X)
#         # y = np.array(y)
#         return X, y







