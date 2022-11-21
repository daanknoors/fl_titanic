"""Server functions for aggregating model updates"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from copy import deepcopy

from src import collaboration
from src import aggregation
from src import stats
from src import model
from src import utils


class Server:
    """Orchestrates the movement of algorithms between clients in collaboration"""

    def __init__(self, collab):
        self.collab = collab
        self.distribute_algorithms()

    def distribute_algorithms(self):
        """Send all algorithms as part of the collaboration to the respective clients"""
        for client_instance in self.collab.clients:
            client_instance.statistics = deepcopy(self.collab.statistics)
            client_instance.transformers = deepcopy(self.collab.transformers)
            client_instance.clf_local = deepcopy(self.collab.classifier)
        print('All algorithms are distributed to clients successfully')

    def run_statistics(self):
        # compute local stats
        for client_instance in self.collab.clients:
            df_train = client_instance.data_pointer.load_train_data(split_xy=False)
            for method_name, stat_method in client_instance.statistics.items():
                stat_method.compute(df_train)

        # aggregate local stats to get global stats
        for method_name, stat_aggregator in self.collab.statistics_aggregators.items():
            # get all local stat results
            local_stats = [c.statistics[method_name] for c in self.collab.clients]

            # combine local results and assign to collab results
            self.collab.statistics[method_name].results_ = stat_aggregator.aggregate(local_stats)
        return self

    # def _run_transformers(self, client_instance):
    #     X_train, y_train = client_instance.data_pointer.load_train_data(split_xy=True)
    #     X_test, y_test = client_instance.data_pointer.load_test_data(split_xy=True)
    #
    #     X_train = self.collab.transformers.fit_transform(X_train, y_train)
    #     X_test = self.collab.transformers.transform(X_test)
    #     return X_train, X_test, y_train, y_test

    # def _init_scores(self):
    #     """Initialize score attributes to evaluate model training iterations"""
    #     self.local_scores_ = {}
    #     self.global_scores_ = {}
    #     for c in self.collab.clients:
    #         self.local_scores_[c.name] = []
    #         self.global_scores_[c.name] = []

    def _update_local_model(self, client_instance):
        # set global classifier to local client instance
        client_instance.clf_local = deepcopy(self.collab.classifier)

    def fit_classifier(self, classes):
        for _ in range(self.collab.classifier_aggregator.n_iterations):
            for client_instance in self.collab.clients:
                self._update_local_model(client_instance)
                client_instance.fit_classifier(classes=classes)

            # get local classifiers
            local_classifiers = [c.clf_local for c in self.collab.clients]

            # combine coefficients, intercept and classes
            coef, intercept, classes = self.collab.classifier_aggregator.aggregate(local_classifiers)

            # update global model
            self.collab.classifier.coef_ = coef
            self.collab.classifier.intercept_ = intercept
            self.collab.classifier.classes_ = classes

    def __repr__(self):
        return utils.simplified_repr(self)

    # def _fit_local_classifiers(self):
    #     for client_instance in self.collab.clients:
    #         X_train, X_test, y_train, y_test = self._run_transformers(client_instance=client_instance)
    #
    #         # set global classifier to local client instance
    #         client_instance.clf_local = deepcopy(self.collab.classifier)
    #
    #         # todo turn into mini batch SGD
    #         if not hasattr(client_instance.clf_local, 'coef_'):
    #             client_instance.clf_local.fit(X_train, y_train)
    #             # set first score to 0 for global since the model hasn't been trained yet
    #             self.global_scores_[client_instance.name].append(0)
    #         else:
    #             # store global scores after averaging model and prior to re-training the model
    #             averaged_model_score = self.collab.classifier.score(X_test, y_test)
    #             self.global_scores_[client_instance.name].append(averaged_model_score)
    #
    #             # todo now it only does one sample i think
    #             client_instance.clf_local.partial_fit(X_train, y_train)
    #
    #         # todo which scores to store? Now only store score after re-training global on locat data
    #         local_score = client_instance.clf_local.score(X_test, y_test)
    #         self.local_scores_[client_instance.name].append(local_score)
    #     return self








