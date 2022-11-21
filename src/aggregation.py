"""Aggregating strategies for combining output from federated algorithms"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from functools import reduce
from src import utils

class Aggregator:

    def __init__(self):
        pass
    def __repr__(self):
        return utils.simplified_repr(self)

class SumValues(Aggregator):

    def aggregate(self, local_statistics):
        combined_local_results = [l.results_ for l in local_statistics]
        sum_local_results = np.sum(combined_local_results)
        return sum_local_results

class ListUnion(Aggregator):

    def __init__(self, unique=True):
        self.unique = True

    def aggregate(self, local_statistics):
        combined_local_results = [l.results_ for l in local_statistics]
        aggregated_results = []
        for r in combined_local_results:
            aggregated_results.extend(r)

        if self.unique:
            aggregated_results = list(set(aggregated_results))
        return aggregated_results


class NestedListUnion(Aggregator):

    def __init__(self, unique=True):
        self.unique = True

    def aggregate(self, local_statistics):
        combined_local_results = [l.results_ for l in local_statistics]

        # merge nested lists
        array_categories = np.array(combined_local_results, dtype='object')
        aggregated_results = reduce(np.add, array_categories)

        # filter unique categories per list
        if self.unique:
            aggregated_results = [list(set(c)) for c in aggregated_results]
        return aggregated_results


class FedAvg(Aggregator):

    def __init__(self, n_global_iterations, weights=None):
        self.n_iterations = n_global_iterations
        self.weights = weights
        super().__init__()

    def aggregate(self, local_classifiers=None):
        # update global classifier by averaging coefficients and intercept from all local classifiers
        global_coef = np.average([c.coef_ for c in local_classifiers], axis=0, weights=self.weights)
        global_intercept = np.average([c.intercept_ for c in local_classifiers], axis=0, weights=self.weights)

        global_classes = []
        for client in local_classifiers:
            global_classes.extend(client.classes_)
        global_classes = np.array(list(set(global_classes)))
        return global_coef, global_intercept, global_classes

    def set_weights(self, local_counts, global_count):
        self.weights = local_counts / global_count
        return self

    def __repr__(self):
        return utils.simplified_repr(self)

# class FedAvg(BaseEstimator, ClassifierMixin):
#     """Federated averaging as proposed by McMahan et al (2017)
#     article: https://arxiv.org/abs/1602.05629
#     """
#
#     def __init__(self, clf_global, clf_clients, n_clients, n_global_iterations, aggregation_strategy='weighted'):
#         self.clf_global = clf_global
#         self.clf_clients = clf_clients
#         self.n_clients = n_clients
#         self.n_global_iterations = n_global_iterations
#         self.aggregation_strategy = aggregation_strategy
#
#     def fit(self, X, y):
#         # todo: case where client sends one model to server to be passed to the others - no average needed
#         if len(self.clf_clients.keys()) == 1:
#             client_name = self.clf_clients.keys()[0]
#             self.clf_global = self.clf_clients[client_name].copy()
#             return self
#
#         # compute weights
#         n_records_client = np.array([c.n_records for _, c in self.clf_clients.items()])
#         n_samples_total = sum(n_records_client)
#         weights = n_records_client / n_samples_total if self.aggregation_strategy == 'weighted' else None
#
#         # update global classifier by averaging coef and intercept from all client models
#         self.clf_global.coef_ = np.average([c for c in self.clf_clients.values.coef], axis=0, weights=weights)
#         self.clf_global.intercept_ = np.average([c for c in self.clf_clients.values.coef], axis=0, weights=weights)
#         return self
#
#     def predict(self, X):
#         return self.clf_global.predict(X)