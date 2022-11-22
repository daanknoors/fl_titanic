"""Aggregating strategies for combining output from federated algorithms"""
import numpy as np
from functools import reduce

from src import utils

class Aggregator:
    """Base aggregator class"""

    def __init__(self):
        pass
    def __repr__(self):
        return utils.simplified_repr(self)


class SumValues(Aggregator):
    """Sum values from statistics, e.g. total count"""

    def aggregate(self, local_statistics):
        combined_local_results = [l.results_ for l in local_statistics]
        sum_local_results = np.sum(combined_local_results)
        return sum_local_results


class ListUnion(Aggregator):
    """Combine two or more lists into one, optionally filter for unique values, e.g. combine target classes"""

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
    """Combine two or more nested lists into one nested listed, e.g. combine all nominal categories"""

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
    """Federated averaging as proposed by McMahan et al (2017)
    article: https://arxiv.org/abs/1602.05629"""

    def __init__(self, n_global_iterations, weights=None):
        self.n_iterations = n_global_iterations
        self.weights = weights

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