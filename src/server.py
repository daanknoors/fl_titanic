"""Server functions for aggregating model updates"""
from copy import deepcopy

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
        """Run local statistics and aggregate them."""
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

    def _update_local_model(self, client_instance):
        # set global classifier to local client instance
        client_instance.clf_local = deepcopy(self.collab.classifier)

    def fit_classifier(self, classes):
        """train local classifiers, aggregate the parameters and re-distribute"""
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

        # update local models after final iteration
        self._update_local_model(client_instance)

    def evaluate_classifier(self):
        """Request clients to evaluate the trained classifier on their test data and collect results"""
        self.collab.clf_results_ = {}
        for client_instance in self.collab.clients:
            client_instance.evaluate_classifier()
            self.collab.clf_results_[client_instance.name] = client_instance.clf_results_

    def __repr__(self):
        return utils.simplified_repr(self)


