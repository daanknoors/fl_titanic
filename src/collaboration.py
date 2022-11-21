"""Collaboration of clients that aim to collaboratively train a model on their local data"""
import pandas as pd

from src import aggregation
from src import client
from src import server
from src import utils
from src import preprocess
from src import model


class Collaboration:
    """Clients, server, and algorithms that can be performed in a collaboration"""

    def __init__(self, clients, statistics, transformers, classifier, classifier_aggregator, statistics_aggregators):
        self.clients = clients
        self.statistics = statistics
        self.transformers = transformers
        self.classifier = classifier
        self.classifier_aggregator = classifier_aggregator
        self.statistics_aggregators = statistics_aggregators

    def merge_collab_scores(self, output_df=False):
        self.scores_ = {}
        for c in self.clients:
            self.scores_[c.name] = {
                'local': c.local_scores_,
                'global': c.global_scores_
            }
        if not output_df:
            return self.scores_
        return pd.json_normalize(self.scores_).apply(pd.Series.explode).reset_index(drop=True)

    # def __repr__(self):
    #     # return f'{self.__class__.__name__}(clients={self.clients}'
    #     attributes = ", ".join([f'{k}={v}' for k, v in vars(self).items()])
    #
    #     return f'{self.__class__.__name__}(\n{attributes})'


