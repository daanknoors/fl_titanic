"""Collaboration of clients that aim to collaboratively train a model on their local data"""
import pandas as pd

from src import aggregation
from src import client
from src import server
from src import utils
from src import preprocess
from src import model


class Collaboration:
    """Clients and algorithms that can be performed in a collaborative federated study"""

    def __init__(self, clients, statistics, transformers, classifier, classifier_aggregator, statistics_aggregators):
        self.clients = clients
        self.statistics = statistics
        self.transformers = transformers
        self.classifier = classifier
        self.classifier_aggregator = classifier_aggregator
        self.statistics_aggregators = statistics_aggregators

    def merge_classifier_scores(self, output_df=False):
        """Combine classifier scores from all clients"""
        self.scores_ = {}
        for c in self.clients:
            self.scores_[c.name] = {}
            score_keys = [k for k in c.clf_scores_.keys() if 'score' in k]
            for s in score_keys:
                self.scores_[c.name][s] = c.clf_scores_[s]
        if not output_df:
            return self.scores_
        return pd.json_normalize(self.scores_).apply(pd.Series.explode).reset_index(drop=True)

    def __repr__(self):
        return utils.simplified_repr(self)

