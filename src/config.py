"""Configuration settings"""
import numpy as np
import os
from pathlib import Path

from src import aggregation
from src import client
from src import collaboration
from src import data
from src import model
from src import server
from src import stats


class ServerTitanic(server.Server):
    def __init__(self):
        super(ServerTitanic, self).__init__(
            collab=CollaborationTitanic()
        )

    # def _run_transformers(self, client_instance):
    #     """Adapt transformers by adding feature categories in OHE function in transformer pipeline"""
    #     if not hasattr(self.collab.statistics['nominal_categories'], 'results_'):
    #         self.run_statistics()
    #
    #     global_categories = self.collab.statistics['nominal_categories'].results_
    #
    #     # specify feature categories to pipeline to ensure categories from both datasets appear after one-hot encoding
    #     self.collab.transformers.named_steps['preprocess'].categories = global_categories
    #     return super(ServerTitanic, self)._run_transformers(client_instance=client_instance)

    def fit_classifier(self):
        if not hasattr(self.collab.statistics['nominal_categories'], 'global_results_'):
            self.run_statistics()

        # set categories of OHE in transformer pipeline
        global_categories = self.collab.statistics['nominal_categories'].results_
        for c in self.collab.clients:
            c.transformers.named_steps['preprocess'].categories = global_categories

        # set weights of FedAvg classifier_aggregator by taking the counts from CountRecord statistic
        local_counts = np.array([c.statistics['count_records'].results_ for c in self.collab.clients])
        global_count = self.collab.statistics['count_records'].results_
        self.collab.classifier_aggregator.set_weights(local_counts=local_counts, global_count=global_count)

        # get target classes
        target_classes = self.collab.statistics['target_classes'].results_
        return super(ServerTitanic, self).fit_classifier(classes=target_classes)


class CollaborationTitanic(collaboration.Collaboration):

    def __init__(self):
        super(CollaborationTitanic, self).__init__(
            clients=[ClientA(), ClientB()],
            statistics={
                'count_records': stats.CountRecords(),
                'nominal_categories': stats.ColumnCategories(FEATURES_NOMINAL, flatten_list=False),
                'target_classes': stats.ColumnCategories(TARGET, flatten_list=True)
            },
            transformers=model.get_standard_transformer_pipeline(),
            # classifier=model.LocalClassifier(clf=model.get_standard_classifier(), n_local_iterations=10, batch_size=50),
            # classifier=model.get_standard_classifier(),
            classifier=model.get_standard_classifier(random_state=RANDOM_STATE),
            classifier_aggregator=aggregation.FedAvg(n_global_iterations=N_GLOBAL_ITERATIONS),
            statistics_aggregators={
                'count_records': aggregation.SumValues(),
                'nominal_categories': aggregation.NestedListUnion(unique=True),
                'target_classes': aggregation.ListUnion(unique=True)
            }
        )


class ClientA(client.Client):

    def __init__(self):
        super(ClientA, self).__init__(
            name='A',
            data_object=DataA(),
            clf_local=None,
            transformers=None,
            statistics=None
        )


class ClientB(client.Client):

    def __init__(self):
        super(ClientB, self).__init__(
            name='B',
            data_object=DataB(),
            clf_local=None,
            transformers=None,
            statistics=None
        )


class DataA(data.LocalData):

    def __init__(self):
        super(DataA, self).__init__(
            path=PATH_DATA,
            filename=FILENAME_DATA_A,
            name='A',
            target_column=TARGET,
            delimiter=',',
            random_state=RANDOM_STATE
        )


class DataB(data.LocalData):

    def __init__(self):
        super(DataB, self).__init__(
            path=PATH_DATA,
            filename=FILENAME_DATA_B,
            name='B',
            target_column=TARGET,
            delimiter=',',
            random_state=RANDOM_STATE
        )





# paths
PATH_PROJECT = Path(os.getcwd()).parent
PATH_DATA = PATH_PROJECT / 'data'
PATH_DATA_RAW = PATH_DATA / 'raw'
PATH_DATA_PREPROCESSED = PATH_DATA / 'preprocessed'
PATH_MODEL = PATH_PROJECT / 'models'

# filenames
FILENAME_DATA_A = 'titanic_setA.csv'
FILENAME_DATA_B = 'titanic_setB.csv'
FILENAME_TRAIN_DATA_A = 'A_train.csv'
FILENAME_TEST_DATA_A = 'A_test.csv'
FILENAME_TRAIN_DATA_B = 'B_train.csv'
FILENAME_TEST_DATA_B = 'B_test.csv'
FILENAME_TEST_UNLABELED = 'test.csv'

# config settings
RANDOM_STATE = 42
N_GLOBAL_ITERATIONS = 20
SCORING_FUNCTION = 'roc_auc'
COlOR_PALETTE = ["#393e46", "#00cde3", "#ff5722", "#d72323"]

# features categories
FEATURES_NOMINAL = ['Sex', 'Cabin', 'Embarked']
FEATURES_ORDINAL = ['Pclass', 'SibSp', 'Parch']
FEATURES_INTERVAL = []
FEATURES_RANGE = ['Age', 'Fare']

FEATURES_MINIMAL = ...
FEATURES_DROP = ['PassengerID', 'pred', 'Name', 'Ticket']

# target columns
TARGET = 'Survived'
