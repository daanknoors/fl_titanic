"""Client functions for training models on local data"""
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, accuracy_score

from src import utils


class Client:
    """Client that has pointer to local data and stores all local versions of the algorithms"""

    def __init__(self, name, data_pointer, clf_local=None, transformers=None, statistics=None):
        self.name = name
        self.data_pointer = data_pointer
        self.clf_local = clf_local
        self.transformers = transformers
        self.statistics = statistics
        self.clf_scores_ = {'local_scores': [], 'global_scores': []}
        self.clf_results_ = {}

    def fit_classifier(self, classes):
        # pre-process data using transformer pipeline
        X_train, X_test, y_train, y_test = self._run_transformers()

        if not hasattr(self.clf_local, 'coef_'):
            # set first score to 0 for global since the model hasn't been trained yet
            self.clf_scores_['global_scores'].append(0)
        else:
            # store global scores after averaging model and prior to re-training the model
            averaged_model_score = self.clf_local.score(X_test, y_test)
            self.clf_scores_['global_scores'].append(averaged_model_score)

        # re-train the algorithm locally
        self.clf_local.partial_fit(X_train, y_train, classes=classes)

        # compute and store local scores
        local_score = self.clf_local.score(X_test, y_test)
        self.clf_scores_['local_scores'].append(local_score)
        return self

    def evaluate_classifier(self):
        """Run evaluation metrics by running the algorithm on the test data"""
        # run transformer pipeline
        X_train, X_test, y_train, y_test = self._run_transformers()

        # get predictions
        y_pred = self.clf_local.predict(X_test)

        # compute and save metrics
        self.clf_results_['accuracy'] = accuracy_score(y_test, y_pred)
        self.clf_results_['roc_auc_score'] = roc_auc_score(y_test, y_pred)
        self.clf_results_['average_precision'] = average_precision_score(y_test, y_pred)
        self.clf_results_['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        return self

    def _run_transformers(self):
        """Apply pre-processing pipeline"""
        X_train, y_train = self.data_pointer.load_train_data(split_xy=True)
        X_test, y_test = self.data_pointer.load_test_data(split_xy=True)

        X_train = self.transformers.fit_transform(X_train, y_train)
        X_test = self.transformers.transform(X_test)
        return X_train, X_test, y_train, y_test

    def __repr__(self):
        return utils.simplified_repr(self)
