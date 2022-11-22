import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle

from src.preprocess import GenericPreprocessor


class BatchSGD(SGDClassifier):
    """SGD Classifier that can run on (mini)-batches of data"""

    def __init__(
            self,
            loss="hinge",
            *,
            penalty="l2",
            alpha=0.0001,
            l1_ratio=0.15,
            fit_intercept=True,
            max_iter=1000,
            tol=1e-3,
            shuffle=True,
            verbose=0,
            epsilon=0.1,
            n_jobs=None,
            random_state=None,
            learning_rate="optimal",
            eta0=0.0,
            power_t=0.5,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5,
            class_weight=None,
            warm_start=False,
            average=False,
            batch_size=0,
            n_local_iterations=1,
    ):
        self.n_local_iterations = n_local_iterations
        self.batch_size = batch_size
        super().__init__(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            n_jobs=n_jobs,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            class_weight=class_weight,
            warm_start=warm_start,
            average=average,
        )

    def fit(self, X, y):
        """Helper function for partial fit, when calling fit directly run partial fit to perform Batch SGD"""
        if hasattr(self, 'classes_'):
            classes = self.classes_
        else:
            classes = np.unique(y)
        return self.partial_fit(X, y, classes=classes)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Perform partial fit on batch of samples"""
        # split data in batches
        X_batches, y_batches = self._split_batches(X, y)

        # train model for specified number of epochs
        for _ in range(0, self.n_local_iterations):
            for X_batch, y_batch in zip(X_batches, y_batches):
                super().partial_fit(X_batch, y_batch, classes=classes, sample_weight=sample_weight)
        return self

    def _split_batches(self, X, y):
        """Split features and target in batches according to batch_size"""
        batch_size = self._check_batch_size(X.shape[0])

        # shuffle X and Y
        X, y = shuffle(X, y)
        n_splits = X.shape[0] / batch_size

        # split batches
        X_batches = np.array_split(X, n_splits)
        y_batches = np.array_split(y, n_splits)
        return X_batches, y_batches

    def _check_batch_size(self, n_records):
        if (self.batch_size is None) or (self.batch_size == 0) or (self.batch_size > n_records):
            return n_records
        return self.batch_size


def get_standard_transformer_pipeline(features_drop, features_nominal, features_ordinal, features_continuous,
                                      categories=None):
    """Standard pipeline of transformers to pre-process and scale data"""
    pipe = Pipeline([
        ('preprocess', GenericPreprocessor(features_drop=features_drop, features_nominal=features_nominal,
                                           features_ordinal=features_ordinal, features_continuous=features_continuous,
                                           categories=categories)),
        ('scale', RobustScaler()),
    ])
    return pipe


def get_standard_classifier(random_state):
    """Get the BatchSGD algorithm with standard parameters"""
    return BatchSGD(loss='log_loss', batch_size=20, n_local_iterations=100, random_state=random_state)
