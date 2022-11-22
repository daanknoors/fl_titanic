"""Statistical functions"""
import numpy as np
from functools import reduce
from src import aggregation
from src import utils

class Statistic:
    """Base class for statistics"""

    def __init__(self):
        pass

    def compute(self, df):
        pass

    def __repr__(self):
        return utils.simplified_repr(self)


class ColumnCategories(Statistic):
    """Create nested list of all column categories"""

    def __init__(self, column_names, flatten_list=False):
        self.column_names = column_names if not isinstance(column_names, str) else [column_names]
        self.flatten_list = flatten_list

    def compute(self, df):
        df = df.copy()

        # convert to category
        df[self.column_names] = df[self.column_names].astype('category')

        # collect categories for each column
        categories = []
        for c in self.column_names:
            # categories need to be ordered in list based input feature indices,
            categories.append(df[c].cat.categories.to_list())

        # optionally flatten list
        if self.flatten_list:
            categories = [item for sublist in categories for item in sublist]

        self.results_ = categories
        return self


class CountRecords(Statistic):
    """Count the number of records in the dataset"""

    def compute(self, df):
        self.results_ = df.shape[0]
        return self
