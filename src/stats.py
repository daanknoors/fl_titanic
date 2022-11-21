"""Statistical functions"""
import numpy as np
from functools import reduce
from src import aggregation


class Statistic:

    def __init__(self):
        pass

    def compute(self, df):
        pass
    #
    # def aggregate(self):
    #     pass


class ColumnCategories(Statistic):

    def __init__(self, column_names, flatten_list=False):
        self.column_names = column_names if not isinstance(column_names, str) else [column_names]
        self.flatten_list = flatten_list

    def compute(self, df):
        df = df.copy()

        df[self.column_names] = df[self.column_names].astype('category')
        categories = []

        for c in self.column_names:
            # categories need to be ordered in list based input feature indices,
            categories.append(df[c].cat.categories.to_list())

        if self.flatten_list:
            categories = [item for sublist in categories for item in sublist]

        self.results_ = categories
        return self

    # def aggregate(self):
    #     # combine categories from all datasets
    #     array_categories = np.array(list(self.local_results_.values()), dtype='object')
    #     combined_categories = reduce(np.add, array_categories)
    #
    #     # filter unique categories per column
    #     combined_categories_unique = [list(set(c)) for c in combined_categories]
    #
    #     self.global_results_ = combined_categories_unique
    #     return self


class CountRecords(Statistic):

    def compute(self, df):
        self.results_ = df.shape[0]
        return self

    # def aggregate(self):
    #     total_count = np.sum(list(self.local_results_.values()))
    #     self.global_results_ = total_count
    #     return self

