"""Data classes"""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src import config


class LocalData:
    """Data class that only stores filenames ways to access the data and preprocessing functions.
    The actual data is not stored in the class to prevent unauthorized access.
    Only supports csv's for now.
    """

    def __init__(self, path, filename, name, target_column, delimiter=',', random_state=None):
        self.data_path = path
        self.filename = filename
        self.name = name
        self.target_column = target_column
        self.delimiter = delimiter
        self.random_state = random_state

    def load_raw_data(self):
        data_path_raw = self.data_path / 'raw'
        return pd.read_csv(data_path_raw / self.filename, delimiter=self.delimiter)

    def load_train_data(self, split_xy=True):
        data_path_train = self.data_path / 'preprocessed'
        filename = f'{self.name}_train.csv'
        df = pd.read_csv(data_path_train / filename, delimiter=self.delimiter)
        if split_xy:
            X, y = self._split_x_y(df)
            return X, y
        return df

    def load_test_data(self, split_xy=True):
        data_path_test = self.data_path / 'preprocessed'
        filename = f'{self.name}_test.csv'
        df = pd.read_csv(data_path_test / filename, delimiter=self.delimiter)
        if split_xy:
            X, y = self._split_x_y(df)
            return X, y
        return df

    def create_train_test(self, test_size=0.2):
        df = self.load_raw_data()

        # split train and test
        df_train, df_test = train_test_split(df, test_size=test_size, stratify=self.target_column,
                                             random_state=self.random_state)

        # save to csv
        data_path_processed = self.data_path / 'processed'
        df_train.to_csv(data_path_processed / f'{self.name}_train.csv', index=False)
        df_test.to_csv(data_path_processed / f'{self.name}_test.csv', index=False)
        print(f'Train and test data saved at: {data_path_processed}')

    def _split_x_y(self, df):
        """Split X and y in dataframe"""
        y = df[self.target_column]
        X = df.drop(columns=[self.target_column])
        return X, y


