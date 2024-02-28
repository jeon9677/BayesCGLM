import os
import sys
import numpy as np
import pandas as pd
from utils import save_np
from params import Params as P


class Dataset(object):
    num_x = 239

    def __init__(self):
        self.data_dir = 'malaria_data'
        self.x_names = [f'x{i + 1}' for i in range(self.num_x)]
        self.y_name = 'y'
        self.cov_names = ['vegetation', 'water', 'rain']

        self.train, self.test = self._load_data()
        self.data = self._build_data()

    def _load_data(self):
        return pd.read_csv(os.path.join(self.data_dir, 'Mat_train.csv')), \
               pd.read_csv(os.path.join(self.data_dir, 'Mat_cv.csv'))

    def _build_data(self):
        return {
            'train_x': self.train[self.x_names].values,
            'test_x': self.test[self.x_names].values,
            'train_y': self.train[self.y_name].values,
            'test_y': self.test[self.y_name].values,
            'train_cov': self.train[self.cov_names].values,
            'test_cov': self.test[self.cov_names].values
        }


def _test():
    dataset = Dataset()
    data = dataset.data
    print(data['train_x'].shape, data['train_y'].shape, data['train_cov'].shape)
    print(data['test_x'].shape, data['test_y'].shape, data['test_cov'].shape)

    Zmat_data, Zmat_cv, X_train_basis, X_cv_basis = \
        data['train_y'], data['test_y'], data['train_x'], data['test_x']

    save_np(Zmat_data, 'malaria_data/y_data.ny')
    save_np(Zmat_cv, 'malaria_data/y_cv.ny')
    save_np(X_train_basis, 'malaria_data/X_train_basis.ny')
    save_np(X_cv_basis, 'malaria_data/X_cv_basis.ny')


if __name__ == '__main__':
    _test()
