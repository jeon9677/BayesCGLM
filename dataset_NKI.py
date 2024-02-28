import os
import sys
import numpy as np
import pandas as pd
from utils import save_np
from scipy.io import loadmat
from sklearn.model_selection import KFold

from params import Params as P


class Dataset(object):
    fold_ind = 0

    def __init__(self):
        self.data_dir = 'NKI_ver2'
        self.y_name = 'stateAnxiety'

        self.x = self._load_x()
        self.cov, self.y = self._load_cov_and_y()

        self.num_fold = 3  # spare
        self.seed = 2021
        self.fold = self._cv()
        self.data = self.fold[self.fold_ind]

    def _load_x(self):
        files = os.listdir(self.data_dir)
        files = list(filter(lambda f: f.startswith('ROICorrelation_FisherZ_sub-') and f.endswith('.txt'), files))
        x = {}
        for f in files:
            id= int(f.split('A')[-1][:-4]) # remove .txt
            # arr = loadmat(os.path.join(self.data_dir, f))['ROICorrelation']
            with open(os.path.join(self.data_dir, f)) as r:
                array2d = []
                for line in r:
                    array1d = np.array([float(item) for item in line.strip().split('\t')])
                    array2d.append(array1d)
            array2d = np.array(array2d)
            np.fill_diagonal(array2d, 0)
            x[id] = array2d

        return x

    def _load_cov_and_y(self):
        df = pd.read_csv(os.path.join(self.data_dir, 'subjects_info.csv'))
        y,cov = {},{}
        for d in df.to_dict(orient='records'):
            id = d['ID']
            cov_value = [d['NT'],d['ET'],d['AT'],d['CT']]
            y_value = d['y']
            #print(y_value,cov_value)
            y[id] = y_value
            cov[id] = cov_value
        return cov,y

    def _cv(self):
        ids = np.array(list(self.x.keys()))
        fold = []
        for train_index, test_index in KFold(n_splits=self.num_fold, random_state=self.seed, shuffle=True).split(ids):
            train_id = [id for id in ids[train_index]]
            test_id = [id for id in ids[test_index]]
            train_x = np.array([self.x[id] for id in train_id])
            test_x = np.array([self.x[id] for id in test_id])
            train_cov = np.array([self.cov[id] for id in train_id])
            test_cov = np.array([self.cov[id] for id in test_id])
            train_y = np.array([self.y[id] for id in train_id])
            test_y = np.array([self.y[id] for id in test_id])
            fold.append({'train_x': train_x,
                         'test_x': test_x,
                         'train_cov': train_cov,
                         'test_cov': test_cov,
                         'train_y': train_y,
                         'test_y': test_y})
        return fold


def _test():
    dataset = Dataset()
    data = dataset.data
    print(dataset.data['train_x'].shape)
    print(dataset.data['test_x'].shape)

    Zmat_data, Zmat_cv, X_train_basis, X_cv_basis, X_train_covariate, X_cv_covariate = \
        data['train_y'], data['test_y'], data['train_x'], data['test_x'], data['train_cov'], data['test_cov']

    save_np(Zmat_data, 'NKI_ver2/y_data.ny')
    save_np(Zmat_cv, 'NKI_ver2/y_cv.ny')
    save_np(X_train_basis, 'NKI_ver2/X_train_basis.ny')
    save_np(X_cv_basis, 'NKI_ver2/X_cv_basis.ny')
    save_np(X_train_covariate, 'NKI_ver2/X_train_covariate.ny')
    save_np(X_cv_covariate, 'NKI_ver2/X_cv_covariate.ny')


if __name__ == '__main__':
    _test()
