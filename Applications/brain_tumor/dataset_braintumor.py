import os
import sys
import numpy as np
import pandas as pd
import cv2
from scipy.io import loadmat
from sklearn.model_selection import KFold

from params import Params as P


class Dataset(object):
    fold_ind = 0

    def __init__(self):
        self.data_dir = 'Brain_Tumor'
        self.y_name = 'Target'

        self.x = self._load_x()
        self.cov, self.y = self._load_cov_and_y()

        self.num_fold = 3  # spare
        self.seed = P.seed
        self.fold = self._cv()
        self.data = self.fold[self.fold_ind]

    def _load_x(self):
        files = os.listdir(self.data_dir)
        files = list(filter(lambda f: f.startswith('Image') and f.endswith('.jpg'), files))
        x = {}
        for f in files:
            id=f[:-4] # remove .txt : = ~
            image = cv2.imread(os.path.join(self.data_dir,f),flags=cv2.IMREAD_GRAYSCALE)
            #print(image)
            #print(image.shape)
            #print(image.reshape(-1).mean())
            #sys.exit(1)
            x[id] = image
        return x

    def _load_cov_and_y(self):
        df = pd.read_csv(os.path.join(self.data_dir, 'Brain_Tumor.csv'))
        y,cov = {},{}
        for d in df.to_dict(orient='records'):
            id = d['Image']
            # cov_value = [d['Mean'],d['Variance'],d['Dissimilarity'],d['Contrast']]
            cov_value = [d['Mean'],d['Variance']]
            y_value = d['Class']
            #print(y_value,cov_value)
            y[id] = y_value
            cov[id] = cov_value
        return cov,y

    def _cv(self):
        # def _y(v):
        #     if v == 0:
        #         return [1, 0]
        #     else:
        #         return [0, 1]

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
    print(dataset.data['train_x'].shape)
    #print(dataset.data['train_y'])
    print(dataset.data['test_cov'].shape)


if __name__ == '__main__':
    _test()
