import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error , accuracy_score, recall_score, precision_score, f1_score

from keras.models import Sequential, Model, Input, load_model
from utils import load_np
from dataset_malaria import Dataset
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':

    exp_dir, n_model = sys.argv[1], int(sys.argv[2])
    size = 10
    dataset = Dataset()
    data = dataset.data

    Zmat_data, Zmat_cv, X_train_basis, X_cv_basis ,X_train_cov, X_cv_cov =\
        data['train_y'], data['test_y'], data['train_x'], data['test_x'],data['train_cov'],data['test_cov']

    post_path = os.path.join(exp_dir, f'posterior_predictive.ny')
    posterior_predictive = load_np(post_path)

    pred_path = os.path.join(exp_dir, f'predictive_sigma.ny')
    pred_sigma = load_np(pred_path)


    y_true = pd.DataFrame(Zmat_cv)

    n_m, n_c = posterior_predictive.shape
    n_mm = n_m

    res = np.zeros((n_mm, n_c, size))
    for m in range(n_mm):
        for c in range(n_c):
            mean = posterior_predictive[m][c]
            std = np.abs(pred_sigma[m][c])
            samples = norm.rvs(loc=mean, scale=std, size=size, random_state=None)
            for s, v in enumerate(samples):
                res[m][c][s] = v


    d1, d2, d3 = res.shape
    print(res.shape)

    _res=[]
    for i in range(d2):
        _res.append(res[:,i,:].reshape(-1))

    _res=np.array(_res)
    print(_res.shape)



    y_dist_final1 = pd.DataFrame(_res)

    y_true = pd.DataFrame(Zmat_cv)


