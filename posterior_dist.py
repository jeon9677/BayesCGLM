import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error , accuracy_score, recall_score, precision_score, f1_score
from scipy.stats import multivariate_normal

def save_np(arr, path):
    with open(path, 'wb') as f:
        np.save(f, arr)


def load_np(path):
    with open(path, 'rb') as f:
        arr = np.load(f)
    return arr



if __name__ == '__main__':

    exp_dir, n_model = sys.argv[1], int(sys.argv[2])
    size = 500
    trace_path = os.path.join(exp_dir, f'trace.npy')
    mc_beta = load_np(trace_path)

    sigma_path = os.path.join(exp_dir, f'posterior_sigma.npy')
    mc_sigma = load_np(sigma_path)
    res = []
    n_m = mc_beta.shape[0]

    for m in range(n_m):
        mean = mc_beta[m]
        std = mc_sigma[m]
        samples = np.random.multivariate_normal(mean, std, size=size).T
        res.append(samples)

    res_array = np.array(res)

    res1 = res_array[:,0,:]
    res2 = res_array[:,1,:]
    # res3 = res_array[:, 2, :]


    b1 = res1.reshape(-1)
    b2 = res2.reshape(-1)
    # b3 = res3.reshape(-1)

    b1 = b1.reshape(b1.shape[0], 1)
    b2 = b2.reshape(b2.shape[0], 1)
    # b3 = b3.reshape(b3.shape[0], 1)


    beta_dist_final = np.hstack([b1,b2])
    name_list = ["b1"] + ["b2"] 

    # beta_dist_final = np.hstack([b1,b2,b3])
    # name_list = ["b1"] + ["b2"] + ["b3"]
    
    beta_dist_final1 = pd.DataFrame(beta_dist_final, columns=name_list)
    df = pd.DataFrame(beta_dist_final1, columns=name_list)

