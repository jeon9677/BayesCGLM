import logging
import tqdm
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky
import statsmodels.api as sm
from tensorflow.keras.optimizers import Adam
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, concatenate
from keras.models import Sequential, Model, Input, load_model
from keras.layers import Dense, Dropout, Flatten, SpatialDropout2D, SpatialDropout1D, AlphaDropout, Conv2D, \
    MaxPooling2D, Conv1D, MaxPooling1D
from scipy.stats import *
from timeit import default_timer as timer
import multiprocessing
from multiprocessing import Pool
from contextlib import contextmanager
from functools import partial
from itertools import product
from dataset_fMRI import Dataset
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score


def get_dropout(input_tensor, p=0.5, mc=False):
    if mc:
        return Dropout(p)(input_tensor, training=True)
    else:
        return Dropout(p)(input_tensor)


def get_model(mc=False, act="relu"):
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, concatenate
    from keras.models import Sequential, Model, Input, load_model
    from keras.layers import Dense, Dropout, Flatten, SpatialDropout2D, SpatialDropout1D, AlphaDropout, Conv2D, \
        MaxPooling2D, Conv1D, MaxPooling1D
    from tensorflow.keras.optimizers import Adam

    inp = Input((278, 278, 1))
    other_input = Input((4,))
    x = Conv2D(8,
               kernel_size=(3, 3),
               strides=(2, 2),
               activation='relu')(inp)
    x = get_dropout(x, p=0.25, mc=mc)
    x = MaxPooling2D()(x)
    x = Conv2D(16,
               kernel_size=(3, 3),
               strides=(2, 2),
               activation='relu')(x)
    x = get_dropout(x, p=0.25, mc=mc)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    x = get_dropout(x, p=0.25, mc=mc)
    x = Dense(8, activation='softplus')(x)
    x = get_dropout(x, p=0.25, mc=mc)
    # x = Dense(16, activation='relu')(x)
    x = concatenate([x, other_input])
    out = Dense(1, activation='linear')(x)  # out[0] , out[1]=1
    # out = Dense(1, activation='sigmoid')(x)  # out[0] , out[1]=1

    model = Model(inputs=[inp, other_input], outputs=out)

    # model.compile(optimizer=Adam(learning_rate=1e-5),
    #                loss='binary_crossentropy',
    #                metrics=['accuracy'])

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='mse',
                  metrics=['mse'])
    return model


def job(weights, X_input, X_cv, X_train_covariate, X_cv_covariate, Zmat_data, Zmat_cv, idx):
    from keras.models import Sequential, Model, Input, load_model
    mc_model = get_model(mc=True, act="relu")
    mc_model.set_weights(weights)
    model = Model(inputs=mc_model.inputs, outputs=mc_model.layers[10].output)   ##### last layer
    rv = model.predict(X_input)
    response = np.array(rv)
    rv_test = model.predict(X_cv)
    response_test = np.array(rv_test)

    x_final = np.hstack([response, X_train_covariate])
    x_final_test = np.hstack([response_test, X_cv_covariate])

    # sys.exit(1)
    Zmat_train = Zmat_data.reshape(Zmat_data.shape[0], 1)
    Zmat_cv_y = Zmat_cv.reshape(Zmat_cv.shape[0], 1)


    x_const_final = sm.add_constant(x_final)
    # print(x_const_final.shape)
    x_const_final_test = sm.add_constant(x_final_test)
    _model = sm.GLM(Zmat_train, x_const_final, family=sm.families.Gaussian())
    _results = _model.fit()

    fisher = _model.information(_results.params)
    fisher = - fisher
    fisher_m = np.array(fisher)
    # fisher_m = np.matrix(fisher)
    # print(fisher_m)
    fisher_inf = np.linalg.inv(fisher_m)
    # print(fisher_inf)
    # print(np.linalg.det(fisher_m))
    # fisher_diag = fisher_inf.diagonal()
    inverse_fisher = fisher_inf[9:, 9:]

    fishers = []

    ### normal
    for subject in x_const_final_test:
        ff = np.dot(np.dot(subject, fisher_inf), subject.T)
        fishers.append(ff)

    pred = _results.predict(x_const_final_test)
    y_true = np.array(Zmat_cv)
    acc = mean_squared_error(y_true, pred) ** 0.5



    return _results.params[-4:], pred, acc, fishers, inverse_fisher  # normal




def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)
    logging.debug("start")


    dataset = Dataset()
    data = dataset.data

    Zmat_data, Zmat_cv, X_train_basis, X_cv_basis, X_train_covariate, X_cv_covariate = \
        data['train_y'], data['test_y'], data['train_x'], data['test_x'], data['train_cov'], data['test_cov']

    standardScaler = StandardScaler()
    standardScaler.fit(X_train_covariate)
    X_train_covariate = standardScaler.transform(X_train_covariate)
    standardScaler.fit(X_cv_covariate)
    X_cv_covariate = standardScaler.transform(X_cv_covariate)


    batch_size = 3
    epochs = 500 # 300

    t0 = timer()  # start time
    mc_model = get_model(mc=True, act="relu")

    h_mc = mc_model.fit([X_train_basis, X_train_covariate], Zmat_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=([X_cv_basis, X_cv_covariate], Zmat_cv))

    logging.debug("load")

    mc_beta = []
    mc_predictions_glmm = []
    mc_predictions_glmm_train = []

    time_result = []

    sigma = []
    pred_sigma = []
    accuracy = []
    n_model = int(sys.argv[1])
    X_input = [X_train_basis, X_train_covariate]
    X_cv = [X_cv_basis, X_cv_covariate]

    weights = mc_model.get_weights()
    logging.debug(f"CPU count: {multiprocessing.cpu_count()}")
    n_process = int(sys.argv[2])

    with Pool(processes=n_process) as pool:
        for mc_beta_, mc_predictions_glmm_, mse, pred_sigma_, sigma_ in \
                pool.map(partial(job,
                                 weights, X_input, X_cv, X_train_covariate, X_cv_covariate, Zmat_data, Zmat_cv),
                         range(n_model)):
            mc_beta.append(mc_beta_)
            mc_predictions_glmm.append(mc_predictions_glmm_)
            accuracy.append(mse)
            pred_sigma.append(pred_sigma_)
            sigma.append(sigma_)
    t1 = timer()  # end time
    logging.debug(f"Time : {t1 - t0}")
    time_temp = np.array(t1 - t0)
    time_result.append(time_temp)


    # sys.exit(1)

    logging.debug('glm')
    exp_name = f'{n_model}'
    exp_dir = f'model_NKI/{exp_name}'

    ### for beta
    np.save(f'{exp_dir}/trace', mc_beta)
    np.save(f'{exp_dir}/posterior_sigma', sigma)
    ### for y
    np.save(f'{exp_dir}/posterior_predictive', mc_predictions_glmm)
    np.save(f'{exp_dir}/predictive_sigma', pred_sigma)
    np.save(f'{exp_dir}/mse', accuracy)
    ### time
    np.save(f'{exp_dir}/time', time_result)


    logging.debug("finsih!")


if __name__ == '__main__':
    main()
