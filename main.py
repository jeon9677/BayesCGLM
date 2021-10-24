import logging
import tqdm
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky
import statsmodels.api as sm
from tensorflow.keras.optimizers import Adam
import sys
import numpy as np
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


def make_grid(grid_shape_tuple):
    '''
    generate grid on [0,grid_shape_tuple[0]]X[0,grid_shape_tuple[1]] in Z^2
    choose mark points as left-lower vertex (in picture, left-upper vertex)
    '''
    gridLoc = []
    for i in range(grid_shape_tuple[0]):
        for j in range(grid_shape_tuple[1]):
            # gridLoc.append([i/grid_shape_tuple[0],j/grid_shape_tuple[1]]) #<- normalize version
            gridLoc.append([i, j])
    return gridLoc


def gp_image_generator(num_image, grid_shape_tuple, matern_param_dict, seed_val):
    '''
    generate image on [0,grid_shape_tuple[0]]X[0,grid_shape_tuple[1]] in Z^2
    the value is normally distributed, structured by matern covariance form
    input:  seed_val 20210415
        num_image : int
        grid_shape_tuple : tuple(int,int). For example, to make a 10 by 10 image, set this argument to (10,10)
        matern_covariance_dict: dictionary{"phi": float, "sigma2": float}. The keys should not be changed (up to order.)
            phi: strengthness of correlation among near points (higher phi -> more stronger corr)
            sigma2: general variance
        seed_val : int ir float

    output:
        numpy array[image_0, image_1, ..., image_'num_image']
        where image_i : 'grid_shape_tuple'-shaped matrix
    '''

    # set the places
    randGenerator = np.random.default_rng(seed=seed_val)
    gridLoc = make_grid(grid_shape_tuple)

    # Matern covariance structure seting
    if "phi" not in matern_param_dict.keys():
        print('hi')
        raise KeyError("matern_param_dict should include the parameter key:value pair -> 'phi':x")
    if "sigma2" not in matern_param_dict.keys():
        raise KeyError("matern_param_dict should include the parameter key:value pair -> 'sigma2':x")

    distMatLoc = cdist(gridLoc, gridLoc, 'euclidean')
    MaternCov_mat = (1 + (5 ** 0.5 * distMatLoc / matern_param_dict["phi"]) + (5 * distMatLoc ** 2) / (
            3 * matern_param_dict["phi"] ** 2)) * \
                    np.exp(-(5 ** 0.5 * (distMatLoc / matern_param_dict["phi"])))
    MaternCov_mat *= matern_param_dict["sigma2"]

    # generate images
    image_list = []
    image_mean = []
    for _ in range(num_image):
        MaternCov_TSseed = randGenerator.normal(loc=0.0, scale=1.0, size=MaternCov_mat.shape[0])
        gp_image = np.matmul(cholesky(MaternCov_mat).T, MaternCov_TSseed)
        gp_image_vector_form = np.array(gp_image)
        gp_image_matrix_form = np.reshape(gp_image_vector_form, grid_shape_tuple)
        image_list.append(gp_image_matrix_form)  # gaussian
        image_mean.append(np.mean(gp_image_matrix_form))
        # image_list.append(np.exp(gp_image_matrix_form)) #poisson

    return np.array(image_list), np.array(image_mean)


def image_filter_generator(loc_knots, grid_shape_tuple, basis="invQuad", **kwargs):
    '''
    generate image filter on [0,grid_shape_tuple[0]]X[0,grid_shape_tuple[1]] in Z^2
    input:
        loc_knots: list of 2-dim-list. the centers of the kernels.
        grid_shape_tuple: tuple(int,int). recommendation: for additional works, set this argument as same as one of image generater
        basis: choose from below
            currently implemented basis :
                "invQuad"
                    needs additional argument 'phi: float', controlling the decay of the kernel as distance grows.
                    if phi is larger, the value of kernel becomes more fastly decrease as distance grows.
                        example : image_filter_generator(..., basis="invQuad", phi=0.05)

    output:
        numpy array[filter_using 'loc_knots_0', filter_using 'loc_knots_1',...]
        where filter_i : 'grid_shape_tuple'-shaped matrix
    '''

    gridLoc = np.array(make_grid(grid_shape_tuple))
    distMat_knot_grid = np.array(cdist(loc_knots, gridLoc, 'euclidean'))
    # distmat[k][p]: from k-th knots to p-th gridpoint
    basis_mat = 0

    # TPSbasis
    # TPSmat = (distMat_knot_grid**2) * (np.log(distMat_knot_grid)) #issue: 1. 제곱이 있나? 2. knot에 거리 0인 점 있으면 error

    # invQuadBasis
    if basis == "invQuad":
        if "phi" in kwargs.keys():
            phi = kwargs["phi"]
        else:
            raise ValueError("please put additional argument 'phi' to control the decay of this kernel")
        basis_mat = 1 / (1 + (phi * distMat_knot_grid) ** 2)

    else:
        raise ValueError("the basis is not yet implemented:", basis)
    basis_mat = np.reshape(basis_mat, (len(loc_knots), grid_shape_tuple[0], grid_shape_tuple[1]))
    return basis_mat


def image_transformer(image_array, filter_array):
    ''' construct f(image) with basis
        now:
            f(image) = sum(image*filter) (*: elementwise multiplication)

        input:
            image_array (from gp_image_generator())
            filter_array (from image_filter_generator())
        output:
            (number of images) by (number_of_filters) matrix
    '''
    feature_list = []
    for image in image_array:
        for filt in filter_array:
            filtered_image = image * filt
            feature_list.append(np.mean(filtered_image))
    feature_array = np.array(feature_list)
    feature_array = np.reshape(feature_array, (len(image_array), len(filter_array)))
    return feature_array


def get_dropout(input_tensor, p=0.5, mc=False):
    if mc:
        return Dropout(p)(input_tensor, training=True)
    else:
        return Dropout(p)(input_tensor)


def simulation_data_generator(num_data, true_beta_vec, image_shape, seed_val):
    # image_array, image_mean = gp_image_generator(num_data, grid_shape_tuple=image_shape,
    #                                              matern_param_dict={"phi": 30, "sigma2": 10},
    #                                              seed_val=seed_val * 10)  #  binary
    image_array, image_mean = gp_image_generator(num_data, grid_shape_tuple=image_shape,
                                                 matern_param_dict={"phi": 15, "sigma2": 1},
                                                 seed_val=seed_val * 10)  # poisson, normal

    # normalization : logit
    basis_array = image_filter_generator([[0, 0], [30, 30], [10, 20], [15, 15]], grid_shape_tuple=image_shape,
                                         basis="invQuad", phi=0.1)  # normal, binary
    # basis_array = image_filter_generator([[30, 30], [10, 20]], grid_shape_tuple=image_shape,
    #                                      basis="invQuad", phi=0.1) # poisson

    features_from_image = image_transformer(image_array, basis_array)
    # print(features_from_image.shape)

    # randGenerator = np.random.default_rng(seed=seed_val)

    randGenerator = np.random.default_rng(seed=seed_val)
    covariateMat = randGenerator.normal(loc=0.0, scale=1.0, size=(num_data, len(true_beta_vec)))
    # mean = [0, 0]
    # cov = [[4, 0.2], [0.2, 4]]
    # covariateMat_beta = randGenerator.multivariate_normal(mean, cov, num_data).T
    # covariateMat_array = np.array(covariateMat_beta)
    # beta1 = covariateMat_array[0, :]
    # beta2 = covariateMat_array[1, :]
    # b1 = beta1.reshape(-1)
    # b2 = beta2.reshape(-1)
    # b1 = b1.reshape(b1.shape[0], 1)
    # b2 = b2.reshape(b2.shape[0], 1)
    # covariateMat = np.hstack([b1, b2])
    Zmat_intensity = np.matmul(covariateMat, true_beta_vec) + np.matmul(features_from_image,
                                                                        np.ones(features_from_image.shape[1]))  # normal
    # Zmat_intensity = np.exp(np.matmul(covariateMat, true_beta_vec) + np.matmul(features_from_image,
    #                                                                  np.ones(features_from_image.shape[1])))  #poisson
    # responseVec = np.random.poisson(lam=Zmat_intensity)
    responseVec = np.random.normal(loc=Zmat_intensity, scale=1)
    # Zmat_intensity = 1 / (1 + np.exp(-(np.matmul(covariateMat, true_beta_vec) + np.matmul(features_from_image,
    #                                                                                       np.ones(
    #                                                                                           features_from_image.shape[
    #                                                                                               1])))))
    # responseVec = np.random.binomial(1, Zmat_intensity)

    return covariateMat, image_array, responseVec, Zmat_intensity


def get_model(mc=False, act="relu"):
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, concatenate
    from keras.models import Sequential, Model, Input, load_model
    from keras.layers import Dense, Dropout, Flatten, SpatialDropout2D, SpatialDropout1D, AlphaDropout, Conv2D, \
        MaxPooling2D, Conv1D, MaxPooling1D
    from tensorflow.keras.optimizers import Adam

    inp = Input((30, 30, 1))
    other_input = Input((2,))
    x = Conv2D(8,
               kernel_size=(4, 4),
               strides=(2, 2),
               activation='relu')(inp)
    x = get_dropout(x, p=0.2, mc=mc)
    x = MaxPooling2D()(x)
    x = Conv2D(16,
               kernel_size=(3, 3),
               strides=(2, 2),
               activation='softmax')(x)
    x = get_dropout(x, p=0.2, mc=mc)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    # x = concatenate([x,other_input])
    x = Dense(32, activation='relu')(x)
    x = get_dropout(x, p=0.2, mc=mc)
    x = Dense(16, activation='relu')(x)
    x = get_dropout(x, p=0.2, mc=mc)
    x = Dense(8, activation='softplus')(x)
    x = get_dropout(x, p=0.2, mc=mc)
    # x = Dense(16, activation='relu')(x)
    x = concatenate([x, other_input])
    out = Dense(1, activation='linear')(x)  # out[0] , out[1]=1
    # out = Dense(1, activation='sigmoid')(x)  # out[0] , out[1]=1

    model = Model(inputs=[inp, other_input], outputs=out)

    # model.compile(optimizer=Adam(learning_rate=1e-5),
    #                loss='binary_crossentropy',
    #                metrics=['accuracy'])

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='mse',
                  metrics=['mse'])

    return model


def job(weights, X_input, X_cv, X_train_covariate, X_cv_covariate, Zmat_data, Zmat_cv, idx):
    from keras.models import Sequential, Model, Input, load_model
    mc_model = get_model(mc=True, act="relu")
    mc_model.set_weights(weights)
    model = Model(inputs=mc_model.inputs, outputs=mc_model.layers[12].output)   ##### last layer
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

    ### poisson
    # for subject in x_const_final_test:
    #     ff = 1 / (np.dot(np.dot(subject, fisher), subject.T))
    #     fishers.append(ff)

    # fishers_train = []
    #
    # for subject in x_const_final:
    #     ff = 1 / (np.dot(np.dot(subject, fisher), subject.T))
    #     fishers_train.append(ff)

    # print(_results.bse)
    # print(_results.bse[-10:])

    # mc_beta.append(_results.params[-2:])
    # mc_beta.append(_results.params[-10:])  # normal final
    # mc_last_layer.append(rv)
    # mc_last_layer_cv.append(rv_test)
    pred = _results.predict(x_const_final_test)
    # mc_predictions_glmm.append(pred)
    pred_train = _results.predict(x_const_final)
    error_train = pred_train - Zmat_train
    mse = np.mean(error_train)
    # mc_predictions_glmm_train.append(pred_train)
    # pred_sigma.append(fishers)
    # pred_sigma_train.append(fishers_train)
    # f_params = gamma_results.params
    # fisher2 = gamma_model.information(f_params)
    # sigma.append(inverse_fisher)
    return _results.params[-2:], pred, mse, fishers, inverse_fisher  # normal



j = 2019311256

def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)
    logging.debug("start")


    covariateMat, image_array, responseVec, Zmat_intensity = simulation_data_generator(1000, [1, 1], (30, 30), j)
    ind_data = 700
    Zmat_data = responseVec[:ind_data, ]  # 1000 x 1
    Zmat_cv = responseVec[ind_data:, ]  # 1000 x 1

    X_train_covariate = covariateMat[:ind_data, :]  # covariate
    X_cv_covariate = covariateMat[ind_data:, :]

    X_train_basis = image_array[:ind_data, :, :]
    X_cv_basis = image_array[ind_data:, :, :]

    batch_size = 32  # 32 normal, 3 poisson, 3 binary
    epochs = 300  # 1000 normal, 100 poisson, 200 binary

    mc_model = get_model(mc=True, act="relu")

    h_mc = mc_model.fit([X_train_basis, X_train_covariate], Zmat_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=([X_cv_basis, X_cv_covariate], Zmat_cv))

    logging.debug("load")

    mc_beta = []
    mc_last_layer = []
    mc_last_layer_cv = []
    mc_predictions_glmm = []
    mc_predictions_glmm_train = []

    time_result = []

    sigma = []
    pred_sigma = []
    pred_sigma_train = []
    n_model = int(sys.argv[1])
    X_input = [X_train_basis, X_train_covariate]
    X_cv = [X_cv_basis, X_cv_covariate]

    weights = mc_model.get_weights()
    logging.debug(f"CPU count: {multiprocessing.cpu_count()}")
    n_process = int(sys.argv[2])
    t0 = timer()  # start time
    with Pool(processes=n_process) as pool:
        for mc_beta_, mc_predictions_glmm_, mse, pred_sigma_, sigma_ in \
                pool.map(partial(job,
                                 weights, X_input, X_cv, X_train_covariate, X_cv_covariate, Zmat_data, Zmat_cv),
                         range(n_model)):
            mc_beta.append(mc_beta_)
            mc_predictions_glmm.append(mc_predictions_glmm_)
            mc_predictions_glmm_train.append(mse)
            pred_sigma.append(pred_sigma_)
            sigma.append(sigma_)
    t1 = timer()  # end time
    logging.debug(f"Time : {t1 - t0}")
    time_temp ={t1 - t0}
    time_result.append(time_temp)


    # sys.exit(1)

    logging.debug('glm')
    exp_name = f'{n_model}'
    exp_dir = f'model_simulation/{exp_name}'

    ### for beta
    np.save(f'{exp_dir}/trace', mc_beta)
    np.save(f'{exp_dir}/posterior_sigma', sigma)
    ### for y
    np.save(f'{exp_dir}/posterior_predictive', mc_predictions_glmm)
    np.save(f'{exp_dir}/predictive_sigma', pred_sigma)
    np.save(f'{exp_dir}/mse', mc_predictions_glmm_train)  # for normal prediction coverage
    ### time
    np.save(f'{exp_dir}/time', time_result)


    logging.debug("finsih!")


if __name__ == '__main__':
    main()
