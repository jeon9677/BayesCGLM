import logging
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky
import statsmodels.api as sm
import sys
from keras.layers import Dense, Dropout, Flatten, SpatialDropout2D, SpatialDropout1D, AlphaDropout, Conv2D, \
    MaxPooling2D 
from scipy.stats import *
from timeit import default_timer as timer
import multiprocessing
from multiprocessing import Pool
from functools import partial
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score



def make_grid(grid_shape_tuple):
    gridLoc = []
    for i in range(grid_shape_tuple[0]):
        for j in range(grid_shape_tuple[1]):
            gridLoc.append([i, j])
    return gridLoc


def gp_image_generator(num_image, grid_shape_tuple, matern_param_dict, seed_val):

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


    return np.array(image_list), np.array(image_mean)


def image_filter_generator(loc_knots, grid_shape_tuple, basis="invQuad", **kwargs):

    gridLoc = np.array(make_grid(grid_shape_tuple))
    distMat_knot_grid = np.array(cdist(loc_knots, gridLoc, 'euclidean'))


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
    image_array, image_mean = gp_image_generator(num_data, grid_shape_tuple=image_shape,
                                                 matern_param_dict={"phi": 15, "sigma2": 1},
                                                 seed_val=seed_val * 10) 

    basis_array = image_filter_generator([[0, 0], [30, 30], [10, 20], [15, 15]], grid_shape_tuple=image_shape,
                                         basis="invQuad", phi=0.1)  # normal, binary
    
#     basis_array = image_filter_generator([[30, 30], [10, 20]], grid_shape_tuple=image_shape,
#                                          basis="invQuad", phi=0.1) # poisson


    features_from_image = image_transformer(image_array, basis_array)

    randGenerator = np.random.default_rng(seed=seed_val)
    covariateMat = randGenerator.normal(loc=0.0, scale=1.0, size=(num_data, len(true_beta_vec)))

    # normal
#     Zmat_intensity = np.matmul(covariateMat, true_beta_vec) + np.matmul(features_from_image,
#                                                                         np.ones(features_from_image.shape[1]))  
#     responseVec = np.random.normal(loc=Zmat_intensity, scale=1)
  
    # poisson 
#     Zmat_intensity = np.exp(np.matmul(covariateMat, true_beta_vec) + np.matmul(features_from_image, np.ones(features_from_image.shape[1])))  
#     responseVec = np.random.poisson(lam=Zmat_intensity)


    # binary 
    Zmat_intensity = 1 / (1 + np.exp(-(np.matmul(covariateMat, true_beta_vec) + np.matmul(features_from_image,
                                                                                        np.ones(features_from_image.shape[1])))))
    responseVec = np.random.binomial(1, Zmat_intensity)


    return covariateMat, image_array, responseVec, Zmat_intensity

def prob2y2(values):
    return [1 if v >= 0.5 else 0 for v in values]

def job(weights, X_input, X_cv, X_train_covariate, X_cv_covariate, Zmat_data, Zmat_cv, idx):
    from keras.models import Sequential, Model, Input, load_model
    mc_model = get_model(mc=True, act="relu")
    mc_model.set_weights(weights)
    model = Model(inputs=mc_model.inputs, outputs=mc_model.layers[10].output)   
    ##### 12th last layer for normal, 10th last layer for binary/poisson
    rv = model.predict(X_input)
    response = np.array(rv)
    rv_test = model.predict(X_cv)
    response_test = np.array(rv_test)

    x_final = np.hstack([response, X_train_covariate])
    x_final_test = np.hstack([response_test, X_cv_covariate])

    Zmat_train = Zmat_data.reshape(Zmat_data.shape[0], 1)


    x_const_final = sm.add_constant(x_final)
    x_const_final_test = sm.add_constant(x_final_test)
#     _model = sm.GLM(Zmat_train, x_const_final, family=sm.families.Gaussian())
#     _model = sm.GLM(Zmat_train, x_const_final, family=sm.families.Poisson()) # poisson
    _model = sm.GLM(Zmat_train, x_const_final, family=sm.families.Binomial()) #binary
    _results = _model.fit()

    fisher = _model.information(_results.params)
    fisher = - fisher
    fisher_m = np.array(fisher)
    fisher_inf = np.linalg.inv(fisher_m)

    inverse_fisher = fisher_inf[9:, 9:] # binary
#     inverse_fisher = fisher_inf[17:, 17:] # normal/poisson

    fishers = []

    for subject in x_const_final_test:
        ff = np.dot(np.dot(subject, fisher_inf), subject.T)
        fishers.append(ff)

   ## normal/poisson case 
    pred = _results.predict(x_const_final_test)
#     pred_train = _results.predict(x_const_final)
#     error_train = pred_train - Zmat_train
#     mse = np.mean(error_train)
    
    ## binary case 
    ys = prob2y2(pred)
    y_true = np.array(Zmat_cv)
    acc = accuracy_score(y_true, ys)
    rec = recall_score(y_true, ys)
    prc = precision_score(y_true, ys)
    f1 = f1_score(y_true, ys)

    acc = np.hstack([acc, rec, prc, f1])
    
#     return _results.params[-2:], pred, mse, fishers, inverse_fisher  # normal/poisson
    return _results.params[-2:], pred, acc, fishers, inverse_fisher  # binary



def get_model(mc=False, act="relu"):
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, concatenate
    from keras.models import Sequential, Model, Input, load_model
    from keras.layers import Dense, Dropout, Flatten, SpatialDropout2D, SpatialDropout1D, AlphaDropout, Conv2D, \
        MaxPooling2D, Conv1D, MaxPooling1D
    from tensorflow.keras.optimizers import Adam

    inp = Input((30, 30, 1))
    other_input = Input((2,))
    x = Conv2D(16,
               kernel_size=(3, 3),    # (4,4) for normal/poisson case,  (3,3) for binary case 
               strides=(1, 1),        # (2,2) for normal/poisson case,  (1,1) for binary case 
               activation='softmax')(inp)
    x = get_dropout(x, p=0.25, mc=mc) # p=0.2 for normal/poisson case, p=0.25 for binary case
    x = MaxPooling2D()(x)
    x = Conv2D(32,                  # 16 for normal case, 32 for binary/poisson case
               kernel_size=(3, 3),  # (3,3) for normal/binary/poisson 
               strides=(1, 1),      # (2,2) for normal case,  (1,1) for binary/poisson case 
               activation='softmax')(x)
    x = get_dropout(x, p=0.25, mc=mc)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    x = get_dropout(x, p=0.25, mc=mc)
    x = Dense(8, activation='linear')(x) # 16 (linear) for normal/pisson case, 8(linear) for binary case  
    x = get_dropout(x, p=0.25, mc=mc)
#     x = Dense(16, activation='softplus')(x) # 16 (softplus) for normal case
#     x = get_dropout(x, p=0.2, mc=mc)
    x = concatenate([x, other_input])
    out = Dense(1, activation='sigmoid')(x) 
    # (linear) for normal case, (exponential) for poisson case, (sigmoid) for binary case 

    model = Model(inputs=[inp, other_input], outputs=out)

    model.compile(optimizer=Adam(learning_rate=1e-4), # 1e-4 for normal/binary case, 1e-3 for poisson case 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # (mse, mse) for normal case, (poisson,mse) for poisson case, (binary_crossentropy, accuracy) for binary case  

    return model

