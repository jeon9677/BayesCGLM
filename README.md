# A Bayesian Convolutional Neural Network-based Generalized Linear Model

### Abstract: Neural networks provide complex function approximations between inputs and a response variable for a wide variety of applications. Examples include a classification for images and regression for spatially or temporally correlated data. Although neural networks can improve prediction performance compared to traditional statistical models, interpreting the impact of covariates is difficult. Furthermore, uncertainty quantification for predictions and inference for model parameters are not trivial. To address these challenges, we propose a new Bayes approach by embedding convolutional neural networks (CNN) within the generalized linear models (GLM) framework. Using extracted features from CNN as informative covariates in GLM, our method can improve prediction accuracy and provide interpretations of regression coefficients. By fitting ensemble GLMs across multiple Monte Carlo realizations, we can fully account for uncertainties. We apply our methods to simulated and real data examples, including non-Gaussian spatial data, brain tumor image data, and fMRI data. The algorithm can be broadly applicable to image regressions or correlated data analysis by providing accurate Bayesian inference quickly. 
