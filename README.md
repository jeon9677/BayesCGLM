# Python codes for 'A Bayesian Convolutional Neural Network-based Generalized Linear Model'

## Codes for training ByasCGLM with simulation datasets: bayescglm_mp.py  (multiprocessing code for BayesCGLM)
* The code require two command arguments: (1) number of Monte Carlo samples (2) number of cores to use 
  
## posterir_dist.py for estimating posterior distribution 
* posterior distribution of (4) in main manuscript.
  
## prediction_dist.py for predictive distribtuion 
* Predictive distribution of (5) in main manuscript.

## simulation.py for generating simulation dataset containing simulated images with four filter images for generating $\Phi$
* get_model() for BayesCNN. 
* job () for training BayesCGLM by combining $\Phi$ from get_model with covariates. 
* simulation_data_generator function() requires four arguments. (1) number of simulated images (2) true coefficient of covariates (3) grid size (4) seed number for random data generation.
* simulation data can generate normal, poisson, and binary responses. 

