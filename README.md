# Python codes for 'A Bayesian Convolutional Neural Network-based Generalized Linear Model'

### Simulation 
## Codes for training ByasCGLM with simulation datasets: bayescglm_mp.py  (multiprocessing code for BayesCGLM)
* The code require two command arguments: (1) number of Monte Carlo samples (2) number of cores to use
* Command example for implementing BayesCGLM for 1000 simulated images : python bayescglm_mp.py 300 3 
  
## posterior_dist.py for estimating posterior distribution 
* Posterior distribution of (4) in main manuscript.
* This code require two arguments : (1)directory where the output of BayesCGLM data is stored.  (2) number of Monte Carlo samples 
* Command example for generating predictive distribution: python posterior_dist model_simulation/300 300

  
## prediction_dist.py for predictive distribtuion 
* Predictive distribution of (5) in main manuscript.
* This code require two arguments : (1)directory where the output of BayesCGLM data is stored.  (2) number of Monte Carlo samples 
* Command example for generating predictive distribution: python predictive_dist model_simulation/300 300


## simulation.py for generating simulation dataset containing simulated images with four filter images for generating $\Phi$
* get_model() for BayesCNN. 
* job () for training BayesCGLM by combining $\Phi$ from get_model with covariates. 
* simulation_data_generator function() requires four arguments:  (1) number of simulated images (2) true coefficient of covariates (3) grid size (4) seed number for random data generation.
* simulation data can generate normal, poisson, and binary responses. 
* Command example for implementing BayesCGLM with 300 Monte Carlo samples by using 5 cores for multiprocessing: python simulation.py 300 5 

### Data 
## Datasets for malaria incidence (malaria_data folder), resting state fMRI (NKI_data folder), and brain tumor images (braintumor_data folder).  
## In each foler, there is code for processing each dataset for training BayesCGLM 

### Applications 
## 00_braintumor_bayescglm_mp.py for training BayesCGLM with braintumor dataset 
## 00_malaria_bayescglm_mp.py for training BayesCGLM with malaria dataset 
## 00_nki_bayescglm_mp.py for training BayesCGLM with fMRI dataset 

