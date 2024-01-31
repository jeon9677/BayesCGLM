# Python codes for 'A Bayesian Convolutional Neural Network-based Generalized Linear Model'

## 1. Simulation dataset code. 
### bayescglm_mp.py for training ByasCGLM with simulation datasets (multiprocessing code for BayesCGLM)
* The code require two command arguments: (1) number of Monte Carlo samples (e.g.300) (2) number of cores to use
* simulation_data_generator function() requires four arguments:  (1) number of simulated images (2) true coefficient of covariates (3) grid size (4) seed number for random data generation.
* simulation data can generate normal, poisson, and binary responses. 
* Example command statement for implementing BayesCGLM for 1000 simulation dataset with 300 Monte Carlo samples and 3 cores for multiprocessing : _python bayescglm_mp.py 300 3_ 


## 2. Codes for generating samples from posterior distributions 
### posterior_dist.py for estimating posterior distribution 
* Posterior distribution of (4) in main manuscript.
* This code require two arguments : (1)directory where the output of BayesCGLM data is stored.  (2) number of Monte Carlo samples 
* Example command statement for generating predictive distribution: _python posterior_dist model_simulation/300 300_

  
## prediction_dist.py for predictive distribtuion 
* Predictive distribution of (5) in main manuscript.
* This code require two arguments : (1)directory where the output of BayesCGLM data is stored.  (2) number of Monte Carlo samples 
* Example command statement for generating predictive distribution: _python predictive_dist model_simulation/300 300_


## 3. Data 
* The brain tumor images dataset is available for download from https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor/data.
* The fMRI data for the anxiety dataset can be obtained from https://fcon_1000.projects.nitrc.org/.
* The malaria incidence data for the African Great Lakes Region can be accessed via https://api.dhsprogram.com of The Demographic and Health Surveys (DHS) Program Application Programming Interface.

## 3. Applications 
* 00_braintumor_bayescglm_mp.py for training BayesCGLM with braintumor dataset
  - Example command statement: 
* 00_malaria_bayescglm_mp.py for training BayesCGLM with malaria dataset
* 00_nki_bayescglm_mp.py for training BayesCGLM with fMRI dataset


