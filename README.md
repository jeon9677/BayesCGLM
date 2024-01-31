# Python codes for 'A Bayesian Convolutional Neural Network-based Generalized Linear Model'

## 1. Simulation dataset code. 
### bayescglm_mp.py for training ByasCGLM with simulation datasets (multiprocessing code for BayesCGLM)
* The code require two command arguments: (1) number of Monte Carlo samples (e.g.300) (2) number of cores to use
* simulation_data_generator function() requires four arguments:  (1) number of simulated images (2) true coefficient of covariates (3) grid size (4) seed number for random data generation.
* simulation data can generate normal, poisson, and binary responses. 
* Example command statement for implementing BayesCGLM for 1000 simulation dataset with 300 Monte Carlo samples and 3 cores for multiprocessing :
```diff
 python bayescglm_mp.py 300 3 
```
## 2. Codes for generating samples from posterior distributions 
### posterior_dist.py for estimating posterior distribution 
* Posterior distribution of (4) in main manuscript.
* This code require two arguments : (1)directory where the output of BayesCGLM data is stored.  (2) number of Monte Carlo samples 
* Example command statement for generating predictive distribution:
```diff
python posterior_dist model_simulation/300 300
``` 
  
### prediction_dist.py for predictive distribtuion 
* Predictive distribution of (5) in main manuscript.
* This code require two arguments : (1)directory where the output of BayesCGLM data is stored.  (2) number of Monte Carlo samples 
* Example command statement for generating predictive distribution:
```diff
python predictive_dist model_simulation/300 300
```
## 3. About applications datasets 
* The brain tumor images dataset is available for download from https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor/data.
* The fMRI data for the anxiety dataset can be obtained from https://fcon_1000.projects.nitrc.org/.
* The malaria incidence data for the African Great Lakes Region can be accessed via https://api.dhsprogram.com of The Demographic and Health Surveys (DHS) Program Application Programming Interface.

## 3-1. Applications BayesCGLM code
* 00_braintumor_bayescglm_mp.py to train BayesCGLM for the braintumor dataset (binary case)  
  - Example command statement:
```diff
python predictive_dist model_simulation/300 300
```
* 00_malaria_bayescglm_mp.py to train BayesCGLM for the malaria dataset (poisson case)
  - Example command statement:
```diff
python predictive_dist model_simulation/300 300
```
* 00_nki_bayescglm_mp.py for to train BayesCGLM for the fMRI dataset (Gaussian case)
  - Example command statement:
```diff
python predictive_dist model_simulation/300 300
```

## Codes for Jupyternotebook 
* I have also uploaded the Jupyter notebook codes for parallel computing of BayesCGLM in _Jupyter_ folder.
* There are outputs as html, providing codes.
