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
* This code require two arguments : (1)directory where the output of BayesCGLM is stored.  (2) number of Monte Carlo samples 
* Example command statement for generating predictive distribution:
```diff
python posterior_dist model_simulation/300 300
``` 
  
### prediction_dist.py for predictive distribtuion 
* Predictive distribution of (5) in main manuscript.
* This code require two arguments : (1)directory where the output of BayesCGLM is stored.  (2) number of Monte Carlo samples 
* Example command statement for generating predictive distribution:
```diff
python prediction_dist model_simulation/300 300
```
## 3. About applications datasets 
* The brain tumor images dataset is available for download from https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor/data.
* The fMRI data for the anxiety dataset can be found in fMRI_data folder.
* The malaria incidence data for the African Great Lakes Region be found in malaria_data folder.

## 4. Applications BayesCGLM code
* dataset_braintumor.py for pre-processing MRI brain tumor dataset
* braintumor_BayesCGLM.py to train BayesCGLM for the braintumor dataset (binary case)
  - To run braintumor_BayesCGLM.py, please also download dataset_braintumor.py for pre-processing MRI brain tumor dataset.
  - Example command statement:
```diff
python braintumor_BayesCGLM.py 500 3 
```
* dataset_malaria.py for pre-processing malaria dataset
* malaria_BayesCGLM.py to train BayesCGLM for the malaria dataset (poisson case)
  - To run malaria_BayesCGLM.py, please also download dataset_malaria.py for pre-processing malaria dataset.
  - Example command statement:
```diff
python malaria_BayesCGLM.py 500 3 
```
* dataset_fMRI.py for pre-processing fMRI dataset
* fMRI_BayesCGLM.py for to train BayesCGLM for the fMRI dataset (Gaussian case)
  - To run fMRI_BayesCGLM.py, please also download dataset_fMRI.py for pre-processing fMRI dataset.
  - Example command statement:
```diff
python fMRI_BayesCGLM.py 500 3 
```
