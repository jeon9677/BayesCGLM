# Python codes for 'A Bayesian Convolutional Neural Network-based Generalized Linear Model'

## 0. Package version
* numpy version: 1.19.5
* pandas version: 1.2.1
* scipy version: 1.6.0
* statsmodels version: 0.12.1
* keras version: 2.4.3
* multiprocess version: 3.12
* functools version: 3.8


## 1. Simulation dataset code. 
### Simulations/bayescglm_mp.py for training ByasCGLM with simulation datasets (multiprocessing code for BayesCGLM)
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
* This code require two arguments : (1) directory where the output of BayesCGLM is stored.  (2) number of Monte Carlo samples 
* Example command statement for generating covariate coefficient posterior distribution: (directory: model_simulation/300, number of MC samples: 300)  
```diff
python posterior_dist model_simulation/300 300
```
* Example command statement for generating covariate coefficient posterior distribution: (directory: model_malaria/500, number of MC samples: 500)  
```diff
python posterior_dist model_malaria/500 500
``` 
  
### prediction_dist.py for predictive distribtuion 
* Predictive distribution of (5) in main manuscript.
* This code require two arguments : (1) directory where the output of BayesCGLM is stored.  (2) number of Monte Carlo samples 
* Example command statement for generating predictive distribution: (directory: model_simulation/300, number of MC samples: 300)  
```diff
python prediction_dist model_simulation/300 300
```
* Example command statement for generating predictive distribution: (directory: model_malaria/500, number of MC samples: 500)  
```diff
python prediction_dist model_malaria/500 500
```

## 3. About applications datasets 
* The brain tumor images dataset is available for download from https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor/data.
* The fMRI data for the anxiety dataset can be found in fMRI_data folder.
* The malaria incidence data for the African Great Lakes Region be found in malaria_data folder.

## 4. Applications BayesCGLM code
* Applications/brain_tumor/dataset_braintumor.py for pre-processing MRI brain tumor dataset
* Applications/brain_tumor/braintumor_BayesCGLM.py to train BayesCGLM for the braintumor dataset (binary case)
  - To run braintumor_BayesCGLM.py, please also download dataset_braintumor.py for pre-processing MRI brain tumor dataset.
  - The code require two command arguments: (1) number of Monte Carlo samples (e.g.500) (2) number of cores to use
  - Example command statement:
```diff
python braintumor_BayesCGLM.py 500 3 
```
* Once run the braintumor_BayesCGLM.py, the output will be stored in model_brain.
--------
* Applications/malaria/dataset_malaria.py for pre-processing malaria dataset
* Applications/malaria/malaria_BayesCGLM.py to train BayesCGLM for the malaria dataset (poisson case)
  - To run malaria_BayesCGLM.py, please also download dataset_malaria.py for pre-processing malaria dataset.
  - The code require two command arguments: (1) number of Monte Carlo samples (e.g.500) (2) number of cores to use
  - Example command statement:
```diff
python malaria_BayesCGLM.py 500 3 
```
* Once run the malaria_BayesCGLM.py, the output will be stored in model_malaria.
---------
* Applications/fMRI/dataset_fMRI.py for pre-processing fMRI dataset
* Applications/fMRI/fMRI_BayesCGLM.py for to train BayesCGLM for the fMRI dataset (Gaussian case)
  - To run fMRI_BayesCGLM.py, please also download dataset_fMRI.py for pre-processing fMRI dataset.
  - The code require two command arguments: (1) number of Monte Carlo samples (e.g.500) (2) number of cores to use
  - Example command statement:
```diff
python fMRI_BayesCGLM.py 500 3 
```
* Once run the fMRI_BayesCGLM.py, the output will be stored in model_NKI.

## 5. Code guidance as html file
* (Gaussian)  https://jeon9677.github.io/BayesCGLM/Code_guidance_html/Gaussian_BayesCGLM_Simulation.html
* (Poisson)  https://jeon9677.github.io/BayesCGLM/Code_guidance_html/Binary_BayesCGLM_Simulation.html
* (Binary)  https://jeon9677.github.io/BayesCGLM/Code_guidance_html/Binary_BayesCGLM_Simulation.html
* (fMRI)  https://jeon9677.github.io/BayesCGLM/Code_guidance_html/fMRI_example.html
* (malaria)  https://jeon9677.github.io/BayesCGLM/Code_guidance_html/malaria_example.html
* (brain tumor)  https://jeon9677.github.io/BayesCGLM/Code_guidance_html/braintumor_example.html
