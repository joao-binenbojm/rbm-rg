# rbm-rg
This repository will be used for all algorithm development, simulations and exploration required for my final year project: "A Mapping Between The Variational Renormalization Group and Deep Learning". This will include the exploration of RBMs/DBNs/DBMs models, MC simulations of the Ising Model, training on simulated data and carrying out statistical analysis on the results obtained.

## Models, Simulations & Functions
All the relevant classes  (RBMs, DBNs, IsingModels) and functions (used to process data and in quantitative methods) are given in the utils.py file. All other scripts import this file and carry out tasks accordingly.

## Models
All the new models (trained with the class DBN2) are present in the final_models folder, containing different training iterations and hyperparamters. The best models were determine to be model4.p with 800 epochs and model3.p from the regularized models.

## Data
Data is available for the Ising model in 1D for different coupling parameters, and for the 2D model for lattices of 64x64 (main set simulated and contains up to 80k samples) and for 128x128. Larger datasets were not uploadable to GitHub.
