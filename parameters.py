"""
Parameters used across scripts
"""

# data directories
import os
from utilities.utils import makeifnot
dir_data = 'data'
dir_figures = 'figures'
dir_ncbi = os.path.join(dir_data, 'ncbi')
dir_google = os.path.join(dir_data, 'google')
dir_esmfold = os.path.join(dir_data, 'esmfold')
makeifnot(dir_data)
makeifnot(dir_ncbi)
makeifnot(dir_figures)
makeifnot(dir_esmfold)

# Number of folds to use (see 9_predict_y.py)
n_folds=5

# Type-I error rate
alpha=0.05

# Number of bootstrap/simulation iterations
n_boot=250

# Reproducability seed
seed = 1

# Which labels to use for 8_debias_y.py script?
ylabels = ['infection', 'PI', 'sweat']

# Name of the reference file (i.e. wildtype)
reference_file = "base"

# Define the MLP regressor as the model class we want to use for the 9_predict_y.py script
from sklearn.neural_network import MLPRegressor
mdl_class = MLPRegressor
# Set up model parameters
di_mdl_class = {'hidden_layer_sizes':[124, 24],
          'activation':'relu',
          'random_state':seed,
          'early_stopping':False,
          'max_iter':700,
          'verbose':False,
          'solver':'adam',
          'learning_rate_init':0.0001}

