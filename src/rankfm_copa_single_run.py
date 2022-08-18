# Conversion of the Jupyter-lab notebook rankfm_for_copa.ipynb
import newlib_monthly as newlib
import copalib as copa
from rankfm.evaluation import hit_rate
from fastcore.all import L, AttrDict
import os

import copa_config
DATA_ROOT = copa_config.copa_config['data']

# Set Parameters. The only ones you really need to adjust are: 
# nb_epochs, alpha, beta, embed_dim, train_file,  valid_file

#--------------------------------------------------------------------------------
config = copa.setup_config_dct()
# PARAMETER UPDATES
# Code is crashing with attributes. 
config.with_attrib = True
config.nb_epochs = 300
config.embed_dim = 30
config.Description = "Randomized monthly pairs (train/valid)"
config.topN = 5
config.learning_rate = 0.1
config.alpha = 0.001
config.beta = 0.001
config.alpha = 0.050
config.beta = 0.050

files_mo_yr, files_monthly, files_yearly = copa.create_file_pairs()
files_monthly_random =  copa.random_file_pairings(files_mo_yr)

use_wandb = False   # Change to True if running wandb

train_file = os.path.join(DATA_ROOT, "attrib_yr_mo_2017_07.csv")
valid_file = os.path.join(DATA_ROOT, "attrib_yr_mo_2017_08.csv")

attr_dct = copa.setup_attributes()
copa.copa_run_single_pair(config, attr_dct, train_file, valid_file, use_wandb)
quit()

#----------------------------------------------------------------------------------
