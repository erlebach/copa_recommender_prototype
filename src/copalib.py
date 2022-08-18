
import newlib_monthly as newlib
import pandas as pd
import matplotlib as plt
import numpy as np
import copalib as copa

from rankfm.evaluation import hit_rate
from fastcore.all import L, AttrDict

import copa_config 

# Set Parameters. The only ones you really need to adjust are: 
# nb_epochs, alpha, beta, embed_dim, train_file,  valid_file

#--------------------------------------------------------------------------------
def setup_config_dct():
    """
    Arguments:
    ----------

    Returns:
    -------

    Notes:
    -----
    """
    config = AttrDict({
        'embed_dim': 20,  
        'alpha' : 0.05,
        'beta' : 0.1,
        'learning_rate' : 0.1,
        'learning_schedule' : 'constant',
        'nb_epochs': 100,
        'train_perc' : 0.8,  # not used
        'valid_perc' : 0.1,  # not used
        'shuffle' : True,               # bool (0 / 1)
        'temporal' : True,              # bool
        'drop_dups' : True,             # bool
        'topN' : 5,
        'loss' : 'bpr',
        'max_samples' : 1, # neg sampliong
        'with_attrib' : 1,         # bool
        'keep_nb_members' : None,
        'continuous_attr' : True,     # bool 
        'verbose' : False,             # bool
        'age_cuts' : [0, 30, 50, 70, 150],  # not used for now
        'train_file' : 'attrib_yr_mo_2017_03.csv',
        'valid_file' : 'attrib_yr_mo_2017_04.csv',
    })
    return config

#-----------------------------------------------------------------------
def setup_attributes():
    """
    Parameters:
    ----------

    Returns:
    -------

    Notes:
    -----
    """
    con_item_attr = ['avg_yr_l', 'avg_yr_h', 'LON_DEC', 'LAT_DEC', 'HEIGHT']
    cat_item_attr = []
    con_user_attr = ['age_departure']
    cat_user_attr = ['TRUE_ORIGIN_COUNTRY', 'ADDR_COUNTRY', 'NATIONALITY']
    attr_dct = {
        'con_item_attr': con_item_attr,
        'cat_item_attr': cat_item_attr,
        'con_user_attr': con_user_attr,
        'cat_user_attr': cat_user_attr,
    }
    return attr_dct

#--------------------------------------------------------------------------------------
def create_file_pairs():
    """
    Arguments:
    ----------

    Returns:
    -------

    Notes:
    -----
    """
    valid_file_template = "attrib_yr_mo_%4d_%02d.csv"
    train_file_template = "attrib_yr_mo_%4d_%02d.csv"
    years = [yr for yr in range(2016,2020)]
    months = [mo for mo in range(1,12)]  # skip last month
    
    files_mo_yr   = []
    files_monthly = []
    files_yearly  = []
    
    files_mo_yr_pairs = []
    for yr in years:
        for mo in months:
            files_mo_yr_pairs.append([valid_file_template%(yr,mo), train_file_template%(yr,mo+1)])
            files_mo_yr.append(valid_file_template%(yr,mo))
            
        files_mo_yr_pairs.append([valid_file_template%(yr,12), train_file_template%(yr+1,1)])
        files_mo_yr.append(valid_file_template%(yr,12))
        
    files_monthly_pairs = []
    template = "attrib_month_%02d.csv"
    for mo in months:
        files_monthly_pairs.append([template%mo, template%(mo+1)])
        files_monthly.append(template%mo)
        
    files_yearly_pairs = []
    template = "attrib_yr_%04d.csv"
    for yr in years:
        files_yearly_pairs.append([template%yr, template%(yr+1)])
        files_yearly.append(template%yr)
       
    return files_mo_yr, files_monthly, files_yearly

#--------------------------------------------------------------------------------------------
def random_file_pairings(files_mo_yr):
    """
    Choose pairs of random monthly files. The validation timestamps come after the training timestamps

    Arguments:
    ----------

    Returns:
    -------

    Notes:
    -----
    """
    # Create random pairings of monthly files. 
    nb_pairs = 100
    nb_files = len(files_mo_yr)
    # Choose random pairs of integer. First integer < last integer. Remove duplicates
    int1 = np.random.randint(0, nb_files, nb_pairs)
    int2 = np.random.randint(0, nb_files, nb_pairs)
    ints = np.array(list(zip(list(int1), list(int2))))

    # Validation set occurs after training set. 
    # However, month_v could occur in an earlier month than month_t, if the years are different. 
    # For example, 2017_05 (May 2017) could be training data, and 2016_03 (March, 2016) could be validation set. 
    # The validation month comes before the training month which is legal if the years are different. 
    new_ints = []
    for idx, (i,j) in enumerate(zip(int1, int2)):
        if i == j: continue
        if j < i:
            ints[idx] = (j,i)
        else:
            ints[idx] = (i,j)
        iii = list(ints[idx])
        new_ints.append(iii)  # ints[idx] is a np array
    
    df = pd.DataFrame(data=new_ints, columns=['int1','int2'])
    pairs = df.drop_duplicates().values
     
    files_monthly_random = []
    for p in pairs:
        pair = (files_mo_yr[p[0]], files_mo_yr[p[1]])
        files_monthly_random.append(pair)

    # month across years. Note that results will improve as testing set grows since the number of destinations flown
    # by a member increases and thus increase the likelihood of a hit. 
    # Perhaps a better way is to calculate the hit rate in the validation set for a given flight, and not consider all flights. 
    # Currently, I compare the topN recommendations with set of destinations flow to by a member. 
    # Instead, one should compare a topN set of recommendations with a single destination in the validation set. 
    # I am not 100% sure tests are done correctly.
    return files_monthly_random

#--------------------------------------------------------------------------------------------
def setup_sweep_config(config):
    """
    Arguments:
    ----------

    Returns:
    -------

    Notes:
    -----
    """
    # Optimal wd: around 3e-3. However, it only small effect on loss function. 
    sweep_config = {
        'name' : 'rankfm_monthly_sweep1',
        'method' : 'grid',
        'parameters' : {
            'lr' : {  'value': 1.e-1, },
            'train_perc' : {'values': [0.1,0.3,0.5] },
            'valid_perc' : {'values': [0.1,0.3,0.5] },
            'batch_size' : {'value' : 1024},
            'nb_epochs'  : {'value' : 100},
            'embed_dim'    : {'value' : 20},
            'temporal'   : {'values' : [True, False]},
            'with_attrib' : {'values' : [True, False]},
        },
    }

    metric = { 
        'name' : 'hr_filtered',
        'goal' : 'maximize'
    }
    
    sweep_config['metric'] = metric
    wandb.config = config

    return sweep_config

#-------------------------------------------------------------------
def hypersweep():
    """
    Arguments:
    ----------

    Returns:
    -------

    Notes:
    -----
    """
    # Make sure that run is specified. Must be None for hypersweep
    # set seed to None for non-reproducibility
    infile = 'activity_reduced_with_attributes.csv'
    all_in_one(config, infile=infile, seed=None, use_wandb=True, run=None)

#-----------------------------------------------------------------------
def copa_run(config, attr_dct, files_monthly_random, use_wandb):
    """
    Arguments:
    ----------

    Returns:
    -------

    Notes:
    -----
    """
    # PARAMETER UPDATES
    config.with_attrib = False
    config.nb_epochs = 300
    config.embed_dim = 30
    config.Description = "Randomized monthly pairs (train/valid)"
    config.topN = 10
    config.learning_rate = 0.1
    config.alpha = 0.001
    config.beta = 0.001
    config.alpha = 0.020
    config.beta = 0.020

    # Try 3 random files
    nb_random_files = 1
    files_random = files_monthly_random[0:nb_random_files]
    hr_filt_list = L()
    hr_notfilt_list = L()
    # pairs (train/valid)
    months_t = L()
    months_v = L()
    years_t = L()
    years_v = L()

    # WHERE DOES THE RANDOMNESS COME FROM? from shuffling the training data? 
    # With enough epochs, shuffling the training data should not an impact unless local minimum. 
    # Therefore, should try to bypass local minima.  <<<< WORK ON THIS. 
    # Looking at random months. should plot hit rates (filt and not filt) as a function of month/yr (training and valid). 
    # That type of correlation is there?   What month should be used for the training data? Same month from previous year, or previous month? 
    # ISSUE: if using only a single month of data for training, many members in the validation set will be cold starts. That is not satisfactory. 
    # Perhaps only consider last year of flights to predict the next months?  More training data will decrease the hit rate if filtering. 
    # So perhaps the unfiltered recommendations are more meaningful? 

    nb_repeats = 1
    
    #for file_t, file_v in files_monthly_random:
    for file_t, file_v in files_random:
        print("==================================================================")
        print(file_t, file_v)
        config.train_file = file_t
        config.valid_file = file_v
        for i in range(nb_repeats):
            hr_notfiltered, hr_filtered, losses = newlib.all_in_one(config, seed=None, use_wandb=use_wandb, run=None, attr_dct=attr_dct)

            hr_filt_list += hr_filtered
            hr_notfilt_list += hr_notfiltered
            years_t += int(file_t[13:17])
            years_v += int(file_v[13:17])
            months_t += int(file_t[18:20])
            months_v += int(file_v[18:20])
        quit()
            
    columns = ["yr_t", "mo_t", "yr_v", "mo_v", "hr_filtered", "hr_notfiltered"]
    data = [years_t, years_v, months_t, months_v, hr_filt_list, hr_notfilt_list]
    df = pd.DataFrame({
          'yr_t':years_t, 'yr_v':years_v,
          'mo_t':months_t, 'mo_v':months_v,
          'hr_filt':hr_filt_list, 'hr_notfilt':hr_notfilt_list
    })
    
    # number of months between training and validation sets
    df['diff1'] = (df.yr_v-df.yr_t)*12 + df.mo_v - df.mo_t

    if use_wandb:
        print("***** Before wandb.Table creation)")
        print("BEFORE wandb.init")
        run = wandb.init(project="RankFM", config=config)
        wandb.Table(data=df)
        wandb.run.log({'hit_rates_df': df}, commit=True)
        run.finish()
        print("run completed")

#--------------------------------------------------------------------------------
def copa_run_single_pair(config, attr_dct, train_file, valid_file, use_wandb):

    """
    Arguments:
    ----------

    Returns:
    -------

    Notes:
    -----
    """

    # Number of times to repeat the simulation to compute mean and standard deviation of hit rates
    nb_repeats = 1
    
    config.train_file = train_file
    config.valid_file = valid_file
    print(train_file, valid_file)

    for i in range(nb_repeats):
        results = newlib.all_in_one(config, seed=None, use_wandb=use_wandb, run=None, attr_dct=attr_dct)
        hr_notfiltered, hr_filtered = results[0:2] 
        if len(results) == 3:
            losses = results[2]

#--------------------------------------------------------------------------------
