# Library to support rankfm

import pandas as pd
import numpy as np
import sys
import os
import pickle

from rankfm.rankfm import RankFM
from rankfm.evaluation import hit_rate, reciprocal_rank, discounted_cumulative_gain, precision, recall, diversity
from fastcore.all import L, AttrDict, Path

# wanda is a framework to collaborate, perform parameters studies, do versioning
import copa_config
DATA_ROOT = copa_config.copa_config['data']

# This file is meant to work when training and validation files are specified separately. 
# Eventually, merge with newlib.py

#----------------------------------------------------------------------------------------
class Rankfm:
    """
    Collect useful routines, and create a dictionary with parameter for every run
    Keys for dct: otpN, keep_nb_members, age_cuts, 
    offset_perc (?), train_perc, valid_perc, temporal
    factors, loss, alpha, beta, learning_rate, learning_schedule
    """
    # config will also be used by wandab.run.config. Values should not be modified
    def __init__(self, config, dct, seed=None, use_wandb=False, run=None, attr_dct=None):
        """
        dct_new: overwrite default dictionary parameters
        """

        # Ensure reproducibility
        if seed != None:
            np.random.seed(seed)

        self.use_wandb = use_wandb

        if use_wandb and not run:
            self.run = wandb.init(project='RankFM', config=config)
        else:
            self.run = run

        if run:
            config.update(wandb.config)  # Configuration now contains sweep parameters
            self.config = config
        else:
            self.config = config

        # con/cat attributes for items and users. All lists L()
        self.attr_dct = attr_dct

        # items to remove from dictionary prior to updating the class dataframe
        self.cols_toremove = ['df_members','df_user_attr','df_item_attr','data_train','data_valid','data_test','model']

    #-----------------------------------------------------------------
    def all_in_one(config, seed=1, use_wandb=True, run=None, attr_dct=None):
        #with wandb.init() as run:
    
            # config will be linked to wandb
        self.read_data()
        self.create_model()
        self.run_model()
        self.finish()

    #-----------------------------------------------------------------
    def finish(self):
        if self.use_wandb and self.run:
            self.run.finish()

    #----------------------------------------------------------------
    def read_data(self, dct_update=AttrDict()): #, continuous_attr=False):
        age_cuts = self.config['age_cuts']
        continuous_attr = self.config.continuous_attr
        train_file = self.config.train_file
        valid_file = self.config.valid_file

        # NOTE: I have to set overwrite_cache to True. For some reason the cache is overwritten in this function. 
        # This creates reproducibility problems when running this function twice in a row
        # MUST INVESTIGATE
        self.dct_train = read_data_attributes_single_file(train_file, 
            age_cuts=age_cuts, dct=self.config, overwrite_cache=True, 
            continuous_attr=continuous_attr, **self.attr_dct)


        self.dct_valid = read_data_attributes_single_file(valid_file, 
            age_cuts=age_cuts, dct=self.config, overwrite_cache=True, 
            continuous_attr=continuous_attr, **self.attr_dct)

    #----------------------------------------------------------------
    def update_storage(self):
        # Append dictionary into the class dataframe
        dct_copy = self.dct.copy()
        try:
            for col in self.cols_toremove:
                dct_copy.pop(col)
        except Exception as err:
            # At least one item cannot be popped
            print("self.dct: ", self.dct.keys())
            print("dct_copy: ", dct_copy.keys())
            print("dct_copy: ", dct_copy.values())
            print("Exception error: \n", err)
            print("===========================================================")

        dct_copy['age_cuts'] = dct_copy['age_cuts']
        try:
            self.df = pd.concat([self.df, pd.DataFrame(dct_copy, index=[0])])
        except:
            self.df = pd.DataFrame(dct_copy, index=[0])
        
    #----------------------------------------------------------------
    def save(self, out_file):
        self.df.to_csv(out_file, index=0)
    
    #----------------------------------------------------------------
    def train_valid(self):
        config = self.config
        train_valid_dct(config, self.dct_train,
            train_perc=1.0,
            valid_perc=0.0,
            shuffle=config.shuffle,
            temporal=config.temporal)

    #----------------------------------------------------------------
    def update_dict(self, dct_update):
        for k,v in dct_update.items():
            self.dct[k] = v
            
    #----------------------------------------------------------------
    def create_model(self):
        config = self.config
        self.dct_train['model'] = RankFM(factors=config['embed_dim'], 
                              loss=config['loss'], 
                              max_samples=config['max_samples'], 
                              alpha=config['alpha'], 
                              beta=config['beta'], 
                              learning_rate=config['learning_rate'], 
                              learning_schedule=config['learning_schedule'])

        self.dct_valid['model'] = self.dct_train['model']  # reference each other
        
    #----------------------------------------------------------------
    def run_model(self):
        dct_train = self.dct_train
        dct_valid = self.dct_valid
        config = self.config
        self.model = dct_train['model']
        try:
            self.hr_notfiltered, self.hr_filtered, self.losses = run_model(self.model, dct_train, dct_valid, **config)
        except:
            self.hr_notfiltered, self.hr_filtered = run_model(self.model, dct_train, dct_valid, **config)

        print("hr_notfiltered: ", self.hr_notfiltered)
        print("hr_filtered: ", self.hr_filtered)

        if self.use_wandb:
            # generate data that might beplottable in wandb
            wandb.Table(self.losses)
            wandb.log({'hr_notfiltered': self.hr_notfiltered, 
                       'hr_filtered': self.hr_filtered,
                       'losses': self.losses},
                       commit=False)
    
#----------------------------------------------------------------------------------------
def read_data_attributes_single_file(in_file, age_cuts=None, overwrite_cache=False, dct=None, continuous_attr=False, cat_user_attr=[], con_user_attr=[], cat_item_attr=[], con_item_attr=[]): 
    """
    in_file (string)
        File containing all records with member and destination attributes. Note that there is 
        no column identification of which columns are which attributes (TODO IN FUTURE)

    age_cuts : list
        Defines the break between different member age categories.

    overwrite_cache: Bool [False]
        If True, overwrite the cache even if present

    continuous_attr: Bool [False]
        If True: temperature, long/lat/altitude are continuous variables

    cat_user_attr: []
        Column names for categorical user attributes. MEMBER_ID is in this list. 

    con_user_attr: []
        Column names for continuous user attributes. MEMBER_ID is this list. 

    cat_item_attr: []
        Column names for categorical item attributes.  D is in the list. 

    con_item_attr: []
        Column names for continuous item attributes D is in the list. 

    Return
    ------
    interact_dct: dictionary
        New keys are: 
            'df_members': data frame of members / destinations
            'df_user_attrib': user attributes, one-hot encoded
    interact_dct['df_members'] = df_members
    interact_dct['df_user_attrib'] = df_user_attrib
    interact_dct['df_item_attr'] = df_item_attr
    df (DataFrame)
       u Dictionary with a minimum of four columns (userID (member), itemID (Destination), rating, year))
        The year column is the year of booking

    Notes
    -----
    - On exit, the data is ordered by flight-departure time
    - The raw data read in is cached in memory (alternatively, stored in binary format) for faster retrieval. 
    - I do not remove duplicate MEMBER_ID/D pairs at this stage.  This is only done for the training/validation/testsing files.
    """

    interact_dct = AttrDict()


    # The leading underscores indicates a private variable by convention
    # Create a dictionary that persists across function invocations
    if not hasattr(read_data_attributes_single_file, '_dct'):
        read_data_attributes_single_file._dct = {'raw_data' : None }
    _dct = read_data_attributes_single_file._dct

    if overwrite_cache:
        read_data_attributes_single_file._dct['raw_data'] = None


    if age_cuts == None:
        age_cuts = [0, 30, 50, 70, 120]

    # Ensure that the last value of age_cuts is >= 120 to capture all age groups
    if age_cuts[-1] < 100:
        age_cuts.append(120)

    if age_cuts[-1] < 120:
        age_cuts[-1] = 120

    # User and item attributes are across entire dataset. 
    # They are restricted to the training set at a later stage.

    #-----------------------------------------------------------------------
    # ALL POSSIBLE USER ATTRIBUTES (should be an argument)
    # age_at_flight changes for different flights, but we ignore this over a 5-year period
    #

    cols_user_attrib = ['MEMBER_ID','TRUE_ORIGIN_COUNTRY','ADDR_COUNTRY','NATIONALITY','age_departure']

    # This should be an argument to the function
    cols_user_attrib = ['MEMBER_ID','age_departure','GENDER']

    #-----------------------------------------------------------------------
    # READ DATA

    attrib_file = in_file
    if isinstance(_dct['raw_data'], pd.DataFrame):
        df_ = _dct['raw_data']
    else:
        df_ = pd.read_csv(attrib_file)
        # Sort data according to flight date
        df_ = df_.sort_values("FLIGHT_DATE")
        _dct['raw_data'] = df_


    # Each member/ID pair should have only a single flight (no not take temporal data into account)

    # This drop_duplicates should not be done at this point, otherwise, I will never be able to predict
    # a destination in the validation set that was present in the original set for a particular member .
    # This explains why I get worse results when I do not filter out previous flights. 
    # df_ = df_.drop_duplicates(['MEMBER_ID','D'])  # already dropped, but let us make sure. <<<< NOT HERE

    # Create categories for member ages
    # NOTE: the monthly files already have the column age_departure. FIX. 
    #df_['age_departure'] = pd.cut(df_.age_at_flight, bins=age_cuts)
    #df_['age_departure'] = df_['age_departure'].astype(float)
    ages = df_.age_departure.values

    #----------------------------------------------------
    prefix = L()
    member_cat_attr_columns = L([]) # categorical attributes
    member_con_attr_columns = L([]) # continuous attributes
    #print("cat_user_attr: ", cat_user_attr)
    #print("con_user_attr: ", con_user_attr)

    age_cat = True  # for now

    cols_user_attr = cat_user_attr
    df_cat_user_attr = None
    df_con_user_attr = None

    # ATTRIBUTE: age_at_departure
    if 'age_departure' in cat_user_attr:
        # CHECK WHETHER age is discrete or continue
        # add argument age_cat: True/False
        for i,c in enumerate(age_cuts):  # Not sure why needed
            age_cuts[i] = float(c)
        interact_dct['age_cuts']      = str(age_cuts)
        df_['age_departure'] = pd.cut(df_.age_departure, bins=age_cuts)
        prefix += 'age_dep'
        member_cat_attr_columns += 'age_departure'

    if 'age_departure' in con_user_attr:
        df_['age_departure'] = df_.age_departure / 100.
        member_con_attr_columns += 'age_departure'

    # ATTRIBUTE: GENDER
    if 'GENDER' in cat_user_attr:
        prefix += 'gender'
        member_cat_attr_columns += 'GENDER'

    # ATTRIBUTE: TRUE_ORIGIN_COUNTRY
    if 'TRUE_ORIGIN_COUNTRY' in cat_user_attr:
        prefix += 'tr_org_co'
        member_cat_attr_columns += 'TRUE_ORIGIN_COUNTRY'

    # ATTRIBUTE: ADDR_COUNTRY
    if 'ADDR_COUNTRY' in cat_user_attr:
        prefix += 'addr_co'
        member_cat_attr_columns += 'ADDR_COUNTRY'

    # ATTRIBUTE: NATIONAlITY
    if 'NATIONALITY' in cat_user_attr:
        prefix += 'natl'
        member_cat_attr_columns += 'NATIONALITY'

    df_members = df_[['MEMBER_ID', 'D']]
    df_cat_user_attr = df_[member_cat_attr_columns]
    df_con_user_attr = df_[member_con_attr_columns]

    # ONE-HOT ENCODING
    if 'MEMBER_ID' in member_cat_attr_columns:
        #member_cat_attr_columns.remove('MEMBER_ID')
        print("'MEMBER_ID' should not be a column of member_cat_attr_columns")
        raise "Error"
    if 'MEMBER_ID' in member_con_attr_columns:
        #member_cat_attr_columns.remove('MEMBER_ID')
        print("'MEMBER_ID' should not be a column of member_con_attr_columns")
        raise "Error"

    lg_pref = len(prefix) > 0
    lg_attr_cols = len(member_con_attr_columns) > 0


    if len(prefix) > 0:
        df_cat_user_attr = pd.get_dummies(df_cat_user_attr, prefix=prefix, columns=member_cat_attr_columns)

    if len(member_con_attr_columns) > 0:
        df_con_user_attr = df_[member_con_attr_columns]

    if lg_pref and lg_attr_cols:
        df_user_attr = pd.concat([df_['MEMBER_ID'], df_con_user_attr, df_cat_user_attr], axis=1)
    elif lg_pref:
        df_user_attr = pd.concat([df_['MEMBER_ID'], df_cat_user_attr], axis=1)
    elif lg_attr_cols:
        df_user_attr = pd.concat([df_['MEMBER_ID'], df_con_user_attr], axis=1)
    else:
        print("1. Should not happen")

    #print("df_user_attrib.columns: ", df_user_attrib.columns)
    #print("user_attrib: ", df_user_attrib.shape)

    ### QUESTION ### 
    ### Should user attributes contain MEMBER_ID?
    ### Should item attributes contain D?

    #------------------------------------------------------------
    # ITEM ATTRIBUTES (Destinations)

    # Read item/Destination attributes
    item_attr_file = os.path.join(DATA_ROOT, "temp_long_lat_height.csv")
    df1 = pd.read_csv(item_attr_file)
    # Initial experimentation: with min/max avg temperatures during the year
    df1 = df1.drop(["avg_wi", "avg_sp", "avg_su", "avg_fa"], axis=1)

    # Tranform temperatures into cateogical variables
    # Try to have the same number in each category? But that might not lead to 
    # relevant categories. 

    # Attributes will be considered as scalars. Divide by 100 for normalization
    # Normalize the continuous variables to (hopefully) improve inference.

    prefix = L()
    item_cat_attr_columns = L([]) # categorical attributes
    item_con_attr_columns = L([]) # continuous attributes

    # ATTRIBUTE: 'avg_yr_l'
    if 'avg_yr_l' in cat_item_attr:
        yr_l_cuts = [-20,40,60,80,120]
        interact_dct['yr_l_cuts']     = str(yr_l_cuts)
        df1['avg_yr_l'] = pd.cut(df1.avg_yr_l, bins=yr_l_cuts)
        item_cat_attr_columns += 'avg_yr_l'
        prefix += 'avg_yr_l'

    if 'avg_yr_l' in con_item_attr:
        df1['avg_yr_l'] = df1.avg_yr_l / 100.
        item_con_attr_columns += 'avg_yr_l'

    # ATTRIBUTE: 'avg_yr_h'
    if 'avg_yr_h' in cat_item_attr:
        yr_h_cuts = [-20,40,60,80,120]
        interact_dct['yr_h_cuts']     = str(yr_h_cuts)
        df1['avg_yr_h'] = pd.cut(df1.avg_yr_h, bins=yr_h_cuts)
        item_cat_attr_columns += 'avg_yr_h'
        prefix += 'avg_yr_h'

    if 'avg_yr_h' in con_item_attr:
        df1['avg_yr_h'] = df1.avg_yr_h / 100.
        item_con_attr_columns += 'avg_yr_h'

    # ATTRIBUTE: 'LON_DEC'
    if 'LON_DEC' in cat_item_attr:
        long_cuts = [-130, -100., -70., -40., 0.]
        interact_dct['long_cuts']     = str(long_cuts)
        df1['LON_DEC'] = pd.cut(df1.LON_DEC, bins=long_cuts)
        item_cat_attr_columns += 'LON_DEC'
        prefix += 'LON_DEC'

    if 'LON_DEC' in con_item_attr:
        df1['LON_DEC'] = df1.LON_DEC / 100.  
        item_con_attr_columns += 'LON_DEC'

    # ATTRIBUTE: 'LAT_DEC'
    if 'LAT_DEC' in cat_item_attr:
        lat_cuts = [-30., -15., 0., 15., 30., 45.]
        interact_dct['lat_cuts']      = str(lat_cuts)
        df1['LAT_DEC'] = pd.cut(df1.LAT_DEC, bins=lat_cuts)
        item_cat_attr_columns += 'LAT_DEC'
        prefix += 'LAT_DEC'

    if 'LAT_DEC' in con_item_attr:
        df1['LAT_DEC'] = df1.LAT_DEC / 100.
        item_con_attr_columns += 'LAT_DEC'

    # ATTRIBUTE: 'HEIGHT'
    if 'HEIGHT' in cat_item_attr:
        altitude_cuts = [0,1000,2000,3000]
        interact_dct['altitude_cuts'] = str(altitude_cuts)
        df1['HEIGHT']  = pd.cut(df1.HEIGHT,  bins=altitude_cuts)
        item_cat_attr_columns += 'HEIGHT'
        prefix += 'HEIGHT'

    if 'HEIGHT' in con_item_attr:
        df1['HEIGHT'] = df1.HEIGHT / df1.HEIGHT.max()
        item_con_attr_columns += 'HEIGHT'

    df_cat_item_attr = df1[item_cat_attr_columns] #.drop_duplicates('D')

    # ONE-HOT ENCODING

    df_cat_item_attr = df1[item_cat_attr_columns]
    df_con_item_attr = df1[item_con_attr_columns]

    ### DUMMIES

    lg_pref = len(prefix) > 0
    lg_attr_cols = len(item_con_attr_columns) > 0

    if lg_pref:
        df_cat_item_attr = pd.get_dummies(df_cat_item_attr, prefix=prefix, columns=item_cat_attr_columns)

    if lg_attr_cols:
        df_con_item_attr = df1[item_con_attr_columns]

    if lg_pref and lg_attr_cols:
        df_item_attr = pd.concat([df1['D'], df_con_item_attr, df_cat_item_attr], axis=1)
    elif lg_pref:
        df_item_attr = pd.concat([df1['D'], df_cat_item_attr], axis=1)
    elif lg_attr_cols:
        df_item_attr = pd.concat([df1['D'], df_con_item_attr], axis=1)
    else:
        print("2. Should not happen")

    # DO WE NEED THIS OR NOT? 
    df_user_attr = df_user_attr.drop_duplicates('MEMBER_ID')
    df_item_attr = df_item_attr.drop_duplicates('D')


    """
    if 'MEMBER_ID' in df_user_attr.columns:
        df_user_attr.drop('MEMBER_ID', inplace=True, axis=1)
    if 'D' in df_item_attr.columns:
        df_item_attr.drop('D', inplace=True, axis=1)
    """

    interact_dct['filename']     = in_file
    interact_dct['df_members']   = df_members
    interact_dct['df_user_attr'] = df_user_attr
    interact_dct['df_item_attr'] = df_item_attr
    return interact_dct

#----------------------------------------------------------------------------------------
def train_valid_dct(config, dct, train_perc, valid_perc, temporal=False, shuffle=True):
    """
    - Split a dataframe into train, validation, testing sets. 
    - Use the dictionary 'dct' to pass information. 
    - There is no return function. 
    - The DataFrame 'df_members' is assumed to be ordered accordng to 'FLIGHT_DATE' timestamp. 
    - See function 'train_valid' for more detail. 
    """
    print("==> INSIDE train_valid_dct")
    df_members = dct['df_members']
    print(df_members.shape)

    dftrain, dfvalid, dftest = train_valid(df_members, train_perc, valid_perc, shuffle=shuffle, temporal=temporal)
    dct['data_train'] = dct_train.df_members
    dct['data_valid'] = dfvalid
    dct['data_test']  = dftest

def train_valid(x, train_perc, valid_perc, temporal=False, shuffle=True):
    """
    Split a dataframe into train, validation, testing sets

    Parameters
    ----------
    x : pandas DataFrame, assumed ordered by flight date

    train_perc, valid_perc: float
        percent of training data [0,1]
        percent of validation data [0,1]

    shuffle:  [True]
        whether to shuffle the data or not, without replacement

    temporal: [True]
        If True, flights in the validation set take place after flights in the training set. 
        IF False, flights are randomized

    Notes:
        - The first two arguments must satisfy the constraint: (train_perc + valid_perc < 1). 
        - The temporal argument suggests I must know the departure times of all flights, and this must be passed in. 
        - This suggests the use of a characteristic timestamp. It also suggests that the division between training/validation/testing
          datesets must occur earlier in the pipeline. Perhaps I should read the data in temporally increasing order, and shuffle
          only this this method. 


    Return
    ------
    x_train, x_valid, x_test : tuple of train, valid, test dataframes

    """

    print("==> INSIDE train_valid")
    #print("temporal: ", temporal)
    #print("shuffle: ", shuffle)
    if not temporal and shuffle:
        x = x.sample(frac=1)

    nb_el = len(x)

    perc_train = train_perc
    perc_valid = valid_perc; 
    perc_test = (1.-valid_perc-train_perc); 

    n_train = int(nb_el * perc_train)
    n_valid = int(nb_el * perc_valid)
    n_test  = int(nb_el * perc_test)

    """
    print("n_train: ", n_train)
    print("n_valid: ", n_valid)
    print("n_test: ", n_test)
    """

    x_train = x.iloc[0:n_train]
    x_valid = x.iloc[n_train:n_train+n_valid]
    x_test  = x.iloc[n_train+n_valid:]

    if shuffle:
        x_train = x_train.sample(frac=1)
        x_valid = x_valid.sample(frac=1)
        x_test  = x_test.sample(frac=1)


    # Ensure that each Member/Destination pair occurs only once
    x_train = x_train.drop_duplicates(['MEMBER_ID','D'])
    x_valid = x_valid.drop_duplicates(['MEMBER_ID','D'])
    x_test  = x_test.drop_duplicates(['MEMBER_ID','D'])

    """
    print("train_valid")
    print("========> x_train: \n", x_train)
    print("========> x_valid: \n", x_valid)
    print("========> x_test: \n", x_test)
    """
    return x_train, x_valid, x_test

#-------------------------------------------------------------------------------------------
def restrict_item_attrib(dct):
    """
    Restrict user_features to items in data_train. The item_features will not 
    change between runs. 

    Return: 
    -------
    item_attrib: DataFrame
        The unique members match the unique members of data_train, a rewquirement of rankfm. 

    Note:
    ----
    The dictionary dct values remain unchanged. On exit, dct['item_attrib'] != item_attr. 
    dct must have the attributes ['df_members','df_item_attr']
    """

    if not 'df_members' in dct or not 'df_item_attr' in dct:
        print("Missing dictionary keys, one of ['df_members', 'df_item_attr']")
        raise "Error"

    data_train = dct['df_members']
    item_attrib = dct['df_item_attr']
    #print("enter restrict_item_attrib, item_attrib: ", item_attrib.shape)
    item_attrib = item_attrib[item_attrib.D.isin(data_train.D)]
    #print("exit restrict_item_attrib, item_attrib: ", item_attrib.shape)  # DECREASED from 90 to 73. WHY? ????
    # Dictionary entry did not change!!!
    #print("dct['df_item_attr'] shape: ", dct['df_item_attr'].shape)

    data_train_D_unique = data_train.D.unique()
    item_attr_D_unique = item_attrib.D.unique()
    #print(f"restrict_item_attrib, unique D, data_train: {data_train_D_unique.shape}, item_attr: {item_attr_D_unique.shape}")
    assert len(data_train_D_unique) == len(item_attr_D_unique),  \
        "Unique D list must be indentical in attrib and df_members"

    # a modifed copy of dct['item_attrib']
    return item_attrib

#-------------------------------------------------------------------------------------------
def restrict_member_attrib(dct):
    """
    Restrict user_features to members in data_train. The user_features will not 
    change between runs. 

    Return: 
    -------
    user_attrib: DataFrame
        The unique members match the unique members of data_train, a rewquirement of rankfm. 

    Notes:
    -----
    The dictionary dct must contain the fields ['df_members', 'df_user_attr']
    """

    if not 'df_members' in dct or not 'df_user_attr' in dct:
        print("Missing dictionary keys, one of ['df_members', 'df_user_attr']")
        raise "Error"

    #data_train = dct['data_train']
    data_train = dct['df_members']
    user_attr = dct['df_user_attr']

    # Why would the shape change if there is no testing set? 
    #print("enter restrict_member_attrib, user_attrib: ", user_attr.shape)
    user_attr = user_attr[user_attr['MEMBER_ID'].isin(data_train.MEMBER_ID)]
    #print("exit restrict_member_attr, user_attr: ", user_attr.shape)
    #print("data_train: ", data_train.MEMBER_ID.nunique()) #8639
    #print("user_attr: ", user_attr.shape) #8639

    data_train_M_unique = data_train.MEMBER_ID.unique()
    user_attr_M_unique = user_attr.MEMBER_ID.unique()

    # I am checking number of elements in the list. I should check the list itself
    assert len(data_train_M_unique) == len(user_attr_M_unique),  \
        "Unique MEMBER_ID list must be indentical in attr and df_members"

    return user_attr
#
#----------------------------------------------------------------------------------------------
#def run_model(model, dct, topN=3, verbose=False, nb_epochs=30, with_attrib=True, **kwds):
def run_model(model, dct_train, dct_valid, topN=3, verbose=False, nb_epochs=30, with_attrib=True, **kwds):
    user_attr = restrict_member_attrib(dct_train)
    item_attr = restrict_item_attrib(dct_train)
    data_train = dct_train['df_members']

    #print(len(item_attr.D.unique()), len(data_train.D.unique()))  # IDENTICAL, so why the ERROR
    assert len(item_attr.D.unique()) == len(data_train.D.unique()), "run_model:: len of item_attr.D.unique() should == len(data_train.D.unique()"
    item_sorted = sorted(item_attr.D.values)
    train_sorted = sorted(data_train.D.values)

    if with_attrib == True:
        model.fit(data_train, user_features=user_attr, item_features=item_attr, sample_weight=None, epochs=nb_epochs, verbose=verbose)
    else:
        model.fit(data_train, sample_weight=None, epochs=nb_epochs, verbose=verbose)

    data_valid = dct_valid['df_members']
    print("data_train/valid: ", data_train.shape, data_valid.shape)
    hr_filtered = hit_rate(model, data_valid, k=topN, filter_previous=True)
    hr_not_filtered = hit_rate(model, data_valid, k=topN, filter_previous=False)

    # ADD topN recommendation using rankfm routines
    # Do the study, include continuous attributes for destinations (height, long/lat, etc.)

    try:
        return hr_not_filtered, hr_filtered, model.losses
    except:
        return hr_not_filtered, hr_filtered

#---------------------------------------------------------------------------------------------------
def all_in_one(config, seed=1, use_wandb=True, run=None, attr_dct=None):
    print("Enter newlib_monthly::all_in_one")
    rankfm = Rankfm(config=config, # wandb configuration
                dct=AttrDict(), 
                seed=seed,
                use_wandb=use_wandb, run=run, attr_dct=attr_dct)

    # config will be linked to wandb
    rankfm.read_data()
    rankfm.create_model()
    rankfm.run_model()
    rankfm.finish()
    try:
        return rankfm.hr_notfiltered, rankfm.hr_filtered, rankfm.losses
    except:
        return rankfm.hr_notfiltered, rankfm.hr_filtered

#------------------------------------------------------------------------------------------
def generate_age_cuts(nb_cuts):
    """
    Generate age cuts and return a list of lists.
    Each age cut has the form [0,x,y,z,150], where x,y are integers.
    """
    age_cuts = []
    for i in range(nb_cuts):
        x = np.random.randint(15, 40, 1)[0]
        z = np.random.randint(60, 80, 1)[0]
        y = np.random.randint(x+10, z-10, 1)[0]
        cut = [0, x, y, z, 150]
        age_cuts.append(cut)
    return age_cuts

#-----------------------------------------------------------------------------------------------------
def read_dict_file():
    """
    Read the index to cat variables pickled file
    """

    file_path = './mapping_index_cat.pickle'
    if not Path(file_path).exists():
        raise f"File '{file_path}' does not exist" 

    with open('mapping_index_cat.pickle', 'rb') as handle:
        dct = pickle.load(handle)

    return dct

#-------------------------------------------------------------------------------------------------
