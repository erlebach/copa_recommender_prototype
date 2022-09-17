# Explore relationship between age and destination
# I should also plot the number of trips (with or without duplication of D) per age

# 2022-09-16
# Read data for one month

import math
import numpy as np
import pandas as pd
from attrdict import AttrDict
import matplotlib.pyplot as plt
import seaborn as sns
import explore_lib as elib

infile = "../data/attrib_yr_mo_2017_07.csv"
attr_file = "../data/temp_long_lat_height.csv"

df = pd.read_csv(infile)
df = df[['D','age_departure']]
df.age_departure = df.age_departure.astype('int')

# Read Attributes
# D,avg_yr_l,avg_yr_h,avg_wi,avg_sp,avg_su,avg_fa,IATA,LAT_DEC,LON_DEC,HEIGHT
attrib_df = pd.read_csv(attr_file)
attrib_df = attrib_df[['D','LAT_DEC','LON_DEC', 'HEIGHT', 'avg_sp', 'avg_su', 'avg_fa', 'avg_wi']]

# Longitude and Latitude at PTY
pty = attrib_df[attrib_df['D'] == 'PTY']
pty_lat_lon = pty.LAT_DEC.values[0], pty.LON_DEC.values[0]

# Merge attribute file with destination file
df1m = df.merge(attrib_df, how='inner')

# Group by age
df1 = df1m.groupby('age_departure')


dct = AttrDict()
dct._setattr('_sequence_type', list) # allow use of append
dct.age = []
dct.avg_dist_g = []
dct.std_dist_g = []
dct.avg_dist_pty = []
dct.std_dist_pty = []
dct.avg_height = []
dct.std_height = []
dct.avg_avg_wi = []
dct.std_avg_wi = []
dct.avg_avg_sp = []
dct.std_avg_sp = []
dct.avg_avg_su = []
dct.std_avg_su = []
dct.avg_avg_fa = []
dct.std_avg_fa = []

for age in range(20,80):
    dist_stat, dist_pty_stat, scalars_dct_stat = elib.cluster_stats(df1, pty_lat_lon, age=age)
    dct.age.append(age)
    dct['age'].append(age)
    dct['avg_dist_g'].append(dist_stat[0])  # does not work with dot notation
    dct['std_dist_g'].append(dist_stat[1])
    dct['avg_dist_pty'].append(dist_pty_stat[0])
    dct['std_dist_pty'].append(dist_pty_stat[1])
    dct['avg_height'].append(scalars_dct_stat.height[0])
    dct['std_height'].append(scalars_dct_stat.height[1])
    dct['avg_avg_su'].append(scalars_dct_stat.avg_su[0])
    dct['std_avg_su'].append(scalars_dct_stat.avg_su[1])
    dct['avg_avg_fa'].append(scalars_dct_stat.avg_fa[0])
    dct['std_avg_fa'].append(scalars_dct_stat.avg_fa[1])
    dct['avg_avg_wi'].append(scalars_dct_stat.avg_wi[0])
    dct['std_avg_wi'].append(scalars_dct_stat.avg_wi[1])
    dct['avg_avg_sp'].append(scalars_dct_stat.avg_sp[0])
    dct['std_avg_sp'].append(scalars_dct_stat.avg_sp[1])

elib.plot_data(dct, infile, attr_file)

#MEMBER_ID,TRUE_OD,D,FLIGHT_DATE,BOOKING_DATE,TICKET_SALES_DATE,TICKET_NUMBER,TRUE_ORIGIN_COUNTRY,ADDR_COUNTRY,PNR,PARTY_SZ,size,booking_date,booking_dowk,booking_mo,flight_date,flight_dowk,flight_mo,TIER_LEVEL,GENDER,BIRTH_DATE,NATIONALITY,ADDR_COUNTRY_y,age_departure,flight_yr

# Plot age statistics: age on x axis, bar charts with error bars. 
# Put the data into pandas dataframe and use Seaborn
