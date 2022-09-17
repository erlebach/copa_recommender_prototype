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

def distance(lat1, lon1, lat2, lon2):
    # D = d (pi/180) R
    # Earth: radius R = 6370 km
    # return distance between two (lat,lon) points on Earth in km
    cos_d = np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(lon2 - lon1)
    D = np.arccos(cos_d) * math.pi * 6370. / 180.
    return D

def centroid(lat_dec, lon_dec):
    sz = lon_dec.shape[0]
    avg_lat = np.mean(lat_dec)
    avg_lon = np.mean(lon_dec)
    return avg_lat, avg_lon

# Compute mean and std of distances between centroid and all points
def mean_std_distances(lat_dec, lon_dec, lat_g, lon_g):
    dist = distance(lat_dec, lon_dec, lat_g, lon_g)
    avg_dist = np.mean(dist)
    std_dist = np.std(dist)
    return avg_dist, std_dist

def mean_std_scalars(scalar_list, headers):
    scalars_dct = AttrDict()
    for i, scalar in enumerate(scalar_list):
        mean = np.mean(scalar)
        std = np.std(scalar)
        scalars_dct[headers[i]] = (mean, std)
    return scalars_dct


def cluster_stats(df, pty_lat_lon, age):
    """
    Given a list of points on Earth, compute their center of gravity, average spoke, and its std
    """
    df1_age = df.get_group(age).sort_values('D')
    df2 = df1_age.drop_duplicates('D')
    # Compute center of gravity of destinations
    lon_dec = df2.LON_DEC.values
    lat_dec = df2.LAT_DEC.values
    height = df2.HEIGHT.values
    avg_wi = df2.avg_wi.values
    avg_sp = df2.avg_sp.values
    avg_su = df2.avg_su.values
    avg_fa = df2.avg_fa.values
    lat_g, lon_g = centroid(lat_dec, lon_dec)
    dist_stat = mean_std_distances(lat_dec, lon_dec, lat_g, lon_g)
    dist_pty_stat = mean_std_distances(lat_dec, lon_dec, *pty_lat_lon)

    scalars_dct_stat = mean_std_scalars([height, avg_wi, avg_sp, avg_su, avg_fa], 
              headers=['height', 'avg_wi', 'avg_sp', 'avg_su', 'avg_fa'])
    return dist_stat, dist_pty_stat, scalars_dct_stat

def plot_data(dct, infile, attr_file):
    """
    Plot data as a function of age [20-80] (brackets of 1)
    """
    df_age = pd.DataFrame(dct)
    fig, axes = plt.subplots(7, 1, figsize=(12, 16))
    df_age.plot(kind='bar', x='age', y='avg_dist_g', yerr='std_dist_g', ax=axes[0], title='Distance Cluster Radius')
    df_age.plot(kind='bar', x='age', y='avg_dist_pty', yerr='std_dist_pty', ax=axes[1], title="Mean Distance from PTY")
    df_age.plot(kind='bar', x='age', y='avg_height', yerr='std_height', ax=axes[2], title='Altitude')
    temp_lim = (50,80)
    df_age.plot(kind='bar', x='age', y='avg_avg_wi', yerr='std_avg_wi', ax=axes[3], title="Avg Winter Temp", ylim=temp_lim)
    df_age.plot(kind='bar', x='age', y='avg_avg_sp', yerr='std_avg_sp', ax=axes[4], title="Avg Spring Temp", ylim=temp_lim)
    df_age.plot(kind='bar', x='age', y='avg_avg_su', yerr='std_avg_su', ax=axes[5], title="Avg Summer Temp", ylim=temp_lim)
    df_age.plot(kind='bar', x='age', y='avg_avg_fa', yerr='std_avg_fa', ax=axes[6], title="Avg Fall Temp", ylim=temp_lim)
    plt.suptitle(f"Data Files: {infile}, {attr_file}")
    plt.tight_layout()
    plt.show()
