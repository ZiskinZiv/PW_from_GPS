#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:04:48 2020

@author: ziskin
"""

def read_ionex_file(file):
    import pandas as pd
    df_sat = pd.read_csv(file, skiprows=48, nrows=32,
                         header=None, delim_whitespace=True)
    df_stations = pd.read_fwf(file, skiprows=80, nrows=50, header=None, widths=[20, 10, 10, 10])
    return df_sat, df_stations

def read_one_sinex(file):
    import pandas as pd
    df = pd.read_fwf(file, skiprows=57)
    df.drop(df.tail(2).index, inplace=True) # drop last n rows
    df.columns = ['bias', 'svn', 'prn', 'station', 'obs1', 'obs2',
                  'bias_start', 'bias_end', 'unit', 'value', 'std']
    ds = xr.Dataset()
    ds.attrs['bias'] = df['bias'].values[0]
    df_sat = df[df['station'].isnull()]
    df_station = df[~df['station'].isnull()]
    return ds


def read_sinex(path, glob='*.BSX'):
    from aux_gps import path_glob
    import xarray as xr
    files = path_glob(path, glob_str=glob)
    for file in files:
        ds = read_one_sinex(file)
        ds_list.append(ds)
    dss = xr.concat(ds, 'time')
    return dss