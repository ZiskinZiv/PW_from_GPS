#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:50:20 2019

@author: shlomi
"""

import pandas as pd
import sys
import xarray as xr
import numpy as np
if sys.platform == 'linux':
    work_path = '/home/shlomi/Desktop/DATA/Work Files/PW_yuval/'
elif sys.platform == 'darwin':  # mac os
    work_path = '/Users/shlomi/Documents/PW_yuval/'

PW_stations_path = work_path + '1minute/'

stations = pd.read_csv(PW_stations_path + 'Zstations', header=0,
                       delim_whitespace=True)
station_names = stations['NAME'].values.tolist()
df_list = []
for st_name in station_names:
    print('Proccessing ' + st_name + ' Station...')
    df = pd.read_csv(PW_stations_path + st_name, delim_whitespace=True)
    df.columns = ['date', 'time', 'PW']
    df.index = pd.to_datetime(df['date'] + 'T' + df['time'])
    df.drop(columns=['date', 'time'], inplace=True)
    df_list.append(df)
df = pd.concat(df_list, axis=1)
print('Concatanting to Xarray...')
# ds = xr.concat([df.to_xarray() for df in df_list], dim="station")
# ds['station'] = station_names
df.columns = station_names
ds = df.to_xarray()
ds = ds.rename({'index': 'time'})
# da = ds.to_array(name='PW').squeeze(drop=True)
comp = dict(zlib=True, complevel=9)  # best compression
encoding = {var: comp for var in ds.data_vars}
print('Saving to PW_2007-2016.nc')
ds.to_netcdf(work_path + 'PW_2007-2016.nc', 'w', encoding=encoding)
print('Done!')
# clean the data:
# da = da.where(da >= 0, np.nan)
# da = da.where(da < 100, np.nan)

# plot the data:
ds.to_array(dim='station').plot(x='time',col='station',col_wrap=4)
# hist:
# df=ds.to_dataframe()
sl=(df>0) & (df<50)
df[sl].hist(bins=30, grid=False, figsize=(15,8))
def desc_nan(data, verbose=True):
    """count only NaNs in data and returns the thier amount and the non-NaNs"""
    import numpy as np
    import xarray as xr

    def nan_da(data):
        nans = np.count_nonzero(np.isnan(data.values))
        non_nans = np.count_nonzero(~np.isnan(data.values))
        if verbose:
            print(str(type(data)))
            print(data.name + ': non-NaN entries: ' + str(non_nans) + ' of total ' +
                  str(data.size) + ', shape:' + str(data.shape) + ', type:' +
                  str(data.dtype))
            print('Dimensions:')
        dim_nn_list = []
        for dim in data.dims:
            dim_len = data[dim].size
            dim_non_nans = np.int(data.dropna(dim)[dim].count())
            dim_nn_list.append(dim_non_nans)
            if verbose:
                print(dim + ': non-NaN labels: ' + str(dim_non_nans) + ' of total ' +
                      str(dim_len))
        return non_nans
    if type(data) == xr.DataArray:
        nn_dict = nan_da(data)
        return nn_dict
    elif type(data) == np.ndarray:
        nans = np.count_nonzero(np.isnan(data))
        non_nans = np.count_nonzero(~np.isnan(data))
        if verbose:
            print(str(type(data)))
            print('non-NaN entries: ' + str(non_nans) + ' of total ' +
                  str(data.size) + ', shape:' + str(data.shape) + ', type:' +
                  str(data.dtype))
    elif type(data) == xr.Dataset:
        for varname in data.data_vars.keys():
            non_nans = nan_da(data[varname])
    return non_nans
