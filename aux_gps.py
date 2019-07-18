#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:33:19 2019

@author: ziskin
"""
from PW_startup import *

def coarse_dem(data, dem_path = work_yuval / 'AW3D30'):
    """coarsen to data coords"""
    # data is lower resolution than awd
    # TODO: add save file option with resolution
    import salem
    awd = salem.open_xr_dataset(dem_path / 'israel_dem.tif')
    awds = data.salem.lookup_transform(awd)
    return awds


def concat_shp(path, shp_file_list, saved_filename):
    import geopandas as gpd
    import pandas as pd
    shapefiles = [path / x for x in shp_file_list]
    gdf = pd.concat([gpd.read_file(shp)
                     for shp in shapefiles]).pipe(gpd.GeoDataFrame)
    gdf.to_file(path / saved_filename)
    print('saved {} to {}'.format(saved_filename, path))
    return


def scale_xr(da, upper=1.0, lower=0.0, unscale=False):
    if not unscale:
        dh = da.max()
        dl = da.min()
        da_scaled = (((da-dl)*(upper-lower))/(dh-dl)) + lower
        da_scaled.attrs = da.attrs
        da_scaled.attrs['scaled'] = True
        da_scaled.attrs['lower'] = dl.item()
        da_scaled.attrs['upper'] = dh.item()
    if unscale and da.attrs['scaled']:
        dh = da.max()
        dl = da.min()
        upper = da.attrs['upper']
        lower = da.attrs['lower']
        da_scaled = (((da-dl)*(upper-lower))/(dh-dl)) + lower
    return da_scaled


def print_saved_file(name, path):
    print(name + ' was saved to ' + str(path))
    return


def dim_intersection(da_list, dim='time', dropna=True):
    import pandas as pd
    if dropna:
        setlist = [set(x.dropna(dim)[dim].values) for x in da_list]
    else:
        setlist = [set(x[dim].values) for x in da_list]
    empty_list = [x for x in setlist if not x]
    if empty_list:
        print('NaN dim drop detected, check da...')
        return
    u = list(set.intersection(*setlist))
    # new_dim = list(set(a.dropna(dim)[dim].values).intersection(
    #     set(b.dropna(dim)[dim].values)))
    if dim == 'time':
        new_dim = sorted(pd.to_datetime(u))
    else:
        new_dim = sorted(u)
    return new_dim


def get_unique_index(da, dim='time'):
    import numpy as np
    _, index = np.unique(da[dim], return_index=True)
    da = da.isel({dim: index})
    return da


def Zscore_xr(da, dim='time'):
    """input is a dattarray of data and output is a dattarray of Zscore
    for the dim"""
    z = (da - da.mean(dim=dim)) / da.std(dim=dim)
    return z


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
                print(dim + ': non-NaN labels: ' +
                      str(dim_non_nans) + ' of total ' + str(dim_len))
        return non_nans
    if isinstance(data, xr.DataArray):
        nn_dict = nan_da(data)
        return nn_dict
    elif isinstance(data, np.ndarray):
        nans = np.count_nonzero(np.isnan(data))
        non_nans = np.count_nonzero(~np.isnan(data))
        if verbose:
            print(str(type(data)))
            print('non-NaN entries: ' + str(non_nans) + ' of total ' +
                  str(data.size) + ', shape:' + str(data.shape) + ', type:' +
                  str(data.dtype))
    elif isinstance(data, xr.Dataset):
        for varname in data.data_vars.keys():
            non_nans = nan_da(data[varname])
    return non_nans
