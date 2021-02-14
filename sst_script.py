#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 13:48:45 2021

@author: shlomi
"""
from PW_paths import work_yuval
from aux_gps import move_or_copy_files_from_doy_dir_structure_to_single_path
from aux_gps import path_glob
import xarray as xr
import numpy as np
sst_path = work_yuval / 'SST/allData/ghrsst/data/GDS2/L4/GLOB/NCEI/AVHRR_OI/v2'
movepath = work_yuval/'SST/allData'
savepath = work_yuval/'SST'


def save_yearly(movepath, savepath, years):
    ps = path_glob(movepath, '*.nc')
    for year in years:
        print('saving year {}...'.format(year))
        ps_year = [x for x in ps if str(year) in x.as_posix()]
        ds = xr.open_mfdataset(ps_year)
        years, datasets = zip(*ds.groupby("time.year"))
        filename = '{}-'.format(year) + '-'.join(ps[0].as_posix().split('/')[-1].split('-')[1:])
        filepath = savepath / filename
        ds.to_netcdf(filepath)
    print('Done!')
    return
    # # now builds the filenames:
    # filepaths = []
    # for year in years:
    #     filename = '{}-'.format(year) + '-'.join(ps[0].as_posix().split('/')[-1].split('-')[1:])
    #     filepath = savepath / filename
    #     filepaths.append(filepath)

    # xr.save_mfdataset(datasets, filepaths)


def save_subset(savepath, subset='med1'):
    ps = path_glob(savepath, '*.nc')
    ds = xr.open_mfdataset(ps)
    if subset == 'med1':
        print('subsetting to med1')
        lat_slice = [30, 50]
        lon_slice = [-20, 45]
        ds = ds.sel(lat=slice(*lat_slice), lon=slice(*lon_slice))
    yrmin = ds['time'].dt.year.min().item()
    yrmax = ds['time'].dt.year.max().item()
    filename = '{}-{}_{}-'.format(subset, yrmin, yrmax) + \
        '-'.join(ps[0].as_posix().split('/')[-1].split('-')[1:])
    ds.to_netcdf(savepath / filename)
    return ds

# years = move_or_copy_files_from_doy_dir_structure_to_single_path(yearly_path=sst_path, movepath=movepath, opr='copy')
# print('opening copied files and saving to {}'.format(movepath))
years = np.arange(1981, 2021)
save_yearly(movepath, savepath, years)
save_subset(savepath, subset='med1')


