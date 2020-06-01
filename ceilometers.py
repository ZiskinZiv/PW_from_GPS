#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:41:11 2020

@author: shlomi
"""
from PW_paths import work_yuval
ceil_path = work_yuval / 'ceilometers'
# available stations: Jerousalem, Nevatim, Ramat_David, Tel_Aviv
stations_dict = {
    'Tel_Aviv': ['TLV', 34.8, 32.1, 5],
    'Nevatim': ['NV', 34.9, 31.2, 400],
    'Ramat_David': ['RD', 35.2, 32.7, 50],
    'Jerusalem': ['JR', 35.2, 31.8, 830]}


def align_pw_to_ceilometers_mlh(path=work_yuval, ceil_path=ceil_path,
                                pw_site='jslm', mlh_site='JR', ax=None,
                                scatter=False):
    import xarray as xr
    mlh = xr.load_dataset(ceil_path / 'MLH_from_ceilometers.nc')
    pw = xr.load_dataset(work_yuval / 'GNSS_PW_hourly_thresh_50_homogenized.nc')
    pw = pw[[x for x in pw if '_error' not in x]]
    pw = pw ['tela', 'klhv', 'jslm', 'nzrt', 'yrcm']
    return

def scatter_plot_pw_mlh(pw, mlh, ax=None):
    
    return ax

def twin_hourly_mean_plot(pw, mlh, ax=None):
    
    return ax


def read_all_ceilometer_stations(path=ceil_path):
    import xarray as xr
    from aux_gps import save_ncfile
    stations = [x for x in stations_dict.keys()]
    da_list = []
    for station in stations:
        print('reading station {}'.format(station))
        da = read_ceilometer_station(path=path, name=station)
        da_list.append(da)
    ds = xr.merge(da_list)
    save_ncfile(ds, path, filename='MLH_from_ceilometers.nc')
    return ds


def read_ceilometer_station(path=ceil_path, name='Jerusalem'):
    from aux_gps import path_glob
    import pandas as pd
    files = path_glob(path, '{}_*.mat'.format(name))
    df_list = []
    for file in files:
        df_list.append(read_one_matfile_ceilometers(file))
    df = pd.concat(df_list, axis=0)
    df.index.name = 'time'
    df.drop_duplicates(inplace=True)
    da = df.to_xarray()
    da.name = stations_dict[name][0]
    da.attrs['full_name'] = 'Mixing Layer Height'
    da.attrs['name'] = 'MLH'
    da.attrs['units'] = 'm'
    da.attrs['station_full_name'] = name
    da.attrs['lon'] = stations_dict[name][1]
    da.attrs['lat'] = stations_dict[name][2]
    da.attrs['alt'] = stations_dict[name][3]
    return da


def read_one_matfile_ceilometers(file):
    from scipy.io import loadmat
    import pandas as pd
    mat = loadmat(file)
    name = [x for x in mat.keys()][-1]
    mdata = mat[name]
    li = []
    days = []
    for i in range(mdata.shape[0]):
        days.append([x.squeeze().item() for x in mdata[i,0]]) 
        li.append([x.squeeze().item() for x in mdata[i,1:]])
    days = [x[0] for x in days]
    df = pd.DataFrame(li[1:], index=days[1:])
    df.columns = [int(x) for x in li[0]]
    df.drop(df.tail(2).index,inplace=True)
    df = df.rename({'201508110': '20150811'}, axis=0)
    df = df.rename({'201608110': '20160811'}, axis=0)
    df.index = pd.to_datetime(df.index)
    # transform to time-series:
    df_list = []
    for date in df.index:
        dts = date + pd.Timedelta(1, unit='H')
        dates = pd.date_range(dts, periods=24, freq='H')
        df1 = pd.DataFrame(df.loc[date].values, index=dates)
        df_list.append(df1)
    s = pd.concat(df_list)[0]
    return s
