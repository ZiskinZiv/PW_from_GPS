#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:20:17 2019

@author: shlomi
"""


def untar_file(path):
    """untar Z file to dir"""
    import subprocess
    import glob
    from subprocess import CalledProcessError
    # cmd = [path + 'tar -xvf *.Z']
    # subprocess.call(cmd, shell=True)
    try:
        subprocess.run(['tar', '-xvf'] + glob.glob(path + '/*.Z') + ['-C'] + [path],
                       stdout=subprocess.PIPE, check=True)
    except CalledProcessError:
        subprocess.run(['tar', '-xvf'] + glob.glob(path + '/*.gz') + ['-C'] + [path],
                       stdout=subprocess.PIPE, check=True)
    return


def delete_all_files_in_dir(path, extension='trop'):
    """delete all trop files in dir after reading them to pandas"""
    import glob
    import subprocess
    subprocess.run(['rm', '-f'] + glob.glob(path + '/*.' + extension),
                   stdout=subprocess.PIPE, check=True)
    return


def read_one_station(path, station):
    """read one station file (e.g., HART.trop) to dataframe"""
    import pandas as pd
    import datetime
    import xarray as xr
    # print('proccessing station: {}'.format(station))
    st = pd.read_csv(path + station + '.trop', comment='#', skiprows=1,
                     delim_whitespace=True, header=None,
                     names=['time_tag', 'apriori', 'estimated', 'sigma',
                            'parameter'])
    # change time_tag to proper datetime index:
    st['time'] = datetime.datetime(2000, 1, 1, 12, 0) + pd.to_timedelta(st['time_tag'], unit='s')
    st.drop('time_tag', axis=1, inplace=True)
    st = st.pivot(index='time', columns='parameter')
    # rename cols:
    new_cols = ['_'.join([x[1], x[0]]) for x in st.columns]
    st.columns = new_cols
    # drop all cols but WET:
    to_drop = [x for x in st.columns if not x.startswith('WET')]
    st.drop(to_drop, axis=1, inplace=True)
    # change units to cm:
    st = st * 100
    # rename and drop more:
    cols = ['to_drop', 'zwd', 'sigma']
    st.columns = cols
    st.drop('to_drop', axis=1, inplace=True)
    stxr = xr.DataArray(st, dims=['time', 'zwd'])
    stxr['zwd']=['value', 'sigma']
    stxr.name = station
    return stxr

def read_all_stations_in_day(path):
    """read all the stations in a day (directory)"""
    import xarray as xr
    import glob
    names = glob.glob(path + '/*.trop')
    station_names =[x.split('/')[-1].split('.')[0] for x in names]
    da_list = []
    for station in station_names:
        da_list.append(read_one_station(path + '/', station))
    ds = xr.merge(da_list)
    return ds

def untar_read_delete_day(path):
    """untar file in directory, read all the station files to datasets
    and delete the remaining files"""
    untar_file(path)
    day = path.split('/')[-1]
    year = path.split('/')[-2]
    print('reading day {}, in year {}'.format(day, year))
    ds = read_all_stations_in_day(path)
    delete_all_files_in_dir(path)
    return ds


def read_entire_year(base_path):
    """read all year all stations, return concateneted dataset"""
    import os
    import xarray as xr
    paths = sorted([x[0] for x in os.walk(base_path)])
    paths.pop(0)
    # ds_list = []
    # ds = xr.Dataset()
    first_ds = untar_read_delete_day(paths[0])
    for path in paths[1:]:
        # ds = ds.merge(untar_read_delete_day(path))
        next_ds = untar_read_delete_day(path)
        ds = first_ds.combine_first(next_ds)
        # ds_list.append(untar_read_delete_day(path))
        first_ds = ds
    # ds = xr.concat(ds_list, 'time')
    return first_ds