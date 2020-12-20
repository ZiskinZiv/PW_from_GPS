#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 12:53:24 2020

@author: ziskin
"""
from PW_paths import work_yuval

def get_dryz_from_one_file(file):
    from aux_gps import line_and_num_for_phrase_in_file
    import re
    i, line = line_and_num_for_phrase_in_file('DryZ', file)
    zhd = re.findall("\d+\.\d+", line)[0]
    return float(zhd)


def get_dryz_from_one_station(gnss_path, station='dsea'):
    from aux_gps import path_glob
    from aux_gps import get_timedate_and_station_code_from_rinex
    import xarray as xr
    files = sorted(path_glob(gnss_path / station / 'rinex/30hr/results/',
                      '{}*_debug.tree'.format(station)))
    dt_list = []
    zhd_list = []
    for file in files:
        rfn = file.as_posix().split('/')[-1][0:12]
        dt = get_timedate_and_station_code_from_rinex(rfn, just_dt=True)
        # print('datetime {}'.format(dt.strftime('%Y-%m-%d')))
        dt_list.append(dt)
        zhd = get_dryz_from_one_file(file)
        zhd_list.append(zhd)
    zhd_da = xr.DataArray(zhd_list, dims=['time'])
    zhd_da['time'] = dt_list
    zhd_da *= 100
    zhd_da.name = station
    zhd_da.attrs['units'] = 'cm'
    zhd_da.attrs['long_name'] = 'Zenith Hydrostatic Delay'
    zhd_da = zhd_da.sortby('time')
    return zhd_da


def get_dryz_from_all_stations(gnss_path, savepath):
    from aux_gps import save_ncfile
    from aux_gps import path_glob
    import xarray as xr
    pathes = path_glob(gnss_path, '*/')
    stations = [x.as_posix().split('/')[-1] for x in pathes]
    ds_list = []
    for station in stations:
        print('obtaining ZHD from station {}'.format(station))
        try:
            zhd = get_dryz_from_one_station(gnss_path, station=station)
            ds_list.append(zhd)
        except FileNotFoundError:
            continue
    ds = xr.merge(ds_list)
    filename = 'ZHD_GNSS.nc'
    save_ncfile(ds, savepath, filename)
    return ds

gnss_path = work_yuval / 'GNSS_stations'
savepath = work_yuval
        
    