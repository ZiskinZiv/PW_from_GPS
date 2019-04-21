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
    # check for empty files:
    if st.empty:
        print('{}.trop is an empty file... skipping'.format(station))
        return None
    else:
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
        stxr['zwd'] = ['value', 'sigma']
        stxr.name = station
        return stxr


def read_all_stations_in_day(path, verbose=False):
    """read all the stations in a day (directory)"""
    import xarray as xr
    import glob
    names = glob.glob(path + '/*.trop')
    station_names = [x.split('/')[-1].split('.')[0] for x in names]
    da_list = []
    for station in station_names:
        if verbose:
            print('reading station {}'.format(station))
        one_station = read_one_station(path + '/', station)
        if one_station is not None:
            da_list.append(one_station)
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


def read_entire_year(base_year_path, save_path=None):
    """read all year all stations, return concateneted dataset"""
    import os
    # import xarray as xr
    paths = sorted([x[0] for x in os.walk(base_year_path)])
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
    if save_path is not None:
        print('saving file to {}'.format(save_path))
        year = paths[0].split('/')[-2]
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in first_ds.data_vars}
        first_ds.to_netcdf(save_path + 'garner_trop_all_stations_' + year +
                           '.nc', 'w', encoding=encoding)
        print('Done!')
    # ds = xr.concat(ds_list, 'time')
    return first_ds


def pick_one_station_assemble_time_series(path,
                                          filenames='garner_trop_all_stations',
                                          station_name=None, save_path=None):
    """pick a GPS station in UPPERCASE (four letters) and return a timeseries
    for all the epochs"""
    import xarray as xr
    if station_name is not None:
        print('Opening all stations...')
        all_data = xr.open_mfdataset(path + filenames + '*.nc')
        try:
            station = all_data[station_name]
        except KeyError:
            print('The station name does not exists...pls pick another.')
            return
        print('picked station: {}'.format(station_name))
        station.compute()
        if save_path is not None:
            print('saving file to {}'.format(save_path))
            comp = dict(zlib=True, complevel=9)  # best compression
            encoding = {var: comp for var in station.to_dataset().data_vars}
            station.to_netcdf(save_path + 'garner_trop_' + station +
                              '.nc', 'w', encoding=encoding)
            print('Done!')
    else:
        raise KeyError('pls pick a station...')
    return station


def check_path(path):
    import os
    path = str(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(path + ' does not exist...')
    return path


def check_station_name(name):
    # import os
    name = str(name)
    if len(name) != 4:
        raise argparse.ArgumentTypeError(name + ' should be 4 letters...')
    return name.upper()


if __name__ == '__main__':
    import argparse
    import sys
    import numpy as np
    start_year = 2011
    end_year = 2019
    parser = argparse.ArgumentParser(description='a command line tool for combining gipsy-tpdl files')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # remove this line: optional = parser...
    required.add_argument('--path', help="a full path to read years and save nc files,\
                          e.g., /data11/ziskin/", type=check_path)
    required.add_argument('--mode', help="mode to operate, e.g., read_years, get_station,",
                          type=str, choices=['read_years', 'get_station'])
    optional.add_argument('--station', help='GPS station name, 4 UPPERCASE letters',
                          type=check_station_name)
#                          metavar=str(cds.start_year) + ' to ' + str(cds.end_year))
#    optional.add_argument('--half', help='a spescific six months to download,\
#                          e.g, 1 or 2', type=int, choices=[1, 2],
#                          metavar='1 or 2')
    parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()
    # print(parser.format_help())
#    # print(vars(args))
    if args.path is None:
        print('path is a required argument, run with -h...')
        sys.exit()
#    elif args.field is None:
#        print('field is a required argument, run with -h...')
#        sys.exit()
    if args.mode == 'read_years':
        years = np.arange(start_year, end_year + 1).astype('str')
        for year in years:
            year_path = args.path + '/' + year
            read_entire_year(year_path, save_path=args.path)
    elif args.mode == 'get_station':
        if args.station is not None:
            pick_one_station_assemble_time_series(args.path,
                                                  filenames='garner_trop_all_stations',
                                                  station_name=args.station,
                                                  save_path=args.path)
        else:
            raise ValueError('need to specify station!')
# command to wget all files and all dirs from http site:
# nohup wget -r --user=anonymous --password='shlomiziskin@gmail.com'
# -e robots=off --no-parent -nH --cut-dirs=2 --reject="index.html*" -U mozilla
# http://garner.ucsd.edu/pub/solutions/gipsy/trop > nohup_wget.out&
