#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 10:32:56 2021

@author: ziskin
"""

from PW_paths import work_yuval
axis_path = work_yuval / 'axis'
gis_path = work_yuval / 'gis'


def process_ims_TD_to_axis_coords(savepath):
    from aux_gps import fill_na_xarray_time_series_with_its_group
    from aux_gps import path_glob
    import xarray as xr
    files = sorted(path_glob(savepath, 'IMS_TD_*.nc'))
    # load latest file:
    ds = xr.load_dataset(files[-1])
    ds_filled = fill_na_xarray_time_series_with_its_group(ds, grp='hour',
                                                          plot=False)
    
    return


def copy_rinex_to_station_dir(main_rinexpath, filenames, suffix='.gz'):
    import shutil
    import os
    from aux_gps import get_timedate_and_station_code_from_rinex
    station = filenames[0][0:4]
    station_dir = main_rinexpath / station
    if not station_dir.is_dir():
        os.mkdir(station_dir)
    cnt = 0
    for filename in filenames:
        year = get_timedate_and_station_code_from_rinex(filename, just_dt=True).year
        doy = filename[4:7]
        if suffix is not None:
            filename += suffix
        to_copy_from = main_rinexpath / str(year) / doy / filename
        to_copy_to = station_dir / filename
        if to_copy_to.is_file():
            print('{} already exists in {}, skipping.'.format(filename, station_dir))
            continue
        try:
            shutil.copy(to_copy_from, to_copy_to)
        except FileNotFoundError:
            print('{} not found, missing ?'.format(filename))
            continue
        cnt += 1
    print('Done copying {} rinex files to {}.'.format(cnt, station_dir))
    return


def produce_rinex_filenames_at_time_window(station='Dimo',
                                           end_dt='2021-04-13T02:00',
                                           window=24):
    """given a end date and a time window (in hours) get the hourly files
    of station going back window hours prior"""
    import pandas as pd
    from aux_gps import get_rinex_filename_from_datetime
    end_dt = pd.to_datetime(end_dt).floor('H')
    print('getting rinex for {} on {} with {} hours backwards.'.format(station, end_dt, window))
    start_dt = end_dt - pd.Timedelta('{} hour'.format(window))
    dt_range = pd.date_range(start_dt, end_dt - pd.Timedelta('1 hour', units='H'), freq='1H')
    # first = get_rinex_filename_from_datetime(station, dt=start_dt, st_lower=False)
    dt_range = [x.strftime('%Y-%m-%dT%H:%M:%S') for x in dt_range]
    filenames = get_rinex_filename_from_datetime(station, dt=dt_range, st_lower=False)
    return filenames


def read_and_concat_smoothFinals(rinexpath, solution='Final'):
    import xarray as xr
    from aux_gps import save_ncfile
    from aux_gps import path_glob
    years = [x.as_posix().split('/')[-1] for x in path_glob(rinexpath, '*/')]
    years = [x for x in years if x.isnumeric()]
    for year in years:
        dsl = []
        # doys = [x.as_posix().split('/')[-1] for x in path_glob(rinexpath/year, '*/')]
        for doypath in path_glob(rinexpath/year, '*/'):
            file = doypath / 'dr' / solution / 'smoothFinal.nc'
            if file.is_file():
                dsl.append(xr.load_dataset(file))
                print('found smoothFinal.nc in {}'.format(doypath))
        if dsl:
            ds = xr.concat(dsl, 'time')
            ds = ds.sortby('time')
            save_ncfile(ds, rinexpath, 'smoothFinal_{}.nc'.format(year))
    return ds


def move_files(path_orig, path_dest, files, out_files=None, verbose=False):
    """move files (a list containing the file names) and move them from
    path_orig to path_dest"""
    import shutil
    import logging
    logger = logging.getLogger('gipsyx')
    if isinstance(files, str):
        files = [files]
    if out_files is not None:
        if isinstance(out_files, str):
            out_files = [out_files]
    orig_filenames_paths = [path_orig / x for x in files]
    if out_files is None:
        out_files = files
    dest_filenames_paths = [path_dest / x for x in out_files]
    # delete files if size =0:
    for file, orig, dest in zip(
            files, orig_filenames_paths, dest_filenames_paths):
        # check for file existance in orig:
        if not orig.is_file():
            if verbose:
                logger.warning('{} does not exist in {}'.format(file, orig))
            continue
        # check if its size is 0:
        if orig.stat().st_size == 0:
            orig.resolve().unlink()
        else:
            shutil.move(orig.resolve(), dest.resolve())
    return


def run_rinex_compression_on_file(path_dir, filename, command='gunzip', cmd_path=None):
    import subprocess
    from subprocess import CalledProcessError
    if not path_dir.is_dir():
        raise ValueError('{} is not a directory!'.format(path_dir))
    if command == 'gunzip':
        cmd = 'gunzip {}'.format(filename)
        try:
            subprocess.run(cmd, shell=True, check=True, cwd=path_dir)
        except CalledProcessError:
            print('{} failed !'.format(command))
            return
    elif command == 'gzip':
        cmd = 'gzip {}'.format(filename)
        try:
            subprocess.run(cmd, shell=True, check=True, cwd=path_dir)
        except CalledProcessError:
            print('{} failed !'.format(command))
            return
    elif command == 'crx2rnx':
        if cmd_path is not None:
            cmd = '{}/CRX2RNX -d {}'.format(cmd_path.as_posix(), filename)
        else:
            cmd = 'CRX2RNX -d {}'.format(filename)
        try:
            subprocess.run(cmd, shell=True, check=True, cwd=path_dir)
        except CalledProcessError:
            print('{} failed !'.format(command))
            return
    elif command == 'rnx2crx':
        if cmd_path is not None:
            cmd = '{}/RNX2CRX -d {}'.format(cmd_path.as_posix(), filename)
        else:
            cmd = 'RNX2CRX -d {}'.format(filename)
        try:
            subprocess.run(cmd, shell=True, check=True, cwd=path_dir)
        except CalledProcessError:
            print('{} failed !'.format(command))
            return
    else:
        raise ValueError('{} not known! '.format(command))


def run_rinex_compression_on_folder(path_dir, command='gunzip', glob='*.14d', cmd_path=None):
    import subprocess
    from subprocess import CalledProcessError
    if not path_dir.is_dir():
        raise ValueError('{} is not a directory!'.format(path_dir))
    # subprocess.call("ls", cwd=path_dir)
    if command == 'gunzip':
        cmd = 'gunzip {}'.format(glob)
        try:
            subprocess.run(cmd, shell=True, check=True, cwd=path_dir)
        except CalledProcessError:
            print('{} failed !'.format(command))
            return
    elif command == 'gzip':
        cmd = 'gzip {}'.format(glob)
        try:
            subprocess.run(cmd, shell=True, check=True, cwd=path_dir)
        except CalledProcessError:
            print('{} failed !'.format(command))
            return
    elif command == 'crx2rnx':
        if cmd_path is not None:
            cmd = 'for f in {} ; do {}/CRX2RNX $f ; done'.format(glob, cmd_path.as_posix())
        else:
            cmd = 'for f in {} ; do CRX2RNX $f ; done'.format(glob)
        try:
            subprocess.run(cmd, shell=True, check=True, cwd=path_dir)
        except CalledProcessError:
            print('{} failed !'.format(command))
            return
    elif command == 'rnx2crx':
        if cmd_path is not None:
            cmd = 'for f in {} ; do {}/RNX2CRX $f ; done'.format(glob, cmd_path.as_posix())
        else:
            cmd = 'for f in {} ; do RNX2CRX $f ; done'.format(glob)
        try:
            subprocess.run(cmd, shell=True, check=True, cwd=path_dir)
        except CalledProcessError:
            print('{} failed !'.format(command))
            return
    else:
        raise ValueError('{} not known! '.format(command))


def teqc_concat_rinex(path_dir, rfn=None, glob='*.14o', cmd_path=None):
    import subprocess
    from subprocess import CalledProcessError
    from aux_gps import path_glob
    from aux_gps import replace_char_at_string_position
    if not path_dir.is_dir():
        raise ValueError('{} is not a directory!'.format(path_dir))
    # subprocess.call("ls", cwd=path_dir)
    if rfn is None:
        files = sorted(path_glob(path_dir, glob))
        rfn = files[0].as_posix().split('/')[-1]
        rfn = replace_char_at_string_position(rfn, char='0', pos=7)
        print('rfn is : {}'.format(rfn))
    if cmd_path is not None:
        cmd = '{}/teqc -phc {} > {}'.format(cmd_path.as_posix(), glob, rfn)
    else:
        cmd = 'teqc -phc {} > {}'.format(glob, rfn)
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=path_dir)
    except CalledProcessError:
        print('{} failed !'.format(cmd))
        return



def read_axis_stations(path=axis_path):
    from aux_gps import path_glob
    import pandas as pd
    file = path_glob(path, 'Axis_StationInformation_*.csv')[-1]
    df = pd.read_csv(file, header=1)
    df.columns = ['station_id', 'unique_id', 'station_code', 'X', 'Y', 'Z',
                  'lat', 'lon', 'alt', 'ant_height', 'ant_name',
                  'station_name', 'menufacturer', 'rec_name', 'rec_firmware',
                  'rec_SN']
    df = df.set_index('station_code')
    return df


def produce_geo_axis_gnss_solved_stations(axis_path=axis_path, path=gis_path,
                                          add_distance_to_coast=False,
                                          plot=True):
    import geopandas as gpd
    from ims_procedures import get_israeli_coast_line
    import pandas as pd
    df = read_axis_stations(path=axis_path)
    df = df[['lat', 'lon', 'alt', 'station_name']]
    isr = gpd.read_file(path / 'Israel_and_Yosh.shp')
    isr.crs = {'init': 'epsg:4326'}
    stations = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon,
                                                                df.lat),
                                crs=isr.crs)
    if add_distance_to_coast:
        isr_coast = get_israeli_coast_line(path=path)
        coast_lines = [isr_coast.to_crs(
            'epsg:2039').loc[x].geometry for x in isr_coast.index]
        for station in stations.index:
            point = stations.to_crs('epsg:2039').loc[station, 'geometry']
            stations.loc[station, 'distance'] = min(
                [x.distance(point) for x in coast_lines]) / 1000.0
    # define groups for longterm analysis, north to south, west to east:
    coastal_dict = {
        key: 0 for (key) in [
            'Mzra',
            'Haif',
            'Maag',
            'TLV_',
            'Ash_',
            'Ashk',
            'Ohad']}
    highland_dict = {key: 1 for (key) in
                     ['Jish', 'Cana', 'Arra', 'kshm', 'Ksm_', 'Jrsl', 'Dora',
                      'Raha', 'Dimo', 'Ramo']}
    eastern_dict = {key: 2 for (key) in
                    ['MSha', 'Alon', 'Gshr', 'Hama', 'Bisa', 'Jeri', 'Ddse',
                     'Yaha', 'Yotv', 'Elat']}
    groups_dict = {**coastal_dict, **highland_dict, **eastern_dict}
    stations['groups_annual'] = pd.Series(groups_dict)
    # define groups with climate code
    # gr1_dict = {
    #     key: 0 for (key) in [
    #         'kabr',
    #         'bshm',
    #         'csar',
    #         'tela',
    #         'alon',
    #         'nzrt',
    #         'mrav',
    #         'yosh',
    #         'jslm',
    #         'elro',
    #         'katz']}
    # gr2_dict = {key: 1 for (key) in
    #             ['slom', 'klhv', 'yrcm', 'drag']}
    # gr3_dict = {key: 2 for (key) in
    #             ['nizn', 'ramo', 'dsea', 'spir', 'nrif', 'elat']}
    # groups_dict = {**gr1_dict, **gr2_dict, **gr3_dict}
    # stations['groups_climate'] = pd.Series(groups_dict)
    # if climate_path is not None:
    #     cc = pd.read_csv(climate_path / 'gnss_station_climate_code.csv',
    #                       index_col='station')
    #     stations = stations.join(cc)
    if plot:
        ax = isr.plot()
        stations.plot(ax=ax, column='alt', cmap='Greens',
                      edgecolor='black', legend=True)
        for x, y, label in zip(stations.lon, stations.lat,
                               stations.index):
            ax.annotate(label, xy=(x, y), xytext=(3, 3),
                        textcoords="offset points")
    return stations


def read_multi_station_tdp_file(file, stations, savepath=None):
    import pandas as pd
    import xarray as xr
    from aux_gps import save_ncfile
    df_raw = pd.read_csv(file, header=None, delim_whitespace=True)
    # first loop over list of stations and extract the data:
    df_stns = [df_raw[df_raw.iloc[:, -1].str.contains(x)] for x in stations]
    # now process each df from df_stns and extract the keys:
    keys = ['DryZ', 'WetZ', 'GradNorth', 'GradEast', 'Pos.X', 'Pos.Y', 'Pos.Z']
    desc = ['Zenith Hydrostatic Delay', 'Zenith Wet Delay',
            'North Gradient of Zenith Wet Delay',
            'East Gradient of Zenith Wet Delay',
            'WGS84(geocentric) X coordinate',
            'WGS84(geocentric) Y coordinate', 'WGS84(geocentric) Z coordinate']
    units = ['cm', 'cm', 'cm/m', 'cm/m', 'm', 'm', 'm']
    desc_dict = dict(zip(keys, desc))
    units_dict = dict(zip(keys, units))
    ppps = []
    for df_stn in df_stns:
        df_list = [df_stn[df_stn.iloc[:, -1].str.contains(x)] for x in keys]
        # make sure that all keys in df have the same length:
        # assert len(set([len(x) for x in df_list])) == 1
        # translate the seconds col to datetime:
        seconds = df_list[-1].iloc[:, 0]
        dt = pd.to_datetime('2000-01-01T12:00:00')
        time = dt + pd.to_timedelta(seconds, unit='sec')
        # build a new df that contains all the vars(from keys):
        ppp = pd.DataFrame(index=time)
        ppp.index.name = 'time'
        for i, df in enumerate(df_list):
            if df.empty:
                continue
            df.columns = ['seconds', 'to_drop', keys[i], keys[i] + '_error',
                          'meta']
            ppp[keys[i]] = df[keys[i]].values
            ppp[keys[i] + '_error'] = df[keys[i] + '_error'].values
            # rename all the Pos. to nothing:
            # ppp.columns = ppp.columns.str.replace('Pos.', '')
        ppps.append(ppp.to_xarray())
    ds = xr.concat(ppps, 'station')
    ds['station'] = stations
    for da in ds:
        if 'Wet' in da or 'Dry' in da or 'Grad' in da:
            ds[da] = ds[da] * 100
            if 'Wet' in da:
                ds[da].attrs['units'] = units_dict.get('WetZ')
            elif 'Grad' in da:
                ds[da].attrs['units'] = units_dict.get('GradNorth')
        ds[da].attrs['long_name'] = desc_dict.get(da, '')
        if 'Pos' in da:
            ds[da].attrs['units'] = 'm'
    pos_names = [x for x in ds if 'Pos' in x]
    pos_new_names = [x.split('.')[-1] for x in pos_names]
    ds = ds.rename(dict(zip(pos_names, pos_new_names)))
    if savepath is not None:
        # filename = file.as_posix().split('/')[-1].split()
        save_ncfile(ds, savepath, 'smoothFinal.nc')
    return ds
