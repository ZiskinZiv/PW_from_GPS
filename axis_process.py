#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 10:32:56 2021

@author: ziskin
"""

from PW_paths import work_yuval
axis_path = work_yuval / 'axis'
gis_path = work_yuval / 'gis'


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
        save_ncfile(ds, savepath, 'smoothFinal.nc')
    return ds