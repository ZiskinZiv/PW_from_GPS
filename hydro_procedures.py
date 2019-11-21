#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:08:43 2019

@author: ziskin
"""

from PW_paths import work_yuval
hydro_path = work_yuval / 'hydro'
gis_path = work_yuval / 'gis'


def read_hydro_metadata(path=hydro_path, gis_path=gis_path, plot=True):
    import pandas as pd
    import geopandas as gpd
    df = pd.read_excel(hydro_path / 'קטלוג_תחנות_הידרומטריות.xlsx',
                       header=4)
    # drop last row:
    df.drop(df.tail(1).index, inplace=True)  # drop last n rows
    df.columns = ['id', 'name', 'active', 'agency', 'type', 'X', 'Y', 'area']
    df.loc[:, 'active'][df['active'] == 'פעילה'] = 1
    df.loc[:, 'active'][df['active'] == 'לא פעילה'] = 0
    df.loc[:, 'active'][df['active'] == 'לא פעילה זמנית'] = 0
    df['active'] = df['active'].astype(float)
    df = df[~df.X.isnull()]
    df = df[~df.Y.isnull()]
    # now, geopandas part:
    geo_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y),
                              crs={'init': 'epsg:2039'})
    # geo_df.crs = {'init': 'epsg:2039'}
    geo_df = geo_df.to_crs({'init': 'epsg:4326'})
    isr = gpd.read_file(gis_path / 'israel_demog2012.shp')
    isr.crs = {'init': 'epsg:4326'}
    geo_df = gpd.sjoin(geo_df, isr, op='within')
    if plot:
        ax = isr.plot()
        geo_df.plot(ax=ax, edgecolor='black', legend=True)
    return geo_df


def read_tides(path=hydro_path):
    from aux_gps import path_glob
    import pandas as pd
    files = path_glob(path, 'דוח_גאויות*.xlsx')
    for file in files:
        df = pd.read_excel(file, header=4)
        df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
        df.columns = [
            'id',
            'name',
            'hydro_year',
            'tide_start_hour',
            'tide_start_date',
            'tide_end_hour',
            'tide_end_date',
            'tide_duration',
            'tide_max_hour',
            'tide_max_date',
            'max_height',
            'max_flow[m^3/sec]',
            'tide_vol[MCM]']
        df = df[~df.hydro_year.isnull()]
        df['id'] = df['id'].astype(int)
        df['tide_start'] = pd.to_datetime(
            df['tide_start_date']) + pd.to_timedelta(df['tide_start_hour'].add(':00'), unit='m', errors='coerce')
        df['tide_end'] = pd.to_datetime(
            df['tide_end_date']) + pd.to_timedelta(df['tide_end_hour'].add(':00'), unit='m', errors='coerce')
        df['tide_max'] = pd.to_datetime(
            df['tide_max_date']) + pd.to_timedelta(df['tide_max_hour'].add(':00'), unit='m', errors='coerce')
        df['tide_duration'] = pd.to_timedelta(df['tide_duration'] + ':00', unit='m', errors='coerce')
        df.loc[:,['max_flow[m^3/sec]']][df['max_flow[m^3/sec]'].str.contains('<')] = 0
        df.loc[:,['tide_vol[MCM]']][df['tide_vol[MCM]'].str.contains('<')] = 0
        to_drop = [ 'tide_start_hour', 'tide_start_date', 'tide_end_hour',
                   'tide_end_date', 'tide_max_hour', 'tide_max_date']
        df = df.drop(to_drop, axis=1)
        return df