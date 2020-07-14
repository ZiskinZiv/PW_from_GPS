#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:42:55 2020

@author: ziskin
"""

from PW_paths import work_yuval
from matplotlib import rcParams
import seaborn as sns
aero_path = work_yuval / 'AERONET'
gis_path = work_yuval / 'gis'

rc = {
    'font.family': 'serif',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium'}
for key, val in rc.items():
    rcParams[key] = val
sns.set(rc=rc, style='white')


def prepare_station_to_pw_comparison(path=aero_path, gis_path=gis_path,
                                     station='boker', mm_anoms=False):
    from aux_gps import keep_iqr
    from aux_gps import anomalize_xr
    ds_dict = load_all_station(path=aero_path, gis_path=gis_path)
    try:
        da = ds_dict[station]['WV(cm)_935nm-AOD']
    except KeyError as e:
        print('station {} has no {} field'.format(station, e))
        return
    da = keep_iqr(da)
    # convert to mm:
    da = da * 10
    da.name = station
    if mm_anoms:
        da_mm = da.resample(time='MS').mean()
        da_mm_anoms = anomalize_xr(da_mm, freq='MS')
        da = da_mm_anoms
    da.attrs['data_source'] = 'AERONET'
    da.attrs['data_field'] = 'WV(cm)_935nm-AOD'
    da.attrs['units'] = 'mm'
    return da

    
def load_all_station(path=aero_path, gis_path=gis_path):
    from aux_gps import path_glob
    import xarray as xr
    files = path_glob(aero_path, '*.nc')
    dsl = [xr.open_dataset(file) for file in files]
    names = [x.attrs['new_name'] for x in dsl]
    ds_dict = dict(zip(names, dsl))
    return ds_dict


def read_all_stations(path=aero_path, savepath=aero_path, glob='*.lev20'):
    from aux_gps import path_glob
    files = path_glob(path, glob)
    for file in files:
        print('reading {}...'.format(file.as_posix().split('/')[-1]))
        read_one_station(file, savepath)
    print('Done!')
    return


def read_one_station(filepath, savepath=None):
    import pandas as pd
    kind = filepath.as_posix().split('.')[-1]
    df = pd.read_csv(filepath, header=6, na_values=-999)
    # create datetime index:
    df['dt'] = df['Date(dd:mm:yyyy)'].astype(str) + ' ' + \
        df['Time(hh:mm:ss)'].astype(str)
    df['dt'] = pd.to_datetime(df['dt'], format='%d:%m:%Y %H:%M:%S')
    df = df.set_index(df['dt'])
    # exctract meta data:
    meta = {}
    meta['AERONET_Site_Name'] = df['AERONET_Site_Name'][0]
    meta['lat'] = df['Site_Latitude(Degrees)'][0]
    meta['lon'] = df['Site_Longitude(Degrees)'][0]
    meta['alt'] = df['Site_Elevation(m)'][0]
    # cols to keep:
    to_keep = [x for x in df.columns if 'AOD' in x]
    to_keep += [x for x in df.columns if 'Angstrom' in x]
    more_fields = ['Precipitable_Water(cm)', 'Solar_Zenith_Angle(Degrees)',
                   'Optical_Air_Mass', 'Sensor_Temperature(Degrees_C)',
                   'Ozone(Dobson)', 'NO2(Dobson)', 'Pressure(hPa)']
    for field in more_fields:
        to_keep += [x for x in df.columns if field == x]
    to_keep = [x for x in to_keep if 'Empty' not in x]
    to_keep = [x for x in to_keep if 'Exact' not in x]
    df = df[to_keep].astype(float)
    df.index.name = 'time'
    ds = df.to_xarray()
    for key, val in meta.items():
        ds.attrs[key] = val
    new_names = {'Dead_Sea': 'dsea', 'Eilat': 'elat', 'Migal': 'migal',
                 'KITcube_Masada': 'masada', 'Weizmann_Institute': 'rehovot',
                 'Technion_Haifa_IL': 'haifa', 'Nes_Ziona': 'ziona',
                 'SEDE_BOKER': 'boker'}
    ds.attrs['new_name'] = new_names.get(ds.attrs['AERONET_Site_Name'])
    if savepath is not None:
        max_year = ds['time'].max().dt.year.item()
        min_year = ds['time'].min().dt.year.item()
        filename = 'AERONET_{}_{}_{}-{}.nc'.format(
                meta['AERONET_Site_Name'], kind, min_year, max_year)
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(savepath / filename, 'w', encoding=encoding)
        print('saved {} to {}.'.format(filename, savepath))
    return ds


def produce_geo_aeronet(path=aero_path, gis_path=gis_path):
    from aux_gps import path_glob
    import pandas as pd
    import geopandas as gpd
    import xarray as xr
    files = path_glob(aero_path, '*.nc')
    dsl = [xr.open_dataset(file) for file in files]
    attrs = [x.attrs for x in dsl]
    df = pd.DataFrame(attrs)
    df.columns = ['old_name', 'lat', 'lon', 'alt', 'name']
    df = df.set_index(df['name'])
    df = df[['lat', 'lon', 'alt']]
    isr = gpd.read_file(gis_path / 'Israel_and_Yosh.shp')
    isr.crs = {'init': 'epsg:4326'}
    aeronet = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon,
                                                               df.lat),
                               crs=isr.crs)
    return aeronet


def plot_stations_and_gps(path=aero_path, gis_path=gis_path):
    from PW_from_gps_figures import plot_israel_map
    from PW_stations import produce_geo_gnss_solved_stations
    from aux_gps import geo_annotate
    ax = plot_israel_map(gis_path=gis_path)
    aero = produce_geo_aeronet(path=path, gis_path=gis_path)
    aero.plot(ax=ax, color='blue', edgecolor='black', marker='^')
    geo_annotate(ax, aero.lon, aero.lat,
                 aero.index, xytext=(4, -6), fmt=None,
                 c='k', fw='bold', fs=None, colorupdown=False)
    gps = produce_geo_gnss_solved_stations(path=gis_path, plot=False)
    gps.plot(ax=ax, color='green', edgecolor='black', marker='s')
    gps_stations = [x for x in gps.index]
    to_plot_offset = ['mrav', 'klhv']
    [gps_stations.remove(x) for x in to_plot_offset]
    gps_normal_anno = gps.loc[gps_stations, :]
    gps_offset_anno = gps.loc[to_plot_offset, :]
    geo_annotate(ax, gps_normal_anno.lon, gps_normal_anno.lat,
                 gps_normal_anno.index, xytext=(3, 3), fmt=None,
                 c='k', fw='bold', fs=None, colorupdown=False)
    geo_annotate(ax, gps_offset_anno.lon, gps_offset_anno.lat,
                 gps_offset_anno.index, xytext=(4, -6), fmt=None,
                 c='k', fw='bold', fs=None, colorupdown=False)
    return

