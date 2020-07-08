#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:11:38 2020

@author: shlomi
"""

from pathlib import Path
from PW_paths import work_yuval
cwd = Path().cwd()
gis_path = work_yuval /'gis'
rc = {
    'font.family': 'serif',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium'}
# TODO: Build mask = (e.index > st) & (e.index <= ed) for dates, choose dates:
# st=df.index[0]-np.timedelta64(1,'M') month each date or half month


def filter_distance_of_earthquake_events(edf, sta_pos, tol_distance=50, plot=False):
    """given a station position (lat, lon), search the erathquake database for
    data with the tolarnace distance of tol_distance km around the sta_pos"""
    from shapely.geometry import Point
    from pyproj import Proj
    import geopandas as gpd
    import contextily as ctx
    import matplotlib.ticker as ticker
    from PW_from_gps_figures import lon_formatter
    from PW_from_gps_figures import lat_formatter
    # use Israel new network in meters for distance calculation:
    isr_proj = Proj(init='EPSG:2039')
    # convert the lat/lon point in sta_pos to shapely point with projection:
    p = Point(sta_pos[0], sta_pos[1])
    p_proj_point = Point(isr_proj(p.y, p.x))
    # create a geodataframe init in WGS84:
    gdf = gpd.GeoDataFrame(edf, geometry=gpd.points_from_xy(edf.lon, edf.lat),
                           crs={'init': 'epsg:4326'})
    # convert to Israel new network:
    gdf = gdf.to_crs({'init': 'epsg:2039'})
    # calculate distance to station position in km:
    gdf['distance'] = gdf.geometry.distance(p_proj_point) / 1000.0 # in km
    # filter close earthquake sources:
    gdf = gdf[gdf['distance'] <= tol_distance]
    if gdf.empty:
        raise KeyError('No Earthquakes in the range of {} found'.format(tol_distance))
    # convert back to WGS84:
    gdf = gdf.to_crs({'init': 'epsg:4326'})
    if plot:
        ax = gdf.plot(column='Md', legend=True)
        d = {'geometry': [Point(sta_pos[1], sta_pos[0])]}
        station = gpd.GeoDataFrame(d, crs="EPSG:4326")
        station.plot(ax=ax, marker='s', color='k')
        ctx.add_basemap(ax,
                        url=ctx.sources.ST_TERRAIN_BACKGROUND,
                        crs='epsg:4326')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.tick_params(top=True, bottom=True, left=True, right=True,
                       direction='out', labelsize=10)
    return gdf


def filter_times_of_earthquake_events(edf, bp_dt, tol=15):
    """given a breakpoint datetime, search the erathquake database for data
    with the tolarnace of tol days around the bp_dt"""
    import numpy as np
    start = bp_dt - np.timedelta64(tol, 'D')
    end = bp_dt + np.timedelta64(tol, 'D')
    mask = (edf.index >= start) & (edf.index <= end)
    df = edf[mask]
    return df


def read_GNSS_station_position_velocity(path=cwd, station='ALON',
                                        return_pos=False):
    import pandas as pd
    df = pd.read_fwf(path / 'GNSS_pos_vel_all.txt')
    cols = [x for x in df.columns]
    cols[0] = 'station'
    cols[1] = 'pos_vel'
    df.columns = cols
    df = df[df['station'] == station]
    if return_pos:
        pos = df[df['pos_vel'] == 'POS'][['N', 'E']].values.squeeze()
        return pos
    else:
        return df


def read_GNSS_station_breakpoints(path=cwd, station='ALON'):
    from aux_gps import decimal_year_to_datetime
    import pandas as pd
    df = pd.read_fwf(path / 'GNSS_breakpoints_all.txt')
    cols = [x for x in df.columns]
    cols[0] = 'station'
    cols[1] = 'time'
    df.columns = cols
    df = df[df['station'] == station]
    df['time'] = df['time'].apply(decimal_year_to_datetime)
    df = df.set_index('time')
    return df


def read_earthquakes_IL(path=cwd):
    import pandas as pd
    df = pd.read_csv(cwd / 'earthquake_IL_1996-2020.csv')
    df = df.set_index('DateTime')
    df.index.name = 'time'
    df.index = pd.to_datetime(df.index)
    df = df.drop('epiid', axis=1)
    cols = [x for x in df.columns]
    cols = ['lat' if x == 'Lat' else x for x in cols]
    cols = ['lon' if x == 'Long' else x for x in cols]
    df.columns = cols
    df = df.sort_index()
    return df
