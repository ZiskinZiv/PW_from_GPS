#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:04:48 2020

@author: ziskin
"""


def add_horizontal_colorbar(fg_obj, rect=[0.1, 0.1, 0.8, 0.025], cbar_kwargs_dict=None):
    # rect = [left, bottom, width, height]
    # add option for just figure object, now, accepts facetgrid object only
    cbar_kws = {'label': '', 'format': '%0.2f'}
    if cbar_kwargs_dict is not None:
        cbar_kws.update(cbar_kwargs_dict)
    cbar_ax = fg_obj.fig.add_axes(rect)
    fg_obj.add_colorbar(cax=cbar_ax, orientation="horizontal", **cbar_kws)
    return fg_obj


def get_dt_from_single_ionex(ionex_str):
    import datetime
    import pandas as pd
    code = ionex_str[0:4]
    days = int(ionex_str[4:7])
    year = ionex_str[-3:-1]
    Year = datetime.datetime.strptime(year, '%y').strftime('%Y')
    dt = datetime.datetime(int(Year), 1, 1) + datetime.timedelta(days - 1)
    dt = pd.to_datetime(dt)
    return dt, code


def read_ionex_xr(file, plot='every_hour', extent=None):
    from getIONEX import read_tec
    import cartopy.crs as ccrs
    import xarray as xr
    import pandas as pd
    dt, code = get_dt_from_single_ionex(file.as_posix().split('/')[-1])
    tecarray, rmsarray, lonarray, latarray, timearray = read_tec(file)
    tec = xr.DataArray(tecarray, dims=['time', 'lat', 'lon'])
    tec_ds = tec.to_dataset(name='tec')
    tec_ds['tec_error'] = xr.DataArray(rmsarray, dims=['time', 'lat', 'lon'])
    tec_ds['lat'] = latarray
    tec_ds['lon'] = lonarray
    time = [pd.Timedelta(x, unit='H') for x in timearray]
    time = [dt + x for x in time]
    tec_ds['time'] = time
    tec_ds = tec_ds.sortby('lat')
    if plot is not None:
        if plot == 'every_hour':
            times = tec_ds['time'].values[::4][:-1]
            da = tec_ds['tec'].sel(time=times)
            if extent is not None:
                da = da.sel(lon=slice(extent[0], extent[1]), lat=slice(extent[0],extent[1]))
            proj = ccrs.PlateCarree()
            fg = da.plot.contourf(col='time', col_wrap=6,
                                  add_colorbar=False,
                                  cmap='viridis', extend=None, levels=41,
                                  subplot_kws={'projection': proj},
                                  transform=ccrs.PlateCarree(),
                                  figsize=(20, 30))
            fg = add_horizontal_colorbar(fg,
                                         cbar_kwargs_dict={'label': 'TECU'})
            fg.fig.subplots_adjust(top=0.965,
                                   bottom=0.116,
                                   left=0.006,
                                   right=0.985,
                                   hspace=0.0,
                                   wspace=0.046)
        elif isinstance(plot, list):
            time1 = dt + pd.Timedelta(plot[0], unit='H')
            time2 = dt + pd.Timedelta(plot[1], unit='H')
            da = tec_ds['tec'].sel(time=slice(time1, time2))
            if extent is not None:
                da = da.sel(lon=slice(extent[0], extent[1]), lat=slice(extent[0],extent[1]))
            proj = ccrs.PlateCarree()
            fg = da.plot.contourf(col='time', col_wrap=6,
                                  add_colorbar=False,
                                  cmap='viridis', extend=None, levels=41,
                                  subplot_kws={'projection': proj},
                                  transform=ccrs.PlateCarree(),
                                  figsize=(20, 30), robust=True)
            fg = add_horizontal_colorbar(fg,
                                         cbar_kwargs_dict={'label': 'TECU'})
            fg.fig.subplots_adjust(top=0.965,
                                   bottom=0.116,
                                   left=0.006,
                                   right=0.985,
                                   hspace=0.0,
                                   wspace=0.046)

        for i, ax in enumerate(fg.axes.flatten()):
            ax.coastlines(resolution='110m')
            gl = ax.gridlines(crs=ccrs.PlateCarree(),
                              linewidth=1,
                              color='black',
                              alpha=0.5,
                              linestyle='--',
                              draw_labels=False)
            # Without this aspect attributes the maps will look chaotic and the
            # "extent" attribute above will be ignored
    return tec_ds


    
def read_ionex_file(file):
    import pandas as pd
    df_sat = pd.read_csv(file, skiprows=48, nrows=32,
                         header=None, delim_whitespace=True)
    df_stations = pd.read_fwf(file, skiprows=80, nrows=50, header=None,
                              widths=[20, 10, 10, 10])
    return df_sat, df_stations

def read_one_sinex(file):
    import pandas as pd
    df = pd.read_fwf(file, skiprows=57)
    df.drop(df.tail(2).index, inplace=True) # drop last n rows
    df.columns = ['bias', 'svn', 'prn', 'station', 'obs1', 'obs2',
                  'bias_start', 'bias_end', 'unit', 'value', 'std']
    ds = xr.Dataset()
    ds.attrs['bias'] = df['bias'].values[0]
    df_sat = df[df['station'].isnull()]
    df_station = df[~df['station'].isnull()]
    return ds


def read_sinex(path, glob='*.BSX'):
    from aux_gps import path_glob
    import xarray as xr
    files = path_glob(path, glob_str=glob)
    for file in files:
        ds = read_one_sinex(file)
        ds_list.append(ds)
    dss = xr.concat(ds, 'time')
    return dss