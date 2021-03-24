#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:51:28 2021

@author: ziskin
"""
from PW_paths import work_yuval
jpl_path = work_yuval/'jpl_products'


def read_seasonal_estimates_jpl_gipsyx_site(path=jpl_path):
    import pandas as pd
    import requests
    from io import StringIO
    from aux_gps import save_ncfile
    url = 'https://sideshow.jpl.nasa.gov/post/tables/table4.html'
    r = requests.get(url)
    data = r.text
    df = pd.read_csv(StringIO(data), delim_whitespace=True, skiprows=5)
    df.drop(df.tail(1).index, inplace=True)  # drop last n rows
    ds = df.to_xarray()
    ds = ds.rename({'level_0': 'station', 'level_1': 'seas_coef'})
    for da in ds:
        ds[da].attrs['units'] = 'mm'
    ds.attrs['units'] = 'mm'
    ds.attrs['annual equation'] = 'AC1*cos(t*2*Pi) + AS1*sin(t*2*Pi)'
    ds.attrs['semi-annual equation'] = 'AC2*cos(t*4*Pi) + AS2*sin(t*4*Pi)'
    ds['N'].attrs['long_name'] = 'north'
    ds['E'].attrs['long_name'] = 'east'
    ds['V'].attrs['long_name'] = 'vertical'
    ds['SN'].attrs['long_name'] = 'north error'
    ds['SE'].attrs['long_name'] = 'east error'
    ds['SV'].attrs['long_name'] = 'vertical error'
    ds.attrs['name'] = 'seasonal estimates'
    filename = 'jpl_seasonal_estimates.nc'
    save_ncfile(ds, path, filename)
    return ds


def read_break_estimates_jpl_gipsyx_site(path=jpl_path):
    import pandas as pd
    import requests
    from io import StringIO
    from aux_gps import save_ncfile
    url = 'https://sideshow.jpl.nasa.gov/post/tables/table3.html'
    r = requests.get(url)
    data = r.text
    df = pd.read_csv(StringIO(data), delim_whitespace=True, skiprows=4)
    df.drop(df.tail(1).index, inplace=True)  # drop last n rows
    ds = df.to_xarray()
    ds = ds.rename({'level_0': 'station', 'level_1': 'year'})
    for da in ds:
        ds[da].attrs['units'] = 'mm'
    ds.attrs['units'] = 'mm'
    ds['N'].attrs['long_name'] = 'north'
    ds['E'].attrs['long_name'] = 'east'
    ds['V'].attrs['long_name'] = 'vertical'
    ds['SN'].attrs['long_name'] = 'north error'
    ds['SE'].attrs['long_name'] = 'east error'
    ds['SV'].attrs['long_name'] = 'vertical error'
    ds.attrs['name'] = 'break estimates'
    filename = 'jpl_break_estimates.nc'
    save_ncfile(ds, path, filename)
    return ds


def read_time_series_jpl_gipsyx_site(station='bshm',
                                     path=jpl_path/'time_series',
                                     verbose=True):
    import pandas as pd
    # from aux_gps import decimal_year_to_datetime
    station = station.upper()
    file = path / '{}.series'.format(station)
    if verbose:
        print('reading {} time series.'.format(file))
    df = pd.read_csv(file, delim_whitespace=True, header=None)
    df.columns = ['decimal_year', 'E', 'N', 'V', 'SE', 'SN', 'SV', 'EN_cor',
                  'EV_cor', 'NV_cor', 'seconds', 'year', 'month', 'day', 'hour',
                  'min', 'sec']
    # convert to mm:
    df[['E', 'N', 'V', 'SE', 'SN', 'SV']] *= 1000
    # df['datetime_from_decimal'] = df['year_decimal'].apply(
    #     decimal_year_to_datetime).round('D')
    dts = []
    for ind, row in df.astype(int).iterrows():
        dt = '{}-{}-{}T{}:{}:{}'.format(row['year'], row['month'],
                                        row['day'], row['hour'], row['min'],
                                        row['sec'])
        dt = pd.to_datetime(dt)
        dts.append(dt.round('D'))
    df['time'] = dts
    df = df.drop(['year', 'month', 'day', 'hour', 'min', 'sec'],
                 axis=1)
    df = df.set_index('time')
    ds = df.to_xarray()
    for da in ds[['N', 'E', 'V', 'SE', 'SN', 'SV']]:
        ds[da].attrs['units'] = 'mm'
    return ds


def produce_seasonal_trend_breakdown_time_series_from_jpl_gipsyx_site(station='bshm',
                                                                      path=jpl_path,
                                                                      var='V', k=2,
                                                                      verbose=True,
                                                                      plot=True):
    import xarray as xr
    from aux_gps import harmonic_da_ts
    from aux_gps import loess_curve
    from aux_gps import keep_iqr
    from aux_gps import get_unique_index
    from aux_gps import xr_reindex_with_date_range
    from aux_gps import decimal_year_to_datetime
    import matplotlib.pyplot as plt
    if verbose:
        print('producing seasonal time series for {} station {}'.format(station, var))
    ds = read_time_series_jpl_gipsyx_site(station=station,
                                          path=path/'time_series', verbose=verbose)
    # dyear = ds['decimal_year']
    da_ts = ds[var]
    da_ts = xr_reindex_with_date_range(get_unique_index(da_ts), freq='D')
    xr.infer_freq(da_ts['time'])
    if k is not None:
        da_ts = keep_iqr(da_ts, k=k)
    da_ts.name = '{}_{}'.format(station, var)
    # detrend:
    trend = loess_curve(da_ts, plot=False)['mean']
    trend.name = da_ts.name + '_trend'
    trend = xr_reindex_with_date_range(trend, freq='D')
    da_ts_detrended = da_ts - trend
    if verbose:
        print('detrended by loess.')
    da_ts_detrended.name = da_ts.name + '_detrended'
    # harmonic cpy fits:
    harm = harmonic_da_ts(da_ts_detrended.dropna('time'), n=2, grp='month',
                          return_ts_fit=True, verbose=verbose)
    harm = xr_reindex_with_date_range(harm, time_dim='time', freq='D')
    harm1 = harm.sel(cpy=1).reset_coords(drop=True)
    harm1.name = da_ts.name + '_annual'
    harm2 = harm.sel(cpy=2).reset_coords(drop=True)
    harm2.name = da_ts.name + '_semiannual'
    resid = da_ts_detrended - harm1 - harm2
    resid.name = da_ts.name + '_residual'
    ds = xr.merge([da_ts, trend, harm1, harm2, resid])
    # load breakpoints:
    breakpoints = xr.open_dataset(
        jpl_path/'jpl_break_estimates.nc').sel(station=station.upper())[var]
    df = breakpoints.dropna('year')['year'].to_dataframe()
    # load seasonal coeffs:
    df['dt'] = df['year'].apply(decimal_year_to_datetime)
    df['dt'] = df['dt'].round('D')
    # seas = xr.load_dataset(
    #     jpl_path/'jpl_seasonal_estimates.nc').sel(station=station.upper())
    # ac1, as1, ac2, as2 = seas[var].values
    # # build seasonal time series:
    # annual = xr.DataArray(ac1*np.cos(dyear*2*np.pi)+as1 *
    #                       np.sin(dyear*2*np.pi), dims=['time'])
    # annual['time'] = da_ts['time']
    # annual.name = '{}_{}_annual'.format(station, var)
    # annual.attrs['units'] = 'mm'
    # annual.attrs['long_name'] = 'annual mode'
    # semiannual = xr.DataArray(ac2*np.cos(dyear*4*np.pi)+as2 *
    #                           np.sin(dyear*4*np.pi), dims=['time'])
    # semiannual['time'] = da_ts['time']
    # semiannual.name = '{}_{}_semiannual'.format(station, var)
    # semiannual.attrs['units'] = 'mm'
    # semiannual.attrs['long_name'] = 'semiannual mode'
    # ds = xr.merge([annual, semiannual, da_ts])
    if plot:
        # plt.figure(figsize=(20, 20))
        axes = ds.to_dataframe().plot(subplots=True, figsize=(20, 20), color='k')
        [ax.grid() for ax in axes]
        [ax.set_ylabel('[mm]') for ax in axes]
        for bp in df['dt']:
            [ax.axvline(bp, color='red') for ax in axes]
        plt.tight_layout()
    return ds
