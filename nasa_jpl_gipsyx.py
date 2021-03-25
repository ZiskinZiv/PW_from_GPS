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
    harm1_keys = [x for x in harm1.attrs.keys() if '_1' in x]
    harm1.attrs = dict(zip(harm1_keys, [harm1.attrs[x] for x in harm1_keys]))
    harm2 = harm.sel(cpy=2).reset_coords(drop=True)
    harm2.name = da_ts.name + '_semiannual'
    harm2_keys = [x for x in harm2.attrs.keys() if '_2' in x]
    harm2.attrs = dict(zip(harm2_keys, [harm2.attrs[x] for x in harm2_keys]))
    resid = da_ts_detrended - harm1 - harm2
    resid.name = da_ts.name + '_residual'
    ds = xr.merge([da_ts, trend, harm1, harm2, resid])
    # load breakpoints:
    try:
        breakpoints = xr.open_dataset(
            jpl_path/'jpl_break_estimates.nc').sel(station=station.upper())[var]
        df = breakpoints.dropna('year')['year'].to_dataframe()
    # load seasonal coeffs:
        df['dt'] = df['year'].apply(decimal_year_to_datetime)
        df['dt'] = df['dt'].round('D')
        bp_da = df.set_index(df['dt'])['dt'].to_xarray()
        bp_da = bp_da.rename({'dt': 'time'})
        ds['{}_{}_breakpoints'.format(station, var)] = bp_da
        no_bp = False
    except KeyError:
        if verbose:
            print('no breakpoints found for {}!'.format(station))
            no_bp = True
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
        dst = ds[[x for x in ds if 'breakpoints' not in x]]
        axes = dst.to_dataframe().plot(subplots=True, figsize=(20, 20), color='k')
        [ax.grid() for ax in axes]
        [ax.set_ylabel('[mm]') for ax in axes]
        if not no_bp:
            for bp in df['dt']:
                [ax.axvline(bp, color='red') for ax in axes]
        plt.tight_layout()
        fig, ax = plt.subplots(figsize=(7, 7))
        harm_mm = harmonic_da_ts(da_ts_detrended.dropna('time'), n=2, grp='month',
                                 return_ts_fit=False, verbose=verbose)
        harm_mm['{}_{}_detrended'.format(station, var)].plot.line(ax=ax, linewidth=0, marker='o', color='k')
        harm_mm['{}_mean'.format(station)].sel(cpy=1).plot.line(ax=ax, marker=None, color='tab:red')
        harm_mm['{}_mean'.format(station)].sel(cpy=2).plot.line(ax=ax, marker=None, color='tab:blue')
        harm_mm['{}_mean'.format(station)].sum('cpy').plot.line(ax=ax, marker=None, color='tab:purple')
        ax.grid()
    return ds


def read_geodetic_positions_and_height(path=jpl_path):
    import pandas as pd
    import requests
    from io import StringIO
    from aux_gps import save_ncfile
    url = 'https://sideshow.jpl.nasa.gov/post/tables/table2.html'
    r = requests.get(url)
    data = r.text
    df = pd.read_csv(StringIO(data), delim_whitespace=True, skiprows=7)
    df.drop(df.tail(1).index, inplace=True)  # drop last n rows
    df = df.unstack()
    cols0 = df.columns.get_level_values(0)
    cols1 = df.columns.get_level_values(1)
    cols = ['{}_{}'.format(x, y) for x, y in zip(cols0, cols1)]
    df.columns = cols
    df.index.name = 'station'
    ds = df.to_xarray()
    pos_das = [x for x in ds if 'POS' in x]
    vel_das = [x for x in ds if 'VEL' in x]
    for da in pos_das:
        if 'V' in da.split('_')[0]:
            ds[da].attrs['units'] = 'mm'
        else:
            ds[da].attrs['units'] = 'deg'
    ds['SN_POS'].attrs['units'] = 'mm'
    ds['SE_POS'].attrs['units'] = 'mm'
    for da in vel_das:
        ds[da].attrs['units'] = 'mm/yr'
    ds.attrs['name'] = 'geodetic positions and height and velocities'
    ds.attrs['reference frame'] = 'IGS14'
    ds.attrs['reference epoch'] = '2020-01-01'
    ds.attrs['reference ellipsoid'] = 'GRS80'
    filename = 'jpl_geodetic_positions_velocities.nc'
    save_ncfile(ds, path, filename)
    return ds


def run_harmonic_analysis_on_all_jpl_products(path=jpl_path, savepath=jpl_path/'harmonic_analysis'):
    from aux_gps import save_ncfile
    dss = read_geodetic_positions_and_height(path=path)
    for i, station in enumerate(dss['station'].values):
        print('processing station {} ({} out of {})'.format(station, i+1, dss['station'].size))
        ds = produce_seasonal_trend_breakdown_time_series_from_jpl_gipsyx_site(station=station, verbose=False, plot=False)
        filename = '{}_V_harmonic_mm.nc'.format(station)
        save_ncfile(ds, savepath, filename)
    return


def build_jpl_station_geodataframe(path=jpl_path, plot=True):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import xarray as xr
    pos_ds = xr.load_dataset(path / 'jpl_geodetic_positions_velocities.nc')
    df = pos_ds[['N_POS', 'E_POS']].to_dataframe()
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df['E_POS'], df['N_POS']))

    # lat = np.arange(-90, 90.25, 0.25)
    # lon = np.arange(-180, 180, 0.25)
    # lat_da = xr.DataArray(lat, dims=['lat'])
    # lat_da['lat'] = lat
    # lon_da = xr.DataArray(lon, dims=['lon'])
    # lon_da['lon'] = lon
    # grid = np.zeros((lat.shape[0], lon.shape[0]), dtype=str)
    # cnt = 0
    # for station in pos_ds['station'].values:
    #     north = pos_ds['N_POS'].sel(station=station)
    #     lat_in_grid = lat_da.sel(lat=north, method='nearest').item()
    #     north_ind = np.where(lat==lat_in_grid)[0]
    #     east = pos_ds['E_POS'].sel(station=station)
    #     lon_in_grid = lon_da.sel(lon=east, method='nearest').item()
    #     east_ind = np.where(lon==lon_in_grid)[0]
    #     if grid[north_ind, east_ind] != '':
    #         print('grid point already taken')
    #         cnt += 1
    #     grid[north_ind, east_ind] = station
    # print('total taken points: {}'.format(cnt))
    if plot:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        base = world.plot(color='white', edgecolor='black', figsize=(15, 15))
        gdf.plot(ax=base, marker='o', color='red', markersize=5)
        plt.tight_layout()
    return gdf


def read_all_jpl_station_harmonic_analysis(path=jpl_path, harm_path=jpl_path/'harmonic_analysis'):
    from aux_gps import path_glob
    import xarray as xr
    import pandas as pd
    files = sorted(path_glob(harm_path, '*_V_harmonic_mm.nc'))
    dsl = [xr.open_dataset(x) for x in files]
    stations = [x.as_posix().split('/')[-1].split('_')[0] for x in files]
    annual_params = []
    semiannual_params = []
    annual_peak_doy = []
    semiannual_peak_doy = []
    for i, ds in enumerate(dsl):
        # print('processing {} station'.format(stations[i]))
        a_name = '{}_V_annual'.format(stations[i])
        sa_name = '{}_V_semiannual'.format(stations[i])
        annual_params.append([x[0] for x in ds[a_name].attrs.values()])
        semiannual_params.append([x[0] for x in ds[sa_name].attrs.values()])
        annual_peak_doy.append(ds[a_name].idxmax().dt.dayofyear)
        semiannual_peak_doy.append(ds[sa_name].idxmax().dt.dayofyear)
        continue
    df = pd.DataFrame(annual_params, index=stations)
    df.columns = ['A_Amp', 'A_offset', 'A_freq', 'A_x0']
    df['SA_Amp'] = [x[0] for x in semiannual_params]
    df['SA_offset'] = [x[1] for x in semiannual_params]
    df['SA_freq'] = [x[2] for x in semiannual_params]
    df['SA_x0'] = [x[3] for x in semiannual_params]
    df['A_peak_doy'] = [x.item() for x in annual_peak_doy]
    df['SA_peak_doy'] = [x.item() for x in semiannual_peak_doy]
    return df
