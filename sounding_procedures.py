#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:24:40 2019

@author: ziskin
"""
# from pathlib import Path
from PW_paths import work_yuval
sound_path = work_yuval / 'sounding'
era5_path = work_yuval / 'ERA5'
edt_path = sound_path / 'edt'
ceil_path = work_yuval / 'ceilometers'
des_path = work_yuval / 'deserve'


def load_field_from_radiosonde(
        path=sound_path, field='Tm', data_type='phys', reduce='min',
        dim='time', plot=True):
    """data_type: phys for 2008-2013, 10 sec sample rate,
    PTU_Wind for 2014-2016 2 sec sample rate,
    edt for 2018-2019 1 sec sample rate with gps"""
    from aux_gps import plot_tmseries_xarray
    from aux_gps import path_glob
    import xarray as xr

    def reduce_da(da):
        if reduce is not None:
            if reduce == 'min':
                da = da.min(dim)
            elif reduce == 'max':
                da = da.max(dim)
        da = da.reset_coords(drop=True)
        return da

    if data_type is not None:
        file = path_glob(
            path, 'bet_dagan_{}_sounding_*.nc'.format(data_type))[-1]
        da = xr.open_dataset(file)[field]
        da = da.sortby('sound_time')
        da = reduce_da(da)
    else:
        files = path_glob(path, 'bet_dagan_*_sounding_*.nc')
        assert len(files) == 3
        ds = [xr.open_dataset(x)[field] for x in files]
        da = xr.concat(ds, 'sound_time')
        da = da.sortby('sound_time')
        da = reduce_da(da)
    if plot:
        plot_tmseries_xarray(da)
    return da


def get_field_from_radiosonde(path=sound_path, field='Tm', data_type='phys',
                              reduce='min', dim='time',
                              times=['2007', '2019'], plot=True):
    """
    old version, to be replaced with load_field_from_radiosonde,
    but still useful for ZWD

    Parameters
    ----------
    path : TYPE, optional
        DESCRIPTION. The default is sound_path.
    field : TYPE, optional
        DESCRIPTION. The default is 'Tm'.
    data_type : TYPE, optional
        DESCRIPTION. The default is 'phys'.
    reduce : TYPE, optional
        DESCRIPTION. The default is 'min'.
    dim : TYPE, optional
        DESCRIPTION. The default is 'time'.
    times : TYPE, optional
        DESCRIPTION. The default is ['2007', '2019'].
    plot : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    da : TYPE
        DESCRIPTION.

    """

    import xarray as xr
    from aux_gps import get_unique_index
    from aux_gps import keep_iqr
    from aux_gps import plot_tmseries_xarray
    from aux_gps import path_glob
    file = path_glob(path, 'bet_dagan_{}_sounding_*.nc'.format(data_type))[0]
    file = path / 'bet_dagan_phys_PW_Tm_Ts_2007-2019.nc'
    ds = xr.open_dataset(file)
    if field is not None:
        da = ds[field]
        if reduce is not None:
            if reduce == 'min':
                da = da.min(dim)
            elif reduce == 'max':
                da = da.max(dim)
            da = da.reset_coords(drop=True)
        da = get_unique_index(da, dim='sound_time')
        da = keep_iqr(da, k=2.0, dim='sound_time', drop_with_freq='12H')
    da = da.sel(sound_time=slice(*times))
    if plot:
        plot_tmseries_xarray(da)
    return da


def calculate_edt_north_east_distance(lat_da, lon_da, method='fast'):
    """fast mode is 11 times faster than slow mode, however fast distance is
    larger than slow...solve this mystery"""
    from shapely.geometry import Point
    from pyproj import Transformer
    import geopandas as gpd
    import pandas as pd
    import numpy as np

    def change_sign(x, y, value):
        if x <= y:
            return -value
        else:
            return value

    if method == 'fast':
        # prepare bet dagan coords:
        bd_lat = 32.01
        bd_lon = 34.81
        fixed_lat = np.ones(lat_da.shape) * bd_lat
        fixed_lon = np.ones(lon_da.shape) * bd_lon
        # define projections:
#        wgs84 = pyproj.CRS('EPSG:4326')
#        isr_tm = pyproj.CRS('EPSG:2039')
        # creare transfrom from wgs84 (lat, lon) to new israel network (meters):
    #    transformer = Transformer.from_crs(wgs84, isr_tm, always_xy=True)
        transformer = Transformer.from_proj(4326, 2039, always_xy=True)
        bd_meters = transformer.transform(bd_lat, bd_lon)
        bd_point_meters = Point(bd_meters[0], bd_meters[1])
    #    # create Points from lat_da, lon_da in wgs84:
    #    dyn_lat = [Point(x, bd_lon) for x in lat_da.values[::2]]
    #    dyn_lon = [Point(bd_lat, x) for x in lon_da.values[::2]]
        # transform to meters:
        dyn_lat_meters = transformer.transform(lat_da.values, fixed_lon)
        dyn_lon_meters = transformer.transform(fixed_lat, lon_da.values)
        # calculate distance in km:
        north_distance = [Point(dyn_lat_meters[0][x],dyn_lat_meters[1][x]).distance(bd_point_meters) / 1000 for x in range(lat_da.size)]
        east_distance = [Point(dyn_lon_meters[0][x],dyn_lon_meters[1][x]).distance(bd_point_meters) / 1000 for x in range(lon_da.size)]
        # sign change:
        new_north_distance = [change_sign(lat_da.values[x], bd_lat, north_distance[x]) for x in range(lat_da.size)]
        new_east_distance = [change_sign(lon_da.values[x], bd_lon, east_distance[x]) for x in range(lon_da.size)]
        north = lat_da.copy(data=new_north_distance)
        north.attrs['units'] = 'km'
        north.attrs['long_name'] = 'distance north'
        east = lon_da.copy(data=new_east_distance)
        east.attrs['long_name'] = 'distance east'
        east.attrs['units'] = 'km'
        return north, east
    elif method == 'slow':
        bet_dagan = pd.DataFrame(index=[0])
        bet_dagan['x'] = 34.81
        bet_dagan['y'] = 32.01
        bet_dagan_gdf = gpd.GeoDataFrame(
            bet_dagan, geometry=gpd.points_from_xy(
                bet_dagan['x'], bet_dagan['y']))
        bet_dagan_gdf.crs = {'init': 'epsg:4326'}
        # transform to israeli meters coords:
        bet_dagan_gdf.to_crs(epsg=2039, inplace=True)
        bd_as_point = bet_dagan_gdf.geometry[0]
        bd_lon = bet_dagan.loc[0, 'x']
        bd_lat = bet_dagan.loc[0, 'y']
        df = lat_da.reset_coords(drop=True).to_dataframe(name='lat')
        df['lon'] = lon_da.reset_coords(drop=True).to_dataframe()
    #    df = ds.reset_coords(drop=True).to_dataframe()
        df['fixed_lon'] = 34.81 * np.ones(df['lon'].shape)
        df['fixed_lat'] = 32.01 * np.ones(df['lat'].shape)
        gdf_fixed_lon = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['fixed_lon'],
                                                                         df.lat))
        gdf_fixed_lon.crs = {'init': 'epsg:4326'}
        gdf_fixed_lon.dropna(inplace=True)
        gdf_fixed_lon.to_crs(epsg=2039, inplace=True)
        gdf_fixed_lat = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon,
                                                                         df['fixed_lat']))
        gdf_fixed_lat.crs = {'init': 'epsg:4326'}
        gdf_fixed_lat.dropna(inplace=True)
        gdf_fixed_lat.to_crs(epsg=2039, inplace=True)
        # calculate distance north from bet dagan coords in km:
        df['north_distance'] = gdf_fixed_lon.geometry.distance(
            bd_as_point) / 1000.0
        # calculate distance east from bet dagan coords in km:
        df['east_distance'] = gdf_fixed_lat.geometry.distance(
            bd_as_point) / 1000.0
        # fix sign to indicate: negtive = south:
        df['north_distance'] = df.apply(
            lambda x: change_sign(
                x.lat, bd_lat, x.north_distance), axis=1)
        # fix sign to indicate: negtive = east:
        df['east_distance'] = df.apply(
            lambda x: change_sign(
                x.lon, bd_lon, x.east_distance), axis=1)
        return df['north_distance'].to_xarray(), df['east_distance'].to_xarray()


#def produce_radiosonde_edt_north_east_distance(path=sound_path, savepath=None,
#                                               verbose=True):
#    from aux_gps import path_glob
#    import geopandas as gpd
#    import pandas as pd
#    import xarray as xr
#    import numpy as np
#
#    def change_sign(x, y, value):
#        if x <= y:
#            return -value
#        else:
#            return value
#    file = path_glob(path, 'bet_dagan_edt_sounding_*.nc')
#    ds = xr.load_dataset(file[0])
#    ds_geo = ds[['lat', 'lon']]
#    # prepare bet dagan coords:
#    bet_dagan = pd.DataFrame(index=[0])
#    bet_dagan['x'] = 34.81
#    bet_dagan['y'] = 32.01
#    bet_dagan_gdf = gpd.GeoDataFrame(
#        bet_dagan, geometry=gpd.points_from_xy(
#            bet_dagan['x'], bet_dagan['y']))
#    bet_dagan_gdf.crs = {'init': 'epsg:4326'}
#    # transform to israeli meters coords:
#    bet_dagan_gdf.to_crs(epsg=2039, inplace=True)
#    bd_as_point = bet_dagan_gdf.geometry[0]
#    bd_lon = bet_dagan.loc[0, 'x']
#    bd_lat = bet_dagan.loc[0, 'y']
#    ds_list = []
#    for i in range(ds['sound_time'].size):
#        record = ds['sound_time'].isel({'sound_time': i})
#        record = record.dt.strftime('%Y-%m-%d %H:%M').values.item()
#        if verbose:
#            print('processing {}.'.format(record))
#        sounding = ds_geo.isel({'sound_time': i})
#        df = sounding.reset_coords(drop=True).to_dataframe()
#        df['fixed_lon'] = 34.81 * np.ones(df['lon'].shape)
#        df['fixed_lat'] = 32.01 * np.ones(df['lat'].shape)
#        gdf_fixed_lon = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['fixed_lon'],
#                                                                         df.lat))
#        gdf_fixed_lon.crs = {'init': 'epsg:4326'}
#        gdf_fixed_lon.dropna(inplace=True)
#        gdf_fixed_lon.to_crs(epsg=2039, inplace=True)
#        gdf_fixed_lat = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon,
#                                                                         df['fixed_lat']))
#        gdf_fixed_lat.crs = {'init': 'epsg:4326'}
#        gdf_fixed_lat.dropna(inplace=True)
#        gdf_fixed_lat.to_crs(epsg=2039, inplace=True)
#        # calculate distance north from bet dagan coords in km:
#        df['north_distance'] = gdf_fixed_lon.geometry.distance(
#            bd_as_point) / 1000.0
#        # calculate distance east from bet dagan coords in km:
#        df['east_distance'] = gdf_fixed_lat.geometry.distance(
#            bd_as_point) / 1000.0
#        # fix sign to indicate: negtive = south:
#        df['north_distance'] = df.apply(
#            lambda x: change_sign(
#                x.lat, bd_lat, x.north_distance), axis=1)
#        # fix sign to indicate: negtive = east:
#        df['east_distance'] = df.apply(
#            lambda x: change_sign(
#                x.lon, bd_lon, x.east_distance), axis=1)
#        # convert to xarray:
#        ds_list.append(df[['east_distance', 'north_distance']].to_xarray())
#    ds_distance = xr.concat(ds_list, 'sound_time')
#    ds_distance['sound_time'] = ds['sound_time']
#    ds_distance.to_netcdf(savepath / 'bet_dagan_edt_distance.nc', 'w')
#    return ds_distance


def analyse_radiosonde_climatology(path=sound_path, data_type='phys',
                                   field='Rho_wv', month=3, season=None,
                                   times=None, hour=None):
    import xarray as xr
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, LogLocator, NullFormatter
    from aux_gps import path_glob
    file = path_glob(path, 'bet_dagan_{}_sounding_*.nc'.format(data_type))
    da = xr.load_dataset(file[0])[field]
    if times is not None:
        da = da.sel(sound_time=slice(*times))
    if hour is not None:
        da = da.sel(sound_time=da['sound_time.hour'] == hour)
    try:
        name = da.attrs['long_name']
    except KeyError:
        name = field
    units = da.attrs['units']
    if season is not None:
        clim = da.groupby('sound_time.season').mean('sound_time')
        df = clim.to_dataset('season').to_dataframe()
        df_copy = df.copy()
        seasons = [x for x in df.columns if season not in x]
        df.reset_index(inplace=True)
        # df.loc[:, season] -= df.loc[:, season]
        df.loc[:, seasons[0]] -= df.loc[:, season]
        df.loc[:, seasons[1]] -= df.loc[:, season]
        df.loc[:, seasons[2]] -= df.loc[:, season]
        fig, ax = plt.subplots(figsize=(12, 5))
        df.plot(x=seasons[0], y='Height', logy=True, color='r', ax=ax)
        df.plot(x=seasons[1], y='Height', logy=True, ax=ax, color='b')
        df.plot(x=seasons[2], y='Height', logy=True, ax=ax, color='g')
        ax.axvline(x=0, color='k')
        # ax.set_xlim(-1, 15)
        ax.legend(seasons+[season], loc='best')
    else:
        clim = da.groupby('sound_time.month').mean('sound_time')
        if month == 12:
            months = [11, 12, 1]
        elif month == 1:
            months = [12, 1, 2]
        else:
            months = [month - 1, month, month + 1]
        df_copy = clim.to_dataset('month').to_dataframe()
        df = clim.sel(month=months).to_dataset('month').to_dataframe()
        month_names = pd.to_datetime(months, format='%m').month_name()
        df.reset_index(inplace=True)
        df.loc[:, months[0]] -= df.loc[:, month]
        df.loc[:, months[2]] -= df.loc[:, month]
        df.loc[:, months[1]] -= df.loc[:, month]
        ax = df.plot(x=months[0], y='Height', logy=True, color='r')
        df.plot(x=months[1], y='Height', logy=True, ax=ax, color='k')
        df.plot(x=months[2], y='Height', logy=True, ax=ax, color='b')
        ax.legend(month_names)
    ax.set_xlabel('{} [{}]'.format(name, units))
    ax.set_ylim(100, 10000)
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    locmaj = LogLocator(base=10,numticks=12)
    ax.yaxis.set_major_locator(locmaj)
    locmin = LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel('height [m]')
    if hour is not None:
        ax.set_title('hour = {}'.format(hour))
    return df_copy


def process_new_field_from_radiosonde_data(phys_ds, dim='sound_time',
                                           field_name='pw', bottom=None,
                                           top=None, verbose=False):
    import xarray as xr
    from aux_gps import keep_iqr
    field_list = []
    for i in range(phys_ds[dim].size):
        record = phys_ds[dim].isel({dim: i})
        if 'time' in dim:
            record = record.dt.strftime('%Y-%m-%d %H:%M').values.item()
        if verbose:
            print('processing {} for {} field.'.format(record, field_name))
        if field_name == 'pw':
            long_name = 'Precipatiable water'
            Dewpt = phys_ds['Dewpt'].isel({dim: i})
            P = phys_ds['P'].isel({dim: i})
            try:
                field, unit = wrap_xr_metpy_pw(Dewpt, P, bottom=bottom, top=top)
            except ValueError:
                field, unit = wrap_xr_metpy_pw(Dewpt, P, bottom=None, top=None)
        elif field_name == 'tm':
            long_name = 'Water vapor mean air temperature'
            P = phys_ds['P'].isel({dim: i})
            T = phys_ds['T'].isel({dim: i})
            RH = phys_ds['RH'].isel({dim: i})
            if 'VP' not in phys_ds:
                if 'MR' not in phys_ds:
                    MR = wrap_xr_metpy_mixing_ratio(P, T, RH, verbose=False)
                VP = wrap_xr_metpy_vapor_pressure(P, MR)
            else:
                VP = phys_ds['VP'].isel({dim: i})
            if 'Rho' not in phys_ds:
                Rho = wrap_xr_metpy_density(P, T, MR, verbose=False)
            else:
                Rho = phys_ds['Rho'].isel({dim: i})
            field, unit = calculate_tm_via_pressure_sum(VP, T, Rho, P,
                                                        bottom=bottom,
                                                        top=top)
        elif field_name == 'ts':
            long_name = 'Surface temperature'
            if 'Height' in phys_ds['T'].dims:
                dropped = phys_ds['T'].isel({dim: i}).dropna('Height')
            elif 'time' in phys_ds['T'].dims:
                dropped = phys_ds['T'].isel({dim: i}).dropna('time')
            field = dropped[0].values.item() + 273.15
            unit = 'K'
        field_list.append(field)
    da = xr.DataArray(field_list, dims=[dim])
    da[dim] = phys_ds[dim]
    da.attrs['units'] = unit
    da.attrs['long_name'] = long_name
    if top is not None:
        da.attrs['top'] = top
    if bottom is not None:
        da.attrs['bottom'] = top
    da = keep_iqr(da, dim=dim, k=1.5)
    if verbose:
        print('Done!')
    return da


def process_radiosonde_data(path=sound_path, savepath=sound_path,
                            data_type='phys', station='bet_dagan', verbose=False):
    import xarray as xr
    from aux_gps import path_glob
    file = path_glob(path, '{}_{}_sounding_*.nc'.format(data_type, station))
    phys_ds = xr.load_dataset(file[0])
    ds = xr.Dataset()
    ds['PW'] = process_new_field_from_radiosonde_data(phys_ds, dim='sound_time',
                                                      field_name='pw',
                                                      bottom=None,
                                                      top=None, verbose=verbose)
    ds['Tm'] = process_new_field_from_radiosonde_data(phys_ds, dim='sound_time',
                                                      field_name='tm',
                                                      bottom=None,
                                                      top=None, verbose=verbose)
    ds['Ts'] = process_new_field_from_radiosonde_data(phys_ds, dim='sound_time',
                                                      field_name='ts',
                                                      bottom=None,
                                                      top=None, verbose=verbose)
    if data_type == 'phys':
        ds['cloud_code'] = phys_ds['cloud_code']
        ds['sonde_type'] = phys_ds['sonde_type']
        ds['min_time'] = phys_ds['min_time']
        ds['max_time'] = phys_ds['max_time']
    yr_min = ds['sound_time'].min().dt.year.item()
    yr_max = ds['sound_time'].max().dt.year.item()
    filename = '{}_{}_PW_Tm_Ts_{}-{}.nc'.format(station, data_type, yr_min, yr_max)
    print('saving {} to {}'.format(filename, savepath))
    ds.to_netcdf(savepath / filename, 'w')
    print('Done!')
    return ds


def calculate_tm_via_trapz_height(VP, T, H):
    from scipy.integrate import cumtrapz
    import numpy as np
    # change T units to K:
    T_copy = T.copy(deep=True) + 273.15
    num = cumtrapz(VP / T_copy, H, initial=np.nan)
    denom = cumtrapz(VP / T_copy**2, H, initial=np.nan)
    tm = num / denom
    return tm


def calculate_tm_via_pressure_sum(VP, T, Rho, P, bottom=None, top=None,
                                  cumulative=False, verbose=False):
    import pandas as pd
    import numpy as np

    def tm_sum(VP, T, Rho, P, bottom=None, top=None):
        # slice for top and bottom:
        if bottom is not None:
            P = P.where(P <= bottom, drop=True)
            T = T.where(P <= bottom, drop=True)
            Rho = Rho.where(P <= bottom, drop=True)
            VP = VP.where(P <= bottom, drop=True)
        if top is not None:
            P = P.where(P >= top, drop=True)
            T = T.where(P >= top, drop=True)
            Rho = Rho.where(P >= top, drop=True)
            VP = VP.where(P >= top, drop=True)
        # convert to Kelvin:
        T_values = T.values + 273.15
        # other units don't matter since it is weighted temperature:
        VP_values = VP.values
        P_values = P.values
        Rho_values = Rho.values
        # now the pressure sum method:
        p = pd.Series(P_values)
        dp = p.diff(-1).abs()
        num = pd.Series(VP_values / (T_values * Rho_values))
        num_sum = num.shift(-1) + num
        numerator = (num_sum * dp / 2).sum()
        denom = pd.Series(VP_values / (T_values**2 * Rho_values))
        denom_sum = denom.shift(-1) + denom
        denominator = (denom_sum * dp / 2).sum()
        tm = numerator / denominator
        return tm
    try:
        T_unit = T.attrs['units']
        assert T_unit == 'degC'
    except KeyError:
        T_unit = 'degC'
        if verbose:
            print('assuming T units are degC...')
    # check that VP and P have the same units:
    assert P.attrs['units'] == VP.attrs['units']
    P_values = P.values
    if cumulative:
        tm_list = []
        # first value is nan:
        tm_list.append(np.nan)
        for pre_val in P_values[1:]:
            if np.isnan(pre_val):
                tm_list.append(np.nan)
                continue
            tm = tm_sum(VP, T, Rho, P, bottom=None, top=pre_val)
            tm_list.append(tm)
        tm = np.array(tm_list)
        return tm, 'K'
    else:
        tm = tm_sum(VP, T, Rho, P, bottom=bottom, top=top)
        return tm, 'K'


def wrap_xr_metpy_pw(dewpt, pressure, bottom=None, top=None, verbose=False,
                     cumulative=False):
    from metpy.calc import precipitable_water
    from metpy.units import units
    import numpy as np
    try:
        T_unit = dewpt.attrs['units']
        assert T_unit == 'degC'
    except KeyError:
        T_unit = 'degC'
        if verbose:
            print('assuming dewpoint units are degC...')
    dew_values = dewpt.values * units(T_unit)
    try:
        P_unit = pressure.attrs['units']
        assert P_unit == 'hPa'
    except KeyError:
        P_unit = 'hPa'
        if verbose:
            print('assuming pressure units are hPa...')
    if top is not None:
        top_with_units = top * units(P_unit)
    else:
        top_with_units = None
    if bottom is not None:
        bottom_with_units = bottom * units(P_unit)
    else:
        bottom_with_units = None
    pressure_values = pressure.values * units(P_unit)
    if cumulative:
        pw_list = []
        # first value is nan:
        pw_list.append(np.nan)
        for pre_val in pressure_values[1:]:
            if np.isnan(pre_val):
                pw_list.append(np.nan)
                continue
            pw = precipitable_water(pressure_values, dew_values, bottom=None,
                                    top=pre_val)
            pw_units = pw.units.format_babel('~P')
            pw_list.append(pw.magnitude)
        pw = np.array(pw_list)
        return pw, pw_units
    else:
        pw = precipitable_water(pressure_values, dew_values,
                                bottom=bottom_with_units, top=top_with_units)
        pw_units = pw.units.format_babel('~P')
        return pw.magnitude, pw_units


def calculate_absolute_humidity_from_partial_pressure(VP, T, verbose=False):
    Rs_v = 461.52  # Specific gas const for water vapour, J kg^{-1} K^{-1}
    try:
        VP_unit = VP.attrs['units']
        assert VP_unit == 'hPa'
    except KeyError:
        VP_unit = 'hPa'
        if verbose:
            print('assuming vapor units are hPa...')
    # convert to Pa:
    VP_values = VP.values * 100.0
    try:
        T_unit = T.attrs['units']
        assert T_unit == 'degC'
    except KeyError:
        T_unit = 'degC'
        if verbose:
            print('assuming temperature units are degree celsius...')
    # convert to Kelvin:
    T_values = T.values + 273.15
    Rho_wv = VP_values/(Rs_v * T_values)
    # resulting units are kg/m^3, convert to g/m^3':
    Rho_wv *= 1000.0
    Rho_wv
    da = VP.copy(data=Rho_wv)
    da.attrs['units'] = 'g/m^3'
    da.attrs['long_name'] = 'Absolute humidity'
    return da


def wrap_xr_metpy_specific_humidity(MR, verbose=False):
    from metpy.calc import specific_humidity_from_mixing_ratio
    from metpy.units import units
    try:
        MR_unit = MR.attrs['units']
        assert MR_unit == 'g/kg'
    except KeyError:
        MR_unit = 'g/kg'
        if verbose:
            print('assuming mixing ratio units are gr/kg...')
    MR_values = MR.values * units(MR_unit)
    SH = specific_humidity_from_mixing_ratio(MR_values)
    da = MR.copy(data=SH.magnitude)
    da.attrs['units'] = MR_unit
    da.attrs['long_name'] = 'Specific humidity'
    return da


def calculate_atmospheric_refractivity(P, T, RH, verbose=False):
    MR = wrap_xr_metpy_mixing_ratio(P, T, RH)
    VP = wrap_xr_metpy_vapor_pressure(P, MR)
    try:
        T_unit = T.attrs['units']
        assert T_unit == 'degC'
    except KeyError:
        T_unit = 'degC'
        if verbose:
            print('assuming temperature units are degree celsius...')
    # convert to Kelvin:
    T_k = T + 273.15
    N = 77.6 * P / T_k + 3.73e5 * VP / T_k**2
    N.attrs['units'] = 'dimensionless'
    N.attrs['long_name'] = 'Index of Refractivity'
    return N


def convert_wind_speed_direction_to_zonal_meridional(WS, WD, verbose=False):
    # make sure it is right!
    import numpy as np
    # drop nans from WS and WD:
    dim = list(set(WS.dims))[0]
    assert dim == list(set(WD.dims))[0]
    DS = WS.to_dataset(name='WS')
    DS['WD'] = WD
#    DS = DS.dropna(dim)
    WS = DS['WS']
    WD = DS['WD']
    assert WS.size == WD.size
    WD = 270 - WD
    try:
        WS_unit = WS.attrs['units']
        if WS_unit != 'm/s':
            if WS_unit == 'knots':
            # 1knots= 0.51444445m/s
                if verbose:
                    print('wind speed in knots, converting to m/s')
                WS = WS * 0.51444445
                WS.attrs.update(units='m/s')
    except KeyError:
        WS_unit = 'm/s'
        if verbose:
            print('assuming wind speed units are m/s...')
    U = WS * np.cos(np.deg2rad(WD))
    V = WS * np.sin(np.deg2rad(WD))
    U.attrs['long_name'] = 'zonal_velocity'
    U.attrs['units'] = 'm/s'
    V.attrs['long_name'] = 'meridional_velocity'
    V.attrs['units'] = 'm/s'
    U.name = 'u'
    V.name = 'v'
    return U, V


#def compare_WW2014_to_Rib_all_seasons(path=sound_path, times=None,
#                                      plot_type='hist', bins=25):
#    import matplotlib.pyplot as plt
#    import seaborn as sns
#    if plot_type == 'hist' or plot_type == 'scatter':
#        fig_hist, axs = plt.subplots(2, 2, sharex=False, sharey=True,
#                                     figsize=(10, 8))
#        seasons = ['DJF', 'MAM', 'JJA', 'SON']
#        cmap = sns.color_palette("colorblind", 2)
#        for i, ax in enumerate(axs.flatten()):
#            ax = compare_WW2014_to_Rib_single_subplot(sound_path=path,
#                                                      season=seasons[i],
#                                                      times=times, ax=ax,
#                                                      colors=[cmap[0],
#                                                              cmap[1]],
#                                                      plot_type=plot_type,
#                                                      bins=bins)
#        fig_hist.tight_layout()
#    return


#def compare_WW2014_to_Rib_single_subplot(sound_path=sound_path, season=None,
#                                         times=None, bins=None,
#                                         ax=None, colors=None,
#                                         plot_type='hist'):
#    from aux_gps import path_glob
#    import xarray as xr
#    from PW_from_gps_figures import plot_two_histograms_comparison
#    ww_file = path_glob(sound_path, 'MLH_WW2014_*.nc')[-1]
#    ww = xr.load_dataarray(ww_file)
#    rib_file = path_glob(sound_path, 'MLH_Rib_*.nc')[-1]
#    rib = xr.load_dataarray(rib_file)
#    ds = ww.to_dataset(name='MLH_WW')
#    ds['MLH_Rib'] = rib
#    if season is not None:
#        ds = ds.sel(sound_time=ds['sound_time.season'] == season)
#        print('selected {} season'.format(season))
#        labels = ['MLH-Rib for {}'.format(season), 'MLH-WW for {}'.format(season)]
#    else:
#        labels = ['MLH-Rib Annual', 'MLH-WW Annual']
#    if times is not None:
#        ds = ds.sel(sound_time=slice(*times))
#        print('selected {}-{} period'.format(*times))
#        title = 'Bet-Dagan radiosonde {}-{} period'.format(*times)
#    else:
#        times = [ds.sound_time.min().dt.year.item(),
#                 ds.sound_time.max().dt.year.item()]
#        title = 'Bet-Dagan radiosonde {}-{} period'.format(*times)
#    if plot_type == 'hist':
#        ax = plot_two_histograms_comparison(ds['MLH_Rib'], ds['MLH_WW'],
#                                            ax=ax, labels=labels,
#                                            colors=colors, bins=bins)
#        ax.legend()
#        ax.set_ylabel('Frequency')
#        ax.set_xlabel('MLH [m]')
#        ax.set_title(title)
#    elif plot_type == 'scatter':
#        if ax is None:
#            fig, ax = plt.subplots()
#        ax.scatter(ds['MLH_Rib'].values, ds['MLH_WW'].values)
#        ax.set_xlabel(labels[0].split(' ')[0] + ' [m]')
#        ax.set_ylabel(labels[1].split(' ')[0] + ' [m]')
#        season_label = labels[0].split(' ')[-1]
#        ax.plot(ds['MLH_Rib'], ds['MLH_Rib'], c='r')
#        ax.legend(['y = x', season_label], loc='upper right')
#        ax.set_title(title)
#    return ax


#def calculate_Wang_and_Wang_2014_MLH_all_profiles(sound_path=sound_path,
#                                                  data_type='phys',
#                                                  hour=12, plot=True,
#                                                  savepath=None):
#    import xarray as xr
#    from aux_gps import smooth_xr
#    import matplotlib.pyplot as plt
#    import seaborn as sns
#    from aux_gps import save_ncfile
#    from PW_from_gps_figures import plot_seasonal_histogram
#    if data_type == 'phys':
#        bd = xr.load_dataset(sound_path / 'bet_dagan_phys_sounding_2007-2019.nc')
#    elif data_type == 'edt':
#        bd = xr.load_dataset(sound_path / 'bet_dagan_edt_sounding_2016-2019.nc')
#    # N = calculate_atmospheric_refractivity(bd['P'], bd['T'], bd['VP'])
#    # assemble all WW vars:
#    WW = bd['N'].to_dataset(name='N')
#    WW['RH'] = bd['RH']
#    WW['PT'] = bd['PT']
#    WW['MR'] = bd['MR']
#    # slice hour:
#    WW = WW.sel(sound_time=WW['sound_time.hour'] == hour)
#    # produce gradients:
#    WW_grad = WW.differentiate('Height', edge_order=2)
#    # smooth them with 1-2-1 smoother:
#    WW_grad_smoothed = smooth_xr(WW_grad, 'Height')
##    return WW_grad_smoothed
#    mlhs = []
#    for dt in WW_grad_smoothed.sound_time:
#        df = WW_grad_smoothed.sel(sound_time=dt).reset_coords(drop=True).to_dataframe()
#        mlhs.append(calculate_Wang_and_Wang_2014_MLH_single_profile(df, plot=False))
#    mlh = xr.DataArray(mlhs, dims=['sound_time'])
#    mlh['sound_time'] = WW_grad_smoothed['sound_time']
#    mlh.name = 'MLH'
#    mlh.attrs['long_name'] = 'Mixing layer height'
#    mlh.attrs['units'] = 'm'
#    mlh.attrs['method'] = 'W&W2014 using PT, N, MR and RH'
#    if savepath is not None:
#        filename = 'MLH_WW2014_{}_{}.nc'.format(data_type, hour)
#        save_ncfile(mlh, sound_path, filename)
#    if plot:
#        cmap = sns.color_palette("colorblind", 5)
#        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 9))
#        df_mean = mlh.groupby('sound_time.month').mean().to_dataframe('mean_MLH')
#        df_mean.plot(color=cmap, ax=ax[0])
#        ax[0].grid()
#        ax[0].set_ylabel('Mean MLH [m]')
#        ax[0].set_title(
#            'Annual mixing layer height from Bet-Dagan radiosonde profiles ({}Z) using W&W2014 method'.format(hour))
#        df_std = mlh.groupby('sound_time.month').std().to_dataframe('std_MLH')
#        df_std.plot(color=cmap, ax=ax[1])
#        ax[1].grid()
#        ax[1].set_ylabel('Std MLH [m]')
#        df_count = mlh.groupby('sound_time.month').count().to_dataframe('count_MLH')
#        df_count.plot(color=cmap, ax=ax[2])
#        ax[2].grid()
#        ax[2].set_ylabel('Count MLH [#]')
#        fig.tight_layout()
#        plot_seasonal_histogram(mlh, dim='sound_time', xlim=(-100, 3000),
#                                xlabel='MLH [m]',
#                                suptitle='MLH histogram using W&W 2014 method')
#    return mlh


#def calculate_Wang_and_Wang_2014_MLH_single_profile(df, alt_cutoff=3000,
#                                                    plot=True):
#    import pandas as pd
#    import numpy as np
#    import matplotlib.pyplot as plt
#    # first , cutoff:
#    df = df.loc[0: alt_cutoff]
#    if plot:
##        df.plot(subplots=True)
#        fig, ax = plt.subplots(1, 4, figsize=(20, 16))
#        df.loc[0: 1200, 'PT'].reset_index().plot.line(y='Height', x='PT', ax=ax[0], legend=False)
#        df.loc[0: 1200, 'RH'].reset_index().plot.line(y='Height', x='RH', ax=ax[1], legend=False)
#        df.loc[0: 1200, 'MR'].reset_index().plot.line(y='Height', x='MR', ax=ax[2], legend=False)
#        df.loc[0: 1200, 'N'].reset_index().plot.line(y='Height', x='N', ax=ax[3], legend=False)
#        [x.grid() for x in ax]
#    ind = np.arange(1, 11)
#    pt10 = df['PT'].nlargest(n=10).index.values
#    n10 = df['N'].nsmallest(n=10).index.values
#    rh10 = df['RH'].nsmallest(n=10).index.values
#    mr10 = df['MR'].nsmallest(n=10).index.values
#    ten = pd.DataFrame([pt10, n10, rh10, mr10]).T
#    ten.columns = ['PT', 'N', 'RH', 'MR']
#    ten.index = ind
#    for i, vc_df in ten.iterrows():
#        mlh_0 = vc_df.value_counts()[vc_df.value_counts() > 2]
#        if mlh_0.empty:
#            continue
#        else:
#            mlh = mlh_0.index.item()
#            return mlh
#    print('MLH Not found using W&W!')
#    return np.nan


def plot_pblh_radiosonde(path=sound_path, reduce='median', fontsize=20):
    import xarray as xr
    import matplotlib.pyplot as plt
    import numpy as np
    pblh = xr.load_dataset(
        sound_path /
        'PBLH_classification_bet_dagan_2s_sounding_2014-2019.nc')
    if reduce == 'median':
        pblh_r = pblh.groupby('sound_time.month').median()
    elif reduce == 'mean':
        pblh_r = pblh.groupby('sound_time.month').mean()
    pblh_c = pblh.groupby('sound_time.month').count()
    count_total = pblh_c.sum()
    df = pblh_r.to_dataframe()
    df[['SBLH_c', 'RBLH_c', 'CBLH_c']] = pblh_c.to_dataframe()
    fig, axes = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(10, 10))
    line_color = 'black'
    bar_color = 'tab:orange'
    df['CBLH'].plot(ax=axes[0], linewidth=2, color=line_color, marker='o', label='CBL', legend=True)
    tw_0 = axes[0].twinx()
    tw_0.bar(x=df.index.values, height=df['CBLH_c'].values, color=bar_color, alpha=0.4)
    df['RBLH'].plot(ax=axes[1], linewidth=2, color=line_color, marker='o', label='RBL', legend=True)
    tw_1 = axes[1].twinx()
    tw_1.bar(x=df.index.values, height=df['RBLH_c'].values, color=bar_color, alpha=0.4)
    df['SBLH'].plot(ax=axes[2], linewidth=2, color=line_color, marker='o', label='SBL', legend=True)
    tw_2 = axes[2].twinx()
    tw_2.bar(x=df.index.values, height=df['SBLH_c'].values, color=bar_color, alpha=0.4)
    axes[0].set_ylabel('CBL [m]', fontsize=fontsize)
    axes[1].set_ylabel('RBL [m]', fontsize=fontsize)
    axes[2].set_ylabel('SBL [m]', fontsize=fontsize)
    tw_0.set_ylabel('Launch ({} total)'.format(count_total['CBLH'].values), fontsize=fontsize)
    tw_1.set_ylabel('Launch ({} total)'.format(count_total['RBLH'].values), fontsize=fontsize)
    tw_2.set_ylabel('Launch ({} total)'.format(count_total['SBLH'].values), fontsize=fontsize)
    [ax.set_xticks(np.arange(1,13,1)) for ax in axes]
    [ax.grid() for ax in axes]
    [ax.tick_params(labelsize=fontsize) for ax in axes]
    [ax.tick_params(labelsize=fontsize) for ax in [tw_0, tw_1, tw_2]]
    fig.suptitle(
        'PBL {} Height from Bet-Dagan radiosonde (2014-2019)'.format(reduce),
        fontsize=fontsize)
    fig.tight_layout()
    return fig


def align_rbl_times_cloud_h1_pwv(rbl_cat, path=work_yuval,
                                 ceil_path=ceil_path, pw_station='tela',
                                 plot_diurnal=True, fontsize=16):
    from ceilometers import read_BD_ceilometer_yoav_all_years
    import xarray as xr
    import matplotlib.pyplot as plt
    from aux_gps import anomalize_xr
    import numpy as np
    # first load cloud_H1 and pwv:
    cld = read_BD_ceilometer_yoav_all_years(path=ceil_path)['cloud_H1']
    cld[cld==0]=np.nan
    ds = cld.to_dataset(name='cloud_H1')
    pwv = xr.open_dataset(
        path /
        'GNSS_PW_thresh_50.nc')[pw_station]
    pwv.load()
    pw_name = 'pwv_{}'.format(pw_station)
    bins_name = '{}'.format(rbl_cat.name)
    ds[pw_name] = pwv.sel(time=pwv['time.season']=='JJA')
    daily_pwv_total = ds[pw_name].groupby('time.hour').count().sum()
    print(daily_pwv_total)
    daily_pwv = anomalize_xr(ds[pw_name]).groupby('time.hour').mean()
    # now load rbl_cat with attrs:
    ds[bins_name] = rbl_cat
    ds = ds.dropna('time')
    # change dtype of bins to int:
    ds[bins_name] = ds[bins_name].astype(int)
    # produce pwv anomalies regarding the bins:
    pwv_anoms = ds[pw_name].groupby(ds[bins_name]) - ds[pw_name].groupby(ds[bins_name]).mean('time')
    counts = ds.groupby('time.hour').count()['cloud_H1']
    ds['pwv_{}_anoms'.format(pw_station)] = pwv_anoms.reset_coords(drop=True)
    if plot_diurnal:
        fig, axes = plt.subplots(figsize=(15, 8))
        df_hour = ds['pwv_tela_anoms'].groupby('time.hour').mean().to_dataframe()
        df_hour['cloud_H1'] = ds['cloud_H1'].groupby('time.hour').mean()
        df_hour['cloud_H1_counts'] = counts
        df_hour['pwv_tela_daily_anoms'] = daily_pwv
        df_hour['pwv_tela_anoms'].plot(marker='s', ax=axes, linewidth=2)
        df_hour['pwv_tela_daily_anoms'].plot(ax=axes, marker='s', color='r', linewidth=2)
#        ax2 = df_hour['cloud_H1'].plot(ax=axes[0], secondary_y=True, marker='o')
        ax2 = df_hour['cloud_H1_counts'].plot(ax=axes, secondary_y=True, marker='o', linewidth=2)
        axes.set_ylabel('PWV TELA anomalies [mm]', fontsize=fontsize)
        axes.set_xlabel('Hour of day [UTC]', fontsize=fontsize)
        ax2.set_ylabel('Cloud H1 data points', fontsize=fontsize)
        axes.set_xticks(np.arange(0, 24, 1))
        axes.xaxis.grid()
        handles,labels = [],[]
        for ax in fig.axes:
            for h,l in zip(*ax.get_legend_handles_labels()):
                handles.append(h)
                labels.append(l)
        axes.legend(handles,labels, fontsize=fontsize)
#        counts.to_dataframe(name='Count').plot(kind='bar', color='tab:blue', alpha=0.5, ax=axes[1], rot=0)
#        axes[1].bar(x=np.arange(0, 24, 1), height=counts.values, color='tab:blue', alpha=0.5)
#        axes[1].set_xticks(np.arange(0, 24, 1))
        axes.tick_params(labelsize=fontsize)
        ax2.tick_params(labelsize=fontsize)
        fig.tight_layout()
        fig.suptitle('PWV TELA anomalies and Cloud H1 counts for JJA', fontsize=fontsize)
        fig.subplots_adjust(top=0.951,
                            bottom=0.095,
                            left=0.071,
                            right=0.936,
                            hspace=0.2,
                            wspace=0.2)
    return ds


def categorize_da_ts(da_ts, season=None, add_hours_to_dt=None, resample=True,
                     bins=[0, 200, 400, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500]):
    # import xarray as xr
    import numpy as np
    import pandas as pd
    time_dim = list(set(da_ts.dims))[0]
    if season is not None:
        da_ts = da_ts.sel(
            {time_dim: da_ts['{}.season'.format(time_dim)] == season})
        print('{} season selected'.format(season))
    # bins = rbl.quantile(
    #     [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
    if da_ts.name is None:
        name = 'MLH'
    else:
        name = da_ts.name
    # rename sound_time to time:
    da_ts = da_ts.rename({time_dim: 'time'})
    df = da_ts.to_dataframe(name=name)
    labels = np.arange(0, len(bins) - 1)
    df['{}_bins'.format(name)] = pd.cut(
        df['{}'.format(name)], bins=bins, labels=labels, retbins=False)
    df_bins = df['{}_bins'.format(name)]
    if add_hours_to_dt is not None:
        print('adding {} hours to datetimes.'.format(add_hours_to_dt))
        df_bins.index += pd.Timedelta(add_hours_to_dt, unit='H')
    if resample:
        re = []
        for row in df_bins.dropna().to_frame().iterrows():
            bin1 = row[1].values
            new_time = pd.date_range(row[0], periods=288, freq='5T')
            new_bins = [bin1 for x in new_time]
            re.append(pd.DataFrame(new_bins, index=new_time, columns=['{}_bins'.format(name)]))
#        df_bins = df_bins.resample('5T').ffill(limit=576).dropna()
        df_bins = pd.concat(re, axis=0)
        print('resampling to 5 mins using ffill.')
    # result = xr.apply_ufunc(np.digitize, rbl, kwargs={'bins': bins})
    # df = result.to_dataframe('bins')
    # df['rbl'] = rbl.to_dataframe(name='rbl')
    # means = df['rbl'].groupby(df['bins']).mean()
    # or just:
    # rbl_bins = rbl.to_dataset(name='rbl').groupby_bins(group='rbl',bins=bins, labels=np.arange(1, len(bins))).groups
    # grp = df.groupby('{}_bins'.format(name)).groups
    print('categorizing to bins: {}'.format(','.join([str(x) for x in bins])))
    df_bins.index.name = 'time'
    da = df_bins.to_xarray().to_array(name='{}_bins'.format(name)).squeeze(drop=True)
    # get the bins borders and insert them as attrs to da:
    dumm = pd.cut(df['{}'.format(name)], bins=bins, labels=None, retbins=False)
    left = [x.left for x in dumm.dtype.categories]
    right = [x.right for x in dumm.dtype.categories]
    for i, label in enumerate(labels):
        da.attrs[str(label)] = [float(left[i]), float(right[i])]
    da.attrs['units'] = da_ts.attrs['units']
    return da


def prepare_radiosonde_and_solve_MLH(ds, method='T', max_height=300):
    import xarray as xr
    import pandas as pd
    ds = ds.drop_sel(time=pd.to_timedelta(0, unit='s'))
    # nullify the first Height:
    ds['Height'] -= ds['Height'].isel(time=0)
    pbls = []
    stimes = []
    for i in range(ds['sound_time'].size):
        if method == 'T':
            mlh = find_surface_inversion_height(
                    ds.isel(
                            sound_time=i).reset_coords(
                                    drop=True), max_height=max_height)
        elif method == 'rig':
            mlh = find_MLH_from_2s_richardson(
                    ds.isel(
                            sound_time=i).reset_coords(
                                    drop=True), method='grad')
        elif method == 'WW':
            mlh = find_MLH_from_2s_WW2014(
                ds.isel(
                    sound_time=i), alt_cutoff=max_height)
        elif method == 'rib':
            mlh = find_MLH_from_2s_richardson(
                ds.isel(
                    sound_time=i), method='bulk')
        if mlh is not None:
            pbls.append(mlh)
            stimes.append(ds.isel(sound_time=i)['sound_time'])
    sound_time = xr.concat(stimes, 'sound_time')
    pbl = xr.DataArray([x.values for x in pbls], dims=['sound_time'])
    pbl['sound_time'] = sound_time
    pbl = pbl.sortby('sound_time')
    pbl.attrs['method'] = method
    if max_height is not None:
        pbl.attrs['max_height'] = max_height
    return pbl


def classify_bet_dagan_pblh(path=sound_path, savepath=None):
    import xarray as xr
    from aux_gps import save_ncfile
    ds = xr.load_dataset(path / 'bet_dagan_2s_sounding_2014-2019.nc')
    sbl = classify_SBL(ds, method='T', sbl_max_height=300, filter_cbl=True)
    rbl = classify_RBL(ds, sbl, method='WW', max_height=3500)
    cbl = classify_CBL(ds, method='WW', max_height=3500)
    dss = xr.merge([sbl, rbl, cbl])
    if savepath is not None:
        filename = 'PBLH_classification_bet_dagan_2s_sounding_2014-2019.nc'
        save_ncfile(dss, savepath, filename)
    return dss


def classify_SBL(ds, method='T', sbl_max_height=300, filter_cbl=True):
    """Run find_surface_inversion using T gradient and filter all 12Z records
    use method=T for temp inversion, rig for gradient richardson"""
    # TODO: fix gradient richardson method
    print('classifying SBL...')
    sbl = prepare_radiosonde_and_solve_MLH(
        ds, method=method, max_height=sbl_max_height)
    if filter_cbl:
        print('filtered {} 12Z records.'.format(
            sbl[sbl['sound_time.hour'] == 12].count().item()))
        sbl = sbl[sbl['sound_time.hour'] == 00]
    sbl.name = 'SBLH'
    sbl.attrs['long_name'] = 'Stable Boundary Layer Height'
    sbl.attrs['units'] = 'm'
    sbl.attrs['method'] = method
    return sbl


def classify_RBL(ds, sbl, method='WW', max_height=3500):
    import pandas as pd
    print('classifying RBL...')
    # filter SBLs, first assemble all 00Z:
    Z00_st = ds['T'].transpose('sound_time', 'time')[
        ds['sound_time.hour'] == 00]['sound_time']
#    ds = ds.sel(sound_time=Z00_st)
    # take out the SBL events:
    sbl_st = sbl['sound_time']
    st = pd.to_datetime(
        list(set(Z00_st.values).difference(set(sbl_st.values))))
    ds = ds.sel(sound_time=st)
    rbl = prepare_radiosonde_and_solve_MLH(
        ds, method=method, max_height=max_height)
    print(
        'found {} RBLs from total of {}'.format(
            rbl['sound_time'].size,
            st.size))
    rbl.name = 'RBLH'
    rbl.attrs['long_name'] = 'Residual Boundary Layer Height'
    rbl.attrs['units'] = 'm'
    rbl.attrs['rbl_candidates'] = st.size
    rbl.attrs['rbl_success'] = rbl['sound_time'].size
    rbl.attrs['method'] = method
    return rbl


def classify_CBL(ds, method='WW', max_height=3500):
    # filter only daytime:
    print('classifying CBL...')
    Z12_st = ds['T'].transpose('sound_time', 'time')[
        ds['sound_time.hour'] == 12]['sound_time']
    ds = ds.sel(sound_time=Z12_st)
    cbl = prepare_radiosonde_and_solve_MLH(
        ds, method=method, max_height=max_height)
    print(
        'found {} CBLs from total of {}'.format(
            cbl['sound_time'].size,
            ds['sound_time'].size))
    cbl.name = 'CBLH'
    cbl.attrs['long_name'] = 'Convective Boundary Layer Height'
    cbl.attrs['units'] = 'm'
    cbl.attrs['cbl_candidates'] = ds['sound_time'].size
    cbl.attrs['cbl_success'] = cbl['sound_time'].size
    cbl.attrs['method'] = method
    return cbl


def find_surface_inversion_height(ds, min_height=None, max_height=300, max_time=None):
    """calculate surface inversion height in meters, surface is defined as
    max_time in seconds after radiosonde launch or max_height in meters,
    if we find the surface invesion layer height is it a candidate for SBL
    (use Rig), if not, and it is at night use Rib, W&W to find RBL"""
    import numpy as np
    from aux_gps import smooth_xr
#    from aux_gps import smooth_xr
#    ds = smooth_xr(ds[['T', 'Height']], dim='time')
    dsize = len(ds.dims)
    if dsize != 1:
        raise('ds dimensions should be 1!')
    new_ds = ds[['T', 'Height']]
    if max_height is None and max_time is None:
        raise('Pls pick either max_time or max_height...')
    if max_height is None and max_time is not None:
        new_ds = new_ds.isel(time=slice(0, max_time))
#        T_diff = (-ds['T'].diff('time').isel(time=slice(0, max_time)))
    elif max_height is not None and max_time is None:
        new_ds = new_ds.where(ds['Height'] <= max_height, drop=True)
#        T_diff = (-ds['T'].diff('time').where(ds['Height']
#                                              <= max_height, drop=True))
    if min_height is not None:
        new_ds = new_ds.where(ds['Height'] >= min_height, drop=True)
    T = new_ds['T']
    H = new_ds['Height']
    T['time'] = H.values
    T = T.rename(time='Height')
    dT = smooth_xr(T.differentiate('Height'), 'Height')
    dT = dT.dropna('Height')
    indLeft = np.searchsorted(-dT, 0, side='left')
    indRight = np.searchsorted(-dT, 0, side='right')
    if indLeft == indRight:
        ind = indLeft
        mlh = H[ind -1: ind + 1].mean()
    else:
        ind = indLeft
        mlh = H[ind]
    # condition for SBL i.e., the temp increases with height until reveres
    positive_dT = (dT[0] - dT[ind - 1]) > 0
    last_ind = dT.size - 1
    if (ind < last_ind) and (ind > 0) and positive_dT:
        return mlh
    else:
        return None
#    T_diff = T_diff.where(T_diff < 0)
#    if T_diff['time'].size != 0:
#        if dsize == 1:
#            inversion_time = T_diff.idxmin('time')
#            inversion_height = ds['Height'].sel(time=inversion_time)
#        elif dsize == 2:
#            inversion_time = T_diff.idxmin('time').dropna('sound_time')
#            inversion_height = ds['Height'].sel(time=inversion_time, sound_time=inversion_time['sound_time'])
#    else:
#        inversion_height = None
#    return inversion_height


def calculate_temperature_lapse_rate_from_2s_radiosonde(ds, radio_time=2,
                                                        Height=None):
    """calculate the \Gamma = -dT/dz (lapse rate), with either radio_time (2 s after launch)
    or Height at certain level"""
    import numpy as np
    # check for dims:
    dsize = len(ds.dims)
    if dsize > 2 or dsize < 1:
        raise('ds dimensions should be 1 or 2...')
    T = ds['T']
    H = ds['Height']
    T0 = T.isel(time=0)
    H0 = H.isel(time=0)
    if radio_time is None and Height is None:
        raise('Pls pick either radio_time or Height...')
    if radio_time is not None and Height is None:
        radio_time = np.timedelta64(radio_time, 's')
        T1 = T.sel(time=radio_time, method='nearest')
        H1 = H.sel(time=radio_time, method='nearest')
        seconds_after = T['time'].sel(time=radio_time, method='nearest').dt.seconds.item()
        if dsize == 1:
            height = H.sel(time=radio_time, method='nearest').item()
            dz = height - H0.item()
        elif dsize == 2:
            height = H.sel(time=radio_time, method='nearest').mean().item()
            dz = (H1 - H0).mean().item()
        method = 'time'
    elif radio_time is None and Height is not None:
        if dsize == 1:
            t1 = (np.abs(H-Height)).idxmin().item()
        elif dsize == 2:
            t1 = (np.abs(H-Height)).idxmin('time')
        H1 = H.sel(time=t1)
        T1 = T.sel(time=t1)
        if dsize == 1:
            height = H1.item()
            seconds_after = T['time'].sel(time=t1).dt.seconds.item()
            dz = height - H0.item()
        elif dsize == 2:
            height = H1.mean().item()
            dz = (H1 - H0).mean().item()
            seconds_after = T['time'].sel(time=t1).dt.seconds.mean().item()
        method = 'height'
    gamma = -1* (T0 - T1) / ((H1 - H0) / 1000)
    gamma.attrs['units'] = 'degC/km'
    gamma.name = 'Gamma'
    gamma.attrs['long_name'] = 'temperature lapse rate'
    gamma.attrs['Height_taken'] = '{:.2f}'.format(height)
    gamma.attrs['dz [m]'] = '{:.2f}'.format(dz)
    gamma.attrs['seconds_after_launch'] = seconds_after
    gamma.attrs['method'] = method
    return gamma


def plot_2s_radiosonde_single_profile(ds, max_height=1000, rib_lims=None,
                                      plot_type='WW'):
    # drop the first time:
    # ds1=ds1.drop_sel(time=pd.to_timedelta(0,unit='s'))
    # nullify the first Height:
    # ds1['Height'] -= ds1['Height'][0]
    # fix 31-35 meters ambiguity
    # fix PT in K and in degC = probably calculated in K and sub 273.15
    import matplotlib.pyplot as plt
    import pandas as pd
    assert len(ds.dims) == 1 and 'time' in ds.dims
    dt = pd.to_datetime(ds['sound_time'].values)
#    ds['Height'] -= ds['Height'][0]
    mlh_rib = find_MLH_from_2s_richardson(ds, method='bulk')
    mlh_rig = find_MLH_from_2s_richardson(ds, method='grad')
    mlh_ww = find_MLH_from_2s_WW2014(ds, alt_cutoff=3500)
    mlh_t = find_surface_inversion_height(ds, max_time=None, max_height=300)
    ds['rib'] = calculate_richardson_from_2s_radiosonde(ds, method='bulk')
    ds['rig'] = calculate_richardson_from_2s_radiosonde(ds, method='grad')
    ds['N'] = calculate_atmospheric_refractivity(ds['P'], ds['T'], ds['RH'])
    T_k = ds['T'] + 273.15
    T_k.attrs['units'] = 'K'
    ds['PT'] = wrap_xr_metpy_potential_temperature(ds['P'], T_k)  # in degC
    ds = ds.assign_coords(time=ds['Height'].values)
    ds = ds.drop('Height')
    ds = ds.rename(time='Height')
    df = ds[['T', 'rib', 'rig', 'PT', 'RH', 'MR', 'N']].to_dataframe()
    if max_height is not None:
        df = df[df.index <= max_height]
    df['Height'] = df.index.values
    df['MR'] *= 1000  # to g/kg
    df['PT'] -= 273.15
    if plot_type == 'WW':
        fig, axes = plt.subplots(
            1, 5, sharey=False, sharex=False, figsize=(
                20, 15))
        df.plot.line(
            ax=axes[0],
            x='rib',
            y='Height',
            marker='.',
            legend=False,
            grid=True,
            color='g')
        axes[0].axhline(y=mlh_rib,color='k', linestyle='-', linewidth=1.5)
        df.plot.line(
            ax=axes[1],
            x='PT',
            y='Height',
            marker='.',
            legend=False,
            grid=True,
            color='g')
        df.plot.line(
            ax=axes[2],
            x='RH',
            y='Height',
            marker='.',
            legend=False,
            grid=True,
            color='g')
        df.plot.line(
            ax=axes[3],
            x='MR',
            y='Height',
            marker='.',
            legend=False,
            grid=True,
            color='g')
        df.plot.line(
            ax=axes[4],
            x='N',
            y='Height',
            marker='.',
            legend=False,
            grid=True,
            color='g')
        [ax.axhline(y=mlh_ww, color='k', linestyle='-', linewidth=1.5) for ax in axes[1:4]]
        axes[0].set_xlabel('Ri$_b$')
        axes[0].axvline(0.25, color='k', linestyle='--')
        axes[1].set_xlabel('$\Theta$ [$\degree$C]')
        axes[2].set_xlabel('RH [%]')
        axes[3].set_xlabel('w [g/kg]')
        axes[4].set_xlabel('N')
        axes[0].set_ylabel('z [m]')
        if rib_lims is not None:
            axes[0].set_xlim(*rib_lims)
    elif plot_type == 'T':
        fig, axes = plt.subplots(
            1, 3, sharey=False, sharex=False, figsize=(
                20, 15))
        df.plot.line(
            ax=axes[0],
            x='T',
            y='Height',
            marker='.',
            legend=False,
            grid=True,
            color='g')
        if mlh_t is not None:
            axes[0].axhline(y=mlh_t,color='k', linestyle='-', linewidth=1.5)
        df.plot.line(
            ax=axes[1],
            x='rig',
            y='Height',
            marker='.',
            legend=False,
            grid=True,
            color='g')
        axes[1].axhline(y=mlh_rig, color='k', linestyle='-', linewidth=1.5)
        df.plot.line(
            ax=axes[2],
            x='rib',
            y='Height',
            marker='.',
            legend=False,
            grid=True,
            color='g')
        axes[1].set_ylim(0, max_height)
        axes[2].axhline(y=mlh_rib, color='k', linestyle='-', linewidth=1.5)
        axes[0].set_xlabel('T [$\degree$C]')
        axes[1].set_xlabel('Ri$_g$')
        axes[2].set_xlabel('Ri$_b$')
        axes[1].axvline(0.25, color='k', linestyle='--')
        axes[2].axvline(0.25, color='k', linestyle='--')
        axes[0].set_ylabel('Z [m]')
        if rib_lims is not None:
            axes[2].set_xlim(*rib_lims)
    fig.suptitle(dt)
    return fig


def calculate_richardson_from_2s_radiosonde(ds, g=9.79474, method='bulk'):
    import numpy as np
    dsize = len(ds.dims)
    if dsize == 2:
        axis=1
    else:
        axis=0
    T_k = ds['T'] + 273.15
    T_k.attrs['units'] = 'K'
    PT = wrap_xr_metpy_potential_temperature(ds['P'], T_k)
    VPT = wrap_xr_metpy_virtual_potential_temperature(ds['P'], ds['T'], ds['MR'])
    U = ds['u']
    V = ds['v']
    H = ds['Height']
#    VPT = wrap_xr_metpy_virtual_potential_temperature(ds['P'], ds['T'], ds['MR']*1000)
#    VPT = PT * (1 +ds['MR']*1000/0.622)/(1+ds['MR']*1000) - 273.15
#    VPT_mean = VPT.cumsum('time') / (np.arange(H.size) + 1)
    if method == 'bulk':
        H0 = H.isel(time=0)
#        H0 = 0
        U0 = U.isel(time=0)
        V0 = V.isel(time=0)
        VPT_0 = VPT.isel(time=0)
        U0 = 0
        V0 = 0
        U2 = (U - U0)**2.0
        V2 = (V - V0)**2.0
        Rib_values = g * (VPT - VPT_0) * (H - H0) / ((VPT_0) * (U2 + V2))
#        Rib_values = g * (VPT - VPT_0) * (H - H0) / ((VPT_mean) * (U2 + V2))
        Ri = VPT.copy(data=Rib_values)
        Ri.name = 'Rib'
        Ri.attrs.update(long_name='Bulk Richardson Number')
        Ri.attrs.update(units='dimensionless')
    elif method == 'grad':
#        PT -= 273.15
        BVF2 = wrap_xr_metpy_brunt_vaisala_f2(H, PT, verbose=False, axis=axis)
#        U.assign_coords(time=H)
#        V.assign_coords(time=H)
#        U = U.rename(time='Height')
#        V = V.rename(time='Height')
#        dU = U.differentiate('Height').values
#        dV = V.differentiate('Height').values
        dU = (U.differentiate('time') / H.differentiate('time')).values
        dV = (V.differentiate('time') / H.differentiate('time')).values
        Ri = BVF2 / (dU**2 + dV**2)
        Ri.name = 'Rig'
        Ri.attrs.update(long_name='Gradient Richardson Number')
        Ri.attrs.update(units='dimensionless')
    return Ri

#
#def calculate_bulk_richardson_from_physical_radiosonde(VPT, U, V, g=9.79474,
#                                                       initial_height_pos=0):
#    import numpy as np
#    z = VPT['Height']  # in meters
#    z0 = VPT['Height'].isel(Height=initial_height_pos)
#    U0 = U.isel(Height=int(initial_height_pos))
#    V0 = V.isel(Height=int(initial_height_pos))
#    VPT_0 = VPT.isel(Height=int(initial_height_pos))
#    VPT_mean = VPT.cumsum('Height') / (np.arange(VPT.Height.size) + 1)
##    U.loc[dict(Height=35)]=0
##    V.loc[dict(Height=35)]=0
#    U2 = (U-U0)**2.0
#    V2 = (V-V0)**2.0
##     WS2 = (WS * 0.51444445)**2
##    Rib_values = g * (VPT - VPT_0) * (z) / ((VPT_mean) * (U2 + V2))
#    Rib_values = g * (VPT - VPT_0) * (z - z0) / ((VPT_0) * (U2 + V2))
#    Rib = VPT.copy(data=Rib_values)
#    Rib.name = 'Rib'
#    Rib.attrs.update(long_name='Bulk Richardson Number')
#    Rib.attrs.update(units='dimensionless')
#    return Rib


#def calculate_gradient_richardson_from_physical_radiosonde(BVF2, U, V):
#    dU = U.differentiate('Height')
#    dV = V.differentiate('Height')
#    Rig = BVF2 / (dU**2 + dV**2)
#    Rig.name = 'Rig'
#    Rig.attrs.update(long_name='Gradient Richardson Number')
#    Rig.attrs.update(units='dimensionless')
#    return Rig


def find_MLH_from_2s_WW2014(ds, alt_cutoff=None, eps=50,
                            return_found_df=False, plot=False):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from aux_gps import smooth_xr
    import xarray as xr
    dsize = len(ds.dims)
    if dsize != 1:
        raise('ds has to be 1D!')
    if 'PT' not in ds:
        T_k = ds['T'] + 273.15
        T_k.attrs['units'] = 'K'
        ds['PT'] = wrap_xr_metpy_potential_temperature(ds['P'], T_k)
    if 'N' not in ds:
        ds['N'] = calculate_atmospheric_refractivity(ds['P'], ds['T'], ds['RH'])
    dt = pd.to_datetime(ds['sound_time'].item())
    dss = ds[['PT', 'RH', 'MR', 'N']].reset_coords(drop=True)
    df_plot = dss.to_dataframe()
    df_plot['Height'] = ds['Height'].values
    df_plot = df_plot.dropna().set_index('Height')
    if alt_cutoff is not None:
        dss = dss.where(ds['Height'] <= alt_cutoff, drop=True)
        H = ds['Height'].where(ds['Height'] <= alt_cutoff, drop=True)
    dss['time'] = H
    dss = dss.rename(time='Height')
#        nearest_cutoff = ds['Height'].dropna('time').sel(Height=alt_cutoff, method='nearest')
#        dss = dss.sel(Height=slice(None, nearest_cutoff))
    dss = dss.differentiate('Height')
    dss = smooth_xr(dss, 'Height')
    df = dss.to_dataframe().dropna()
    ind = np.arange(1, 11)
    pt10 = df['PT'].nlargest(n=10).index.values
    n10 = df['N'].nsmallest(n=10).index.values
    rh10 = df['RH'].nsmallest(n=10).index.values
    mr10 = df['MR'].nsmallest(n=10).index.values
    ten = pd.DataFrame([pt10, n10, rh10, mr10]).T
    ten.columns = ['PT', 'N', 'RH', 'MR']
    ten.index = ind
    found = []
    for i, vc_df in ten.iterrows():
        row_sorted = vc_df.sort_values()
        diff_3_1 = row_sorted[3] - row_sorted[1]
        diff_2_0 = row_sorted[2] - row_sorted[0]
        if (diff_3_1 <= eps) and (diff_2_0 <= eps):
            found.append(row_sorted)
        elif diff_3_1 <= eps:
            found.append(row_sorted[1:4])
        elif diff_2_0 <= eps:
            found.append(row_sorted[0:3])

#        mlh_0 = vc_df.value_counts()[vc_df.value_counts() > 2]
#        if mlh_0.empty:
#            continue
#        else:
#            mlh = mlh_0.index.item()
#            return mlh
    if not found:
        print('MLH Not found for {} using W&W!'.format(dt))
        return None
    found_df = pd.concat(found, axis=1).T
    mlh_mean = found_df.iloc[0].mean()
    if return_found_df:
        return found_df
    if plot:
#        df.plot(subplots=True)
        fig, ax = plt.subplots(1, 4, figsize=(20, 16))
        df_plot.loc[0: 1200, 'PT'].reset_index().plot.line(y='Height', x='PT', ax=ax[0], legend=False)
        df_plot.loc[0: 1200, 'RH'].reset_index().plot.line(y='Height', x='RH', ax=ax[1], legend=False)
        df_plot.loc[0: 1200, 'MR'].reset_index().plot.line(y='Height', x='MR', ax=ax[2], legend=False)
        df_plot.loc[0: 1200, 'N'].reset_index().plot.line(y='Height', x='N', ax=ax[3], legend=False)
        [x.grid() for x in ax]
        [ax[0].axhline(y=mlh, color='r', linestyle='--') for mlh in found_df['PT'].dropna()]
        [ax[1].axhline(y=mlh, color='r', linestyle='--') for mlh in found_df['RH'].dropna()]
        [ax[2].axhline(y=mlh, color='r', linestyle='--') for mlh in found_df['MR'].dropna()]
        [ax[3].axhline(y=mlh, color='r', linestyle='--') for mlh in found_df['N'].dropna()]
        [axx.axhline(y=mlh_mean,color='k', linestyle='-', linewidth=1.5) for axx in ax]
        [axx.set_ylim(0, 1200) for axx in ax]
        [axx.autoscale(enable=True, axis='x', tight=True) for axx in ax]
    return xr.DataArray(mlh_mean)


def find_MLH_from_2s_richardson(ds, crit=0.25, method='bulk'):
    import numpy as np
    ri_dict = {'bulk': 'rib', 'grad': 'rig'}
    ri_name = ri_dict.get(method)
    if ri_name not in ds:
        ds[ri_name] = calculate_richardson_from_2s_radiosonde(
            ds, method=method)
#        if ri_name == 'rig':
#            ds[ri_name] = smooth_xr(ds[ri_name])
    indLeft = np.searchsorted(ds[ri_name], crit, side='left')
    indRight = np.searchsorted(ds[ri_name], crit, side='right')
    if indLeft == indRight:
        ind = indLeft
        last_ind = ds[ri_name].size - 1
        if (ind < last_ind) and (ind > 0):
            assert ds[ri_name][ind - 1] < crit
            mlh = ds['Height'][ind - 1: ind + 1].mean()
        else:
            return None
    else:
        if (ind < last_ind) and (ind > 0):
            ind = indLeft
            mlh = ds['Height'][indLeft]
        else:
            return None

#    mlh_time = np.abs(ds[ri_name] - crit).idxmin('time')
#    mlh = ds['Height'].sel(time=mlh_time)
    mlh.name = 'MLH'
    mlh.attrs['long_name'] = 'Mixing Layer Height'
    return mlh


#def calculate_MLH_from_Rib_single_profile(Rib_df, crit=0.25):
#    import numpy as np
#    # drop first row:
#    # df = Rib_df.drop(35)
##    # get index position of first closest to crit:
##    i = df['Rib'].sub(crit).abs().argmin() + 1
##    df_c = df.iloc[i-2:i+2]
#    indLeft = np.searchsorted(Rib_df['Rib'], crit, side='left')
#    indRight = np.searchsorted(Rib_df['Rib'], crit, side='right')
#    if indLeft == indRight:
#        ind = indLeft
#    else:
#        ind = indLeft
#    mlh = Rib_df.index[ind]
#    # mlh = df['Rib'].sub(crit).abs().idxmin()
#    return mlh


#def calculate_Rib_MLH_all_profiles(sound_path=sound_path, crit=0.25, hour=12,
#                                   data_type='phys', savepath=None):
#    from aux_gps import save_ncfile
#    import xarray as xr
#    from PW_from_gps_figures import plot_seasonal_histogram
#    if data_type == 'phys':
#        bd = xr.load_dataset(sound_path /
#                             'bet_dagan_phys_sounding_2007-2019.nc')
#        pos = 0
#    elif data_type == 'edt':
#        bd = xr.load_dataset(sound_path /
#                             'bet_dagan_edt_sounding_2016-2019.nc')
#        pos = 2
#    Rib = calculate_bulk_richardson_from_physical_radiosonde(bd['VPT'], bd['U'],
#                                                             bd['V'], g=9.79474,
#                                                             initial_height_pos=pos)
#    mlh = calculate_MLH_time_series_from_all_profiles(Rib, crit=crit,
#                                                      hour=hour, plot=False)
#    plot_seasonal_histogram(mlh, dim='sound_time', xlim=(-100, 3000),
#                            xlabel='MLH [m]',
#                            suptitle='MLH histogram using Rib method')
#    if savepath is not None:
#        filename = 'MLH_Rib_{}_{}_{}.nc'.format(
#            str(crit).replace('.', 'p'), data_type, hour)
#        save_ncfile(mlh, sound_path, filename)
#    return mlh


#def calculate_MLH_time_series_from_all_profiles(Rib, crit=0.25, hour=12,
#                                                dim='sound_time', plot=True):
#    from aux_gps import keep_iqr
#    import matplotlib.pyplot as plt
#    import xarray as xr
#    rib = Rib.sel(sound_time=Rib['sound_time.hour'] == hour)
#    mlhs = []
#    for time in rib[dim]:
#        #        print('proccessing MLH retreival of {} using Rib at {}'.format(
#        #            time.dt.strftime('%Y-%m-%d:%H').item(), crit))
#        df = rib.sel({dim: time}).reset_coords(drop=True).to_dataframe()
#        mlhs.append(calculate_MLH_from_Rib_single_profile(df, crit=crit))
#    da = xr.DataArray(mlhs, dims=[dim])
#    da[dim] = rib[dim]
#    da.name = 'MLH'
#    da.attrs['long_name'] = 'Mixing layer height'
#    da.attrs['units'] = 'm'
#    da.attrs['method'] = 'Rib@{}'.format(crit)
#    if plot:
#        da = keep_iqr(da, dim)
#        fig, ax = plt.subplots(figsize=(15, 6))
#        ln = da.plot(ax=ax)
#        ln200 = da.where(da >= 200).plot(ax=ax)
#        lnmm = da.where(da > 200).resample(
#            {dim: 'MS'}).mean().plot(ax=ax, linewidth=3, color='r')
#        ax.legend(ln + ln200 + lnmm,
#                  ['MLH', 'MLH above 200m', 'MLH above 200m monthly means'])
#        ax.grid()
#        ax.set_ylabel('MLH from Rib [m]')
#        ax.set_xlabel('')
#        fig.tight_layout()
#    return da


#def solve_MLH_with_all_crits(RiB, mlh_all=None, hour=12, cutoff=200,
#                             plot=True):
#    import xarray as xr
#    import matplotlib.pyplot as plt
#    import seaborn as sns
#    from aux_gps import keep_iqr
#    if mlh_all is None:
#        mlhs = []
#        crits = [0.25, 0.33, 0.5, 0.75, 1.0]
#        for crit in crits:
#            print('solving mlh for {} critical RiB value.'.format(crit))
#            mlh = calculate_MLH_time_series_from_all_profiles(RiB, crit=crit,
#                                                              hour=hour,
#                                                              plot=False)
#            mlh = keep_iqr(mlh, dim='sound_time')
#            mlhs.append(mlh)
#
#        mlh_all = xr.concat(mlhs, 'crit')
#        mlh_all['crit'] = crits
#        if cutoff is not None:
#            mlh_all = mlh_all.where(mlh_all >= cutoff)
#    if plot:
#        cmap = sns.color_palette("colorblind", 5)
#        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 9))
#        df_mean = mlh_all.groupby('sound_time.month').mean().to_dataset('crit').to_dataframe()
#        df_mean.plot(color=cmap, style=['-+', '-.', '-', '--', '-.'], ax=ax[0])
#        ax[0].grid()
#        ax[0].set_ylabel('Mean MLH [m]')
#        ax[0].set_title(
#            'Annual mixing layer height from Bet-Dagan radiosonde profiles ({}Z) using RiB method'.format(hour))
#        df_std = mlh_all.groupby('sound_time.month').std().to_dataset('crit').to_dataframe()
#        df_std.plot(color=cmap, style=['-+', '-.', '-', '--', '-.'], ax=ax[1])
#        ax[1].grid()
#        ax[1].set_ylabel('Std MLH [m]')
#        df_count = mlh_all.groupby('sound_time.month').count().to_dataset('crit').to_dataframe()
#        df_count.plot(color=cmap, style=['-+', '-.', '-', '--', '-.'], ax=ax[2])
#        ax[2].grid()
#        ax[2].set_ylabel('Count MLH [#]')
#        fig.tight_layout()
#    return mlh_all


def scatter_plot_MLH_PWV(ds, season='JJA', crit=0.25):
    import matplotlib.pyplot as plt
    if season is not None:
        ds = ds.sel(sound_time=ds['sound_time.season'] == season)
    ds = ds.sel(crit=crit)
    hour = list(set(ds['sound_time'].dt.hour.values))[0]
    days = ds['PWV_MLH'].dropna('sound_time').size
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ds['PWV_MLH'], ds['MLH'], alpha=0.75, marker='o')
    ax.scatter(ds['PWV_max']-ds['PWV_MLH'], ds['MLH'], alpha=0.75, marker='s')
    ax.grid()
    ax.set_xlabel('PWV [mm]')
    ax.set_ylabel('MLH [m]')
    ax.set_title(
        'MLH from Bet_Dagan profiles (RiB={},{}Z) in {} vs. PWV ({} days)'.format(
            crit, hour,
            season, days))
    ax.legend(['PWV below MLH', 'PWV above MLH'])
    return fig


def process_all_MLH_with_PWV(MLH_all, PWV):
    import xarray as xr
    mlhs = []
    for crit in MLH_all.crit:
        print('proccesing mlh-pwv for {} critical RiB value.'.format(crit.item()))
        ds = return_PWV_with_MLH_values(PWV, MLH_all.sel(crit=crit))
        mlhs.append(ds)
    ds = xr.concat(mlhs, 'crit')
    ds['crit'] = MLH_all.crit
    return ds


def return_PWV_with_MLH_values(PW, MLH, dim='sound_time'):
    import xarray as xr
    pws = []
    pw_max = []
    MLH = MLH.dropna(dim)
    for time in MLH[dim]:
        pws.append(PW.sel({dim: time}).sel(Height=MLH.sel({dim: time})))
        pw_max.append(PW.sel({dim: time}).max())
    pw_da = xr.concat(pws, dim)
    pw_da_max = xr.concat(pw_max, dim)
    ds = xr.Dataset()
    ds['PWV_MLH'] = pw_da
    ds['PWV_max'] = pw_da_max
    ds['MLH'] = MLH
    return ds


def wrap_xr_metpy_brunt_vaisala_f2(Height, PT, axis=0, verbose=False):
    from metpy.calc import brunt_vaisala_frequency_squared
    from metpy.units import units
    try:
        PT_unit = PT.attrs['units']
        assert PT_unit == 'K'
    except KeyError:
        PT_unit = 'K'
        if verbose:
            print('assuming potential temperature units are degree kelvin...')
    PT_values = PT.values * units('kelvin')
    try:
        H_unit = Height.attrs['units']
        assert H_unit == 'm'
    except KeyError:
        H_unit = 'm'
        if verbose:
            print('assuming Height units are m...')
    H_values = Height.values * units('m')
    bvf2 = brunt_vaisala_frequency_squared(H_values, PT_values, axis=axis)
    da = PT.copy(data=bvf2.magnitude)
    da.name = 'BVF2'
    da.attrs['units'] = '1/sec**2'
    da.attrs['long_name'] = 'Brunt-Vaisala Frequency squared'
    return da


def wrap_xr_metpy_virtual_temperature(T, MR, verbose=False):
    from metpy.calc import virtual_temperature
    from metpy.units import units
    try:
        T_unit = T.attrs['units']
        assert T_unit == 'degC'
    except KeyError:
        T_unit = 'degC'
        if verbose:
            print('assuming temperature units are degree celsius...')
    # convert to Kelvin:
    T_values = T.values + 273.15
    T_values = T_values * units('K')
    try:
        MR_unit = MR.attrs['units']
        assert MR_unit == 'kg/kg'
    except KeyError:
        MR_unit = 'kg/kg'
        if verbose:
            print('assuming mixing ratio units are gr/kg...')
    MR_values = MR.values * units(MR_unit)
    Theta = virtual_temperature(T_values, MR_values)
    da = MR.copy(data=Theta.magnitude) #/ 1000  # fixing for g/kg
    da.name = 'VPT'
    da.attrs['units'] = 'K'
    da.attrs['long_name'] = 'Virtual Potential Temperature'
    return da


def wrap_xr_metpy_virtual_potential_temperature(P, T, MR, verbose=False):
    from metpy.calc import virtual_potential_temperature
    from metpy.units import units
    try:
        P_unit = P.attrs['units']
        assert P_unit == 'hPa'
    except KeyError:
        P_unit = 'hPa'
        if verbose:
            print('assuming pressure units are hpa...')
    P_values = P.values * units(P_unit)
    try:
        T_unit = T.attrs['units']
        assert T_unit == 'degC'
    except KeyError:
        T_unit = 'degC'
        if verbose:
            print('assuming temperature units are degree celsius...')
    # convert to Kelvin:
    T_values = T.values + 273.15
    T_values = T_values * units('K')
    try:
        MR_unit = MR.attrs['units']
        assert MR_unit == 'kg/kg'
    except KeyError:
        MR_unit = 'kg/kg'
        if verbose:
            print('assuming mixing ratio units are gr/kg...')
    MR_values = MR.values * units(MR_unit)
    Theta = virtual_potential_temperature(P_values, T_values, MR_values)
    da = P.copy(data=Theta.magnitude)#   / 1000  # fixing for g/kg
    da.name = 'VPT'
    da.attrs['units'] = 'K'
    da.attrs['long_name'] = 'Virtual Potential Temperature'
    return da


def wrap_xr_metpy_potential_temperature(P, T, verbose=False):
    from metpy.calc import potential_temperature
    from metpy.calc import exner_function
    from metpy.units import units
    try:
        P_unit = P.attrs['units']
        assert P_unit == 'hPa'
    except KeyError:
        P_unit = 'hPa'
        if verbose:
            print('assuming pressure units are hpa...')
    P_values = P.values * units(P_unit)
#    try:
#        T_unit = T.attrs['units']
#        assert T_unit == 'degC'
#    except KeyError:
#        T_unit = 'degC'
#        if verbose:
#            print('assuming temperature units are degree celsius...')
#     convert to Kelvin:
#    T_values = T.values + 273.15
#    T_values = T_values * units('K')
#    Theta = potential_temperature(P_values, T)
    Theta = T / exner_function(P_values)
    da = P.copy(data=Theta.values)
    da.name = 'PT'
    da.attrs['units'] = T.attrs['units']
    da.attrs['long_name'] = 'Potential Temperature'
    return da


def wrap_xr_metpy_vapor_pressure(P, MR, verbose=False):
    from metpy.calc import vapor_pressure
    from metpy.units import units
    try:
        P_unit = P.attrs['units']
        assert P_unit == 'hPa'
    except KeyError:
        P_unit = 'hPa'
        if verbose:
            print('assuming pressure units are hPa...')
    try:
        MR_unit = MR.attrs['units']
        assert MR_unit == 'kg/kg'
    except KeyError:
        MR_unit = 'kg/kg'
        if verbose:
            print('assuming mixing ratio units are kg/kg...')
    P_values = P.values * units(P_unit)
    MR_values = MR.values * units(MR_unit)
    VP = vapor_pressure(P_values, MR_values)
    da = P.copy(data=VP.magnitude)
    da.attrs['units'] = P_unit
    da.attrs['long_name'] = 'Water vapor partial pressure'
    return da


def wrap_xr_metpy_mixing_ratio(P, T, RH, verbose=False):
    from metpy.calc import mixing_ratio_from_relative_humidity
    import numpy as np
    from metpy.units import units
    if np.max(RH) > 1.2:
        RH_values = RH.values / 100.0
    else:
        RH_values = RH.values * units('dimensionless')
    try:
        T_unit = T.attrs['units']
        assert T_unit == 'degC'
    except KeyError:
        T_unit = 'degC'
        if verbose:
            print('assuming temperature units are degC...')
        T.attrs['units'] = T_unit
    try:
        P_unit = P.attrs['units']
        assert P_unit == 'hPa'
    except KeyError:
        P_unit = 'hPa'
        if verbose:
            print('assuming pressure units are hPa...')
    T_values = T.values * units(T_unit)
    P_values = P.values * units(P_unit)
    mixing_ratio = mixing_ratio_from_relative_humidity(
        RH_values, T_values, P_values)
    da = T.copy(data=mixing_ratio.magnitude)
    da.name = 'MR'
    da.attrs['units'] = 'kg/kg'
    da.attrs['long_name'] = 'Water vapor mass mixing ratio'
    return da


def wrap_xr_metpy_density(P, T, MR, verbose=False):
    from metpy.calc import density
    from metpy.units import units
    try:
        MR_unit = MR.attrs['units']
    except KeyError:
        MR_unit = 'g/kg'
        if verbose:
            print('assuming mixing ratio units are g/kg...')
        MR.attrs['units'] = MR_unit
    try:
        T_unit = T.attrs['units']
        assert T_unit == 'degC'
    except KeyError:
        T_unit = 'degC'
        if verbose:
            print('assuming temperature units are degC...')
        T.attrs['units'] = T_unit
    try:
        P_unit = P.attrs['units']
        assert P_unit == 'hPa'
    except KeyError:
        P_unit = 'hPa'
        if verbose:
            print('assuming pressure units are hPa...')
    T_values = T.values * units(T_unit)
    P_values = P.values * units(P_unit)
    MR_values = MR.values * units(MR_unit)
    Rho = density(P_values, T_values, MR_values)
    Rho = Rho.to('g/m^3')
    da = P.copy(data=Rho.magnitude)
    da.attrs['units'] = 'g/m^3'
    da.attrs['long_name'] = 'Air density'
    return da


def wrap_xr_metpy_dewpoint(T, RH, verbose=False):
    import numpy as np
    from metpy.calc import dewpoint_from_relative_humidity
    from metpy.units import units
    if np.max(RH) > 1.2:
        RH_values = RH.values / 100.0
    else:
        RH_values = RH.values
    try:
        T_unit = T.attrs['units']
        assert T_unit == 'degC'
    except KeyError:
        T_unit = 'degC'
        if verbose:
            print('assuming temperature units are degC...')
        T.attrs['units'] = T_unit
    T_values = T.values * units(T_unit)
    dewpoint = dewpoint_from_relative_humidity(T_values, RH_values)
    da = T.copy(data=dewpoint.magnitude)
    da.attrs['units'] = T_unit
    da.attrs['long_name'] = 'Dew point'
    return da


class Constants:
    def __init__(self):
        import astropy.units as u
        # Specific gas const for water vapour, J kg^{-1} K^{-1}:
        self.Rs_v = 461.52 * u.joule / (u.kilogram * u.Kelvin)
        # Specific gas const for dry air, J kg^{-1} K^{-1}:
        self.Rs_da = 287.05 * u.joule / (u.kilogram * u.Kelvin)
        self.MW_dry_air = 28.9647  * u.gram / u.mol  # gr/mol
        self.MW_water = 18.015 * u.gram / u.mol  # gr/mol
        self.Water_Density = 1000.0 * u.kilogram / u.m**3
        self.Epsilon = self.MW_water / self.MW_dry_air  # Epsilon=Rs_da/Rs_v;

    def show(self):
        from termcolor import colored
        for attr, value in vars(self).items():
            print(colored('{} : '.format(attr), color='blue', attrs=['bold']), end='')
            print(colored('{:.2f}'.format(value), color='white', attrs=['bold']))


class WaterVaporVar:
    def __init__(self, name, value, unit):
        try:
            setattr(self, name, value.values)
        except AttributeError:
            setattr(self, name, value)
        if value is not None:
            value = getattr(self, name)
            setattr(self, name, value * unit)

class WaterVapor:
    def __init__(self,
                 Z=None,
                 P=None,
                 T=None,
                 DWPT=None,
                 MR=None,
                 Q=None,
                 RH=None,
                 PPMV=None,
                 MD=None,
                 KMOL=None,
                 ND=None,
                 PP=None,
                 verbose=True):
        import astropy.units as u
        self.verbose = verbose
        self.Z = WaterVaporVar('Z', Z, u.meter).Z  # height in meters
        self.P = WaterVaporVar('P', P, u.hPa).P  # pressure in hPa
        self.T = WaterVaporVar('T', T, u.deg_C).T  # temperature in deg_C
        self.DWPT = WaterVaporVar('DWPT', DWPT, u.deg_C).DWPT  # dew_point in deg_C
        self.MR = WaterVaporVar('MR', MR, u.gram / u.kilogram).MR  # mass_mixing_ratio in gr/kg
        self.Q = WaterVaporVar('Q', Q, u.gram / u.kilogram).Q  # specific humidity in gr/kg
        self.RH = WaterVaporVar('RH', RH, u.percent).RH  # relative humidity in %
        self.PPMV = WaterVaporVar('PPMV', PPMV, u.cds.ppm).PPMV  # volume_mixing_ratio in ppm
        self.MD = WaterVaporVar('MD', MD, u.gram / u.meter**3).MD  # water vapor density in gr/m^3
        self.KMOL = WaterVaporVar('KMOL', KMOL, u.kilomole / u.cm**2).KMOL  # water vapor column density in kmol/cm^2
        self.ND = WaterVaporVar('ND', ND, u.dimensionless_unscaled / u.m**3).ND  # number density in molecules / m^3
        self.PP = WaterVaporVar('PP', PP, u.hPa).PP  # water vapor partial pressure in hPa
    # update attrs from dict containing keys as attrs and vals as attrs vals
    # to be updated

    def from_dict(self, d):
        self.__dict__.update(d)
        return self

    def show(self, name='all'):
        from termcolor import colored
        if name == 'all':
            for attr, value in vars(self).items():
                print(colored('{} : '.format(attr), color='blue', attrs=['bold']), end='')
                print(colored(value, color='white', attrs=['bold']))
        elif hasattr(self, name):
            print(colored('{} : '.format(name), color='blue', attrs=['bold']), end='')
            print(colored(self.name, color='white', attrs=['bold']))

    def convert(self, from_to='PP_to_MR'):
        import astropy.units as u
        C = Constants()
        from_name = from_to.split('_')[0]
        to_name = from_to.split('_')[-1]
        from_ = getattr(self, from_name)
        to_ = getattr(self, to_name)
        print('converting {} to {}:'.format(from_name, to_name))
        if to_ is not None:
            if self.verbose:
                print('{} already exists, overwriting...'.format(to_name))
        if from_ is not None:
            if from_name == 'PP' and to_name == 'MR':
                # convert wv partial pressure to mass mixing ratio:
                if self.P is None:
                    raise Exception('total pressure is needed for this conversion')
                self.MR = C.Epsilon * self.PP / (self.P - self.PP) / self.MR.unit.decompose()
                return self.MR
            elif from_name == 'MR' and to_name == 'PP':
                # convert mass mixing ratio to wv partial pressure:
                if self.P is None:
                    raise Exception('total pressure is needed for this conversion')
                e_tag = self.MR * self.MR.unit.decompose() / (C.Epsilon)
                self.PP = self.P * e_tag / (1. * e_tag.unit + e_tag)
                return self.PP
        else:
            raise Exception('{} is needed to perform conversion'.format(from_))
        return self


def check_sound_time_datetime(dt):
    import pandas as pd
    if dt.hour >= 21 and dt.hour <= 23:
        sound_time = (
            dt +
            pd.Timedelta(
                1,
                unit='d')).replace(
            hour=0,
            minute=0,
            second=0)
    elif dt.hour >= 0 and dt.hour <= 2:
        sound_time = dt.replace(hour=0, minute=0, second=0)
    elif dt.hour >= 9 and dt.hour <= 14:
        sound_time = dt.replace(hour=12, minute=0, second=0)
    else:
        raise ValueError('{} time is not midnight nor noon'.format(dt))
    return sound_time


def check_sound_time(df):
    import pandas as pd
    # check for validity of radiosonde air time, needs to be
    # in noon or midnight:
    if not df.between_time('22:00', '02:00').empty:
        if df.index[0].hour <= 23 and df.index[0].hour >= 21:
            sound_time = pd.to_datetime((df.index[0] + pd.Timedelta(1, unit='d')).strftime('%Y-%m-%d')).replace(hour=0, minute=0)
        else:
            sound_time = pd.to_datetime(df.index[0].strftime('%Y-%m-%d')).replace(hour=0, minute=0)
    elif not df.between_time('10:00', '14:00').empty:
        sound_time = pd.to_datetime(
            df.index[0].strftime('%Y-%m-%d')).replace(hour=12, minute=0)
    elif (df.between_time('22:00', '02:00').empty and
          df.between_time('10:00', '14:00').empty):
        raise ValueError(
            '{} time is not midnight nor noon'.format(
                df.index[0]))
    return sound_time


#def calculate_tm_edt(wvpress, tempc, height, mixratio,
#                     press, method='mr_trapz'):
#    def calculate_tm_with_method(method='mr_trapz'):
#        import numpy as np
#        import pandas as pd
#        nonlocal wvpress, tempc, height, mixratio, press
#        # conver Mixing ratio to WV-PP(e):
#        MW_dry_air = 28.9647  # gr/mol
#        MW_water = 18.015  # gr/mol
#        Epsilon = MW_water / MW_dry_air  # Epsilon=Rs_da/Rs_v;
#        mr = pd.Series(mixratio)
#        p = pd.Series(press)
#        eps_tag = mr / (1000.0 * Epsilon)
#        e = p * eps_tag / (1.0 + eps_tag)
#        try:
#            # directly use WV-PP with trapz:
#            if method == 'wv_trapz':
#                numerator = np.trapz(
#                    wvpress /
#                    (tempc +
#                     273.15),
#                    height)
#                denominator = np.trapz(
#                    wvpress / (tempc + 273.15)**2.0, height)
#                tm = numerator / denominator
#            # use WV-PP from mixing ratio with trapz:
#            elif method == 'mr_trapz':
#                numerator = np.trapz(
#                    e /
#                    (tempc +
#                     273.15),
#                    height)
#                denominator = np.trapz(
#                    e / (tempc + 273.15)**2.0, height)
#                tm = numerator / denominator
#            # use WV-PP from mixing ratio with sum:
#            elif method == 'mr_sum':
#                e_ser = pd.Series(e)
#                height = pd.Series(height)
#                e_sum = (e_ser.shift(-1) + e_ser).dropna()
#                h = height.diff(-1).abs()
#                numerator = 0.5 * (e_sum * h / (pd.Series(tempc) + 273.15)).sum()
#                denominator = 0.5 * \
#                    (e_sum * h / (pd.Series(tempc) + 273.15)**2.0).sum()
#                tm = numerator / denominator
#        except ValueError:
#            return np.nan
#        return tm
#    if method is None:
#        tm_wv_trapz = calculate_tm_with_method('wv_trapz')
#        tm_mr_trapz = calculate_tm_with_method('mr_trapz')
#        tm_sum = calculate_tm_with_method('sum')
#    else:
#        tm = calculate_tm_with_method(method)
#        chosen_method = method
#    return tm, chosen_method
#
#
#def calculate_tpw_edt(mixratio, rho, rho_wv, height, press, g, method='trapz'):
#    def calculate_tpw_with_method(method='trapz'):
#        nonlocal mixratio, rho, rho_wv, height, press, g
#        import numpy as np
#        import pandas as pd
#        # calculate specific humidity q:
#        q = (mixratio / 1000.0) / \
#            (1 + 0.001 * mixratio / 1000.0)
#        try:
#            if method == 'trapz':
#                tpw = np.trapz(q * rho,
#                               height)
#            elif method == 'sum':
#                rho_wv = pd.Series(rho_wv)
#                height = pd.Series(height)
#                rho_sum = (rho_wv.shift(-1) + rho_wv).dropna()
#                h = height.diff(-1).abs()
#                tpw = 0.5 * (rho_sum * h).sum()
#            elif method == 'psum':
#                q = pd.Series(q)
#                q_sum = q.shift(-1) + q
#                p = pd.Series(press) * 100.0  # to Pa
#                dp = p.diff(-1).abs()
#                tpw = (q_sum * dp / (2.0 * 1000 * g)).sum() * 1000.0
#        except ValueError:
#            return np.nan
#        return tpw
#    if method is None:
#        # calculate all the methods and choose trapz if all agree:
#        tpw_trapz = calculate_tpw_with_method(method='trapz')
#        tpw_sum = calculate_tpw_with_method(method='sum')
#        tpw_psum = calculate_tpw_with_method(method='psum')
#    else:
#        tpw = calculate_tpw_with_method(method=method)
#        chosen_method = method
#    return tpw, chosen_method


def read_all_deserve_soundings(path=des_path, savepath=None):
    from aux_gps import path_glob
    import xarray as xr
    from aux_gps import save_ncfile
    radio_path = path / 'radiosonde'
    files = path_glob(radio_path, '*.txt')
    mas_files = [x for x in files if 'MAS' in x.as_posix()]
    maz_files = [x for x in files if 'MAZ' in x.as_posix()]
    ds_list = []
    for file in mas_files:
        ds_list.append(read_one_deserve_record(file))
    ds_mas = xr.concat(ds_list, 'sound_time')
    ds_mas = ds_mas.sortby('sound_time')
    ds_list = []
    for file in maz_files:
        ds_list.append(read_one_deserve_record(file))
    ds_maz = xr.concat(ds_list, 'sound_time')
    ds_maz = ds_maz.sortby('sound_time')
    if savepath is not None:
        filename = 'deserve_massada_sounding_2014-2014.nc'
        save_ncfile(ds_mas, savepath, filename)
        filename = 'deserve_mazzra_sounding_2014-2014.nc'
        save_ncfile(ds_maz, savepath, filename)
    return ds_mas, ds_maz


def get_loc_sound_time_from_deserve_filepath(filepath):
    import pandas as pd
    txt_filepath = filepath.as_posix().split('/')[-1]
    loc = txt_filepath.split('_')[0]
    sound_time = txt_filepath.split('_')[1]
    sound_time = pd.to_datetime(sound_time, format='%Y%m%d%H')
    return loc, sound_time


def read_one_deserve_record(filepath):
    import pandas as pd
    loc, sound_time = get_loc_sound_time_from_deserve_filepath(filepath)
    df = pd.read_csv(filepath, header=None, skiprows=1, delim_whitespace=True)
    df.columns = [
        'time',
        'P',
        'T',
        'RH',
        'WS',
        'WD',
        'lon',
        'lat',
        'alt',
        'Dewpt',
        'vi_te']
    units = [
        'mm:ss',
        'hPa',
        'degC',
        '%',
        'm/s',
        'deg',
        'deg',
        'deg',
        'm',
        'degC',
        'degC']
    units_dict = dict(zip(df.columns, units))
    df['time'] = '00:' + df['time'].astype(str)
    df['time'] = pd.to_timedelta(df['time'], errors='coerce', unit='sec')
    df['time'] = df['time'].dt.total_seconds()
    df.set_index('time', inplace=True)
    ds = df.to_xarray()
    ds = ds.sortby('time')
    for key, val in units_dict.items():
        try:
            ds[key].attrs['units'] = val
        except KeyError:
            pass
    long_names = {'T': 'Air temperature', 'Dewpt': 'Dew point', 'RH': 'Relative humidity',
                  'WS': 'Wind speed', 'WD': 'Wind direction',
                  'P': 'Air pressure', 'lon': 'Longitude', 'lat': 'Latitude',
                  'alt': 'Altitude a.s.l'}
    for var, long_name in long_names.items():
        ds[var].attrs['long_name'] = long_name
    ds['sound_time'] = sound_time
    ds['location'] = loc
    return ds


def read_one_EDT_record(filepath, year=None):
    import pandas as pd
    import numpy as np
    import xarray as xr
#    from aux_gps import get_unique_index
    from aux_gps import remove_duplicate_spaces_in_string
    from scipy.integrate import cumtrapz
    with open(filepath) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    meta = {}
    for i, line in enumerate(content):
        if line.startswith('Station:'):
            line = remove_duplicate_spaces_in_string(line)
            meta['station'] = line.split(' ')[1]
        if line.startswith('Launch time:'):
            line = remove_duplicate_spaces_in_string(line)
            dt = 'T'.join(line.split(' ')[2:4])
            meta['launch_time'] = pd.to_datetime(dt)
        if line.startswith('RS type:'):
            line = remove_duplicate_spaces_in_string(line)
            rs_type = line.split(' ' )[-1]
            meta['RS_type'] = rs_type
        if line.startswith('RS number:'):
            line = remove_duplicate_spaces_in_string(line)
            rs_num = line.split(' ' )[-1]
            meta['RS_number'] = rs_num
        if line.startswith('Reason for termination:'):
            line = remove_duplicate_spaces_in_string(line)
            termination = line.split(' ' )[-1]
            meta['termination_reason'] = termination
        if line.startswith('Record name:'):
            meta_header = remove_duplicate_spaces_in_string(line)
            meta_header = meta_header.split(': ')
            meta_header[-1] = meta_header[-1].split(':')[0]
        if i >42:
            break
    sound_time = check_sound_time_datetime(meta['launch_time'])
    # select only one year:
    if year is not None:
        if sound_time.year != year:
            return None
    # read units table:        global mixratio, rho, rho_wv, height, press, g

    table = pd.read_fwf(
        filepath,
        skiprows=22,
        nrows=19,
        header=None,
        widths=[16, 16, 23, 9, 9])
    table.columns = meta_header
    # drop 3 last rows:
    table = table.iloc[:-3, :]
    # units dict:
    units_dict = dict(zip(table['Record name'], table['Unit']))
    units_dict.pop('Pscl')
    # read all data:
    data = pd.read_csv(
        filepath,
        delim_whitespace=True,
        skiprows=44,
        na_values=-32768)
    # drop 3 last cols:
    data = data.iloc[:, :-3]
    data = data.drop('Pscl', axis=1)
    # datetime index:
    # Time = data['time'].copy(deep=True)
    data['time'] = pd.to_timedelta(data['time'], errors='coerce', unit='sec')
#    data['time'] += meta['launch_time']
    data['time'] = data['time'].dt.total_seconds()
    data.set_index('time', inplace=True)
#    sound_time = check_sound_time(data)
#    # now index as height:
#    data.set_index('Height', inplace=True)
    # data['Time'] = Time
    data = data.astype('float')
    data.index = data.index.astype('float')
    # to xarray:
    ds = data.to_xarray()
#    return ds
#    ds = ds.dropna('Height')
    ds = ds.sortby('time')
#    ds = get_unique_index(ds, 'Height')
    # interpolate:
    new_time = np.arange(0, float(int(ds.time.max())))
#    ds = ds.interpolate_na(
#        'time',
#        method='linear',
#        max_gap=3,
#        fill_value='extrapolate')
#    ds = ds.interp(coords={'time': new_time}, method='cubic')
    ds_list = []
    for da in ds.data_vars.values():
        # dropna in all vars:
        dropped = da.dropna('time')
        # if some data remain, interpolate:
        if dropped.size > 0:
            interpolated = dropped.interp(coords={'time': new_time}, method='cubic',
                                          kwargs={'fill_value': np.nan})
            ds_list.append(interpolated)
        # if nothing left, create new ones with nans:
        else:
            nan_da = np.nan * ds['time']
            nan_da.name = dropped.name
            ds_list.append(dropped)
    ds = xr.merge(ds_list)
    ds['Height'].attrs['units'] = 'm'
#    ds['time'].attrs['units'] = 'sec'
    ds.attrs['station'] = meta['station']
    # copy unit attrs:
    for key, val in units_dict.items():
        try:
            ds[key].attrs['units'] = val
        except KeyError:
            pass
    ds = ds.rename({'Lat': 'lat', 'Lon': 'lon', 'Range': 'range'})
    # add more variables: Tm, Tpw, etc...
#    ds['east_distance'], ds['north_distance'] = calculate_edt_north_east_distance(
#        ds['lat'],ds['lon'], method='fast')
    ds['T'] -= 273.15
    ds['T'].attrs['units'] = 'degC'
    # rename for consistency with phys and PTU data:
    ds = ds.rename({'TD': 'Dewpt', 'FF': 'WS', 'DD': 'WD'})
    ds['Dewpt'] -= 273.15
    ds['Dewpt'].attrs['units'] = 'degC'
    long_names = {'T': 'Air temperature', 'Dewpt': 'Dew point', 'RH': 'Relative humidity',
                  'v': 'v component of wind velocity', 'u': 'u component of wind velocity',
                  'P': 'Air pressure', 'MR': 'Water vapor mass mixing ratio',
                  'WS': 'Wind speed', 'lon': 'Longitude', 'lat': 'Latitude'}
    for var, long_name in long_names.items():
        ds[var].attrs['long_name'] = long_name
    # fix MR:
#    ds['MR'] /= 1000.0
#    ds['Rho'] = wrap_xr_metpy_density(ds['P'], ds['T'], ds['MR'])
#    ds['Rho'].attrs['long_name'] = 'Air density'
#    ds['Q'] = wrap_xr_metpy_specific_humidity(ds['MR'])
#    ds['Q'].attrs['long_name'] = 'Specific humidity'
#    ds['VP'] = wrap_xr_metpy_vapor_pressure(ds['P'], ds['MR'])
#    ds['VP'].attrs['long_name'] = 'Water vapor partial pressure'
#    ds['Rho_wv'] = calculate_absolute_humidity_from_partial_pressure(
#        ds['VP'], ds['T'])
#    pw = cumtrapz((ds['Q']*ds['Rho']).fillna(0), ds['Height'], initial=0)
#    ds['PW'] = xr.DataArray(pw, dims=['time'])
#    ds['PW'].attrs['units'] = 'mm'
#    ds['PW'].attrs['long_name'] = 'Precipitable water'
#    tm = calculate_tm_via_trapz_height(
#        ds['VP'],
#        ds['T'],
#        ds['Height'])
#    ds['VPT'] = wrap_xr_metpy_virtual_potential_temperature(ds['P'], ds['T'],
#                                                            ds['MR'])
#    ds['PT'] = wrap_xr_metpy_potential_temperature(ds['P'], ds['T'])
#    ds['VT'] = wrap_xr_metpy_virtual_temperature(ds['T'], ds['MR'])
#    U, V = convert_wind_speed_direction_to_zonal_meridional(ds['WS'], ds['DD'])
#    ds['U'] = U
#    ds['V'] = V
#    ds['N'] = calculate_atmospheric_refractivity(ds['P'], ds['T'], ds['VP'])
#    ds['Tm'] = xr.DataArray(tm, dims=['time'])
#    ds['Tm'].attrs['units'] = 'K'
#    ds['Tm'].attrs['long_name'] = 'Water vapor mean air temperature'
#    ds['Ts'] = ds['T'].dropna('time').values[0] + 273.15
#    ds['Ts'].attrs['units'] = 'K'
#    ds['Ts'].attrs['long_name'] = 'Surface temperature'
    ds['sound_time'] = sound_time
    ds['min_time'] = meta['launch_time']
    ds['max_time'] = ds['min_time'] + pd.Timedelta(ds['time'][-1].item(), unit='s')
    ds['sonde_type'] = meta['RS_type']
    ds['RS_number'] = meta['RS_number']
    ds['termination_reason'] = meta['termination_reason']
    # convert to timedelta for resampling purposes:
    ds['time'] = pd.to_timedelta(ds['time'].values, unit='sec')
    return ds


def transform_model_levels_to_pressure(path, field_da, plevel=85.0, mm=True):
    """takes a field_da (like t, u) that uses era5 L137 model levels and
    interpolates it using pf(only monthly means) to desired plevel"""
    # TODO: do scipy.interpolate instead of np.interp (linear)
    # TODO: add multiple plevel support
    import xarray as xr
    from aux_functions_strat import dim_intersection
    import numpy as np
    if mm:
        pf = xr.open_dataset(path / 'era5_full_pressure_mm_1979-2018.nc')
    pf = pf.pf
    levels = dim_intersection([pf, field_da], dropna=False, dim='level')
    pf = pf.sel(level=levels)
    field_da = field_da.sel(level=levels)
    field_da = field_da.transpose('time', 'latitude', 'longitude', 'level')
    field = field_da.values
    da = np.zeros((pf.time.size, pf.latitude.size, pf.longitude.size, 1),
                  'float32')
    pf = pf.values
    for t in range(pf.shape[0]):
        print(t)
        for lat in range(pf.shape[1]):
            for lon in range(pf.shape[2]):
                pressures = pf[t, lat, lon, :]
                vals = field[t, lat, lon, :]
                da[t, lat, lon, 0] = np.interp(plevel, pressures, vals)
    da = xr.DataArray(da, dims=['time', 'lat', 'lon', 'level'])
    da['level'] = [plevel]
    da['time'] = field_da['time']
    da['level'].attrs['long_name'] = 'pressure_level'
    da['level'].attrs['units'] = 'hPa'
    da['lat'] = field_da['latitude'].values
    da['lat'].attrs = field_da['latitude'].attrs
    da['lon'] = field_da['longitude'].values
    da['lon'].attrs = field_da['longitude'].attrs
    da.name = field_da.name
    da.attrs = field_da.attrs
    da = da.sortby('lat')
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in da.to_dataset(name=da.name).data_vars}
    filename = 'era5_' + da.name + '_' + str(int(plevel)) + 'hPa.nc'
    da.to_netcdf(path / filename, encoding=encoding)
    return da


def create_model_levels_map_from_surface_pressure(work_path):
    import numpy as np
    import xarray as xr
    """takes ~24 mins for monthly mean with 1.25x1.25 degree for full 137
    levels,~=2.9GB file. Don't try with higher sample rate"""
    ds = read_L137_to_ds(work_path)
    a = ds.a.values
    b = ds.b.values
    n = ds.n.values
    sp = xr.open_dataset(work_path / 'era5_SP_mm_1979-2018.nc')
    sp = sp.sp / 100.0
    pf = np.zeros((sp.time.size, sp.latitude.size, sp.longitude.size, len(a)),
                  'float32')
    sp_np = sp.values
    for t in range(sp.time.size):
        print(t)
        for lat in range(sp.latitude.size):
            for lon in range(sp.longitude.size):
                ps = sp_np[t, lat, lon]
                pf[t, lat, lon, :] = pressure_from_ab(ps, a, b, n)
    pf_da = xr.DataArray(pf, dims=['time', 'latitude', 'longitude', 'level'])
    pf_da.attrs['units'] = 'hPa'
    pf_da.attrs['long_name'] = 'full_pressure_level'
    pf_ds = pf_da.to_dataset(name='pf')
    pf_ds['level'] = n
    pf_ds['level'].attrs['long_name'] = 'model_level_number'
    pf_ds['latitude'] = sp.latitude
    pf_ds['latitude'].attrs = sp.latitude.attrs
    pf_ds['longitude'] = sp.longitude
    pf_ds['longitude'].attrs = sp.longitude.attrs
    pf_ds['time'] = sp.time
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in pf_ds.data_vars}
    pf_ds.to_netcdf(work_path / 'era5_full_pressure_mm_1979-2018.nc',
                    encoding=encoding)
    return pf_ds


def calculate_era5_tm(q_ml, t_ml, pf_ml, g=9.79):
    """given pressure in model levels(ps_ml) and specific humidity(q_ml)
    in era5 model levels and temperature, compute the tm for each lat/lon grid point."""
    assert t_ml.attrs['units'] == 'K'
    assert q_ml.attrs['units'] == 'kg kg**-1'
    assert pf_ml.attrs['units'] == 'Pa'
    factor = g * 2   # g in mks
    WVpress = VaporPressure(t_ml - 273.15, method='Buck', units='Pa')
    wvpress = WVpress + WVpress.shift(level=1)
    t = t_ml + t_ml.shift(level=1)
    t2 = t_ml**2.0 + t_ml.shift(level=1)**2.0
    Rho = DensHumid(t_ml - 273.15, pf_ml / 100.0, WVpress / 100.0, out='both')
    rho = Rho + Rho.shift(level=1)
    nume = ((wvpress * pf_ml.diff('level')) / (t * rho * factor)).sum('level')
    denom = ((wvpress * pf_ml.diff('level')) / (t2 * rho * factor)).sum('level')
    tm = nume / denom
    tm.name = 'Tm'
    tm.attrs['units'] = 'K'
    return tm


def calculate_era5_pw(q_ml, pf_ml, g=9.79):
    """given pressure in model levels(ps_ml) and specific humidity(q_ml)
    in era5 model levels, compute the IWV for each lat/lon grid point."""
    assert q_ml.attrs['units'] == 'kg kg**-1'
    assert pf_ml.attrs['units'] == 'Pa'
    factor = g * 2 * 1000  # density of water in mks, g in mks
    pw = (((q_ml + q_ml.shift(level=1)) * pf_ml.diff('level')) / factor).sum('level')
    pw = pw * 1000.0
    pw.name = 'PW'
    pw.attrs['units'] = 'mm'
    return pw


def pressure_from_ab(ps_da, ds_l137):
    """takes the surface pressure(hPa) and  a, b coeffs from L137(n) era5
    table and produces the pressure level point in hPa"""
    try:
        unit = ps_da.attrs['units']
    except AttributeError:
        # assume Pascals in ps
        unit = 'Pa'
    a = ds_l137.a
    b = ds_l137.b
    if unit == 'Pa':
        ph = a + b * ps_da
    pf_da = 0.5 * (ph + ph.shift(n=1))
    pf_da.name = 'pressure'
    pf_da.attrs['units'] = unit
    pf_da = pf_da.dropna('n')
    pf_da = pf_da.rename({'n': 'level'})
    return pf_da


def read_L137_to_ds(path):
    import pandas as pd
    l137_df = pd.read_csv(path / 'L137_model_levels_1976_climate.txt',
                          header=None, delim_whitespace=True, na_values='-')
    l137_df.columns = ['n', 'a', 'b', 'ph', 'pf', 'Geopotential Altitude',
                       'Geometric Altitude', 'Temperature', 'Density']
    l137_df.set_index('n')
    l137_df.drop('n', axis=1, inplace=True)
    l137_df.index.name = 'n'
    ds = l137_df.to_xarray()
    ds.attrs['long_name'] = 'L137 model levels and 1976 ICAO standard atmosphere 1976'
    ds.attrs['surface_pressure'] = 1013.250
    ds.attrs['pressure_units'] = 'hPa'
    ds.attrs['half_pressure_formula'] = 'ph(k+1/2) = a(k+1/2) + ps*b(k+1/2)'
    ds.attrs['full_pressure_formula'] = 'pf(k) = 1/2*(ph(k-1/2) + ph(k+1/2))'
    ds['n'].attrs['long_name'] = 'model_level_number'
    ds['a'].attrs['long_name'] = 'a_coefficient'
    ds['a'].attrs['units'] = 'Pa'
    ds['b'].attrs['long_name'] = 'b_coefficient'
    ds['ph'].attrs['long_name'] = 'half_pressure_level'
    ds['ph'].attrs['units'] = 'hPa'
    ds['pf'].attrs['long_name'] = 'full_pressure_level'
    ds['pf'].attrs['units'] = 'hPa'
    ds['Geopotential Altitude'].attrs['units'] = 'm'
    ds['Geometric Altitude'].attrs['units'] = 'm'
    ds['Temperature'].attrs['units'] = 'K'
    ds['Density'].attrs['units'] = 'kg/m^3'
    return ds


def merge_era5_fields_and_save(path=era5_path, field='SP'):
    from aux_gps import path_glob
    import xarray as xr
    files = path_glob(path, 'era5_{}*.nc'.format(field))
    ds = xr.open_mfdataset(files)
    ds = concat_era5T(ds)
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds = ds.sortby('lat')
    yr_min = ds.time.min().dt.year.item()
    yr_max = ds.time.max().dt.year.item()
    filename = 'era5_{}_israel_{}-{}.nc'.format(field, yr_min, yr_max)
    print('saving {} to {}'.format(filename, path))
    ds.to_netcdf(path / filename, 'w')
    print('Done!')
    return ds


def concat_era5T(ds):
    import xarray as xr
    field_0001 = [x for x in ds.data_vars if '0001' in x]
    field_0005 = [x for x in ds.data_vars if '0005' in x]
    field = [x for x in ds.data_vars if '0001' not in x and '0005' not in x]
    if field_0001 and field_0005:
        da = xr.concat([ds[field_0001[0]].dropna('time'),
                        ds[field_0005[0]].dropna('time')], 'time')
        da.name = field_0001[0].split('_')[0]
    elif not field_0001 and not field_0005:
        return ds
    if field:
        da = xr.concat([ds[field[0]].dropna('time'), da], 'time')
    dss = da.to_dataset(name=field[0])
    return dss


def merge_concat_era5_field(path=era5_path, field='Q', savepath=None):
    import xarray as xr
    from aux_gps import path_glob
    strato_files = path_glob(path, 'era5_{}_*_pl_1_to_150.nc'.format(field))
    strato_list = [xr.open_dataset(x) for x in strato_files]
    strato = xr.concat(strato_list, 'time')
    tropo_files = path_glob(path, 'era5_{}_*_pl_175_to_1000.nc'.format(field))
    tropo_list = [xr.open_dataset(x) for x in tropo_files]
    tropo = xr.concat(tropo_list, 'time')
    ds = xr.concat([tropo, strato], 'level')
    ds = ds.sortby('level', ascending=False)
    ds = ds.sortby('time')
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    if savepath is not None:
        yr_min = ds.time.min().dt.year.item()
        yr_max = ds.time.max().dt.year.item()
        filename = 'era5_{}_israel_{}-{}.nc'.format(field, yr_min, yr_max)
        print('saving {} to {}'.format(filename, savepath))
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds}
        ds.to_netcdf(savepath / filename, 'w', encoding=encoding)
    print('Done!')
    return ds


def calculate_PW_from_era5(path=era5_path, glob_str='era5_Q_israel*.nc',
                           water_density=1000.0, savepath=None):
    import xarray as xr
    from aux_gps import path_glob
    from aux_gps import calculate_g
    import numpy as np
    file = path_glob(path, glob_str)[0]
    Q = xr.open_dataset(file)['q']
    g = calculate_g(Q.lat)
    g.name = 'g'
    g = g.mean('lat')
    plevel_in_pa = Q.level * 100.0
    # P_{i+1} - P_i:
    plevel_diff = np.abs(plevel_in_pa.diff('level'))
    # Q_i + Q_{i+1}:
    Q_sum = Q.shift(level=-1) + Q
    pw_in_mm = ((Q_sum * plevel_diff) /
                (2.0 * water_density * g)).sum('level') * 1000.0
    pw_in_mm.name = 'pw'
    pw_in_mm.attrs['units'] = 'mm'
    if savepath is not None:
        yr_min = pw_in_mm.time.min().dt.year.item()
        yr_max = pw_in_mm.time.max().dt.year.item()
        filename = 'era5_PW_israel_{}-{}.nc'.format(yr_min, yr_max)
        print('saving {} to {}'.format(filename, savepath))
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in pw_in_mm.to_dataset(name='pw')}
        pw_in_mm.to_netcdf(savepath / filename, 'w', encoding=encoding)
    print('Done!')
    return pw_in_mm


def calculate_Tm_from_era5(path=era5_path, Tfile='era5_T_israel*.nc',
                           RHfile='era5_RH_israel*.nc', savepath=None):
    import xarray as xr
    from aux_gps import path_glob
    tfile = path_glob(path, Tfile)
    rhfile = path_glob(path, RHfile)
    T = xr.open_dataarray(tfile)
    RH = xr.open_dataarray(rhfile)
    Dewpt = dewpoint_rh(T, RH)
    WVpress = VaporPressure(Dewpt, units='hPa', method='Buck')
    nom = WVpress / T
    nom = nom.integrate('level')
    denom = WVpress / T ** 2.0
    denom = denom.integrate('level')
    Tm = nom / denom
    Tm.name = 'Tm'
    Tm.attrs['units'] = 'K'
    if savepath is not None:
        yr_min = Tm.time.min().dt.year.item()
        yr_max = Tm.time.max().dt.year.item()
        filename = 'era5_Tm_israel_{}-{}.nc'.format(yr_min, yr_max)
        print('saving {} to {}'.format(filename, savepath))
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in Tm.to_dataset(name='Tm')}
        Tm.to_netcdf(savepath / filename, 'w', encoding=encoding)
    print('Done!')
    return Tm


def calculate_pw_from_physical_with_params(temp, rh, press, height, **params):
    """calculate the pw from radiosonde temp, rh, press, height, params are
    kwargs that hold the parameters for scaling the temp, rh, press, etc.."""
    import numpy as np
    dewpt = dewpoint_rh(temp, rh)
    wvpress = VaporPressure(dewpt, units='hPa', method='Buck')
    mixratio = MixRatio(wvpress, press)  # both in hPa
    # Calculate density of air (accounting for moisture)
    rho = DensHumid(temp, press, wvpress, out='both')
    specific_humidity = (mixratio / 1000.0) / (1 + 0.001 * mixratio / 1000.0)
    pw = np.trapz(specific_humidity * rho, height)
    return pw


def evaluate_sin_to_tmsearies(
        da,
        time_dim='time',
        plot=True,
        params_up={
                'amp': 6.0,
                'period': 365.0,
                'phase': 30.0,
                'offset': 253.0,
                'a': 0.0,
                'b': 0.0},
        func='sine',
        bounds=None,
        just_eval=False):
    import pandas as pd
    import xarray as xr
    import numpy as np
    from aux_gps import dim_intersection
    from scipy.optimize import curve_fit
    from sklearn.metrics import mean_squared_error

    def sine(time, amp, period, phase, offset):
        f = amp * np.sin(2 * np.pi * (time / period + 1.0 / phase)) + offset
        return f

    def sine_on_linear(time, amp, period, phase, offset, a):
        f = a * time / 365.25 + amp * \
            np.sin(2 * np.pi * (time / period + 1.0 / phase)) + offset
        return f

    def sine_on_quad(time, amp, period, phase, offset, a, b):
        f = a * (time / 365.25) ** 2.0 + b * time / 365.25 + amp * \
            np.sin(2 * np.pi * (time / period + 1.0 / phase)) + offset
        return f
    params={'amp': 6.0,
            'period': 365.0,
            'phase': 30.0,
            'offset': 253.0,
            'a': 0.0,
            'b': 0.0}
    params.update(params_up)
    print(params)
    lower = {}
    upper = {}
    if bounds is not None:
        # lower = [(x - y) for x, y in zip(params, perc2)]
        for key in params.keys():
            lower[key] = -np.inf
            upper[key] = np.inf
        lower['phase'] = 0.01
        upper['phase'] = 0.05
        lower['offset'] = 46.2
        upper['offset'] = 46.3
        lower['a'] = 0.0001
        upper['a'] = 0.002
        upper['amp'] = 0.04
        lower['amp'] = 0.02
    else:
        for key in params.keys():
            lower[key] = -np.inf
            upper[key] = np.inf
    lower = [x for x in lower.values()]
    upper = [x for x in upper.values()]
    params = [x for x in params.values()]
    da_no_nans = da.dropna(time_dim)
    time = da_no_nans[time_dim].values
    time = pd.to_datetime(time)
    jul = time.to_julian_date()
    jul -= jul[0]
    jul_with_nans = pd.to_datetime(da[time_dim].values).to_julian_date()
    jul_with_nans -= jul[0]
    ydata = da_no_nans.values
    if func == 'sine':
        print('Model chosen: y = amp * sin (2*pi*(x/T + 1/phi)) + offset')
        if not just_eval:
            popt, pcov = curve_fit(sine, jul, ydata, p0=params[:-2],
                                   bounds=(lower[:-2], upper[:-2]), ftol=1e-9,
                                   xtol=1e-9)
            amp = popt[0]
            period = popt[1]
            phase = popt[2]
            offset = popt[3]
            perr = np.sqrt(np.diag(pcov))
            print('amp: {:.4f} +- {:.2f}'.format(amp, perr[0]))
            print('period: {:.2f} +- {:.2f}'.format(period, perr[1]))
            print('phase: {:.2f} +- {:.2f}'.format(phase, perr[2]))
            print('offset: {:.2f} +- {:.2f}'.format(offset, perr[3]))
        new = sine(jul_with_nans, amp, period, phase, offset)
    elif func == 'sine_on_linear':
        print('Model chosen: y = a * x + amp * sin (2*pi*(x/T + 1/phi)) + offset')
        if not just_eval:
            popt, pcov = curve_fit(sine_on_linear, jul, ydata, p0=params[:-1],
                                   bounds=(lower[:-1], upper[:-1]), xtol=1e-11,
                                   ftol=1e-11)
            amp = popt[0]
            period = popt[1]
            phase = popt[2]
            offset = popt[3]
            a = popt[4]
            perr = np.sqrt(np.diag(pcov))
            print('amp: {:.4f} +- {:.2f}'.format(amp, perr[0]))
            print('period: {:.2f} +- {:.2f}'.format(period, perr[1]))
            print('phase: {:.2f} +- {:.2f}'.format(phase, perr[2]))
            print('offset: {:.2f} +- {:.2f}'.format(offset, perr[3]))
            print('a: {:.7f} +- {:.2f}'.format(a, perr[4]))
        new = sine_on_linear(jul_with_nans, amp, period, phase, offset, a)
    elif func == 'sine_on_quad':
        print('Model chosen: y = a * x^2 + b * x + amp * sin (2*pi*(x/T + 1/phi)) + offset')
        if not just_eval:
            popt, pcov = curve_fit(sine_on_quad, jul, ydata, p0=params,
                                   bounds=(lower, upper))
            amp = popt[0]
            period = popt[1]
            phase = popt[2]
            offset = popt[3]
            a = popt[4]
            b = popt[5]
            perr = np.sqrt(np.diag(pcov))
            print('amp: {:.4f} +- {:.2f}'.format(amp, perr[0]))
            print('period: {:.2f} +- {:.2f}'.format(period, perr[1]))
            print('phase: {:.2f} +- {:.2f}'.format(phase, perr[2]))
            print('offset: {:.2f} +- {:.2f}'.format(offset, perr[3]))
            print('a: {:.7f} +- {:.2f}'.format(a, perr[4]))
            print('b: {:.7f} +- {:.2f}'.format(a, perr[5]))
        new = sine_on_quad(jul_with_nans, amp, period, phase, offset, a, b)
    new_da = xr.DataArray(new, dims=[time_dim])
    new_da[time_dim] = da[time_dim]
    resid = new_da - da
    rmean = np.mean(resid)
    new_time = dim_intersection([da, new_da], time_dim)
    rmse = np.sqrt(mean_squared_error(da.sel({time_dim: new_time}).values,
                                      new_da.sel({time_dim: new_time}).values))
    print('MEAN : {}'.format(rmean))
    print('RMSE : {}'.format(rmse))
    if plot:
        da.plot.line(marker='.', linewidth=0., figsize=(20, 5))
        new_da.plot.line(marker='.', linewidth=0.)
    return new_da


def move_bet_dagan_physical_to_main_path(bet_dagan_path):
    """rename bet_dagan physical radiosonde filenames and move them to main
    path i.e., bet_dagan_path! warning-DO ONCE """
    from aux_gps import path_glob
    import shutil
    year_dirs = sorted([x for x in path_glob(bet_dagan_path, '*/') if x.is_dir()])
    for year_dir in year_dirs:
        month_dirs = sorted([x for x in path_glob(year_dir, '*/') if x.is_dir()])
        for month_dir in month_dirs:
            day_dirs = sorted([x for x in path_glob(month_dir, '*/') if x.is_dir()])
            for day_dir in day_dirs:
                hour_dirs = sorted([x for x in path_glob(day_dir, '*/') if x.is_dir()])
                for hour_dir in hour_dirs:
                    file = [x for x in path_glob(hour_dir, '*/') if x.is_file()]
                    splitted = file[0].as_posix().split('/')
                    name = splitted[-1]
                    hour = splitted[-2]
                    day = splitted[-3]
                    month = splitted[-4]
                    year = splitted[-5]
                    filename = '{}_{}{}{}{}'.format(name, year, month, day, hour)
                    orig = file[0]
                    dest = bet_dagan_path / filename
                    shutil.move(orig.resolve(), dest.resolve())
                    print('moving {} to {}'.format(filename, bet_dagan_path))
    return year_dirs


def read_all_radiosonde_data(path, savepath=None, data_type='phys',
                             verbose=True, year=None):
    """use year for edt,then concat, otherwise it collapses"""
    from aux_gps import path_glob
#    from aux_gps import get_unique_index
    from aux_gps import keep_iqr
    import xarray as xr
    import pandas as pd
    from aux_gps import save_ncfile
    ds_list = []
    cnt = 0
    if data_type == 'phys':
        glob = '*/'
    elif data_type == 'edt':
        glob = '*EDT'
    elif data_type == 'PTU':
        glob = '*PTULevels'
    elif data_type == 'Wind':
        glob = '*WindLevels'
    for path_file in sorted(path_glob(path, glob)):
        cnt += 1
        if path_file.is_file():
            if data_type == 'phys':
                date_ff = path_file.as_posix().split('/')[-1].split('_')[-1]
                date_ff = pd.to_datetime(date_ff, format='%Y%m%d%H')
                ds = read_one_physical_radiosonde_report(path_file)
            elif data_type == 'edt':
                date_ff = path_file.as_posix().split('/')[-1].split('_')[0]
                date_ff = pd.to_datetime(date_ff, format='%Y%m%d%H')
                ds = read_one_EDT_record(path_file, year=year)
            elif data_type == 'PTU' or data_type == 'Wind':
                date_ff = path_file.as_posix().split('/')[-1].split('_')[0]
                date_ff = pd.to_datetime(date_ff, format='%Y%m%d%H')
                ds = read_one_PTU_Wind_levels_radiosonde_report(path_file)
            if ds is None:
                print('{} is corrupted or skipped...'.format(
                    path_file.as_posix().split('/')[-1]))
                continue
            date = pd.to_datetime(ds['sound_time'].item())
            if verbose:
                print('reading {} {} radiosonde report'.format(date.strftime('%Y-%m-%d %H:%M'), data_type))
            try:
                assert date == date_ff
            except AssertionError:
                print('date from filename : {}, date from report: {}'.format(date_ff, date))
                return -1
            ds_list.append(ds)
    dss = xr.concat(ds_list, 'sound_time')
    print('concatenating...')
    dss = dss.sortby('sound_time')
#     dss = get_unique_index(dss, 'sound_time', verbose=True)
    # filter iqr:
    print('filetring...')
    da_list = []
    for da in dss.data_vars.values():
        if len(da.dims) == 1:
            if da.dims[0] == 'sound_time' and da.dtype == 'float64':
                da_list.append(keep_iqr(da, dim='sound_time', verbose=True))
    new_ds = xr.merge(da_list)
    # replace with filtered vars:
    for da in new_ds.data_vars:
        dss[da] = new_ds[da]
    if savepath is not None:
        yr_min = dss.sound_time.min().dt.year.item()
        yr_max = dss.sound_time.max().dt.year.item()
        filename = 'bet_dagan_{}_sounding_{}-{}.nc'.format(data_type, yr_min, yr_max)
        save_ncfile(dss, savepath, filename)
#        print('clearing temp files...')
#        [x.unlink() for x in tempfiles]
    print('Done!')
    return dss


def combine_EDT_and_PTU_radiosonde(sound_path=sound_path, savepath=None,
                                   resample='2s'):
    from aux_gps import path_glob
    from aux_gps import save_ncfile
    import pandas as pd
    import numpy as np
    import xarray as xr

    def interpolate_one_da(da, dim_over, freq='2s'):
        ds_list = []
        for stime in dim_over:
            daa = da.sel({dim_over.name: stime})
            # dropna in all vars:
            dropped = daa.dropna('time')
            last_time = pd.Timedelta(daa.dropna('time')['time'].max().item(),units='s')
            new_time = pd.timedelta_range(pd.Timedelta(0,unit='s'), last_time, freq=freq)
            # if some data remain, interpolate:
            if dropped.size > 0:
                interpolated = dropped.interp(coords={'time': new_time}, method='cubic',
                                              kwargs={'fill_value': np.nan})
                ds_list.append(interpolated)
        ds = xr.concat(ds_list, dim_over.name)
        return ds

    ptu_file = path_glob(sound_path, 'bet_dagan_PTU_Wind_sounding_*.nc')[-1]
    edt_file = path_glob(sound_path, 'bet_dagan_edt_sounding*.nc')[-1]
    ptu = xr.load_dataset(ptu_file)
    edt = xr.load_dataset(edt_file)
    if resample is not None:
        time_vars = [x for x in edt if 'time' in edt[x].dims]
        other_vars = [x for x in edt if 'time' not in edt[x].dims]
        other_ds = edt[other_vars]
        msg = 'interpolated to {} seconds'.format(resample)
        print(msg)
        edt = edt[time_vars].map(interpolate_one_da, dim_over=edt['sound_time'], freq=resample)
#        edt = edt[time_vars].interp(coords={'time': new_time}, method='cubic',
#                             kwargs={'fill_value': np.nan})
#        edt = edt[time_vars].resample(time=resample, keep_attrs=True).mean(keep_attrs=True)
        edt = edt.update(other_ds)
        edt.attrs['action'] = msg
    # convert edt['MR'] to kg/kg:
    edt['MR'] /= 1000
    edt['MR'].attrs['units'] = 'kg/kg'
    # add some vars to ptu for consistensy with edt:
    ptu['u'], ptu['v'] = convert_wind_speed_direction_to_zonal_meridional(
        ptu['WS'], ptu['WD'])
    ptu['MR'] = wrap_xr_metpy_mixing_ratio(ptu['P'], ptu['T'], ptu['RH'])
    # intersection fields:
    inter_fields = list(set(edt.data_vars).intersection(set(ptu.data_vars)))
    ptu = ptu[inter_fields]
    edt = edt[inter_fields]
    # finally concat:
    ds = xr.concat([ptu, edt], 'sound_time')
    if savepath is not None:
        yr_min = ds.sound_time.min().dt.year.item()
        yr_max = ds.sound_time.max().dt.year.item()
        save_ncfile(
            ds, savepath, 'bet_dagan_2s_sounding_{}-{}.nc'.format(yr_min, yr_max))
    return ds


def combine_PTU_and_Wind_levels_radiosonde(sound_path=sound_path,
                                           savepath=None, resample='2s',
                                           drop_period=['2016-11-17',
                                                        '2016-12-31']):
    from aux_gps import path_glob
    from aux_gps import save_ncfile
    import xarray as xr
    import pandas as pd
    ptu_file = path_glob(sound_path, 'bet_dagan_PTU_sounding*.nc')[-1]
    wind_file = path_glob(sound_path, 'bet_dagan_Wind_sounding*.nc')[-1]
    ptu = xr.load_dataset(ptu_file)
    wind = xr.load_dataset(wind_file)
    ptu['WS'] = wind['WS']
    ptu['WD'] = wind['WD']
#    ptu['U'] = wind['U']
#    ptu['V'] = wind['V']
#    ptu['time'].attrs['units'] = 'sec'
    ptu.attrs['lat'] = 32.01
    ptu.attrs['lon'] = 34.81
    if drop_period is not None:
        first_date = pd.to_datetime(drop_period[0])
        last_date = pd.to_datetime(drop_period[1])
        mask = ~(pd.to_datetime(ptu['sound_time'].values) >= first_date) & (
            pd.to_datetime(ptu['sound_time'].values) <= last_date)
        ptu = ptu.loc[{'sound_time': mask}]
        msg = 'dropped datetimes {} to {}'.format(
                drop_period[0],
                drop_period[1])
        print(msg)
        ptu.attrs['operation'] = msg
    if resample is not None:
        time_vars = [x for x in ptu if 'time' in ptu[x].dims]
        other_vars = [x for x in ptu if 'time' not in ptu[x].dims]
        other_ds = ptu[other_vars]
        msg = 'resampled to {} seconds'.format(resample)
        print(msg)
        ptu = ptu[time_vars].resample(time=resample, keep_attrs=True).mean(keep_attrs=True)
        ptu = ptu.update(other_ds)
        ptu.attrs['action'] = msg
    if savepath is not None:
        yr_min = ptu.sound_time.min().dt.year.item()
        yr_max = ptu.sound_time.max().dt.year.item()
        save_ncfile(
            ptu, savepath, 'bet_dagan_PTU_Wind_sounding_{}-{}.nc'.format(yr_min, yr_max))
    return ptu


def read_one_PTU_Wind_levels_radiosonde_report(path_file, verbose=False):
    """read one (12 or 00) PTU levels bet dagan radiosonde reports and return a df
    containing time series, PW for the whole sounding and time span of the
    sounding"""
    # if we save the index with datetimes , when we concatenate on sound_time
    # the memory crashes, so we save the index with seconds from launch
    import pandas as pd
    from aux_gps import line_and_num_for_phrase_in_file
    from aux_gps import remove_duplicate_spaces_in_string
    from scipy.integrate import cumtrapz
    import xarray as xr
    # first determine if PTU or Wind:
    _, type_str = line_and_num_for_phrase_in_file('Levels', path_file)
    if 'PTULevels' in type_str:
        report_type = 'ptu'
        cols = [
            'min',
            'sec',
            'Pressure',
            'Height',
            'Temperature',
            'RH',
            'Dewpt',
            'Significance',
            'flags']
    elif 'WindLevels' in type_str:
        report_type = 'wind'
        cols = [
            'min',
            'sec',
            'Pressure',
            'Height',
            'dum',
            'Speed',
            'Direction',
            'Significance',
            'flags']
    else:
        raise('Could not determine if PTU or Wind')
    # then, read to df:
    skip_to_line, _ = line_and_num_for_phrase_in_file(
        'min  s', path_file)
    skip_to_line += 2
    df = pd.read_csv(
        path_file,
        skiprows=skip_to_line,
        encoding="windows-1252",
        delim_whitespace=True, names=cols, na_values=['///', '/////',
                                                      '//////', 'dfv'])
    # get cloud code:
    _, cld_str = line_and_num_for_phrase_in_file('Clouds', path_file)
    cld_code = cld_str.split(':')[-1].split(' ')[-1].split('\n')[0]
    # get sounding_type:
    _, snd_str = line_and_num_for_phrase_in_file('Sounding type ', path_file)
    if snd_str is None:
        _, snd_str = line_and_num_for_phrase_in_file('RS type ', path_file)
    snd_type = snd_str.rstrip().split(':')[-1].lstrip()
    # get datetime:
    _, dt_str = line_and_num_for_phrase_in_file('Started', path_file)
    dt_str = remove_duplicate_spaces_in_string(dt_str).rstrip().split(' ')
    dt = '{}-{}-{} {}'.format(dt_str[-5], dt_str[-4], dt_str[-3], dt_str[-2])
    dt = pd.to_datetime(dt, format='%d-%B-%Y %H:%M')
    # validate sound_time:
    sound_time = check_sound_time_datetime(dt)
    dt_col = pd.to_timedelta(df['min'], unit='min') + pd.to_timedelta(df['sec'], unit='s')
    df['time'] = dt_col.dt.total_seconds()
    df = df.set_index('time')
#    # validate sound_time:
#    sound_time = check_sound_time(df)
    if report_type == 'ptu':
        df = df[['Pressure', 'Height', 'Temperature', 'RH', 'Dewpt']]
        units = ['hPa', 'm', 'degC', '%', 'degC']
        df.columns = ['P', 'Height', 'T', 'RH', 'Dewpt']
    elif report_type == 'wind':
        df = df[['Pressure', 'Height', 'Speed', 'Direction']]
        units = ['hpa', 'm', 'm/s', 'deg']
        df.columns = ['P', 'Height', 'WS', 'WD']
        df['WD'] = df['WD'].astype(float)
        df = df[df['WS'] >= 0]
    # check for duplicate index entries (warning: could be bad profile):
    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        df = df[~df.index.duplicated(keep='last')]
        print('found {} duplicate times in {}'.format(n_dup, sound_time.strftime('%Y-%m-%d %H:%M')))
    # convert to dataset, add units and attrs:
    ds = df.to_xarray()
    for i, da in enumerate(ds):
        ds[da].attrs['units'] = units[i]
    ds['sound_time'] = sound_time
    ds['cloud_code'] = cld_code
    ds['sonde_type'] = snd_type
    # get the minimum and maximum times:
    ds['min_time'] = dt + pd.Timedelta(df.index[0], unit='s')
    ds['max_time'] = ds['min_time'] + pd.Timedelta(df.index[-1], unit='s')
    # caclculate different variables:
#    if report_type == 'ptu':
#        ds['MR'] = wrap_xr_metpy_mixing_ratio(ds['P'], ds['T'], ds['RH'])
#        ds['MR'].attrs['long_name'] = 'Water vapor mass mixing ratio'
#        ds['Rho'] = wrap_xr_metpy_density(ds['P'], ds['T'], ds['MR'])
#        ds['Rho'].attrs['long_name'] = 'Air density'
#        ds['Q'] = wrap_xr_metpy_specific_humidity(ds['MR'])
#        ds['Q'].attrs['long_name'] = 'Specific humidity'
#        ds['VP'] = wrap_xr_metpy_vapor_pressure(ds['P'], ds['MR'])
#        ds['VP'].attrs['long_name'] = 'Water vapor partial pressure'
#        ds['Rho_wv'] = calculate_absolute_humidity_from_partial_pressure(
#            ds['VP'], ds['T'])
#        pw = cumtrapz(ds['Q']*ds['Rho'], ds['Height'], initial=0)
#        ds['PW'] = xr.DataArray(pw, dims=['time'])
#        ds['PW'].attrs['units'] = 'mm'
#        ds['PW'].attrs['long_name'] = 'Precipitable water'
#        tm = calculate_tm_via_trapz_height(ds['VP'], ds['T'], ds['Height'])
#    #    tm, unit = calculate_tm_via_pressure_sum(ds['VP'], ds['T'], ds['Rho'], ds['P'],
#    #                                             cumulative=True, verbose=False)
#        ds['VPT'] = wrap_xr_metpy_virtual_potential_temperature(ds['P'], ds['T'],
#                                                                ds['MR'])
#        ds['PT'] = wrap_xr_metpy_potential_temperature(ds['P'], ds['T'])
#        ds['VT'] = wrap_xr_metpy_virtual_temperature(ds['T'], ds['MR'])
#        ds['N'] = calculate_atmospheric_refractivity(ds['P'], ds['T'], ds['VP'])
#        ds['Tm'] = xr.DataArray(tm, dims=['time'])
#        ds['Tm'].attrs['units'] = 'K'
#        ds['Tm'].attrs['long_name'] = 'Water vapor mean air temperature'
#        ds['Ts'] = ds['T'].dropna('time').values[0] + 273.15
#        ds['Ts'].attrs['units'] = 'K'
#        ds['Ts'].attrs['long_name'] = 'Surface temperature'
#    elif report_type == 'wind':
#        U, V = convert_wind_speed_direction_to_zonal_meridional(ds['WS'], ds['WD'])
#        ds['U'] = U
#        ds['V'] = V
    # convert to time delta:
    ds['time'] = pd.to_timedelta(ds['time'].values, unit='sec')
    return ds


def read_one_physical_radiosonde_report(path_file, skip_from_year=2014,
                                        verbose=False):
    """read one(12 or 00) physical bet dagan radiosonde reports and return a df
    containing time series, PW for the whole sounding and time span of the
    sounding"""
#    import numpy as np
#    from aux_gps import get_unique_index
    import pandas as pd
    import xarray as xr
    from scipy.integrate import cumtrapz

#    def df_to_ds_and_interpolate(df, h='Height'):
#        import numpy as np
#        # set index the height, and transform to xarray:
#        df = df.set_index(h).squeeze()
#        ds = df.to_xarray()
#        ds = ds.sortby(h)
#        ds = get_unique_index(ds, dim=h)
#        # do cubic interpolation:
#        height = np.linspace(35, 25000, 500)
#        ds_list = []
#        for da in ds.data_vars.values():
#            # dropna in all vars:
#            dropped = da.dropna(h)
#            # if some data remain, interpolate:
#            if dropped.size > 0:
#                ds_list.append(dropped.interp({h: height}, method='cubic'))
#            # if nothing left, create new ones with nans:
#            else:
#                nan_da = np.nan * ds[h]
#                nan_da.name = dropped.name
#                ds_list.append(dropped)
#        ds = xr.merge(ds_list)
#        ds.attrs['operation'] = 'all physical fields were cubic interpolated to {}'.format(h)
#        return ds

    df = pd.read_csv(
         path_file,
         header=None,
         encoding="windows-1252",
         delim_whitespace=True,
         skip_blank_lines=True,
         na_values=['/////'])
    time_str = path_file.as_posix().split('/')[-1].split('_')[-1]
    dt = pd.to_datetime(time_str, format='%Y%m%d%H')
    sound_time = check_sound_time_datetime(dt)
    if sound_time.year >= skip_from_year:
        return None
    if not df[df.iloc[:, 0].str.contains('PILOT')].empty:
        return None
    # drop last two cols:
    df.drop(df.iloc[:, -2:len(df)], inplace=True, axis=1)
    # extract datetime:
    date = df[df.loc[:, 0].str.contains("PHYSICAL")].loc[0, 3]
    time = df[df.loc[:, 0].str.contains("PHYSICAL")].loc[0, 4]
    dt = pd.to_datetime(date + 'T' + time, dayfirst=True)
    # extract station number:
    station_num = int(df[df.loc[:, 0].str.contains("STATION")].iloc[0, 3])
    # extract cloud code:
    cloud_code = df[df.loc[:, 0].str.contains("CLOUD")].iloc[0, 3]
    # extract sonde_type:
    sonde_type = df[df.loc[:, 5].fillna('NaN').str.contains("TYPE")].iloc[0, 7:9]
    sonde_type = sonde_type.dropna()
    sonde_type = '_'.join(sonde_type.to_list())
    # change col names to:
    df.columns = ['Time', 'T', 'RH', 'P', 'H-Sur', 'Height', 'EL',
                  'AZ', 'WD', 'WS']
    radio_units = dict(zip(df.columns.to_list(),
                           ['sec', 'degC', '%', 'hPa', 'm', 'm', 'deg', 'deg', 'deg',
                            'knots']))
    # iterate over the cols and change all to numeric(or timedelta) values:
    for col in df.columns:
        if col == 'Time':
            df[col] = pd.to_timedelta(df[col], errors='coerce')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # filter all entries that the first col is not null:
    df = df[~df.loc[:, 'Time'].isnull()]
    # get the minimum and maximum times:
    max_time = dt + df['Time'].iloc[-1]
    # reindex with total_seconds:
    total_sec = df['Time'].dt.total_seconds()
#    df['datetime_index'] = df['Time'] + dt
    df = df.set_index(total_sec)
    df.index.name = 'time'

#    # now, convert Time to decimal seconds:
#    df['Time'] = df['Time'] / np.timedelta64(1, 's')
    # check for EL and AZ that are not empty and WD and WS that are NaN's are
    # switch them:
    WD_empty = df['WD'].isnull().all()
    WS_empty = df['WS'].isnull().all()
    EL_empty = df['EL'].isnull().all()
    AZ_empty = df['AZ'].isnull().all()
    if WD_empty and WS_empty and not EL_empty and not AZ_empty:
        print('switching AZ, EL and WS, WD cols...')
        df = df.rename(columns={'EL':'WD','WD':'EL', 'AZ': 'WS', 'WS': 'AZ'})
    df.drop('Time', axis=1, inplace=True)
    ds = df.to_xarray()
    ds['time'].attrs['units'] = 'sec'
#    ds = df_to_ds_and_interpolate(df)
    # add meta data:
    for name in ds.data_vars:
        ds[name].attrs['units'] = radio_units[name]
    # add more fields:
#    ds['Dewpt'] = wrap_xr_metpy_dewpoint(ds['T'], ds['RH'])
#    ds['Dewpt'].attrs['long_name'] = 'Dew point'
#    ds['MR'] = wrap_xr_metpy_mixing_ratio(ds['P'], ds['T'], ds['RH'])
#    ds['MR'].attrs['long_name'] = 'Water vapor mass mixing ratio'
#    ds['Rho'] = wrap_xr_metpy_density(ds['P'], ds['T'], ds['MR'])
#    ds['Rho'].attrs['long_name'] = 'Air density'
#    ds['Q'] = wrap_xr_metpy_specific_humidity(ds['MR'])
#    ds['Q'].attrs['long_name'] = 'Specific humidity'
#    ds['VP'] = wrap_xr_metpy_vapor_pressure(ds['P'], ds['MR'])
#    ds['VP'].attrs['long_name'] = 'Water vapor partial pressure'
#    ds['Rho_wv'] = calculate_absolute_humidity_from_partial_pressure(
#        ds['VP'], ds['T'])
#    pw = cumtrapz(ds['Q']*ds['Rho'], ds['Height'], initial=0)
#    ds['PW'] = xr.DataArray(pw, dims=['time'])
#    ds['PW'].attrs['units'] = 'mm'
#    ds['PW'].attrs['long_name'] = 'Precipitable water'
#    tm = calculate_tm_via_trapz_height(ds['VP'], ds['T'], ds['Height'])
##    tm, unit = calculate_tm_via_pressure_sum(ds['VP'], ds['T'], ds['Rho'], ds['P'],
##                                             cumulative=True, verbose=False)
#    ds['VPT'] = wrap_xr_metpy_virtual_potential_temperature(ds['P'], ds['T'],
#                                                            ds['MR'])
#    ds['PT'] = wrap_xr_metpy_potential_temperature(ds['P'], ds['T'])
#    ds['VT'] = wrap_xr_metpy_virtual_temperature(ds['T'], ds['MR'])
#    U, V = convert_wind_speed_direction_to_zonal_meridional(ds['WS'], ds['WD'])
#    ds['U'] = U
#    ds['V'] = V
#    ds['N'] = calculate_atmospheric_refractivity(ds['P'], ds['T'], ds['VP'])
#    ds['Tm'] = xr.DataArray(tm, dims=['time'])
#    ds['Tm'].attrs['units'] = 'K'
#    ds['Tm'].attrs['long_name'] = 'Water vapor mean air temperature'
#    ds['Ts'] = ds['T'].dropna('time').values[0] + 273.15
#    ds['Ts'].attrs['units'] = 'K'
#    ds['Ts'].attrs['long_name'] = 'Surface temperature'
    ds['sound_time'] = sound_time
    ds['min_time'] = dt
    ds['max_time'] = max_time
    ds['cloud_code'] = cloud_code
    ds['sonde_type'] = sonde_type
    ds.attrs['station_number'] = station_num
    ds.attrs['lat'] = 32.01
    ds.attrs['lon'] = 34.81
#    ds.attrs['alt'] = 31.0
    ds['Height'].attrs['units'] = 'm'
    # convert to time delta:
    ds['time'] = pd.to_timedelta(ds['time'].values, unit='sec')
    return ds


def classify_clouds_from_sounding(sound_path=sound_path):
    import xarray as xr
    import numpy as np
    da = xr.open_dataarray(sound_path / 'ALL_bet_dagan_soundings.nc')
    ds = da.to_dataset(dim='var')
    cld_list = []
    for date in ds.time:
        h = ds['HGHT'].sel(time=date).dropna('mpoint').values
        T = ds['TEMP'].sel(time=date).dropna('mpoint').values
        dwT = ds['DWPT'].sel(time=date).dropna('mpoint').values
        h = h[0: len(dwT)]
        cld = np.empty(da.mpoint.size, dtype='float')
        cld[:] = np.nan
        T = T[0: len(dwT)]
        try:
            dT = np.abs(T - dwT)
        except ValueError:
            print('valueerror')
            cld_list.append(cld)
            continue
        found = h[dT < 0.5]
        found_pos = np.where(dT < 0.5)[0]
        if found.any():
            for i in range(len(found)):
                if found[i] < 2000:
                    cld[found_pos[i]] = 1.0  # 'LOW' clouds
                elif found[i] < 7000 and found[i] > 2000:
                    cld[found_pos[i]] = 2.0  # 'MIDDLE' clouds
                elif found[i] < 13000 and found[i] > 7000:
                    cld[found_pos[i]] = 3.0  # 'HIGH' clouds
        cld_list.append(cld)
    ds['CLD'] = ds['HGHT'].copy(deep=False, data=cld_list)
    ds.to_netcdf(sound_path / 'ALL_bet_dagan_soundings_with_clouds.nc', 'w')
    print('ALL_bet_dagan_soundings_with_clouds.nc saved to {}'.format(sound_path))
    return ds


def compare_interpolated_to_real(Tint, date='2014-03-02T12:00',
                                 sound_path=sound_path):
    from metpy.plots import SkewT
    from metpy.units import units
    import pandas as pd
    import xarray as xr
    import matplotlib.pyplot as plt
    da = xr.open_dataarray(sound_path / 'ALL_bet_dagan_soundings.nc')
    p = da.sel(time=date, var='PRES').values * units.hPa
    dt = pd.to_datetime(da.sel(time=date).time.values)
    T = da.sel(time=dt, var='TEMP').values * units.degC
    Tm = Tint.sel(time=dt).values * units.degC
    pm = Tint['pressure'].values * units.hPa
    fig = plt.figure(figsize=(9, 9))
    title = da.attrs['description'] + ' ' + dt.strftime('%Y-%m-%d')
    skew = SkewT(fig)
    skew.plot(p, T, 'r', linewidth=2)
    skew.plot(pm, Tm, 'g', linewidth=2)
    skew.ax.set_title(title)
    skew.ax.legend(['Original', 'Interpolated'])
    return


def average_meteogram(sound_path=sound_path, savepath=None):
    import xarray as xr
    import numpy as np
    from scipy.interpolate import interp1d
    da = xr.open_dataarray(sound_path / 'ALL_bet_dagan_soundings.nc')
    # pressure = np.linspace(1010, 20, 80)
    height = da.sel(var='HGHT').isel(time=0).dropna('mpoint').values
    T_list = []
    for i in range(da.time.size):
        x = da.isel(time=i).sel(var='HGHT').dropna('mpoint').values
        y = da.isel(time=i).sel(var='TEMP').dropna('mpoint').values
        x = x[0: len(y)]
        f = interp1d(x, y, kind='linear', fill_value='extrapolate')
        T_list.append(f(height))
    T = xr.DataArray(T_list, dims=['time', 'height'])
    ds = T.to_dataset(name='temperature')
    ds['height'] = height
    ds['time'] = da.time
    ds['temperature'].attrs['units'] = 'degC'
    ds['temperature'] = ds['temperature'].where(ds['temperature'].isel(height=0)>-13,drop=True)
    ds['height'].attrs['units'] = 'm'
    ts_list = [ds['temperature'].isel(height=0).sel(time=x) + 273.15 for x in
               ds.time]
    tm_list = [Tm(ds['temperature'].sel(time=x), from_raw_sounding=False)
               for x in ds.time]
    tm = xr.DataArray(tm_list, dims='time')
    tm.attrs['description'] = 'mean atmospheric temperature calculated by water vapor pressure weights'
    tm.attrs['units'] = 'K'
    ts = xr.concat(ts_list, 'time')
    ts.attrs['description'] = 'Surface temperature from BET DAGAN soundings'
    ts.attrs['units'] = 'K'
    hours = [12, 0]
    hr_dict = {12: 'noon', 0: 'midnight'}
    seasons = ['DJF', 'SON', 'MAM', 'JJA']
    for season in seasons:
        for hour in hours:
            da = ds['temperature'].sel(time=ds['time.season'] == season).where(
                ds['time.hour'] == hour).dropna('time').mean('time')
            name = ['T', season, hr_dict[hour]]
            ds['_'.join(name)] = da
    ds['season'] = ds['time.season']
    ds['hour'] = ds['time.hour'].astype(str)
    ds['hour'] = ds.hour.where(ds.hour != '12', 'noon')
    ds['hour'] = ds.hour.where(ds.hour != '0', 'midnight')
    ds['ts'] = ts
    ds['tm'] = tm
    ds = ds.dropna('time')
    if savepath is not None:
        filename = 'bet_dagan_sounding_pw_Ts_Tk1.nc'
        print('saving {} to {}'.format(filename, savepath))
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds}
        ds.to_netcdf(savepath / filename, 'w', encoding=encoding)
    print('Done!')
    return ds


def plot_skew(sound_path=sound_path, date='2018-01-16T12:00', two=False):
    from metpy.plots import SkewT
    from metpy.units import units
    import matplotlib.pyplot as plt
    import pandas as pd
    import xarray as xr
    da = xr.open_dataarray(sound_path / 'ALL_bet_dagan_soundings.nc')
    p = da.sel(time=date, var='PRES').values * units.hPa
    dt = pd.to_datetime(da.sel(time=date).time.values)
    if not two:
        T = da.sel(time=date, var='TEMP').values * units.degC
        Td = da.sel(time=date, var='DWPT').values * units.degC
        Vp = VaporPressure(da.sel(time=date, var='TEMP').values) * units.Pa
        dt = pd.to_datetime(da.sel(time=date).time.values)
        fig = plt.figure(figsize=(9, 9))
        title = da.attrs['description'] + ' ' + dt.strftime('%Y-%m-%d %H:%M')
        skew = SkewT(fig)
        skew.plot(p, T, 'r', linewidth=2)
        skew.plot(p, Td, 'g', linewidth=2)
        # skew.ax.plot(p, Vp, 'k', linewidth=2)
        skew.ax.set_title(title)
        skew.ax.legend(['Temp', 'Dewpoint'])
    elif two:
        dt1 = pd.to_datetime(dt.strftime('%Y-%m-%dT00:00'))
        dt2 = pd.to_datetime(dt.strftime('%Y-%m-%dT12:00'))
        T1 = da.sel(time=dt1, var='TEMP').values * units.degC
        T2 = da.sel(time=dt2, var='TEMP').values * units.degC
        fig = plt.figure(figsize=(9, 9))
        title = da.attrs['description'] + ' ' + dt.strftime('%Y-%m-%d')
        skew = SkewT(fig)
        skew.plot(p, T1, 'r', linewidth=2)
        skew.plot(p, T2, 'b', linewidth=2)
        # skew.ax.plot(p, Vp, 'k', linewidth=2)
        skew.ax.set_title(title)
        skew.ax.legend(['Temp at ' + dt1.strftime('%H:%M'),
                        'Temp at ' + dt2.strftime('%H:%M')])
    return


#def es(T):
#    """ARM function for water vapor saturation pressure"""
#    # T in celsius:
#    import numpy as np
#    es = 6.1094 * np.exp(17.625 * T / (T + 243.04))
#    # es in hPa
#    return es

def run_pyigra_save_xarray(station, path=sound_path):
    import subprocess
    filepath = path / 'igra_{}_raw.txt'.format(station)
    command = '/home/ziskin/miniconda3/bin/PyIGRA --id {} -o {}'.format(station,filepath)
    subprocess.call([command], shell=True)
    # pyigra_to_xarray(station + '_pt.txt', path=path)
    return


def pyigra_to_xarray(station, path=sound_path):
    import pandas as pd
    import xarray as xr
    # import numpy as np
    filepath = path / 'igra_{}_raw.txt'.format(station)
    df = pd.read_csv(filepath, delim_whitespace=True, na_values=[-9999.0, 9999])
    dates = df['NOMINAL'].unique().tolist()
    print('splicing dataframe and converting to xarray dataset...')
    ds_list = []
    for date in dates:
        dff = df.loc[df.NOMINAL == date]
        # release = dff.iloc[0, 1]
        dff = dff.drop(['NOMINAL', 'RELEASE'], axis=1)
        dss = dff.to_xarray()
        dss = dss.drop('index')
        dss = dss.rename({'index': 'point'})
        dss['point'] = range(dss.point.size)
        # dss.attrs['release'] = release
        ds_list.append(dss)
    print('concatenating to time-series dataset')
    datetimes = pd.to_datetime(dates, format='%Y%m%d%H')
    # max_ind = np.max([ds.index.size for ds in ds_list])
    # T = np.nan * np.ones((len(dates), max_ind))
    # P = np.nan * np.ones((len(dates), max_ind))
    # for i, ds in enumerate(ds_list):
    #     tsize = ds['TEMPERATURE'].size
    #     T[i, 0:tsize] = ds['TEMPERATURE'].values
    #     P[i, 0:tsize] = ds['PRESSURE'].values
    # Tda = xr.DataArray(T, dims=['time', 'point'])
    # Tda.name = 'Temperature'
    # Tda.attrs['units'] = 'deg C'
    # Tda['time'] = datetimes
    # Pda = xr.DataArray(P, dims=['time', 'point'])
    # Pda.name = 'Pressure'
    # Pda.attrs['units'] = 'hPa'
    # Pda['time'] = datetimes
    # ds = Tda.to_dataset(name='Temperature')
    # ds['Pressure'] = Pda
    units = {'PRESSURE': 'hPa', 'TEMPERATURE': 'deg_C', 'RELHUMIDITY': '%',
             'DEWPOINT': 'deg_C', 'WINDSPEED': 'm/sec',
             'WINDDIRECTION': 'azimuth'}
    ds = xr.concat(ds_list, 'time')
    ds['time'] = datetimes
    for da, unit in units.items():
        ds[da].attrs['unit'] = unit
    ds.attrs['name'] = 'radiosonde soundings from IGRA'
    ds.attrs['station'] = station
    filename = 'igra_{}.nc'.format(station)
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(path / filename, 'w', encoding=encoding)
    print('saved {} to {}.'.format(filename, path))
    return ds


def process_mpoint_da_with_station_num(path=sound_path, station='08001', k_iqr=1):
    from aux_gps import path_glob
    import xarray as xr
    from aux_gps import keep_iqr
    file = path_glob(sound_path, 'ALL*{}*.nc'.format(station))
    da = xr.open_dataarray(file[0])
    ts, tm, tpw = calculate_ts_tm_tpw_from_mpoint_da(da)
    ds = xr.Dataset()
    ds['Tm'] = xr.DataArray(tm, dims=['time'], name='Tm')
    ds['Tm'].attrs['unit'] = 'K'
    ds['Tm'].attrs['name'] = 'Water vapor mean atmospheric temperature'
    ds['Ts'] = xr.DataArray(ts, dims=['time'], name='Ts')
    ds['Ts'].attrs['unit'] = 'K'
    ds['Ts'].attrs['name'] = 'Surface temperature'
    ds['Tpw'] = xr.DataArray(tpw, dims=['time'], name='Tpw')
    ds['Tpw'].attrs['unit'] = 'mm'
    ds['Tpw'].attrs['name'] = 'precipitable_water'
    ds['time'] = da.time
    ds = keep_iqr(ds, k=k_iqr, dim='time')
    yr_min = ds.time.min().dt.year.item()
    yr_max = ds.time.max().dt.year.item()
    ds = ds.rename({'time': 'sound_time'})
    filename = 'station_{}_soundings_ts_tm_tpw_{}-{}.nc'.format(station, yr_min, yr_max)
    print('saving {} to {}'.format(filename, path))
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in ds}
    ds.to_netcdf(path / filename, 'w', encoding=encoding)
    print('Done!')
    return ds


def calculate_ts_tm_tpw_from_mpoint_da(da):
    """ calculate the atmospheric mean temperature and precipitable_water"""
    import numpy as np
    times_tm = []
    times_tpw = []
    times_ts = []
    station_num = [x for x in da.attrs['description'].split(' ') if x.isdigit()]
    print('calculating ts, tm and tpw from station {}'.format(station_num))
    ds = da.to_dataset('var')
    for itime in range(ds.time.size):
        tempc = ds['TEMP'].isel(time=itime).dropna('mpoint').reset_coords(drop=True)
        # dwptc = ds['DWPT'].isel(time=itime).dropna('mpoint').reset_coords(drop=True)
        hghtm = ds['HGHT'].isel(time=itime).dropna('mpoint').reset_coords(drop=True)
        preshpa = ds['PRES'].isel(time=itime).dropna('mpoint').reset_coords(drop=True)   # in hPa
        WVpress = VaporPressure(tempc, units='hPa', method='Buck')
        Mixratio = MixRatio(WVpress, preshpa)  # both in hPa
        tempk = tempc + 273.15
        Rho = DensHumid(tempc, preshpa, WVpress, out='both')
        specific_humidity = (Mixratio / 1000.0) / \
             (1 + 0.001 * Mixratio / 1000.0)
        try:
            numerator = np.trapz(WVpress / tempk, hghtm)
            denominator = np.trapz(WVpress / tempk**2.0, hghtm)
            tm = numerator / denominator
        except ValueError:
            tm = np.nan
        try:
            tpw = np.trapz(specific_humidity * Rho, hghtm)
        except ValueError:
            tpw = np.nan
        times_tm.append(tm)
        times_tpw.append(tpw)
        times_ts.append(tempk[0].values.item())
    Tm = np.array(times_tm)
    Ts = np.array(times_ts)
    Tpw = np.array(times_tpw)
    return Ts, Tm, Tpw


def MixRatio(e, p):
    """Mixing ratio of water vapour
    INPUTS
    e (Pa) Water vapor pressure
    p (Pa) Ambient pressure
    RETURNS
    qv (g kg^-1) Water vapor mixing ratio`
    """
    MW_dry_air = 28.9647  # gr/mol
    MW_water = 18.015  # gr/mol
    Epsilon = MW_water / MW_dry_air  # Epsilon=Rs_da/Rs_v;
    # The ratio of the gas constants
    return 1000 * Epsilon * e / (p - e)


def DensHumid(tempc, pres, e, out='dry_air'):
    """Density of moist air.
    This is a bit more explicit and less confusing than the method below.
    INPUTS:
    tempc: Temperature (C)
    pres: static pressure (hPa)
    e: water vapor partial pressure (hPa)
    OUTPUTS:
    rho_air (kg/m^3)
    SOURCE: http://en.wikipedia.org/wiki/Density_of_air
    """
    tempk = tempc + 273.15
    prespa = pres * 100.0
    epa = e * 100.0
    Rs_v = 461.52  # Specific gas const for water vapour, J kg^{-1} K^{-1}
    Rs_da = 287.05  # Specific gas const for dry air, J kg^{-1} K^{-1}
    pres_da = prespa - epa
    rho_da = pres_da / (Rs_da * tempk)
    rho_wv = epa/(Rs_v * tempk)
    if out == 'dry_air':
        return rho_da
    elif out == 'wv_density':
        return rho_wv
    elif out == 'both':
        return rho_da + rho_wv


def VaporPressure(tempc, phase="liquid", units='hPa', method=None):
    import numpy as np
    """Water vapor pressure over liquid water or ice.
    INPUTS:
    tempc: (C) OR dwpt (C), if SATURATION vapour pressure is desired.
    phase: ['liquid'],'ice'. If 'liquid', do simple dew point. If 'ice',
    return saturation vapour pressure as follows:
    Tc>=0: es = es_liquid
    Tc <0: es = es_ice
    RETURNS: e_sat  (Pa) or (hPa) units parameter choice
    SOURCE: http://cires.colorado.edu/~voemel/vp.html (#2:
    CIMO guide (WMO 2008), modified to return values in Pa)
    This formulation is chosen because of its appealing simplicity,
    but it performs very well with respect to the reference forms
    at temperatures above -40 C. At some point I'll implement Goff-Gratch
    (from the same resource).
    """
    if units == 'hPa':
        unit = 1.0
    elif units == 'Pa':
        unit = 100.0
    if method is None:
        over_liquid = 6.112 * np.exp(17.67 * tempc / (tempc + 243.12)) * unit
        over_ice = 6.112 * np.exp(22.46 * tempc / (tempc + 272.62)) * unit
    elif method == 'Buck':
        over_liquid = 6.1121 * \
            np.exp((18.678 - tempc / 234.5) * (tempc / (257.4 + tempc))) * unit
        over_ice = 6.1125 * \
            np.exp((23.036 - tempc / 333.7) * (tempc / (279.82 + tempc))) * unit
    # return where(tempc<0,over_ice,over_liquid)

    if phase == "liquid":
        # return 6.112*exp(17.67*tempc/(tempc+243.12))*100.
        return over_liquid
    elif phase == "ice":
        # return 6.112*exp(22.46*tempc/(tempc+272.62))*100.
        return np.where(tempc < 0, over_ice, over_liquid)
    else:
        raise NotImplementedError


def dewpoint(e):
    """Calculate the ambient dewpoint given the vapor pressure.

    Parameters
    ----------
    e : Water vapor partial pressure in mb (hPa)
    Returns
    -------
    Dew point temperature in deg_C
    See Also
    --------
    dewpoint_rh, saturation_vapor_pressure, vapor_pressure
    Notes
    -----
    This function inverts the [Bolton1980]_ formula for saturation vapor
    pressure to instead calculate the temperature. This yield the following
    formula for dewpoint in degrees Celsius:

    .. math:: T = \frac{243.5 log(e / 6.112)}{17.67 - log(e / 6.112)}

    """
    import numpy as np
    # units are degC
    sat_pressure_0c = 6.112  # in millibar
    val = np.log(e / sat_pressure_0c)
    return 243.5 * val / (17.67 - val)


def dewpoint_rh(temperature, rh):
    """Calculate the ambient dewpoint given air temperature and relative
    humidity.
    Parameters
    ----------
    temperature : in deg_C
        Air temperature
    rh : in %
        Relative Humidity
    Returns
    -------
       The dew point temperature
    See Also
    --------
    dewpoint, saturation_vapor_pressure
    """
    import numpy as np
    import warnings
    if np.any(rh > 120):
        warnings.warn('Relative humidity >120%, ensure proper units.')
    return dewpoint(rh / 100.0 * VaporPressure(temperature, units='hPa',
                    method='Buck'))

#def ea(T, rh):
#    """water vapor pressure function using ARM function for sat."""
#    # rh is relative humidity in %
#    return es(T) * rh / 100.0
