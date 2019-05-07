#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:50:20 2019

@author: shlomi
"""

import pandas as pd
import sys
import xarray as xr
import numpy as np
if sys.platform == 'linux':
    work_path = '/home/shlomi/Desktop/DATA/Work Files/PW_yuval/'
elif sys.platform == 'darwin':  # mac os
    work_path = '/Users/shlomi/Documents/PW_yuval/'
PW_stations_path = work_path + '1minute/'
stations = pd.read_csv('stations.txt', header=0, delim_whitespace=True,
                       index_col='NAME')

def proc_1minute(path):
    stations = pd.read_csv(path + 'Zstations', header=0,
                           delim_whitespace=True)
    station_names = stations['NAME'].values.tolist()
    df_list = []
    for st_name in station_names:
        print('Proccessing ' + st_name + ' Station...')
        df = pd.read_csv(PW_stations_path + st_name, delim_whitespace=True)
        df.columns = ['date', 'time', 'PW']
        df.index = pd.to_datetime(df['date'] + 'T' + df['time'])
        df.drop(columns=['date', 'time'], inplace=True)
        df_list.append(df)
    df = pd.concat(df_list, axis=1)
    print('Concatanting to Xarray...')
    # ds = xr.concat([df.to_xarray() for df in df_list], dim="station")
    # ds['station'] = station_names
    df.columns = station_names
    ds = df.to_xarray()
    ds = ds.rename({'index': 'time'})
    # da = ds.to_array(name='PW').squeeze(drop=True)
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in ds.data_vars}
    print('Saving to PW_2007-2016.nc')
    ds.to_netcdf(work_path + 'PW_2007-2016.nc', 'w', encoding=encoding)
    print('Done!')
    # clean the data:
    # da = da.where(da >= 0, np.nan)
    # da = da.where(da < 100, np.nan)

    # plot the data:
    ds.to_array(dim='station').plot(x='time', col='station', col_wrap=4)
    # hist:
    # df=ds.to_dataframe()
    sl = (df > 0) & (df < 50)
    df[sl].hist(bins=30, grid=False, figsize=(15, 8))
    return


def read_stations_to_dataset(path, group_name='israeli', save=False,
                             names=None):
    import xarray as xr
    import glob
    if names is None:
        stations = []
        file_list_with_path = sorted(glob.glob(path + 'garner_trop_' + '[!all_stations]*.nc'))
        for filename in file_list_with_path:
            st_name = filename.split('/')[-1].split('.')[0].split('_')[-1]
            print('Reading station {}'.format(st_name))
            da = xr.open_dataarray(filename)
            da = da.dropna('time')
            stations.append(da)
        ds = xr.merge(stations)
    if save:
        savefile = 'garner_' + group_name + '_stations.nc'
        print('saving {} to {}'.format(savefile, path))
        ds.to_netcdf(path + savefile, 'w')
        print('Done!')
    return ds


def filter_stations(path, group_name='israeli', save=False):
    """filter bad values in trop products stations"""
    import xarray as xr
    filename = 'garner_' + group_name + '_stations'
    print('Reading {}.nc from {}'.format(filename, path))
    ds = xr.open_dataset(path + filename + '.nc')
    ds['zwd'].attrs['units'] = 'Zenith Wet Delay in cm'
    stations = [x for x in ds.data_vars.keys()]
    for station in stations:
        print('filtering station {}'.format(station))
        # first , remove negative values:
        ds[station] = ds[station].where(ds[station].sel(zwd='value') > 0)
        # get zscore of data:
        zscore = Zscore_xr(ds[station].sel(zwd='value'), dim='time')
        ds[station] = ds[station].where(np.abs(zscore) < 5)
    if save:
        filename = filename + '_filtered.nc'
        print('saving {} to {}'.format(filename, path))
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(path + filename, 'w', encoding=encoding)
        print('Done!')
    return ds

#def overlap_time_xr(*args, union=False):
#    """return the intersection of datetime objects from time field in *args"""
#    # caution: for each arg input is xarray with dim:time
#    time_list = []
#    for ts in args:
#        time_list.append(ts.time.values)
#    if union:
#        union = set.union(*map(set, time_list))
#        un = sorted(list(union))
#        return un
#    else:
#        intersection = set.intersection(*map(set, time_list))
#        intr = sorted(list(intersection))
#        return intr


def read_ims(path, filename):
    import pandas as pd
    """parse ims stations meta-data"""
    if '10mins' in filename:
        ims = pd.read_excel(path + filename,
                            sheet_name='מטה-דטה', skiprows=1)
        # drop two last cols and two last rows:
        ims = ims.drop(ims.columns[[-1, -2]], axis=1)
        ims = ims.drop(ims.tail(2).index)
        cols = ['#', 'ID', 'name_hebrew', 'name_english', 'east', 'west',
                'lon', 'lat', 'alt', 'starting_date', 'variables', 'model',
                'eq_position', 'wind_meter_height', 'notes']
        ims.columns = cols
        ims.index = ims['#'].astype(int)
        ims = ims.drop('#', axis=1)
        # fix lat, lon cols:
        ims['lat'] = ims['lat'].str.replace(u'\xba', '').astype(float)
        ims['lon'] = ims['lon'].str.replace(u'\xba', '').astype(float)
        # fix alt col:
        ims['alt'] = ims['alt'].replace('~', '', regex=True).astype(float)
        # fix starting date col:
        ims['starting_date'] = pd.to_datetime(ims['starting_date'])
    else:
        ims = pd.read_excel(path + filename,
                            sheet_name='תחנות אקלים', skiprows=1)
        cols = ['ID', 'name_hebrew', 'name_english', 'station_type', 'east',
                'west', 'lon', 'lat', 'alt', 'starting_date', 'closing_date',
                'date_range']
        ims.columns = cols
        # ims.index = ims['ID'].astype(int)
        # ims = ims.drop('ID', axis=1)
        # fix lat, lon cols:
        ims['lat'] = ims['lat'].str.replace(u'\xba', '').astype(float)
        ims['lon'] = ims['lon'].str.replace(u'\xba', '').astype(float)
        # fix alt col:
        ims['alt'] = ims['alt'].replace('~', '', regex=True).astype(float)
        # fix starting date, closing_date col:
        ims['starting_date'] = pd.to_datetime(ims['starting_date'])
        ims['closing_date'] = pd.to_datetime(ims['closing_date'])
    return ims


def produce_geo_ims(path, filename, closed_stations=False, plot=True):
    import geopandas as gpd
    import numpy as np
    isr = gpd.read_file(path + 'israel_demog2012.shp')
    isr.crs = {'init': 'epsg:4326'}
    ims = read_ims(path, filename)
    if closed_stations:
        ims = ims[np.isnat(ims.closing_date)]
    geo_ims = gpd.GeoDataFrame(ims, geometry=gpd.points_from_xy(ims.lon,
                                                                ims.lat),
                               crs=isr.crs)
    if plot:
        ax = isr.plot()
        geo_ims.plot(ax=ax, column='alt', cmap='Reds', edgecolor='black',
                     legend=True)
    return geo_ims


def produce_geo_gps_stations(path, plot=True):
    import geopandas as gpd
    stations_df = pd.read_csv('stations.txt', index_col='NAME',
                              delim_whitespace=True)
    isr = gpd.read_file(path + 'israel_demog2012.shp')
    isr.crs = {'init': 'epsg:4326'}
    stations = gpd.GeoDataFrame(stations_df,
                                geometry=gpd.points_from_xy(stations_df.LON,
                                                            stations_df.LAT),
                                crs=isr.crs)
    stations_isr = gpd.sjoin(stations, isr, op='within')
    if plot:
        ax = isr.plot()
        stations_isr.plot(ax=ax, column='ALT', cmap='Greens',
                          edgecolor='black', legend=True)
        for x, y, label in zip(stations_isr.LON, stations_isr.LAT,
                               stations_isr.index):
            ax.annotate(label, xy=(x, y), xytext=(3, 3),
                        textcoords="offset points")
    return stations_isr


def get_minimum_distance(geo_ims, geo_gps, path, plot=True):
    def min_dist(point, gpd2):
        gpd2['Dist'] = gpd2.apply(
                    lambda row: point.distance(
                            row.geometry), axis=1)
        geoseries = gpd2.iloc[gpd2['Dist'].values.argmin()]
        geoseries.loc['distance'] = gpd2['Dist'].values.min()
        return geoseries
    min_list = []
    for gps_rows in geo_gps.iterrows():
        ims_min_series = min_dist(gps_rows[1]['geometry'], geo_ims)
        min_list.append(ims_min_series[['ID', 'name_hebrew', 'name_english',
                                        'lon', 'lat', 'alt', 'starting_date',
                                        'distance']])
    geo_df = pd.concat(min_list, axis=1).T
    geo_df['lat'] = geo_df['lat'].astype(float)
    geo_df['lon'] = geo_df['lon'].astype(float)
    geo_df['alt'] = geo_df['alt'].astype(float)
    geo_df.index = geo_gps.index
    stations_meta = ims_api_get_meta()
    # select ims_stations that appear in the geo_df (closest to gps stations):
    ims_selected = stations_meta.loc[stations_meta.stationId.isin(geo_df.ID.values.tolist())]
    # get the channel of temperature measurment of the selected stations: 
    cid = []
    for index, row in geo_df.iterrows():
        channel = [irow['TD_channel'] for ind, irow in ims_selected.iterrows()
                   if irow['stationId'] == row['ID']]
        if channel:
            cid.append(channel[0])
        else:
            cid.append(None)
    # put the channel_id in the geo_df so later i can d/l the exact channel
    # for each stations needed for the gps station:
    geo_df['channel_id'] = cid
    geo_df['channel_id'] = geo_df['channel_id'].fillna(0).astype(int)
    geo_df['ID'] = geo_df.ID.astype(int)
    geo_df['distance'] = geo_df.distance.astype(float)
    geo_df['starting_date'] = pd.to_datetime(geo_df.starting_date)
    if plot:
        import geopandas as gpd
        isr = gpd.read_file(path + 'israel_demog2012.shp')
        isr.crs = {'init': 'epsg:4326'}
        geo_gps_new = gpd.GeoDataFrame(geo_df,
                                       geometry=gpd.points_from_xy(geo_df.lon,
                                                                   geo_df.lat),
                                       crs=isr.crs)
        ax = isr.plot()
        geo_gps.plot(ax=ax, color='green',
                     edgecolor='black', legend=True)
        for x, y, label in zip(geo_gps.LON, geo_gps.LAT,
                               geo_gps.ALT):
            ax.annotate(label, xy=(x, y), xytext=(3, 3),
                        textcoords="offset points")
        geo_gps_new.plot(ax=ax, color='red', edgecolor='black', legend=True)
        for x, y, label in zip(geo_gps_new.lon, geo_gps_new.lat,
                               geo_gps_new.alt):
            ax.annotate(label, xy=(x, y), xytext=(3, 3),
                        textcoords="offset points")
    return geo_df


def ims_api_get_meta(active_only=True):
    import requests
    import pandas as pd
    """temperature is channelId 11"""
    myToken = 'f058958a-d8bd-47cc-95d7-7ecf98610e47'
    headers = {'Authorization': 'ApiToken ' + myToken}
    r = requests.get('https://api.ims.gov.il/v1/envista/stations/',
                     headers=headers)
    stations_10mins = pd.DataFrame(r.json())
    # filter inactive stations:
    if active_only:
        stations_10mins = stations_10mins[stations_10mins.active]
    # arrange lat lon nicely and add channel num for dry temp:
    lat_ = []
    lon_ = []
    TD_channelId = []
    for index, row in stations_10mins.iterrows():
        lat_.append(row['location']['latitude'])
        lon_.append(row['location']['longitude'])
        channel = [x['channelId'] for x in row.monitors if x['name'] ==
                   'TD']
        if channel:
            TD_channelId.append(channel[0])
        else:
            TD_channelId.append(None)
    stations_10mins['lat'] = lat_
    stations_10mins['lon'] = lon_
    stations_10mins['TD_channel'] = TD_channelId
    stations_10mins.drop(['location', 'StationTarget', 'stationsTag'],
                         axis=1, inplace=True)
    return stations_10mins


def parse_ims_to_df(raw_data):
    """gets ims station raw data, i.e., r.json()['data'] and returns
    a pandas dataframe"""
    import pandas as pd
    datetimes = [x['datetime'] for x in raw_data]
    data = [x['channels'][0] for x in raw_data]
    df = pd.DataFrame.from_records(data, index=pd.to_datetime(datetimes,
                                                              utc=True))
    # init an empty list dict:
#    fields = {k: [] for k in df.channels.iloc[0][0].keys()}
#    for index, row in df.iterrows():
#        for k in fields.keys():
#            fields[k].append(row.channels[0][k])
#    for k in fields.keys():
#        df[k] = fields[k]
#    df.drop(['channels', 'datetime'], axis=1, inplace=True)
    df.drop(['alias', 'description'], axis=1, inplace=True)
    return df


def download_ims_data(geo_df, path, end_date='2019-04-15'):
    import requests
    import glob

    def to_dataarray(df, index, row):
        import pandas as pd
        ds = df.to_xarray()
        ds['time'] = pd.to_datetime(ds.time)
        channel_name = ds.name.isel(time=0).values
        channel_id = ds.id.isel(time=0).values
        ds = ds.drop(['id', 'name'])
        da = ds.to_array(dim='TD', name=str(index))
        da.attrs['channel_id'] = channel_id.item()
        da.attrs['channel_name'] = channel_name.item()
        da.attrs['station_name'] = row.name_english
        da.attrs['station_id'] = row.ID
        da.attrs['station_lat'] = row.lat
        da.attrs['station_lon'] = row.lon
        da.attrs['station_alt'] = row.alt
        return da

    def get_dates_list(starting_date, end_date):
        """divide the date span into full 1 years and a remainder, tolist"""
        import numpy as np
        import pandas as pd
        end_date = pd.to_datetime(end_date)
        s_year = starting_date.year
        e_year = end_date.year
        years = np.arange(s_year, e_year + 1)
        dates = [starting_date.replace(year=x) for x in years]
        if (end_date - dates[-1]).days > 0:
            dates.append(end_date)
        return dates

    myToken = 'f058958a-d8bd-47cc-95d7-7ecf98610e47'
    headers = {'Authorization': 'ApiToken ' + myToken}
    already_dl = []
    for paths in glob.glob(path+'*_TD.nc'):
        already_dl.append(paths.split('/')[-1].split('.')[0].split('_')[0])
    to_download = list(set(geo_df.index.values.tolist()).difference(set(already_dl)))
    if to_download:
        geo_df = geo_df.loc[to_download]
    for index, row in geo_df.iterrows():
        # get a list of dates to download: (1 year old parts)
        dates = get_dates_list(row.starting_date, end_date)
        # get station id and channel id(only dry temperature):
        name = row.name_english
        station_id = row.ID
        channel_id = row.channel_id
        # if tempertue is not measuered in station , skip:
        if channel_id == 0:
            continue
        print ('Getting IMS data for {} station(ID={}) from channel {}'.format(name, station_id, channel_id))
        # loop over one year time span and download:
        df_list = []
        for i in range(len(dates) - 1):
            first_date = dates[i].strftime('%Y/%m/%d')
            last_date = dates[i + 1].strftime('%Y/%m/%d')
            print('proccesing dates: {} to {}'.format(first_date, last_date))
            dl_command = ('https://api.ims.gov.il/v1/envista/stations/' +
                          str(station_id) + '/data/' + str(channel_id) +
                          '?from=' + first_date + '&to=' + last_date)
            r = requests.get(dl_command, headers=headers)
            if r.status_code == 204:  # i.e., no content:
                print('no content for this search, skipping...')
                break
            print('parsing to dataframe...')
            df_list.append(parse_ims_to_df(r.json()['data']))
        print('concatanating df and transforming to xarray...')
        df_all = pd.concat(df_list)
        # only valid results:
        df_valid = df_all[df_all['valid']]
        df_valid.index.name = 'time'
        da = to_dataarray(df_valid, index, row)
        filename = index + '_TD.nc'
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in da.to_dataset().data_vars}
        print('saving to {} to {}'.format(filename, path))
        da.to_netcdf(path + filename, 'w', encoding=encoding)
        print('done!')
    #    return df_list
    # pick station and time span
    # download
    # call parse_ims_to_df
    # concatanate and save to nc
    return


def post_proccess_ims(da):
    """fill in the missing time data for the ims temperature stations"""
    import pandas as pd
    import numpy as np
    import xarray as xr
    da = da.sel(TD='value')
    da = da.reset_coords(drop=True)
    # first compute the climatology and the anomalies:
    print('computing anomalies:')
    climatology = da.groupby('time.month').mean('time')
    anom = da.groupby('time.month') - climatology
    # then comupte the diurnal cycle:
    print('computing diurnal change:')
    diurnal = anom.groupby('time.hour').mean('time')
    # assemble old and new time and comupte the difference:
    print('assembeling missing data:')
    old_time = pd.to_datetime(da.time.values)
    new_time = pd.date_range(da.time.min().item(), da.time.max().item(),
                             freq='10min')
    missing_time = pd.to_datetime(sorted(set(new_time).difference(set(old_time))))
    missing_data = np.empty((missing_time.shape))
    print('proccessing missing data...')
    for i in range(len(missing_data)):
        # replace data as to monthly long term mean and diurnal hour:
        missing_data[i] = (climatology.sel(month=missing_time[i].month) +
                           diurnal.sel(hour=missing_time[i].hour))
    series = pd.Series(data=missing_data, index=missing_time)
    series.index.name='time'
    mda = series.to_xarray()
    mda.name = da.name
    new_data = xr.concat([mda, da], 'time')
    new_data = new_data.sortby('time')
    print('done!')
    return new_data


def kappa(T, k2=17.0, k3=3.776e5):
    """T in celsious"""
    Tm = (273.15 + T) * 0.72 + 70.2
    Rv = 461.52  # J*Kg^-1*K^-1
    k = 1e-6 * (k3 / Tm + k2) * Rv
    k = 1.0 / k
    return k

def kappa_yuval(T, k2=64.79, k3=3.776e5):
    """T in celsious"""
    Tm = (273.15 + T) * 0.72 + 70.2
    Rv = 461.52  # J*Kg^-1*K^-1
    k = 1e-6 * (k3 / Tm + k2 / Rv)
    k = 1.0 / k
    return k

def Zscore_xr(da, dim='time'):
    """input is a dattarray of data and output is a dattarray of Zscore
    for the dim"""
    z = (da - da.mean(dim=dim)) / da.std(dim=dim)
    return z

def desc_nan(data, verbose=True):
    """count only NaNs in data and returns the thier amount and the non-NaNs"""
    import numpy as np
    import xarray as xr

    def nan_da(data):
        nans = np.count_nonzero(np.isnan(data.values))
        non_nans = np.count_nonzero(~np.isnan(data.values))
        if verbose:
            print(str(type(data)))
            print(data.name + ': non-NaN entries: ' + str(non_nans) + ' of total ' +
                  str(data.size) + ', shape:' + str(data.shape) + ', type:' +
                  str(data.dtype))
            print('Dimensions:')
        dim_nn_list = []
        for dim in data.dims:
            dim_len = data[dim].size
            dim_non_nans = np.int(data.dropna(dim)[dim].count())
            dim_nn_list.append(dim_non_nans)
            if verbose:
                print(dim + ': non-NaN labels: ' +
                      str(dim_non_nans) + ' of total ' + str(dim_len))
        return non_nans
    if isinstance(data, xr.DataArray):
        nn_dict = nan_da(data)
        return nn_dict
    elif isinstance(data, np.ndarray):
        nans = np.count_nonzero(np.isnan(data))
        non_nans = np.count_nonzero(~np.isnan(data))
        if verbose:
            print(str(type(data)))
            print('non-NaN entries: ' + str(non_nans) + ' of total ' +
                  str(data.size) + ', shape:' + str(data.shape) + ', type:' +
                  str(data.dtype))
    elif isinstance(data, xr.Dataset):
        for varname in data.data_vars.keys():
            non_nans = nan_da(data[varname])
    return non_nans
