#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:22:51 2019

@author: ziskin
"""
ims_path = work_yuval / 'IMS_T'


def read_ims(path, filename):
    import pandas as pd
    """parse ims stations meta-data"""
    if '10mins' in filename:
        ims = pd.read_excel(path / filename,
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


def produce_geo_ims(path, filename='IMS_10mins_meta_data.xlsx',
                    closed_stations=False, plot=True):
    import geopandas as gpd
    import numpy as np
    isr = gpd.read_file(path / 'israel_demog2012.shp')
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
    df.drop(['alias', 'description'], axis=1, inplace=True)
    return df


def download_ims_data(geo_df, path, end_date='2019-04-15'):
    import requests
    import glob
    import pandas as pd

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
        to_download = list(set(geo_df.index.values.tolist()
                               ).difference(set(already_dl)))
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
        print(
            'Getting IMS data for {} station(ID={}) from channel {}'.format(
                name,
                station_id,
                channel_id))
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


def produce_T_dataset(path, save=True, unique_index=True,
                      clim_period='dayofyear', resample_method='ffill'):
    import xarray as xr
    da_list = []
    for file_and_path in path.glob('*TD.nc'):
        da = xr.open_dataarray(file_and_path)
        print('post-proccessing temperature data for {} station'.format(da.name))
        da_list.append(post_proccess_ims(da, unique_index, clim_period,
                                         resample_method))
    ds = xr.merge(da_list)
    if save:
        filename = 'IMS_TD_israeli_for_gps.nc'
        print('saving {} to {}'.format(filename, path))
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(path / filename, 'w', encoding=encoding)
        print('Done!')
    return ds