#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:22:51 2019

@author: ziskin
"""
from PW_startup import *
ims_path = work_yuval / 'IMS_T'
gis_path = work_yuval / 'gis'


def get_meta_data_hourly_ims_climate_database(ds):
    import pandas as pd
    name_list = []
    for name, da in ds.data_vars.items():
        data = [name.split('_')[0], da.attrs['station_id'], da.attrs['lat'],
                da.attrs['lon'], da.attrs['height']]
        name_list.append(data)
    df = pd.DataFrame(name_list)
    df.columns = ['name', 'id', 'lat', 'lon', 'height']
    return df


def proccess_hourly_ims_climate_database(path=ims_path, var='tas',
                                         times=('1996', '2019')):
    import xarray as xr
    import numpy as np
    ds = xr.open_dataset(path / 'hourly_ims.nc')
    if var is not None:
        ds = ds.sel({'var': var})
        print('selecting {} variables'.format(var))
        if times is not None:
            print('selecting times from {} to {}'.format(times[0], times[1]))
            ds = ds.sel(time=slice(times[0], times[1]))
            to_drop_list = []
            for name, da in ds.data_vars.items():
                if (np.isnan(da) == True).all().item():
                    to_drop_list.append(name)
            ds = ds.drop(to_drop_list)
    return ds


def read_hourly_ims_climate_database(path=ims_path / 'ground',
                                     savepath=ims_path):
    """downloaded from tau...ds is a dataset of all stations,
    times is a time period"""
    import pandas as pd
    import xarray as xr
    from aux_gps import print_saved_file
    da_list = []
    for file in sorted(path.glob('*.csv')):
        name = file.as_posix().split('/')[-1].split('_')[0]
        sid = file.as_posix().split('/')[-1].split('_')[1]
        array_name = '_'.join([name, sid])
        print('reading {} station...'.format(array_name))
        df = pd.read_csv(file, index_col='time')
        df.index = pd.to_datetime(df.index)
        df.drop(labels=['Unnamed: 0', 'name'], axis=1, inplace=True)
        lat = df.loc[:, 'lat'][0]
        lon = df.loc[:, 'lon'][0]
        height = df.loc[:, 'height'][0]
        df.drop(labels=['lat', 'lon', 'height'], axis=1, inplace=True)
        da = df.to_xarray().to_array(dim='var')
        da.name = array_name
        da.attrs['station_id'] = sid
        da.attrs['lat'] = lat
        da.attrs['lon'] = lon
        da.attrs['height'] = height
        da_list.append(da)
    ds = xr.merge(da_list)
    print('Done!')
    if savepath is not None:
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(savepath / 'hourly_ims.nc', 'w', encoding=encoding)
        print_saved_file('hourly_ims.nc', path)
    return ds


def read_ims_metadata_from_files(path=gis_path,
                                 filename='IMS_10mins_meta_data.xlsx'):
    # for longer climate archive data use filename = IMS_climate_archive_meta_data.xls
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
    ims = read_ims_metadata_from_files(path, filename)
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


def ims_api_get_meta(active_only=True, channel_name='TD'):
    import requests
    import pandas as pd
    """get meta data on 10mins ims stations"""
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
    channelId_list = []
    for index, row in stations_10mins.iterrows():
        lat_.append(row['location']['latitude'])
        lon_.append(row['location']['longitude'])
        channel = [x['channelId'] for x in row.monitors if x['name'] ==
                   channel_name]
        if channel:
            channelId_list.append(channel[0])
        else:
            channelId_list.append(None)
    stations_10mins['lat'] = lat_
    stations_10mins['lon'] = lon_
    stations_10mins[channel_name + '_channel'] = channelId_list
    stations_10mins.drop(['location', 'StationTarget', 'stationsTag'],
                         axis=1, inplace=True)
    return stations_10mins


def download_ims_single_station(stationid, savepath=ims_path,
                                channel_name='TD'):
    """download single station with channel_name from earliest to latest.
    if chanel_name is None, download all channels"""
    import requests
    import pandas as pd
    # TODO: add all channels download...
    def parse_ims_to_df(raw_data, ch_name):
        """gets ims station raw data, i.e., r.json()['data'] and returns
        a pandas dataframe"""
        import pandas as pd
        if ch_name is not None:
            datetimes = [x['datetime'] for x in raw_data]
            data = [x['channels'][0] for x in raw_data]
            df = pd.DataFrame.from_records(data,
                                           index=pd.to_datetime(datetimes,
                                                                utc=True))
            df.drop(['alias', 'description'], axis=1, inplace=True)
            cols = [ch_name + '_' + x for x in df.columns]
            df.columns = cols
        elif ch_name is None:
            # add all channels d/l here:
            datetimes = [x['datetime'] for x in raw_data]
            names = [x['name'] for x in data['channels']]
            keys = [*data['channels'][0].keys()]
        return df

    def to_dataarray(df, meta):
        # add all channels d/l here:
        import pandas as pd
        ds = df.to_xarray()
        ds['time'] = pd.to_datetime(ds.time)
        channel_name = [*ds.data_vars.keys()][0].split('_')[0]
        channel_id = ds[channel_name + '_id'].isel(time=0).values.item()
        to_drop = [x for x in ds.data_vars.keys() if 'value' not in x]
        ds = ds.drop(to_drop)
        da = ds[channel_name + '_value'].reset_coords(drop=True)
        da.name = meta['name']
        da.attrs['channel_id'] = int(channel_id)
        da.attrs['channel_name'] = channel_name
        da.attrs['station_name'] = meta['name']
        da.attrs['station_id'] = meta['id']
        da.attrs['active'] = meta['active']
        da.attrs['station_lat'] = str(meta['loc']['latitude'])
        da.attrs['station_lon'] = str(meta['loc']['longitude'])
        for key, value in da.attrs.items():
            print(key, value)
        return da

    def get_dates_list(start_date, end_date):
        """divide the date span into full 1 years and a remainder, tolist"""
        import numpy as np
        import pandas as pd
        end_date = pd.to_datetime(end_date)
        start_date = pd.to_datetime(start_date)
        s_year = start_date.year
        e_year = end_date.year
        years = np.arange(s_year, e_year + 1)
        dates = [start_date.replace(year=x) for x in years]
        if (end_date - dates[-1]).days > 0:
            dates.append(end_date)
        return dates

    myToken = 'f058958a-d8bd-47cc-95d7-7ecf98610e47'
    headers = {'Authorization': 'ApiToken ' + myToken}
    r = requests.get('https://api.ims.gov.il/v1/envista/stations/',
                     headers=headers)
    stations_10mins = pd.DataFrame(r.json())
    meta = {}
    st_name = stations_10mins['name'].where(
            stations_10mins['stationId'] == stationid).dropna().item()
    location = stations_10mins['location'].where(
            stations_10mins['stationId'] == stationid).dropna().item()
    active = stations_10mins['active'].where(
            stations_10mins['stationId'] == stationid).dropna().item()
    meta['name'] = '-'.join(st_name.split(' '))
    meta['id'] = stationid
    meta['loc'] = location
    meta['active'] = active
    r_early = requests.get('https://api.ims.gov.il/v1/envista/stations/' +
                           str(stationid) + '/data/earliest', headers=headers)
    r_late = requests.get('https://api.ims.gov.il/v1/envista/stations/' +
                          str(stationid) + '/data/latest', headers=headers)
    data = r_early.json()['data'][0]
    earliest = pd.to_datetime(data['datetime']).strftime('%Y-%m-%d')
    data = r_late.json()['data'][0]
    latest = pd.to_datetime(data['datetime']).strftime('%Y-%m-%d')
    print(
         'Downloading station {} with id: {}, from {} to {}'.format(
                 st_name,
                 stationid,
                 earliest,
                 latest))
    # one channel download:
    if channel_name is not None:
        channel_id = [x['id'] for x in data['channels']
                      if x['name'] == channel_name]
        if channel_id:
            print('getting just {} channel with id: {}'.format(channel_name,
                                                               channel_id[0]))
            ch_id = channel_id[0]
            dates = get_dates_list(earliest, latest)
            df_list = []
            for i in range(len(dates) - 1):
                first_date = dates[i].strftime('%Y/%m/%d')
                last_date = dates[i + 1].strftime('%Y/%m/%d')
                print('proccesing dates: {} to {}'.format(first_date,
                                                          last_date))
                dl_command = ('https://api.ims.gov.il/v1/envista/stations/' +
                              str(stationid) + '/data/' + str(ch_id) +
                              '?from=' + first_date + '&to=' + last_date)
                r = requests.get(dl_command, headers=headers)
                if r.status_code == 204:  # i.e., no content:
                    print('no content for this search, skipping...')
                    continue
                print('parsing to dataframe...')
                df_list.append(parse_ims_to_df(r.json()['data'], channel_name))
            print('concatanating df and transforming to xarray...')
            df_all = pd.concat(df_list)
            # only valid results:
            # df_valid = df_all[df_all['valid']]
            df_all.index.name = 'time'
            # remove duplicated index values:
            df_all = df_all[~df_all.index.duplicated()]
            first = df_all.index[0]
            last = df_all.index[-1]
            new_index = pd.date_range(first, last, freq='10min')
            df_all = df_all.reindex(new_index)
            valid_name = channel_name + '_valid'
            value_name = channel_name + '_value'
            df_all[valid_name].fillna(False, inplace=True)
            # replace non valid measurments with nans
            new_vals = df_all[value_name].where(df_all[valid_name])
            df_all[value_name] = new_vals
            df_all.index.name = 'time'
            da = to_dataarray(df_all, meta)
            filename = '_'.join([meta['name'], str(meta['id']), channel_name,
                                 '10mins']) + '.nc'
            comp = dict(zlib=True, complevel=9)  # best compression
            encoding = {var: comp for var in da.to_dataset().data_vars}
            print('saving to {} to {}'.format(filename, savepath))
            da.to_netcdf(savepath / filename, 'w', encoding=encoding)
            print('done!')
    # all channels download add support here:
    elif channel_name is None:
        print('getting all channels...')
        dates = get_dates_list(earliest, latest)
        df_list = []
        for i in range(len(dates) - 1):
            first_date = dates[i].strftime('%Y/%m/%d')
            last_date = dates[i + 1].strftime('%Y/%m/%d')
            print('proccesing dates: {} to {}'.format(first_date,
                                                      last_date))
            dl_command = ('https://api.ims.gov.il/v1/envista/stations/' +
                          str(stationid) + '/data?from=' + first_date +
                          '&to=' + last_date)
            r = requests.get(dl_command, headers=headers)
            if r.status_code == 204:  # i.e., no content:
                print('no content for this search, skipping...')
                break
            print('parsing to dataframe...')
            df_list.append(parse_ims_to_df(r.json()['data'], None))
    return 

#def download_ims_data(geo_df, path, end_date='2019-04-15'):
#    import requests
#    import glob
#    import pandas as pd
#
#    def to_dataarray(df, index, row):
#        import pandas as pd
#        ds = df.to_xarray()
#        ds['time'] = pd.to_datetime(ds.time)
#        channel_name = ds.name.isel(time=0).values
#        channel_id = ds.id.isel(time=0).values
#        ds = ds.drop(['id', 'name'])
#        da = ds.to_array(dim='TD', name=str(index))
#        da.attrs['channel_id'] = channel_id.item()
#        da.attrs['channel_name'] = channel_name.item()
#        da.attrs['station_name'] = row.name_english
#        da.attrs['station_id'] = row.ID
#        da.attrs['station_lat'] = row.lat
#        da.attrs['station_lon'] = row.lon
#        da.attrs['station_alt'] = row.alt
#        return da
#
#    def get_dates_list(starting_date, end_date):
#        """divide the date span into full 1 years and a remainder, tolist"""
#        import numpy as np
#        import pandas as pd
#        end_date = pd.to_datetime(end_date)
#        s_year = starting_date.year
#        e_year = end_date.year
#        years = np.arange(s_year, e_year + 1)
#        dates = [starting_date.replace(year=x) for x in years]
#        if (end_date - dates[-1]).days > 0:
#            dates.append(end_date)
#        return dates
#
#    myToken = 'f058958a-d8bd-47cc-95d7-7ecf98610e47'
#    headers = {'Authorization': 'ApiToken ' + myToken}
#    already_dl = []
#    for paths in glob.glob(path+'*_TD.nc'):
#        already_dl.append(paths.split('/')[-1].split('.')[0].split('_')[0])
#        to_download = list(set(geo_df.index.values.tolist()
#                               ).difference(set(already_dl)))
#    if to_download:
#        geo_df = geo_df.loc[to_download]
#    for index, row in geo_df.iterrows():
#        # get a list of dates to download: (1 year old parts)
#        dates = get_dates_list(row.starting_date, end_date)
#        # get station id and channel id(only dry temperature):
#        name = row.name_english
#        station_id = row.ID
#        channel_id = row.channel_id
#        # if tempertue is not measuered in station , skip:
#        if channel_id == 0:
#            continue
#        print(
#            'Getting IMS data for {} station(ID={}) from channel {}'.format(
#                name,
#                station_id,
#                channel_id))
#        # loop over one year time span and download:
#        df_list = []
#        for i in range(len(dates) - 1):
#            first_date = dates[i].strftime('%Y/%m/%d')
#            last_date = dates[i + 1].strftime('%Y/%m/%d')
#            print('proccesing dates: {} to {}'.format(first_date, last_date))
#            dl_command = ('https://api.ims.gov.il/v1/envista/stations/' +
#                          str(station_id) + '/data/' + str(channel_id) +
#                          '?from=' + first_date + '&to=' + last_date)
#            r = requests.get(dl_command, headers=headers)
#            if r.status_code == 204:  # i.e., no content:
#                print('no content for this search, skipping...')
#                break
#            print('parsing to dataframe...')
#            df_list.append(parse_ims_to_df(r.json()['data']))
#        print('concatanating df and transforming to xarray...')
#        df_all = pd.concat(df_list)
#        # only valid results:
#        # df_valid = df_all[df_all['valid']]
#        df_all.index.name = 'time'
#        da = to_dataarray(df_all, index, row)
#        filename = index + '_TD.nc'
#        comp = dict(zlib=True, complevel=9)  # best compression
#        encoding = {var: comp for var in da.to_dataset().data_vars}
#        print('saving to {} to {}'.format(filename, path))
#        da.to_netcdf(path + filename, 'w', encoding=encoding)
#        print('done!')
#    #    return df_list
#    # pick station and time span
#    # download
#    # call parse_ims_to_df
#    # concatanate and save to nc
#    return


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


def fill_missing_single_ims_station(da, unique_index=True,
                                    clim_period='dayofyear'):
    """fill in the missing time data for the ims station of any variable with
    clim_period is the fine tuning of the data replaced, options are:
        month, weekofyear, dayofyear. return a dataset with original and filled
        dataarray"""
    # da should be dattaarray and not dataset!
    import pandas as pd
    import numpy as np
    import xarray as xr
    from aux_gps import get_unique_index
    if unique_index:
        ind_diff = da.size - get_unique_index(da).size
        da = get_unique_index(da)
        print('dropped {} non-unique datetime index.'.format(ind_diff))
    # make sure no coords are in xarray:
    da = da.reset_coords(drop=True)
    # make sure nans are dropped:
    nans_diff = da.size - da.dropna('time').size
    print('dropped {} nans.'.format(nans_diff))
    da_no_nans = da.dropna('time')
    if clim_period == 'month':
        grpby = 'time.month'
        print('long term monthly mean data replacment selected')
    elif clim_period == 'weekofyear':
        print('long term weekly mean data replacment selected')
        grpby = 'time.weekofyear'
    elif clim_period == 'dayofyear':
        print('long term daily mean data replacment selected')
        grpby = 'time.dayofyear'
    # first compute the climatology and the anomalies:
    print('computing anomalies:')
    climatology = da_no_nans.groupby(grpby).mean('time')
    anom = da_no_nans.groupby(grpby) - climatology
    # then comupte the diurnal cycle:
    print('computing diurnal change:')
    diurnal = anom.groupby('time.hour').mean('time')
    # assemble old and new time and comupte the difference:
    print('assembeling missing data:')
    old_time = pd.to_datetime(da_no_nans.time.values)
    freq = pd.infer_freq(da.time.values)
    new_time = pd.date_range(da_no_nans.time.min().item(),
                             da_no_nans.time.max().item(), freq=freq)
    missing_time = pd.to_datetime(
        sorted(
            set(new_time).difference(
                set(old_time))))
    missing_data = np.empty((missing_time.shape))
    print('proccessing missing data...')
    for i in range(len(missing_data)):
        # replace data as to monthly long term mean and diurnal hour:
        # missing_data[i] = (climatology.sel(month=missing_time[i].month) +
        missing_data[i] = (climatology.sel({clim_period: getattr(missing_time[i],
                                                                 clim_period)}) +
                           diurnal.sel(hour=missing_time[i].hour))
    series = pd.Series(data=missing_data, index=missing_time)
    series.index.name = 'time'
    mda = series.to_xarray()
    mda.name = da.name
    new_data = xr.concat([mda, da_no_nans], 'time')
    new_data = new_data.sortby('time')
    # copy attrs:
    new_data.attrs = da.attrs
    new_data.attrs['description'] = 'missing data was '\
                                    'replaced by using ' + clim_period \
                                    + ' mean and hourly signal.'
    # put new_data and missing data into a dataset:
    dataset = new_data.to_dataset(name=new_data.name)
    dataset[new_data.name + '_original'] = da_no_nans
    print('done!')
    return dataset

#    # resample to 5min with resample_method: (interpolate is very slow)
#    print('resampling to 5 mins using {}'.format(resample_method))
#    # don't resample the missing data:
#    dataset = dataset.resample(time='5min').ffill()