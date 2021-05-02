#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 15:46:46 2021

@author: shlomi
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:29:10 2019
This script needs more work, mainly on updating new data
@author: ziskin
implement a mode arg which will retain the long term update behaviour but will also
add a real-time mode, which with a timestamp and window takes a parameter (TD)
snapshot of all the stations togather and saves it to disk
write another script with click!
"""


# def load_saved_station(path, station_id, channel):
#    from aux_gps import path_glob
#    import xarray as xr
#    files = path_glob(path, '*_{}_{}_10mins.nc'.format(station_id, channel))
#    if len(files) == 0:
#        return False
#    elif len(files) == 1:
#        return xr.load_dataset(files[0])
#    elif len(files) > 1:
#        raise ValueError('too many files with the same glob str')
#
#
# def parse_filename(file_path):
##    filename = file_path.as_posix().split('/')[-1].split('.')[0]
##    station_name = filename.split('_')[0]
##    station_id = filename.split('_')[1]
##    channel = filename.split('_')[2]
# return station_name, station_id, channel
#
#


import click
from loguru import logger
channels = ['BP', 'DiffR', 'Grad', 'NIP', 'Rain', 'RH', 'STDwd', 'TD',
            'TDmax', 'TDmin', 'TG', 'Time', 'WD', 'WDmax', 'WS', 'WS10mm',
            'WS1mm', 'WSmax']


def parse_single_station(data):
    import pandas as pd
    datetimes = [x['datetime'] for x in data]
    dfl = []
    for i, dt in enumerate(datetimes):
        # get all channels into dataframe for each datetime:
        df = pd.DataFrame(data[i]['channels'])
        df = df[df['valid'] == True]
        df = df[['name', 'value']]
        df = df.T
        df.columns = df.loc['name'].tolist()
        df = df.drop('name', axis=0)
        df.index = [pd.to_datetime(dt)]
        dfl.append(df)
    df = pd.concat(dfl)
    return df


@click.command()
@click.option('--savepath', '-s', help='a full path to download the files, e.g., /home/ziskin/Work_Files/PW_yuval/IMS_T/10mins.', type=click.Path(exists=True))
@click.option('--window', '-w', nargs=1, default=30, help='how many hours before now to get the data', type=click.IntRange(min=1, max=120))
def ims_download(*args, **kwargs):
    """
    Downloads a 10mins parameter from the IMS for the last <window> hours for all stations.

    Parameters
    ----------
    savepath : Path or string
        a full path to download the files, e.g., /home/ziskin/Work_Files/PW_yuval/IMS_T/10mins.
       Returns
    -------
    None.

    """
    import pandas as pd
    import requests
    from requests.exceptions import SSLError
    window = kwargs['window']
    savepath = kwargs['savepath']
    # channel = kwargs['channel']
    now_dt = pd.Timestamp.now().floor('H')
    start_dt = now_dt - pd.Timedelta('{} hour'.format(window))
    logger.info('Downloading IMS to {} and window {}'.format(
        savepath, window))
    logger.info('Fetching IMS from {} to {}.'.format(start_dt, now_dt))
    # use API from IMS
    myToken = 'f058958a-d8bd-47cc-95d7-7ecf98610e47'
    headers = {'Authorization': 'ApiToken ' + myToken}
    # download meta-data on stations:
    r = requests.get('https://api.ims.gov.il/v1/envista/stations/',
                     headers=headers)
    stations_df = pd.DataFrame(r.json())
    # use only active 10mins stations:
    stations_df = stations_df[stations_df['timebase'] == 10]
    stations_df = stations_df[stations_df['active'] == True]
    for i, row in stations_df.iterrows():
        st_id = row['stationId']
        st_name = row['name']
        last = now_dt.strftime('%Y/%m/%dT%H:00:00')
        first = start_dt.strftime('%Y/%m/%dT%H:00:00')
        dl_command = ('https://api.ims.gov.il/v1/envista/stations/' +
                      str(st_id) + '/data/?from=' + first + '&to=' + last)
        try:
            r = requests.get(dl_command, headers=headers)
        except SSLError:
            logger.warning('SSLError')
            r = requests.get(dl_command, headers=headers)
        if r.status_code == 204:  # i.e., no content:
            logger.warning('no content for this search, skipping...')
            continue
        logger.info('parsing data from {} to dataframe.'.format(st_name))
        parse_single_station(r.json()['data'])
    return

# def check_ds_last_datetime(ds, fmt=None):
#     """return the last datetime of the ds"""
#     import pandas as pd
#     import xarray as xr
#     if isinstance(ds, xr.DataArray):
#         ds = ds.to_dataset(n.strftime('%Y/%m/%d')ame=ds.name)
#     # assume time series with one time dim:
#     time_dim = list(set(ds.dims))[0]
#     dvars = [x for x in ds.data_vars]
#     if dvars:
#         dt = ds[dvars[0]].dropna(time_dim)[time_dim][-1].values
#         dt = pd.to_datetime(dt)
#         if fmt is None:
#             return dt
#         else:
#             return dt.strftime(fmt)
#     else:
#         raise KeyError("dataset is empty ( no data vars )")


def download_ims_single_station(stationid, savepath=None,
                                channel_name='TD', update=None):
    """download single station with channel_name from earliest to latest.
    if chanel_name is None, download all channels"""
    import requests
    import pandas as pd
    import logging
    from requests.exceptions import SSLError

    def parse_ims_to_df(raw_data, ch_name):
        """gets ims station raw data, i.e., r.json()['data'] and returns
        a pandas dataframe"""
        import pandas as pd
        from pytz import timezone
        if ch_name is not None:
            datetimes = [x['datetime'] for x in raw_data]
            # Local datetimes:
            dts = [x.split('+')[0] for x in datetimes]
            # bool mask for DST:
            dts_dst = [x.split('+')[-1] for x in datetimes]
            dst_bool = [True if x == '03:00' else False for x in dts_dst]
            jer = timezone('Asia/Jerusalem')
            data = [x['channels'][0] for x in raw_data]
            df = pd.DataFrame.from_records(data, index=pd.to_datetime(dts))
            df.drop(['alias', 'description'], axis=1, inplace=True)
            cols = [ch_name + '_' + x for x in df.columns]
            df.columns = cols
            df = df.tz_localize(jer, ambiguous=dst_bool,
                                nonexistent='shift_forward')
            df = df.tz_convert('UTC')
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
        ds['time'] = pd.to_datetime(ds.time.values)
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

    logger = logging.getLogger('ims_downloader')
    myToken = 'f058958a-d8bd-47cc-95d7-7ecf98610e47'
    headers = {'Authorization': 'ApiToken ' + myToken}
    r = requests.get('https://api.ims.gov.il/v1/envista/stations/',
                     headers=headers)
    stations_10mins = pd.DataFrame(r.json())
    meta = {}
    st_name = stations_10mins['name'].where(
        stations_10mins['stationId'] == stationid).dropna()
    location = stations_10mins['location'].where(
        stations_10mins['stationId'] == stationid).dropna()
    active = stations_10mins['active'].where(
        stations_10mins['stationId'] == stationid).dropna()
    meta['name'] = '-'.join(st_name.iloc[0].split())
    meta['id'] = stationid
    meta['loc'] = location.iloc[0]
    meta['active'] = active.iloc[0]
    r_early = requests.get('https://api.ims.gov.il/v1/envista/stations/' +
                           str(stationid) + '/data/earliest', headers=headers)
    r_late = requests.get('https://api.ims.gov.il/v1/envista/stations/' +
                          str(stationid) + '/data/latest', headers=headers)
    data = r_early.json()['data'][0]
    if update is not None:
        earliest = update + pd.Timedelta(10, unit='m')
    else:
        earliest = pd.to_datetime(data['datetime']).strftime('%Y-%m-%d')
    data = r_late.json()['data'][0]
    latest = pd.to_datetime(data['datetime']).strftime('%Y-%m-%d')
    # check if trying to update stations in the same day:
    if earliest == latest:
        logger.error('Wait for at least one day before trying to update...')
    logger.info(
        'Downloading station {} with id: {}, from {} to {}'.format(
            st_name.values[0],
            stationid,
            earliest,
            latest))
    # one channel download:
    if channel_name is not None:
        channel_id = [x['id'] for x in data['channels']
                      if x['name'] == channel_name]
        if channel_id:
            logger.info('getting just {} channel with id: {}'.format(channel_name,
                                                                     channel_id[0]))
            ch_id = channel_id[0]
            dates = get_dates_list(earliest, latest)
            df_list = []
            for i in range(len(dates) - 1):
                first_date = dates[i].strftime('%Y/%m/%d')
                last_date = dates[i + 1].strftime('%Y/%m/%d')
                logger.info('proccesing dates: {} to {}'.format(first_date,
                                                                last_date))
                dl_command = ('https://api.ims.gov.il/v1/envista/stations/' +
                              str(stationid) + '/data/' + str(ch_id) +
                              '?from=' + first_date + '&to=' + last_date)
                try:
                    r = requests.get(dl_command, headers=headers)
                except SSLError:
                    logger.warning('SSLError')
                    r = requests.get(dl_command, headers=headers)
                if r.status_code == 204:  # i.e., no content:
                    logger.warning('no content for this search, skipping...')
                    continue
                logger.info('parsing to dataframe...')
                df_list.append(parse_ims_to_df(r.json()['data'], channel_name))
            logger.info('concatanating df and transforming to xarray...')
            try:
                df_all = pd.concat(df_list)
            except ValueError:
                logger.warning('no new data on station {}.'.format(stationid))
                return None
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
            if update is not None:
                return da
            else:
                filename = '_'.join(['-'.join(meta['name'].split(' ')), str(meta['id']), channel_name,
                                     '10mins']) + '.nc'
                comp = dict(zlib=True, complevel=9)  # best compression
                encoding = {var: comp for var in da.to_dataset().data_vars}
                logger.info('saving to {} to {}'.format(filename, savepath))
                da.to_netcdf(savepath / filename, 'w', encoding=encoding)
                # print('done!')
    # all channels download add support here:
    elif channel_name is None:
        logger.info('getting all channels...')
        dates = get_dates_list(earliest, latest)
        df_list = []
        for i in range(len(dates) - 1):
            first_date = dates[i].strftime('%Y/%m/%d')
            last_date = dates[i + 1].strftime('%Y/%m/%d')
            logger.info('proccesing dates: {} to {}'.format(first_date,
                                                            last_date))
            dl_command = ('https://api.ims.gov.il/v1/envista/stations/' +
                          str(stationid) + '/data?from=' + first_date +
                          '&to=' + last_date)
            r = requests.get(dl_command, headers=headers)
            if r.status_code == 204:  # i.e., no content:
                logger.warning('no content for this search, skipping...')
                break
            logger.info('parsing to dataframe...')
            df_list.append(parse_ims_to_df(r.json()['data'], None))
    return


def download_all_10mins_ims(savepath, channel_name='TD'):
    """download all 10mins stations per specified channel, updateing fields is
    automatic"""
    from aux_gps import path_glob
    import xarray as xr
    import logging
    logger = logging.getLogger('ims_downloader')
    glob = '*_{}_10mins.nc'.format(channel_name)
    files = sorted(path_glob(savepath, glob, return_empty_list=True))
    files = [x for x in files if x.is_file()]
    if files:
        time_dim = list(set(xr.open_dataarray(files[0]).dims))[0]
    last_dates = [check_ds_last_datetime(xr.open_dataarray(x)) for x in files]
    st_id_downloaded = [int(x.as_posix().split(
        '/')[-1].split('_')[1]) for x in files]
    d = dict(zip(st_id_downloaded, last_dates))
    stations = ims_api_get_meta(active_only=True, channel_name=channel_name)
    for index, row in stations.iterrows():
        st_id = row['stationId']
        if st_id not in d.keys():
            download_ims_single_station(savepath=savepath,
                                        channel_name=channel_name,
                                        stationid=st_id, update=None)
        elif st_id in d.keys():
            logger.info('updating station {}...'.format(st_id))
            da = download_ims_single_station(savepath=savepath,
                                             channel_name=channel_name,
                                             stationid=st_id, update=d[st_id])
            if da is not None:
                file = path_glob(
                    savepath, '*_{}_{}_10mins.nc'.format(st_id, channel_name))[0]
                da_old = xr.load_dataarray(file)
                da = xr.concat([da, da_old], time_dim)
                filename = '_'.join(['-'.join(row['name'].split(' ')), str(st_id), channel_name,
                                     '10mins']) + '.nc'
                comp = dict(zlib=True, complevel=9)  # best compression
                encoding = {var: comp for var in da.to_dataset().data_vars}
                logger.info('saving to {} to {}'.format(filename, savepath))
                try:
                    da.to_netcdf(savepath / filename, 'w', encoding=encoding)
                except PermissionError:
                    (savepath / filename).unlink()
                    da.to_netcdf(savepath / filename, 'w', encoding=encoding)
            # print('done!')
        else:
            logger.warning('station {} is already in {}, skipping...'.format(st_id,
                                                                             savepath))
    return


if __name__ == '__main__':

    ims_download()

# if __name__ == '__main__':
#     # import argparse
#     import sys
#     from ims_procedures import ims_api_get_meta
#     from pathlib import Path
#     from aux_gps import configure_logger
#     logger = configure_logger('ims_downloader')
#     channels = ['BP', 'DiffR', 'Grad', 'NIP', 'Rain', 'RH', 'STDwd', 'TD',
#                 'TDmax', 'TDmin', 'TG', 'Time', 'WD', 'WDmax', 'WS', 'WS10mm',
#                 'WS1mm', 'WSmax']
#     savepath = Path('/home/ziskin/Work_Files/PW_yuval/IMS_T/10mins')
#     parser = argparse.ArgumentParser(description='a command line tool for downloading all 10mins stations from the IMS with specific variable')
#     optional = parser._action_groups.pop()
#     required = parser.add_argument_group('required arguments')
#     # remove this line: optional = parser...
#     required.add_argument('--savepath', help="a full path to download the files, e.g., /home/ziskin/Work_Files/PW_yuval/IMS_T/10mins", type=check_path)
#     required.add_argument('--channel', help="10 mins channel name , e.g., TD, BP or RH",
#                           choices=channels)
#     required.add_argument('--delete', action='store_true')  # its False
#     #optional.add_argument('--station', nargs='+',
#     #                      help='GPS station name, 4 UPPERCASE letters',
#     #                      type=check_station_name)
# #                          metavar=str(cds.start_year) + ' to ' + str(cds.end_year))
# #    optional.add_argument('--half', help='a spescific six months to download,\
# #                          e.g, 1 or 2', type=int, choices=[1, 2],
# #                          metavar='1 or 2')
#     parser._action_groups.append(optional)  # added this line
#     args = parser.parse_args()
#     # print(parser.format_help())
# #    # print(vars(args))
#     if args.savepath is None:
#         print('savepath is a required argument, run with -h...')
#         sys.exit()
# #    elif args.field is None:
# #        print('field is a required argument, run with -h...')
# #        sys.exit()
#     if args.channel is not None and not args.delete:
#         download_all_10mins_ims(args.savepath, channel_name=args.channel)
#         logger.info('Done!')
#     elif args.delete:
#         generate_delete(args.savepath, args.channel)
#     else:
#         raise ValueError('need to specify channel name!')