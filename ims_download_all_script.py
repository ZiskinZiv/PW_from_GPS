#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:29:10 2019
UPDATING is working well.
Will add a post-proccessing procedures for dividing for years, NaN filling and
PWV production.
@author: ziskin
Another script is ims_stations_download and is for real-time e.g., AXIS
(implemented using click)

"""


#def load_saved_station(path, station_id, channel):
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
##def parse_filename(file_path):
##    filename = file_path.as_posix().split('/')[-1].split('.')[0]
##    station_name = filename.split('_')[0]
##    station_id = filename.split('_')[1]
##    channel = filename.split('_')[2]
##    return station_name, station_id, channel
#


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


def configure_logger(name='general', filename=None):
    import logging
    import sys
    stdout_handler = logging.StreamHandler(sys.stdout)
    if filename is not None:
        file_handler = logging.FileHandler(filename=filename, mode='a')
        handlers = [file_handler, stdout_handler]
    else:
        handlers = [stdout_handler]

    logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
            handlers=handlers
            )
    logger = logging.getLogger(name=name)
    return logger


def query_yes_no(question, default="no"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    import sys
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def path_glob(path, glob_str='*.Z', return_empty_list=False):
    """returns all the files with full path(pathlib3 objs) if files exist in
    path, if not, returns FilenotFoundErro"""
    from pathlib import Path
#    if not isinstance(path, Path):
#        raise Exception('{} must be a pathlib object'.format(path))
    path = Path(path)
    files_with_path = [file for file in path.glob(glob_str) if file.is_file]
    if not files_with_path and not return_empty_list:
        raise FileNotFoundError('{} search in {} found no files.'.format(glob_str,
                                path))
    elif not files_with_path and return_empty_list:
        return files_with_path
    else:
        return files_with_path


def check_ds_last_datetime(ds, fmt=None):
    """return the last datetime of the ds"""
    import pandas as pd
    import xarray as xr
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name=ds.name)
    # assume time series with one time dim:
    time_dim = list(set(ds.dims))[0]
    dvars = [x for x in ds.data_vars]
    if dvars:
        dt = ds[dvars[0]].dropna(time_dim)[time_dim][-1].values
        dt = pd.to_datetime(dt)
        if fmt is None:
            return dt
        else:
            return dt.strftime(fmt)
    else:
        raise KeyError("dataset is empty ( no data vars )")


def check_path(path):
    import os
    from pathlib import Path
    path = str(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(path + ' does not exist...')
    return Path(path)


def generate_delete(savepath, channel_name):
    # from aux_gps import query_yes_no
    # from aux_gps import path_glob
    try:
        glob = '*_{}_10mins.nc'.format(channel_name)
        files_to_delete = path_glob(savepath, glob)
    except FileNotFoundError:
        print('skipping {} , because its empty or not existant..'.format(savepath))
        return
    print('WARNING for channel {}, ALL nc files in {} WILL BE DELETED!'.format(channel_name, savepath))
    to_delete = query_yes_no('ARE YOU SURE ?')
    if not to_delete:
        print('files NOT deleted...')
    else:
        [x.unlink() for x in files_to_delete]
        print('FILES DELETED!')
    return


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
            df = df.tz_localize(jer, ambiguous=dst_bool, nonexistent='shift_forward')
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
        da.attrs['active'] = str(meta['active'])
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
    # from aux_gps import path_glob
    import xarray as xr
    import logging
    logger = logging.getLogger('ims_downloader')
    glob = '*_{}_10mins.nc'.format(channel_name)
    files = sorted(path_glob(savepath, glob, return_empty_list=True))
    files = [x for x in files if x.is_file()]
    if files:
        time_dim = list(set(xr.open_dataarray(files[0]).dims))[0]
    last_dates = [check_ds_last_datetime(xr.open_dataarray(x)) for x in files]
    st_id_downloaded = [int(x.as_posix().split('/')[-1].split('_')[1]) for x in files]
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
                file = path_glob(savepath, '*_{}_{}_10mins.nc'.format(st_id, channel_name))[0]
                da_old = xr.load_dataarray(file)
                da = xr.concat([da, da_old], time_dim)
                da = da.sortby(time_dim)
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

def merge_stations_and_divide_to_yearly_monthly_files(savepath, channel_name='TD',
                                                      year_months=None):
    import xarray as xr
    import logging
    # import numpy as np
    import os
    from aux_gps import get_unique_index

    def save_yearly_monthly_file(ds_ym, month_savepath, filename):
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds_ym.data_vars}
        logger.info('saving to {} to {}'.format(filename, month_savepath))
        try:
            ds_ym.to_netcdf(month_savepath / filename, 'w', encoding=encoding)
        except PermissionError:
            (month_savepath / filename).unlink()
            ds_ym.to_netcdf(month_savepath / filename, 'w', encoding=encoding)

    logger = logging.getLogger('ims_downloader')
    glob = '*_{}_10mins.nc'.format(channel_name)
    files = sorted(path_glob(savepath, glob, return_empty_list=True))
    files = [x for x in files if x.is_file()]
    if files:
        time_dim = list(set(xr.open_dataarray(files[0]).dims))[0]
    logger.info('Reading all {} stations, merging them and saving as monthly files.'.format(len(files)))
    # create year savepath:
    month_savepath = savepath / 'monthly'
    if not month_savepath.is_dir():
        os.mkdir(month_savepath)
        logger.info('created {}.'.format(month_savepath))
    else:
        logger.info('{} already exist.'.format(month_savepath))
    # load stations list:
    dsl = [xr.open_dataset(x) for x in files]
    dsl = [x.sortby(time_dim) for x in dsl]
    if year_months is None:
        # this merge over the years is very slow, run it only once:
        # ds = xr.merge(dsl)
        yr_min = min([x[time_dim].min().dt.year.item() for x in dsl])
        mnth_min = min([x[time_dim].min().dt.month.item() for x in dsl])
        yr_max = max([x[time_dim].max().dt.year.item() for x in dsl])
        mnth_max = max([x[time_dim].max().dt.month.item() for x in dsl])
        start = pd.to_datetime('{}-{}'.format(yr_min, mnth_min), format='%Y-%m')
        end = pd.to_datetime('{}-{}'.format(yr_max, mnth_max), format='%Y-%m')
        # years = np.arange(yr_min, yr_max + 1)
        dts = pd.date_range(start=start, end=end, freq='m')
        dts=[x.strftime('%Y-%m') for x in dts]
        logger.info('Found {}-{} as years.'.format(yr_min, yr_max))
        for dt in dts:
            ds_dt_list = []
            for ds in dsl:
                try:
                    ds_ym = ds.sel({time_dim: dt})
                    ds_ym = get_unique_index(ds_ym, dim=time_dim)
                except KeyError:
                    continue
                ds_dt_list.append(ds_ym)
            # ds_year = [x.load() for x in ds_year]
            ds_yms = xr.merge(ds_dt_list)
            # ds_year = ds.sel({time_dim: str(year)})
            filename = 'IMS_ALL_{}_{}.nc'.format(channel_name, dt)
            save_yearly_monthly_file(ds_yms, month_savepath, filename)
    else:
        logger.info('Using user supplied dts {}.'.format(year_months))
        for dt in year_months:
            ds_dt_list = []
            for ds in dsl:
                try:
                    ds_ym = ds.sel({time_dim: dt})
                    ds_ym = get_unique_index(ds_ym, dim=time_dim)
                except KeyError:
                    continue
                ds_dt_list.append(ds_ym)
            # ds_year = [x.load() for x in ds_year]
            ds_yms = xr.merge(ds_dt_list)
            # ds_year = ds.sel({time_dim: str(year)})
            filename = 'IMS_ALL_{}_{}.nc'.format(channel_name, dt)
            save_yearly_monthly_file(ds_yms, month_savepath, filename)
    logger.info('Done saving IMS yearly {} files.'.format(channel_name))


def post_process_ims_stations(month_savepath, gis_path, dem_path,
                              stats_path, year_months=None):
    """fill TD with hourly mean if NaN and smooth, then fill in station_lat
    and lon and alt from DEM, finally interpolate to SOI coords and save"""
    # from aux_gps import fill_na_xarray_time_series_with_its_group
    from aux_gps import fillna_xarray_da_time_series_with_long_term_stats
    from ims_procedures import analyse_10mins_ims_field
    # from axis_process import produce_rinex_filenames_at_time_window
    from ims_procedures import IMS_interpolating_to_GNSS_stations_israel
    from aux_gps import save_ncfile
    from aux_gps import path_glob
    # import pandas as pd
    import xarray as xr
    # load IMS stats data for the stations:
    ds_stats = xr.load_dataset(stats_path/'IMS_TD_month_hour_stats.nc')
    # first select all or some years of IMS data from month_savepath
    files = sorted(path_glob(month_savepath, 'IMS_ALL_TD_*.nc'))
    if year_months is not None:
        new_files = []
        for file in files:
            year_month = file.as_posix().split('/')[-1].split('.')[0].split('_')[-1]
            if year_month in year_months:
                new_files.append(file)
        files = new_files
    for file in files:
        year_month = file.as_posix().split('/')[-1].split('.')[0].split('_')[-1]
        year = int(year_month.split('-')[0])
        logger.info('Performing post proccess on IMS TD for {}.'.format(year_month))
        ds = xr.load_dataset(file)
        das = []
        for da in ds:
            da_filled = fillna_xarray_da_time_series_with_long_term_stats(ds[da], ds_stats)
            if da_filled is None:
                logger.warning('could not find {} station in stats, skipping...'.format(da))
                continue
            das.append(da_filled)
        ds = xr.merge(das)
        ds = analyse_10mins_ims_field(ds=ds, var='TD', gis_path=gis_path,
                                      dem_path=dem_path)
        if int(year) >= 1996:
            ds_soi = IMS_interpolating_to_GNSS_stations_israel(
                        dt=None, start_year=str(year), verbose=True,
                        savepath=None, network='soi-apn', ds_td=ds,
                        cut_days_ago=None, axis_path=None, concat_all_TD=False)
            filename = 'SOI_TD_{}.nc'.format(year_month)
            save_ncfile(ds_soi, month_savepath, filename)
        # now_dt = pd.Timestamp.utcnow().floor('H')
        # names = produce_rinex_filenames_at_time_window(end_dt=now_dt,
        #                                                window=window)
        # st_str = names[0][4:8]
        # end_str = names[-1][4:8]
        # filename = 'AXIS_TD_{}-{}.nc'.format(st_str, end_str)
        # save_ncfile(ds_axis, savepath, filename)
    return ds


def produce_pwv_all_stations(td_month_path, rinex_path, mda_path):
    from PW_stations import load_mda
    from PW_stations import produce_GNSS_station_PW
    from aux_gps import fill_na_xarray_time_series_with_its_group
    from aux_gps import path_glob
    from aux_gps import save_ncfile
    import xarray as xr
    # first load mda:
    mda = load_mda(mda_path)
    # now load and concat all TD with GNSS - SOI stations:
    td_files = sorted(path_glob(td_month_path, 'SOI_TD_*.nc'))
    td_list = [xr.load_dataset(x) for x in td_files]
    td = xr.concat(td_list, 'time')
    td = td.sortby('time')
    # now loop over each station path, produce pwv and save:
    st_dirs = path_glob(rinex_path, '*/')
    st_dirs = [x for x in st_dirs if x.is_dir()]
    # st_dirs = [x for x in st_dirs if not x.as_posix().split('/')[-1].isnumeric()]
    # assert len(st_dirs) == 27
    pwv_list = []
    for st_dir in st_dirs:
        station = st_dir.as_posix().split('/')[-1]
        if station not in td:
            logger.error('{} not found in temperature database, skipping...'.format(station))
            continue
        file = st_dir/'gipsyx_solutions/{}_PPP_all_years.nc'.format(station.upper())
        if not file.is_file():
            logger.error('{} not found in PPP gipsyx solutions, skipping...'.format(station))
            continue
        wet = xr.open_dataset(file)['WetZ'].squeeze(drop=True)
        logger.info('loaded {}.'.format(file))
        wet_error = xr.open_dataset(file)['WetZ_error'].squeeze(drop=True)
        wet.name = station
        wet_error.name = station
        # resample temp to 5 mins and reindex to wet delay time:
        t = td[station].resample(time='5T').ffill().reindex_like(wet.time)
        # fill in NaNs with mean hourly signal:
        try:
            t_new = fill_na_xarray_time_series_with_its_group(t, grp='hour')
        except ValueError as e:
            logger.warning('encountered error: {}, skipping {}'.format(e, file))
            continue
        try:
            pwv = produce_GNSS_station_PW(wet, t_new, mda=mda,
                                          model_name='LR', plot=False)
            pwv_error = produce_GNSS_station_PW(wet_error, t_new, mda=mda,
                                                model_name='LR', plot=False)
            pwv_error.name = '{}_error'.format(pwv.name)
            pwv_ds = xr.merge([pwv, pwv_error])
            filename = '{}_PWV_all_years.nc'.format(station.upper())
            save_ncfile(pwv_ds, st_dir/'gipsyx_solutions', filename)
            pwv_list.append(pwv_ds)
        except ValueError as e:
            logger.warning('encountered error: {}, skipping {}'.format(e, file))
            continue


if __name__ == '__main__':
    import argparse
    import sys
    import pandas as pd
    # from ims_procedures import ims_api_get_meta
    from pathlib import Path
    # from aux_gps import configure_logger
    logger = configure_logger('ims_downloader')
    channels = ['BP', 'DiffR', 'Grad', 'NIP', 'Rain', 'RH', 'STDwd', 'TD',
                'TDmax', 'TDmin', 'TG', 'Time', 'WD', 'WDmax', 'WS', 'WS10mm',
                'WS1mm', 'WSmax']
    savepath = Path('/home/ziskin/Work_Files/PW_yuval/IMS_T/10mins')
    dem_path = Path('/home/ziskin/Work_Files/PW_yuval/AW3D30')
    gis_path = Path('/home/ziskin/Work_Files/PW_yuval/gis')
    work_yuval = Path('/home/ziskin/Work_Files/PW_yuval/')
    parser = argparse.ArgumentParser(description='a command line tool for downloading all 10mins stations from the IMS with specific variable')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # remove this line: optional = parser...
    required.add_argument('--savepath', help="a full path to download the files, e.g., /home/ziskin/Work_Files/PW_yuval/IMS_T/10mins", type=check_path)
    required.add_argument('--channel', help="10 mins channel name , e.g., TD, BP or RH",
                          choices=channels)
    required.add_argument('--delete', action='store_true')  # its False
    # don't need to specify folder unless files were moved:
    required.add_argument('--dem_path', help="a full path to DEM data", const=dem_path)
    required.add_argument('--gis_path', help="a full path to GIS data", const=dem_path)
    required.add_argument('--mda_path', help="a full path to mda model (ts-tm)", const=work_yuval)

    #optional.add_argument('--station', nargs='+',
    #                      help='GPS station name, 4 UPPERCASE letters',
    #                      type=check_station_name)
#                          metavar=str(cds.start_year) + ' to ' + str(cds.end_year))
#    optional.add_argument('--half', help='a spescific six months to download,\
#                          e.g, 1 or 2', type=int, choices=[1, 2],
#                          metavar='1 or 2')
    required.add_argument('--last_2_months', action='store_true')
    optional.add_argument('--datetimes', help="select the year-months that the IMS stations are saved as yearly files",
                          type=str,
                          nargs='+')
    parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()
    # print(parser.format_help())
#    # print(vars(args))
    if args.savepath is None:
        print('savepath is a required argument, run with -h...')
        sys.exit()
#    elif args.field is None:
#        print('field is a required argument, run with -h...')
#        sys.exit()
    if args.channel is not None and not args.delete:
        download_all_10mins_ims(args.savepath, channel_name=args.channel)
        if args.last_2_months:
            yr = pd.Timestamp.today().year
            month = pd.Timestamp.today().month
            last_month = (pd.Timestamp.today() - pd.Timedelta(30, unit='d')).month
            last_year = (pd.Timestamp.today() - pd.Timedelta(30, unit='d')).year
            args.datetimes = ['{}-{}'.format(last_year, last_month), '{}-{}'.format(yr, month)]
        # first merge all IMS TD stations and divide into year-monthly files:
        merge_stations_and_divide_to_yearly_monthly_files(args.savepath,
                                                          channel_name=args.channel,
                                                          year_months=args.datetimes)
        # then, post-process them and produce temperature at GNSS stations coords:
        post_process_ims_stations(args.savepath/'monthly', args.gis_path, args.dem_path,
                                  args.savepath, year_months=args.datetimes)
        # now use ts-tm model to convert WetZ into PWV and save:
        produce_pwv_all_stations(args.savepath/'monthly', work_yuval/'GNSS_stations', args.mda_path)
        logger.info('Done!')
    elif args.delete:
        generate_delete(args.savepath, args.channel)
    else:
        raise ValueError('need to specify channel name!')
