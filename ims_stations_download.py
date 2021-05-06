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

import click
from loguru import logger
from PW_paths import work_yuval
gis_path = work_yuval / 'gis'
awd_path = work_yuval / 'AW3D30'
axis_path = work_yuval / 'axis'
channels = ['BP', 'DiffR', 'Grad', 'NIP', 'Rain', 'RH', 'STDwd', 'TD',
            'TDmax', 'TDmin', 'TG', 'Time', 'WD', 'WDmax', 'WS', 'WS10mm',
            'WS1mm', 'WSmax']
ch_units = ['hPa', 'W/m^2', 'W/m^2', 'W/m^2', 'mm', '%', 'deg', 'degC', 'degC',
            'degC', 'degC', 'hhmm', 'deg', 'deg', 'm/s', 'm/s', 'm/s', 'm/s']
ch_desc = ['surface pressure', 'diffuse radiation', 'global radiation',
           'direct radiation', 'rain', 'relative humidity', 'wind direction std',
           'dry temperature', 'maximum temperature', 'minimum temperature',
           'ground temperature', 'maximum end of 10 mins', 'wind direction',
           'maximum wind direction', 'wind speed', 'maximum 10 mins wind speed',
           'maximum 1 mins wind speed', 'maximum wind speed']
ch_units_dict = dict(zip(channels, ch_units))
ch_desc_dict = dict(zip(channels, ch_desc))


def parse_single_station(data):
    import pandas as pd
    datetimes = [x['datetime'] for x in data]
    dfl = []
    for i, dt in enumerate(datetimes):
        # get all channels into dataframe for each datetime:
        df = pd.DataFrame(data[i]['channels'])
        df = df[df['valid']]
        df = df[df['status'] == 1]
        df = df[['name', 'value']]
        df = df.T
        df.columns = df.loc['name'].tolist()
        df = df.drop('name', axis=0)
        df.index = [pd.to_datetime(dt)]
        dfl.append(df)
    df = pd.concat(dfl)
    df = df.astype(float)
    df.index = df.index.tz_convert('UTC')
    df.index.name = 'time'
    ds = df.to_xarray()
    ds['time'] = pd.to_datetime(ds['time'].values)
    for da in ds:
        if da in channels:
            ds[da].attrs['units'] = ch_units_dict.get(da)
            ds[da].attrs['long_name'] = ch_desc_dict.get(da)
    return ds


@click.command()
@click.option('--savepath', '-s', help='a full path to download the files, e.g., /home/ziskin/Work_Files/PW_yuval/IMS_T/10mins.',
              type=click.Path(exists=True))
@click.option('--window', '-w', nargs=1, default=30,
              help='how many hours before now to get the data',
              type=click.IntRange(min=1, max=120))
@click.option('--gis_path', help='a full path to gis folder',
              type=click.Path(exists=True), default=gis_path)
@click.option('--awd_path', help='a full path to dem folder',
              type=click.Path(exists=True), default=awd_path)
@click.option('--axis_path', help='a full path to dem folder',
              type=click.Path(exists=True), default=axis_path)
@click.option('--mda_path', help='a full path to where the ts-tm model files are',
              type=click.Path(exists=True), default=work_yuval)
def main_program(*args, **kwargs):
    from pathlib import Path
    window = kwargs['window']
    savepath = Path(kwargs['savepath'])
    gis_path = Path(kwargs['gis_path'])
    awd_path = Path(kwargs['awd_path'])
    axis_path = Path(kwargs['axis_path'])
    mda_path = Path(kwargs['mda_path'])
    dsl = ims_download(savepath, window, save=False)
    ds = process_ims_stations(savepath, window, var='TD', ds=dsl)
    ds_axis = post_process_ims_stations(ds, window, savepath / 'TD', gis_path,
                                        awd_path, axis_path)
    produce_pw_all_stations(ds_axis, axis_path, mda_path)
    return


def ims_download(savepath, window, save=False):
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
    from requests.exceptions import ConnectionError
    from aux_gps import save_ncfile
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
    stations_df = stations_df[stations_df['active']]
    ds_list = []
    for i, row in stations_df.iterrows():
        st_id = row['stationId']
        st_name = row['name']
        last = now_dt.strftime('%Y/%m/%dT%H:00:00')
        first = start_dt.strftime('%Y/%m/%dT%H:00:00')
        lat = row['location']['latitude']
        if lat is None:
            lat = ''
        lon = row['location']['longitude']
        if lon is None:
            lon = ''
        dl_command = ('https://api.ims.gov.il/v1/envista/stations/' +
                      str(st_id) + '/data/?from=' + first + '&to=' + last)
        try:
            r = requests.get(dl_command, headers=headers)
        except SSLError:
            logger.warning('SSLError')
            r = requests.get(dl_command, headers=headers)
        except ConnectionError:
            logger.warning('ConnectionError')
            r = requests.get(dl_command, headers=headers)
        if r.status_code == 204:  # i.e., no content:
            logger.warning('no content for this search, skipping...')
            continue
        logger.info('parsing data from {} to dataframe.'.format(st_name))
        ds = parse_single_station(r.json()['data'])
        ds.attrs['station_name'] = '-'.join(st_name.split(' '))
        ds.attrs['lat'] = lat
        ds.attrs['lon'] = lon
        ds.attrs['station_id'] = st_id
        if save:
            filename = '{}.nc'.format('-'.join(st_name.split(' ')))
            save_ncfile(ds, savepath, filename)
        else:
            ds_list.append(ds)
    return ds_list


def process_ims_stations(mainpath, window, var='TD', ds=None):
    import os
    import xarray as xr
    from aux_gps import path_glob
    from axis_process import produce_rinex_filenames_at_time_window
    from aux_gps import save_ncfile
    import pandas as pd
    logger.info('processing IMS stations with {} variable'.format(var))
    savepath = mainpath / var
    if not savepath.is_dir():
        os.mkdir(savepath)
        logger.info('created {}.'.format(savepath))
    else:
        logger.info('{} already exist.'.format(savepath))
    if ds is None:
        files = path_glob(mainpath, '*.nc')
        dsl = [xr.load_dataset(x) for x in files]
    else:
        dsl = ds
    ds_list = []
    for ds in dsl:
        try:
            ds_var = ds[var]
        except KeyError:
            logger.warning('no {} in {}.'.format(
                var, ds.attrs['station_name']))
            continue
        ds_var.name = ds.attrs['station_name']
        ds_var.attrs['lat'] = ds.attrs['lat']
        ds_var.attrs['lon'] = ds.attrs['lon']
        ds_var.attrs['station_id'] = ds.attrs['station_id']
        ds_list.append(ds_var)
    ds = xr.merge(ds_list)
    now_dt = pd.Timestamp.utcnow().floor('H')
    names = produce_rinex_filenames_at_time_window(end_dt=now_dt,
                                                   window=window)
    st_str = names[0][4:8]
    end_str = names[-1][4:8]
    filename = 'IMS_{}_{}-{}.nc'.format(var, st_str, end_str)
    save_ncfile(ds, savepath, filename)
    # finally delete all nc files:
    if ds is None:
        [x.unlink() for x in files]
    return ds


def post_process_ims_stations(ds, window, savepath, gis_path, dem_path,
                              axis_path):
    """fill TD with hourly mean if NaN and smooth, then fill in station_lat
    and lon and alt from DEM, finally interpolate to AXIS coords and save"""
    from aux_gps import fill_na_xarray_time_series_with_its_group
    from ims_procedures import analyse_10mins_ims_field
    from axis_process import produce_rinex_filenames_at_time_window
    from ims_procedures import IMS_interpolating_to_GNSS_stations_israel
    from aux_gps import save_ncfile
    import pandas as pd
    now_dt = pd.Timestamp.utcnow().floor('H')
    ds = fill_na_xarray_time_series_with_its_group(ds, grp='hour')
    ds = analyse_10mins_ims_field(ds=ds, var='TD', gis_path=gis_path,
                                  dem_path=dem_path)
    ds_axis = IMS_interpolating_to_GNSS_stations_israel(
        dt=None, start_year=str(now_dt.year), verbose=True, savepath=None,
        network='axis', ds_td=ds, cut_days_ago=None, axis_path=axis_path)
    now_dt = pd.Timestamp.utcnow().floor('H')
    names = produce_rinex_filenames_at_time_window(end_dt=now_dt,
                                                   window=window)
    st_str = names[0][4:8]
    end_str = names[-1][4:8]
    filename = 'AXIS_TD_{}-{}.nc'.format(st_str, end_str)
    save_ncfile(ds_axis, savepath, filename)
    return ds_axis


def produce_pw_all_stations(ds, axis_path, mda_path):
    from PW_stations import load_mda
    from PW_stations import produce_GNSS_station_PW
    from aux_gps import fill_na_xarray_time_series_with_its_group
    from aux_gps import path_glob
    from aux_gps import save_ncfile
    import xarray as xr
    # first load mda:
    mda = load_mda(mda_path)
    # now loop over each station, produce pwv and save:
    st_dirs = path_glob(axis_path, '*/')
    st_dirs = [x for x in st_dirs if x.is_dir()]
    st_dirs = [x for x in st_dirs if not x.as_posix().split('/')[-1].isnumeric()]
    assert len(st_dirs) == 27
    for st_dir in st_dirs:
        station = st_dir.as_posix().split('/')[-1]
        last_file = sorted(path_glob(st_dir/'dr/ultra', '*.nc'))[-1]
        last_file_str = last_file.as_posix().split('/')[-1][4:13]
        wet = xr.load_dataset(last_file)['WetZ']
        wet_error = xr.load_dataset(last_file)['WetZ_error']
        wet.name = station
        wet_error = station
        # resample temp to 5 mins and reindex to wet delay time:
        t = ds[station].resample(time='5T').ffill().reindex_like(wet.time)
        # fill in NaNs with mean hourly signal:
        t_new = fill_na_xarray_time_series_with_its_group(t, grp='hour')
        pwv = produce_GNSS_station_PW(wet, t_new, mda=mda,
                                      model_name='LR', plot=False)
        pwv_error = produce_GNSS_station_PW(wet_error, t_new, mda=mda,
                                            model_name='LR', plot=False)
        pwv_error.name = '{}_error'.format(pwv.name)
        pwv_ds = xr.merge([pwv, pwv_error])
        filename = '{}{}_PWV.nc'.format(station, last_file_str)
        save_ncfile(pwv_ds, st_dir/'dr/ultra', filename)

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


if __name__ == '__main__':
    main_program()
