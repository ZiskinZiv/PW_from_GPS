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
        df = df[df['valid'] == True]
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
def main_program(*args, **kwargs):
    from pathlib import Path
    window = kwargs['window']
    savepath = Path(kwargs['savepath'])
    ims_download(savepath, window)
    process_ims_stations(savepath, window, var='TD')
    return


def ims_download(savepath, window):
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
        if r.status_code == 204:  # i.e., no content:
            logger.warning('no content for this search, skipping...')
            continue
        logger.info('parsing data from {} to dataframe.'.format(st_name))
        ds = parse_single_station(r.json()['data'])
        ds.attrs['station_name'] = '-'.join(st_name.split(' '))
        ds.attrs['lat'] = lat
        ds.attrs['lon'] = lon
        ds.attrs['station_id'] = st_id
        filename = '{}.nc'.format('-'.join(st_name.split(' ')))
        save_ncfile(ds, savepath, filename)
    return


def process_ims_stations(mainpath, window, var='TD'):
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

    files = path_glob(mainpath, '*.nc')
    dsl = [xr.load_dataset(x) for x in files]
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
    return ds

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

