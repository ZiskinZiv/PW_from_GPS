#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:50:20 2019

@author: shlomi
"""

import pandas as pd
import numpy as np
from PW_paths import work_yuval
from PW_paths import work_path
from PW_paths import geo_path

garner_path = work_yuval / 'garner'
ims_path = work_yuval / 'IMS_T'
gis_path = work_yuval / 'gis'
sound_path = work_yuval / 'sounding'
phys_soundings = sound_path / 'bet_dagan_phys_sounding_2007-2019.nc'
tela_zwd = work_yuval / 'gipsyx_results/tela_newocean/TELA_PPP_1996-2019.nc'
jslm_zwd = work_yuval / 'gipsyx_results/jslm_newocean/JSLM_PPP_2001-2019.nc'
alon_zwd = work_yuval / 'gipsyx_results/alon_newocean/ALON_PPP_2005-2019.nc'
tela_zwd_aligned = work_yuval / 'TELA_zwd_aligned_with_physical_bet_dagan.nc'
alon_zwd_aligned = work_yuval / 'ALON_zwd_aligned_with_physical_bet_dagan.nc'
jslm_zwd_aligned = work_yuval / 'JSLM_zwd_aligned_with_physical_bet_dagan.nc'
tela_ims = ims_path / '10mins/TEL-AVIV-COAST_178_TD_10mins_filled.nc'
alon_ims = ims_path / '10mins/ASHQELON-PORT_208_TD_10mins_filled.nc'
jslm_ims = ims_path / '10mins/JERUSALEM-CENTRE_23_TD_10mins_filled.nc'
station_on_geo = geo_path / 'Work_Files/PW_yuval/GNSS_stations'
PW_stations_path = work_yuval / '1minute'
stations = pd.read_csv('All_gps_stations.txt', header=0, delim_whitespace=True,
                       index_col='name')
GNSS = work_yuval / 'GNSS_stations'

# TODO: finish clouds formulation in ts-tm modeling
# TODO: finish playing with ts-tm modeling, various machine learning algos.
# TODO: redo the hour, season and cloud selection in formulate_plot


def build_df_lat_lon_alt_gnss_stations(gnss_path=GNSS, savepath=None):
    from aux_gps import path_glob
    import pandas as pd
    stations_in_gnss = [x.as_posix().split('/')[-1]
                        for x in path_glob(GNSS, '*')]
    dss = [
        load_gipsyx_results(
            x,
            sample_rate='MS',
            plot_fields=None) for x in stations_in_gnss]
    stations_not_found = [x for x in dss if isinstance(x, str)]
    [stations_in_gnss.remove(x) for x in stations_not_found]
    dss = [x for x in dss if not isinstance(x, str)]
    lats = [x.dropna('time').lat[0].values.item() for x in dss]
    lons = [x.dropna('time').lon[0].values.item() for x in dss]
    alts = [x.dropna('time').alt[0].values.item() for x in dss]
    df = pd.DataFrame(lats)
    df.index = stations_in_gnss
    df['lon'] = lons
    df['alt'] = alts
    df.columns = ['lat', 'lon', 'alt']
    df.sort_index(inplace=True)
    if savepath is not None:
        filename = 'israeli_gnss_coords.txt'
        df.to_csv(savepath/filename, sep=' ')
    return df


def run_error_analysis(station='tela', task='edit30hr'):
    station_on_geo = geo_path / 'Work_Files/PW_yuval/GNSS_stations'
    if task == 'edit30hr':
        path = station_on_geo / station / 'rinex/30hr'
        err, df = gipsyx_runs_error_analysis(path, glob_str='*.dr.gz')
    elif task == 'run':
        path = station_on_geo / station / 'rinex/30hr/results'
        err, df = gipsyx_runs_error_analysis(path, glob_str='*.tdp')
    return err, df


def gipsyx_runs_error_analysis(path, glob_str='*.tdp'):
    from collections import Counter
    from aux_gps import get_timedate_and_station_code_from_rinex
    from aux_gps import path_glob
    import pandas as pd
    import logging

    def find_errors(content_list, name):
        keys = [x for x in content_list if 'KeyError' in x]
        vals = [x for x in content_list if 'ValueError' in x]
        excpt = [x for x in content_list if 'Exception' in x]
        err = [x for x in content_list if 'Error' in x]
        trouble = [x for x in content_list if 'Trouble' in x]
        problem = [x for x in content_list if 'Problem' in x]
        fatal = [x for x in content_list if 'FATAL' in x]
        timed = [x for x in content_list if 'Timed' in x]
        errors = keys + vals + excpt + err + trouble + problem + fatal + timed
        if not errors:
            dt, _ = get_timedate_and_station_code_from_rinex(name)
            logger.warning('found new error on {} ({})'.format(name,  dt.strftime('%Y-%m-%d')))
        return errors

    logger = logging.getLogger('gipsyx_post_proccesser')
    rfns = []
    files = path_glob(path, glob_str, True)
    for file in files:
        # first get all the rinex filenames that gipsyx ran successfuly:
        rfn = file.as_posix().split('/')[-1][0:12]
        rfns.append(rfn)
    if files:
        logger.info('running error analysis for station {}'.format(rfn[0:4].upper()))
    all_errors = []
    errors = []
    dates = []
    rinex = []
    files = path_glob(path, '*.err')
    for file in files:
        rfn = file.as_posix().split('/')[-1][0:12]
        # now, filter the error files that were copyed but there is tdp file
        # i.e., the gipsyx run was successful:
        if rfn in rfns:
            continue
        else:
            dt, _ = get_timedate_and_station_code_from_rinex(rfn)
            dates.append(dt)
            rinex.append(rfn)
            with open(file) as f:
                content = f.readlines()
                # you may also want to remove whitespace characters like `\n` at
                # the end of each line
                content = [x.strip() for x in content]
                all_errors.append(content)
                errors.append(find_errors(content, rfn))
    er = [','.join(x) for x in all_errors]
    df = pd.DataFrame(data=rinex, index=dates, columns=['rinex'])
    df['error'] = er
    df = df.sort_index()
    total = len(rfns) + len(df)
    good = len(rfns)
    bad = len(df)
    logger.info('total files: {}, successful runs: {}, errornous runs: {}'.format(
            total, good, bad))
    logger.info('success percent: {0:.1f}%'.format(100.0 * good / total))
    logger.info('error percent: {0:.1f}%'.format(100.0 * bad / total))
    # now count the similar errors and sort:
    flat_list = [item for sublist in errors for item in sublist]
    counted_errors = Counter(flat_list)
    errors_sorted = sorted(counted_errors.items(), key=lambda x: x[1],
                           reverse=True)
    return errors_sorted, df


def get_zwd_from_sounding_and_compare_to_gps(phys_sound_file=phys_soundings,
                                             zwd_file=tela_zwd_aligned,
                                             tm=None, plot=True):
    import xarray as xr
    import matplotlib.pyplot as plt
    station = zwd_file.as_posix().split('/')[-1].split('_')[0]
    zwd_and_tpw = xr.open_dataset(zwd_file)
    tpw = zwd_and_tpw['Tpw']
    pds = get_ts_tm_from_physical(phys_sound_file, plot=False)
    if tm is None:
        k = kappa(pds['tm'], Tm_input=True)
    else:
        k = kappa(tm, Tm_input=True)
    zwd_sound = tpw / k
    zwd_and_tpw['WetZ_from_bet_dagan'] = zwd_sound
    radio = zwd_and_tpw['WetZ_from_bet_dagan']
    gps = zwd_and_tpw['{}_WetZ'.format(station)]
    if plot:
        radio.plot.line(marker='.', linewidth=0.)
        gps.plot.line(marker='.', linewidth=0.)
        plt.title('{} station ZWD and Bet-Dagan ZWD using {} temperatures'.format(station, station))
        plt.figure()
        (radio - gps).plot.line(marker='.', linewidth=0.)
        plt.title('{}-BET_DAGAN ZWD Residuals'.format(station))
        plt.figure()
        (radio - gps).plot.hist(bins=100)
    return zwd_and_tpw


def fit_ts_tm_produce_ipw_and_compare_TELA(phys_sound_file=phys_soundings,
                                           zwd_file=tela_zwd_aligned,
                                           IMS_file=tela_ims,
                                           sound_path=sound_path,
                                           categories=None, model='LR',
                                           times=['2005', '2019'],
                                           **compare_kwargs):
    """categories can be :'bevis', None, 'season' and/or 'hour'. None means
    whole dataset ts-tm.
    models can be 'LR' or 'TSEN'. compare_kwargs is for
    compare_to_sounding2 i.e., times, season, hour, title"""
    import xarray as xr
    if categories == 'bevis':
        results = None
    else:
        results = ml_models_T_from_sounding(sound_path, categories, model,
                                            physical_file=phys_sound_file,
                                            times=times)
    if categories is None:
        compare_kwargs.update({'title': 'whole'})
    else:
        if isinstance(categories, str):
            compare_kwargs.update({'title': [categories][0]})
        elif isinstance(categories, list):
            compare_kwargs.update({'title': 'hour_season'})
    zwd_and_tpw = xr.open_dataset(zwd_file)
    if times is not None:
        zwd_and_tpw = zwd_and_tpw.sel(time=slice(*times))
    station = zwd_file.as_posix().split('/')[-1].split('_')[0]
    wetz = zwd_and_tpw['{}_WetZ'.format(station)]
    tpw = zwd_and_tpw['Tpw']
    wetz_error = zwd_and_tpw['{}_WetZ_error'.format(station)]
    # load the 10 mins temperature data from IMS:
    T = xr.open_dataset(IMS_file)
    T = T.to_array(name='t').squeeze(drop=True)
    pw_gps = produce_single_station_IPW(wetz, T, mda=results, model_name=model)
    compare_to_sounding2(pw_gps, tpw, station=station, **compare_kwargs)
    return pw_gps


def get_ts_tm_from_physical(phys=phys_soundings, plot=True):
    import xarray as xr
    from aux_gps import get_unique_index
    from aux_gps import keep_iqr
    from aux_gps import plot_tmseries_xarray
    pds = xr.open_dataset(phys)
    pds = pds[['Tm', 'Ts']]
    pds = pds.rename({'Ts': 'ts', 'Tm': 'tm'})
    pds = pds.rename({'sound_time': 'time'})
    pds = get_unique_index(pds)
    pds = keep_iqr(pds, k=2.0)
    pds = pds.dropna('time')
    if plot:
        plot_tmseries_xarray(pds)
    return pds


def align_physical_bet_dagan_soundings_pw_to_gps_station_zwd(
        phys_sound_file, zwd_file, IMS_file,
        savepath=work_yuval, model=None, plot=True):
    """compare the IPW of the physical soundings of bet dagan station to
    the TELA gps station - using IMS temperature Tel-aviv station"""
    from aux_gps import get_unique_index
    from aux_gps import keep_iqr
    from aux_gps import dim_intersection
    import xarray as xr
    import numpy as np
    station = zwd_file.as_posix().split('/')[-1].split('_')[0]
    filename = '{}_zwd_aligned_with_physical_bet_dagan.nc'.format(station)
    if not (savepath / filename).is_file():
        print('saving {} to {}'.format(filename, savepath))
        # first load physical bet_dagan Tpw, Ts, Tm and dt_range:
        phys = xr.open_dataset(phys_sound_file)
        # clean and merge:
        p_list = [get_unique_index(phys[x], 'sound_time')
                  for x in ['Ts', 'Tm', 'Tpw', 'dt_range']]
        phys_ds = xr.merge(p_list)
        phys_ds = keep_iqr(phys_ds, 'sound_time', k=2.0)
        phys_ds = phys_ds.rename({'Ts': 'ts', 'Tm': 'tm'})
        # load the zenith wet daley for GPS (e.g.,TELA) station:
        zwd = xr.open_dataset(zwd_file)
        zwd = zwd[['WetZ', 'WetZ_error']]
        # loop over dt_range and average the results on PW:
        wz_list = []
        wz_std = []
        wz_error_list = []
        for i in range(len(phys_ds['dt_range'].sound_time)):
            min_time = phys_ds['dt_range'].isel(sound_time=i).sel(bnd='Min').values
            max_time = phys_ds['dt_range'].isel(sound_time=i).sel(bnd='Max').values
            wetz = zwd['WetZ'].sel(time=slice(min_time, max_time)).mean('time')
            wetz_std = zwd['WetZ'].sel(time=slice(min_time, max_time)).std('time')
            wetz_error = zwd['WetZ_error'].sel(time=slice(min_time, max_time)).mean('time')
            wz_std.append(wetz_std)
            wz_list.append(wetz)
            wz_error_list.append(wetz_error)
        wetz_gps = xr.DataArray(wz_list, dims='sound_time')
        wetz_gps.name = '{}_WetZ'.format(station)
        wetz_gps_error = xr.DataArray(wz_error_list, dims='sound_time')
        wetz_gps_error.name = 'TELA_WetZ_error'
        wetz_gps_std = xr.DataArray(wz_list, dims='sound_time')
        wetz_gps_std.name = 'TELA_WetZ_std'
        wetz_gps['sound_time'] = phys_ds['sound_time']
        wetz_gps_error['sound_time'] = phys_ds['sound_time']
        new_time = dim_intersection([wetz_gps, phys_ds['Tpw']], 'sound_time')
        wetz_gps = wetz_gps.sel(sound_time=new_time)
        tpw_bet_dagan = phys_ds.Tpw.sel(sound_time=new_time)
        zwd_and_tpw = xr.merge([wetz_gps, wetz_gps_error, wetz_gps_std,
                                tpw_bet_dagan])
        zwd_and_tpw = zwd_and_tpw.rename({'sound_time': 'time'})
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in zwd_and_tpw.data_vars}
        zwd_and_tpw.to_netcdf(savepath / filename, 'w', encoding=encoding)
        print('Done!')
        return
    else:
        print('found file!')
        zwd_and_tpw = xr.open_dataset(savepath / filename)
        wetz = zwd_and_tpw['{}_WetZ'.format(station)]
        wetz_error = zwd_and_tpw['{}_WetZ_error'.format(station)]
        # load the 10 mins temperature data from IMS:
        tela_T = xr.open_dataset(IMS_file)
        # tela_T = tela_T.resample(time='5min').ffill()
        # compute the kappa function and multiply by ZWD to get PW(+error):
        k, dk = kappa_ml(tela_T.to_array(name='Ts').squeeze(drop=True),
                         model=model, verbose=True)
        kappa = k.to_dataset(name='{}_kappa'.format(station))
        kappa['{}_kappa_error'.format(station)] = dk
        PW = (
            kappa['{}_kappa'.format(station)] *
            wetz).to_dataset(
            name='{}_PW'.format(station)).squeeze(
                drop=True)
        PW['{}_PW_error'.format(station)] = np.sqrt(
            wetz_error**2.0 +
            kappa['{}_kappa_error'.format(station)]**2.0)
        PW['TPW_bet_dagan'] = zwd_and_tpw['Tpw']
        PW = PW.dropna('time')
    return PW


def read_log_files(path, savepath=None, fltr='updated_by_shlomi',
                   suff='*.log'):
    """read gnss log files for putting them into ocean tides model"""
    import pandas as pd
    from aux_gps import path_glob
    from tabulate import tabulate

    def to_fwf(df, fname, showindex=False):
        from tabulate import simple_separated_format
        tsv = simple_separated_format("   ")
        # tsv = 'plain'
        content = tabulate(
            df.values.tolist(), list(
                df.columns), tablefmt=tsv, showindex=showindex, floatfmt='f')
        open(fname, "w").write(content)

    files = sorted(path_glob(path, glob_str=suff))
    record = {}
    for file in files:
        filename = file.as_posix().split('/')[-1]
        if fltr not in filename:
            continue
        station = filename.split('_')[0]
        print('reading station {} log file'.format(station))
        with open(file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        posnames = ['X', 'Y', 'Z']
        pos_list = []
        for pos in posnames:
            text = [
                x for x in content if '{} coordinate (m)'.format(pos) in x][0]
            xyz = float(text.split(':')[-1])
            pos_list.append(xyz)
        record[station] = pos_list
    df = pd.DataFrame.from_dict(record, orient='index')
    df.columns = posnames
    if savepath is not None:
        savefilename = 'stations_approx_loc.txt'
        show_index = [x + '                   ' for x in df.index.tolist()]
        to_fwf(df, savepath / savefilename, show_index)
        # df.to_csv(savepath / savefilename, sep=' ')
        print('{} was saved to {}.'.format(savefilename, savepath))
    return df


def analyze_missing_rinex_files(path, savepath=None):
    from aux_gps import get_timedate_and_station_code_from_rinex
    from aux_gps import datetime_to_rinex_filename
    from aux_gps import path_glob
    import pandas as pd
    dt_list = []
    files = path_glob(path, '*.Z')
    for file in files:
        filename = file.as_posix().split('/')[-1][:-2]
        dt, station = get_timedate_and_station_code_from_rinex(filename)
        dt_list.append(dt)
    dt_list = sorted(dt_list)
    true = pd.date_range(dt_list[0], dt_list[-1], freq='1D')
    # df = pd.DataFrame(dt_list, columns=['downloaded'], index=true)
    dif = true.difference(dt_list)
    dts = [datetime_to_rinex_filename(station, x) for x in dif]
    df_missing = pd.DataFrame(data=dts, index=dif.strftime('%Y-%m-%d'),
                              columns=['filenames'])
    df_missing.index.name = 'dates'
    if savepath is not None:
        filename = station + '_missing_rinex_files.txt'
        df_missing.to_csv(savepath / filename)
        print('{} was saved to {}'.format(filename, savepath))
    return df_missing


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


def parameter_study_ts_tm_TELA_bet_dagan(tel_aviv_IMS_file, path=work_yuval,
                                         coef=[-3, 3], inter=[-300, 300],
                                         span=10, breakdown=True, plot=True):
    import xarray as xr
    import numpy as np
    from aux_gps import dim_intersection
    import matplotlib.pyplot as plt
    filename = 'TELA_zwd_aligned_with_physical_bet_dagan.nc'
    zwd_and_tpw = xr.open_dataset(path / filename)
    wetz = zwd_and_tpw['TELA_WetZ']
    tpw = zwd_and_tpw['Tpw']
    # load the 10 mins temperature data from IMS:
    tela_T = xr.open_dataset(tel_aviv_IMS_file)
    coef_space = np.linspace(*coef, span)
    intercept_space = np.linspace(*inter, span)
    model = np.stack([coef_space, intercept_space], axis=0)
    if breakdown:
        seasons = ['DJF', 'MAM', 'JJA', 'SON']
        hours = [0, 12]
        rds_list = []
        for season in seasons:
            for hour in hours:
                print('calculating kappa of season {} and hour {}'.format(season, hour))
                T = tela_T.to_array(name='TELA_T').squeeze(drop=True)
                T = T.where(T['time.season'] == season).dropna('time')
                T = T.where(T['time.hour'] == hour).dropna('time')
                k, _ = kappa_ml(T, model=model, no_error=True)
                print('building results...')
                pw = k * wetz
                new_time = dim_intersection([pw, tpw])
                pw = pw.sel(time=new_time)
                tpw_sel = tpw.sel(time=new_time)
                rmse = (tpw_sel - pw)**2.0
                rmse = np.sqrt(rmse.mean('time'))
                mean_error = (tpw_sel - pw).mean('time')
                rmse.name = 'RMSE'.format(season, hour)
                mean_error.name = 'MEAN'.format(season, hour)
                merged = xr.merge([mean_error, rmse])
                merged = merged.expand_dims(['season', 'hour'])
                merged['season'] = [season]
                merged['hour'] = [hour]
                rds_list.append(merged.stack(prop=['season', 'hour']))
        rds = xr.concat(rds_list, 'prop').unstack('prop')
        print('Done!')
    else:
        print('calculating kappa of for all data!')
        T = tela_T.to_array(name='TELA_T').squeeze(drop=True)
        k, _ = kappa_ml(T, model=model, no_error=True)
        print('building results...')
        pw = k * wetz
        new_time = dim_intersection([pw, tpw])
        pw = pw.sel(time=new_time)
        tpw_sel = tpw.sel(time=new_time)
        rmse = (tpw_sel - pw)**2.0
        rmse = np.sqrt(rmse.mean('time'))
        mean_error = (tpw_sel - pw).mean('time')
        rmse.name = 'RMSE_all'
        mean_error.name = 'MEAN_all'
        rds = xr.merge([mean_error, rmse])
        print('Done!')
    if plot:
        if not breakdown:
            fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            rds.MEAN.plot.pcolormesh(ax=ax[0])
            rds.RMSE.plot.pcolormesh(ax=ax[1])
        else:
            fg_mean = rds.MEAN.plot.pcolormesh(row='hour', col='season',
                                               figsize=(20, 10), cmap='seismic')
            [ax.grid() for ax in fg_mean.fig.axes]
            # fg_mean.fig.tight_layout()
            # fg_mean.fig.subplots_adjust(right=0.9)
            fg_rmse = rds.RMSE.plot.pcolormesh(row='hour', col='season',
                                               figsize=(20, 10))
            [ax.grid() for ax in fg_rmse.fig.axes]
            # fg_mean.fig.tight_layout()
            # fg_rmse.fig.subplots_adjust(right=0.9)
    return rds


#def get_geo_data_from_gps_stations(gps_names):
#    import requests
#    from bs4 import BeautifulSoup as bs
#    user = "anonymous"
#    passwd = "shlomiziskin@gmail.com"
#    # Make a request to the endpoint using the correct auth values
#    auth_values = (user, passwd)
#    response = requests.get(url, auth=auth_values)
#    soup = bs(response.text, "lxml")
#    allLines = soup.text.split('\n')
#    X = [x for x in allLines if 'X coordinate' in x][0].split()[-1]
#    Y = [x for x in allLines if 'Y coordinate' in x][0].split()[-1]
#    Z = [x for x in allLines if 'Z coordinate' in x][0].split()[-1]
# 
## Convert JSON to dict and print
#print(response.json())


def read_stations_to_dataset(path, group_name='israeli', save=False,
                             names=None):
    import xarray as xr
    if names is None:
        stations = []
        for filename in sorted(path.glob('garner_trop_[!all_stations]*.nc')):
            st_name = filename.as_posix().split('/')[-1].split('.')[0].split('_')[-1]
            print('Reading station {}'.format(st_name))
            da = xr.open_dataarray(filename)
            da = da.dropna('time')
            stations.append(da)
        ds = xr.merge(stations)
    if save:
        savefile = 'garner_' + group_name + '_stations.nc'
        print('saving {} to {}'.format(savefile, path))
        ds.to_netcdf(path / savefile, 'w')
        print('Done!')
    return ds


def filter_stations(path, group_name='israeli', save=False):
    """filter bad values in trop products stations"""
    import xarray as xr
    from aux_gps import Zscore_xr
    filename = 'garner_' + group_name + '_stations.nc'
    print('Reading {}.nc from {}'.format(filename, path))
    ds = xr.open_dataset(path / filename)
    ds['zwd'].attrs['units'] = 'Zenith Wet Delay in cm'
    stations = [x for x in ds.data_vars.keys()]
    for station in stations:
        print('filtering station {}'.format(station))
        # first , remove negative values:
        ds[station] = ds[station].where(ds[station].sel(zwd='value') > 0)
        # get zscore of data and errors:
        zscore_val = Zscore_xr(ds[station].sel(zwd='value'), dim='time')
        zscore_sig = Zscore_xr(ds[station].sel(zwd='sigma'), dim='time')
        # filter for zscore <5 for data and <3 for error:
        ds[station] = ds[station].where(np.abs(zscore_val) < 5)
        ds[station] = ds[station].where(np.abs(zscore_sig) < 3)
    if save:
        filename = filename + '_filtered.nc'
        print('saving {} to {}'.format(filename, path))
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(path / filename, 'w', encoding=encoding)
        print('Done!')
    return ds

# def overlap_time_xr(*args, union=False):
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


def produce_geo_gps_stations(path=gis_path, file='All_gps_stations.txt',
                             plot=True):
    import geopandas as gpd
    import xarray as xr
    stations_df = pd.read_csv(file, index_col='name',
                              delim_whitespace=True)
    isr_dem = xr.open_rasterio(path / 'israel_dem.tif')
    alt_list = []
    for index, row in stations_df.iterrows():
        lat = row['lat']
        lon = row['lon']
        alt = isr_dem.sel(band=1, x=lon, y=lat, method='nearest').values.item()
        alt_list.append(float(alt))
    stations_df['alt'] = alt_list
    isr = gpd.read_file(path / 'israel_demog2012.shp')
    isr.crs = {'init': 'epsg:4326'}
    stations = gpd.GeoDataFrame(stations_df,
                                geometry=gpd.points_from_xy(stations_df.lon,
                                                            stations_df.lat),
                                crs=isr.crs)
    stations_isr = gpd.sjoin(stations, isr, op='within')
    if plot:
        ax = isr.plot()
        stations_isr.plot(ax=ax, column='alt', cmap='Greens',
                          edgecolor='black', legend=True)
        for x, y, label in zip(stations_isr.lon, stations_isr.lat,
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
    ims_selected = stations_meta.loc[stations_meta.stationId.isin(
        geo_df.ID.values.tolist())]
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
    geo_df['gps_lat'] = geo_gps.lat
    geo_df['gps_lon'] = geo_gps.lon
    geo_df['gps_alt'] = geo_gps.alt
    geo_df['alt_diff'] = geo_df.alt - geo_gps.alt
    if plot:
        import geopandas as gpd
        isr = gpd.read_file(path / 'israel_demog2012.shp')
        isr.crs = {'init': 'epsg:4326'}
        geo_gps_new = gpd.GeoDataFrame(geo_df,
                                       geometry=gpd.points_from_xy(geo_df.lon,
                                                                   geo_df.lat),
                                       crs=isr.crs)
        ax = isr.plot()
        geo_gps.plot(ax=ax, color='green',
                     edgecolor='black', legend=True)
        for x, y, label in zip(geo_gps.lon, geo_gps.lat,
                               geo_gps.alt):
            ax.annotate(label, xy=(x, y), xytext=(3, 3),
                        textcoords="offset points")
        geo_gps_new.plot(ax=ax, color='red', edgecolor='black', legend=True)
        for x, y, label in zip(geo_gps_new.lon, geo_gps_new.lat,
                               geo_gps_new.alt):
            ax.annotate(label, xy=(x, y), xytext=(3, 3),
                        textcoords="offset points")
    return geo_df


def fix_T_height(path, geo_df, lapse_rate=6.5):
    """fix the temperature diffrence due to different height between the IMS
    and GPS stations"""
    # use lapse rate of 6.5 K/km = 6.5e-3 K/m
    import xarray as xr
    lr = 1e-3 * lapse_rate  # convert to K/m
    Tds = xr.open_dataset(path / 'IMS_TD_israeli_for_gps.nc')
    stations = [x for x in Tds.data_vars.keys() if 'missing' not in x]
    ds_list = []
    for st in stations:
        try:
            alt_diff = geo_df.loc[st, 'alt_diff']
            # correction is lapse_rate in K/m times alt_diff in meteres
            # if alt_diff is positive, T should be higher and vice versa
            Tds[st].attrs['description'] += ' The data was fixed using {} K/km '\
                                            'lapse rate bc the difference'\
                                            ' between the temperature station '\
                                            'and the gps station is {}'\
                                            .format(lapse_rate, alt_diff)
            Tds[st].attrs['lapse_rate_fix'] = lapse_rate
            ds_list.append(Tds[st] + lr * alt_diff)
        except KeyError:
            print('{} station not found in gps data'.format(st))
        continue
    ds = xr.merge(ds_list)
    # copy attrs:
    for da in ds:
        ds[da].attrs = Tds[da].attrs
    return ds


def produce_geo_df(gis_path=gis_path):
    print('getting IMS temperature stations metadata...')
    ims = produce_geo_ims(gis_path, filename='IMS_10mins_meta_data.xlsx',
                          closed_stations=False, plot=False)
    print('getting GPS stations ZWD from garner...')
    gps = produce_geo_gps_stations(gis_path, file='stations.txt', plot=False)
    print('combining temperature and GPS stations into one dataframe...')
    geo_df = get_minimum_distance(ims, gps, gis_path, plot=False)
    print('Done!')
    return geo_df


def save_GNSS_PW_israeli_stations(savepath=work_yuval):
    from pathlib import Path
    import pandas as pd
    import xarray as xr
    sample = {'1H': 'hourly', '3H': '3hourly', 'D': 'Daily', 'W': 'weekly',
              'MS': 'monthly'}
    filename = 'israeli_gnss_coords.txt'
    df = pd.read_csv(Path().cwd() / filename, header=0, delim_whitespace=True)
    stations = df.index.tolist()
    ds_list = []
    for sta in stations:
        print(sta, '5mins')
        pw = produce_GNSS_station_PW(sta, None, plot=False)
        ds_list.append(pw)
    ds = xr.merge(ds_list)
    if savepath is not None:
        filename = 'GNSS_PW.nc'
        print('saving {} to {}'.format(filename, savepath))
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(savepath / filename, 'w', encoding=encoding)
    for skey in sample.keys():
        ds_list = []
        for sta in stations:
            print(sta, sample[skey])
            pw = produce_GNSS_station_PW(sta, skey, plot=False)
            ds_list.append(pw)
        ds = xr.merge(ds_list)
        if savepath is not None:
            filename = 'GNSS_{}_PW.nc'.format(sample[skey])
            print('saving {} to {}'.format(filename, savepath))
            comp = dict(zlib=True, complevel=9)  # best compression
            encoding = {var: comp for var in ds.data_vars}
            ds.to_netcdf(savepath / filename, 'w', encoding=encoding)
    print('Done!')
    return


def produce_GNSS_station_PW(station='tela', sample_rate=None, model=None,
                            phys=None, plot=True):
    from aux_gps import plot_tmseries_xarray
    """use phys=phys_soundings to use physical bet dagan radiosonde report,
    otherwise phys=None to use wyoming data. model=None is LR, model='bevis'
    is Bevis 1992-1994 et al. model
    use sample_rate=None to use full 5 min sample_rate"""
    sample = {'1H': 'hourly', '3H': '3hourly', 'D': 'Daily', 'W': 'weekly',
              'MS': 'monthly'}
    zwd = load_gipsyx_results(station, sample_rate, plot_fields=None)
    Ts = load_GNSS_TD(station, sample_rate, False)
    if model is None:
        mda = ml_models_T_from_sounding(categories=None, models=['LR'],
                                        physical_file=phys, plot=False)
    elif model == 'bevis':
        mda = None
    PW = produce_single_station_IPW(zwd, Ts, mda)
    PW = PW.rename({'PW': station})
    PW = PW.rename({'PW_error': '{}_error'.format(station)})
    if plot:
        plot_tmseries_xarray(PW)
    return PW


def produce_single_station_IPW(zwd, Tds, mda=None, model_name='LR'):
    """input is zwd from gipsy or garner, Tds is the temperature of the
    station, mda is the Ts-Tm relationsship ml models dataarray, model is
    the ml model chosen."""
    import xarray as xr
    # hours = dict(zip([12, 0], ['noon', 'midnight']))
    if isinstance(zwd, xr.Dataset):
        try:
            zwd_error = zwd['WetZ_error']
            zwd = zwd['WetZ']
        except KeyError:
            raise('no error field in zwd dataset...')
    if mda is None:
        # Bevis 1992 relationship:
        print('Using Bevis 1992-1994 Ts-Tm relationship.')
        kappa_ds, kappa_err = kappa_ml(Tds, model=None)
        ipw = kappa_ds * zwd
        ipw_error = kappa_ds * zwd_error + zwd * kappa_err
        ipw_error.name = 'PW_error'
        ipw_error.attrs['long_name'] = 'Precipitable Water standard error'
        ipw_error.attrs['units'] = 'kg / m^2'
        ipw.name = 'PW'
        ipw.attrs['long_name'] = 'Precipitable Water'
        ipw.attrs['units'] = 'kg / m^2'
        ipw = ipw.to_dataset(name='PW')
        ipw['PW_error'] = ipw_error
        ipw.attrs['description'] = 'whole data Tm formulation using Bevis etal. 1992'
        print('Done!')
        return ipw
    time_dim = mda.attrs['time_dim']
    hours = None
    seasons = None
    if 'season' in [x.split('.')[-1] for x in list(mda.dims)]:
        val = mda['{}.season'.format(time_dim)].values.tolist()
        key = '{}.season'.format(time_dim)
        seasons = {key: val}
    if 'hour' in [x.split('.')[-1] for x in list(mda.dims)]:
        val = mda['{}.hour'.format(time_dim)].values.tolist()
        key = '{}.hour'.format(time_dim)
        hours = {key: val}
    if 'any_cld' in mda.dims:
        any_clds = mda.any_cld.values.tolist()
    if len(mda.dims) == 1 and 'name' in mda.dims:
        print('Found whole data Ts-Tm relationship.')
#        Tmul = mda.sel(parameter='slope').values.item()
#        Toff = mda.sel(parameter='intercept').values.item()
        m = mda.sel(name=model_name).values.item()
        kappa_ds, kappa_err = kappa_ml(Tds, model=m, slope_err=mda.attrs['LR_whole_stderr_slope'])
        ipw = kappa_ds * zwd
        ipw_error = kappa_ds * zwd_error + zwd * kappa_err
        ipw_error.name = 'PW_error'
        ipw_error.attrs['long_name'] = 'Precipitable Water standard error'
        ipw_error.attrs['units'] = 'kg / m^2'
        ipw.name = 'PW'
        ipw.attrs['long_name'] = 'Precipitable Water'
        ipw.attrs['units'] = 'kg / m^2'
        ipw = ipw.to_dataset(name='PW')
        ipw['PW_error'] = ipw_error
        ipw.attrs['description'] = 'whole data Tm formulation using {} model'.format(
            model_name)
        print('Done!')
        return ipw
    elif len(mda.dims) == 2 and hours is not None:
        print('Found hourly Ts-Tm relationship slice.')
        kappa_list = []
        h_key = [x for x in hours.keys()][0]
        for hr_num in [x for x in hours.values()][0]:
            print('working on hour {}'.format(hr_num))
            sliced = Tds.where(Tds[h_key] == hr_num).dropna(time_dim)
            m = mda.sel({'name': model_name, h_key: hr_num}).values.item()
            kappa_part, kappa_err = kappa_ml(sliced, model=m)
            kappa_list.append(kappa_part)
        des_attrs = 'hourly data Tm formulation using {} model'.format(
            model_name)
    elif len(mda.dims) == 2 and seasons is not None:
        print('Found season Ts-Tm relationship slice.')
        kappa_list = []
        s_key = [x for x in seasons.keys()][0]
        for season in [x for x in seasons.values()][0]:
            print('working on season {}'.format(season))
            sliced = Tds.where(Tds[s_key] == season).dropna('time')
            m = mda.sel({'name': model_name, s_key: season}).values.item()
            kappa_part, kappa_err = kappa_ml(sliced, model=m)
            kappa_list.append(kappa_part)
        des_attrs = 'seasonly data Tm formulation using {} model'.format(
            model_name)
    elif len(mda.dims) == 2 and set(mda.dims) == set(['any_cld', 'name']):
        print('Found clouds Ts-Tm relationship slice.')
    elif (len(mda.dims) == 3 and set(mda.dims) ==
          set(['any_cld', 'season', 'name'])):
        print('Found clouds and seasonly Ts-Tm relationship slice.')
    elif (len(mda.dims) == 3 and set(mda.dims) ==
          set(['any_cld', 'hour', 'name'])):
        print('Found clouds and hour Ts-Tm relationship slice.')
        # no way to find clouds in historical data ??
        kappa_list = []
#        mda_list = []
#        mda_vals = []
        for hr_num in hours.keys():
            for any_cld in any_clds:
                print('working on any_cld {}, hour {}'.format(
                    any_cld, hours[hr_num]))
#                Tmul = models.sel(any_cld=any_cld, hour=hours[hr_num],
#                                   parameter='slope')
#                Toff = models.sel(any_cld=any_cld, hour=hours[hr_num],
#                                   parameter='intercept')
                sliced = Tds.where(Tds['time.season'] == season).dropna(
                    'time').where(Tds['time.hour'] == hr_num).dropna('time')
                m = mda.sel(any_cld=any_cld, hour=hours[hr_num],
                            name=model_name)
                kappa_part = kappa_ml(sliced, model=m)
                kappa_keys = ['T_multiplier', 'T_offset', 'k2', 'k3']
                kappa_keys = [x + '_' + season + '_' + hours[hr_num] for x in
                              kappa_keys]
                mda_list.append(kappa_keys)
                mda_vals.append([Tmul.values.item(), Toff.values.item(),
                                 k2, k3])
                kappa_list.append(kappa_part)
    elif (len(mda.dims) == 3 and seasons is not None and hours is not None):
        print('Found hourly and seasonly Ts-Tm relationship slice.')
        kappa_list = []
        h_key = [x for x in hours.keys()][0]
        s_key = [x for x in seasons.keys()][0]
        for hr_num in [x for x in hours.values()][0]:
            for season in [x for x in seasons.values()][0]:
                print('working on season {}, hour {}'.format(
                    season, hr_num))
                sliced = Tds.where(Tds[s_key] == season).dropna(
                    time_dim).where(Tds[h_key] == hr_num).dropna(time_dim)
                m = mda.sel({'name': model_name, s_key: season,
                             h_key: hr_num}).values.item()
                kappa_part, kappe_err = kappa_ml(sliced, model=m)
                kappa_list.append(kappa_part)
        des_attrs = 'hourly and seasonly data Tm formulation using {} model'.format(model_name)
    kappa_ds = xr.concat(kappa_list, time_dim)
    ipw = kappa_ds * zwd
    ipw_error = kappa_ds * zwd_error + zwd * kappa_err
    ipw_error.name = 'PW_error'
    ipw_error.attrs['long_name'] = 'Precipitable Water standard error'
    ipw_error.attrs['units'] = 'kg / m^2'
    ipw.name = 'PW'
    ipw.attrs['long_name'] = 'Precipitable Water'
    ipw.attrs['units'] = 'kg / m^2'
    ipw = ipw.to_dataset(name='PW')
    ipw['PW_error'] = ipw_error
    ipw.attrs['description'] = 'whole data Tm formulation using Bevis etal. 1992'
    print('Done!')
    ipw = ipw.reset_coords(drop=True)
    return ipw


def produce_IPW_field(geo_df, ims_path=ims_path, gps_path=garner_path,
                      savepath=None, lapse_rate=6.5, Tmul=0.72,
                      T_offset=70.2, k2=22.1, k3=3.776e5, station=None,
                      plot=True, hist=True):
    import xarray as xr
    """produce IPW field from zwd and T, for one station or all stations"""
    # IPW = kappa[kg/m^3] * ZWD[cm]
    print('fixing T data for height diffrences with {} K/km lapse rate'.format(
            lapse_rate))
    Tds = fix_T_height(ims_path, geo_df, lapse_rate)
    print(
        'producing kappa multiplier to T data with k2: {}, and k3: {}.'.format(
            k2,
            k3))
    Tds = kappa(Tds, Tmul, T_offset, k2, k3)
    kappa_dict = dict(zip(['T_multiplier', 'T_offset', 'k2', 'k3'],
                          [Tmul, T_offset, k2, k3]))
    garner_zwd = xr.open_dataset(gps_path /
                                 'garner_israeli_stations_filtered.nc')
    if station is not None:
        print('producing IPW field for station: {}'.format(station))
        try:
            ipw = Tds[station] * garner_zwd[station.upper()]
            ipw.name = station.upper()
            ipw.attrs['gps_lat'] = geo_df.loc[station, 'gps_lat']
            ipw.attrs['gps_lon'] = geo_df.loc[station, 'gps_lon']
            ipw.attrs['gps_alt'] = geo_df.loc[station, 'gps_alt']
            for k, v in kappa_dict.items():
                ipw.attrs[k] = v
        except KeyError:
            raise('{} station not found in garner gps data'.format(station))
        ds = ipw.to_dataset(name=ipw.name)
        ds = ds.rename({'zwd': 'ipw'})
        ds['ipw'].attrs['name'] = 'IPW'
        ds['ipw'].attrs['long_name'] = 'Integrated Precipitable Water'
        ds['ipw'].attrs['units'] = 'kg / m^2'
        print('Done!')
    else:
        print('producing IPW fields:')
        ipw_list = []
        for st in Tds:
            try:
                # IPW = kappa(T) * Zenith Wet Delay:
                ipw = Tds[st] * garner_zwd[st.upper()]
                ipw.name = st.upper()
                ipw.attrs['gps_lat'] = geo_df.loc[st, 'gps_lat']
                ipw.attrs['gps_lon'] = geo_df.loc[st, 'gps_lon']
                ipw.attrs['gps_alt'] = geo_df.loc[st, 'gps_alt']
                for k, v in kappa_dict.items():
                    ipw.attrs[k] = v
                ipw_list.append(ipw)
            except KeyError:
                print('{} station not found in garner gps data'.format(st))
            continue
        ds = xr.merge(ipw_list)
        ds = ds.rename({'zwd': 'ipw'})
        ds['ipw'].attrs['name'] = 'IPW'
        ds['ipw'].attrs['long_name'] = 'Integrated Precipitable Water'
        ds['ipw'].attrs['units'] = 'kg / m^2'
        print('Done!')
        if savepath is not None:
            filename = 'IPW_israeli_from_gps.nc'
            print('saving {} to {}'.format(filename, savepath))
            comp = dict(zlib=True, complevel=9)  # best compression
            encoding = {var: comp for var in ds.data_vars}
            ds.to_netcdf(savepath / filename, 'w', encoding=encoding)
            print('Done!')
        if plot:
            ds.sel(ipw='value').to_array(dim='station').sortby('station').plot(
                x='time',
                col='station',
                col_wrap=4,
                figsize=(15, 8))
        if hist:
            ds.sel(ipw='value').to_dataframe().hist(bins=100, grid=False,
                                                    figsize=(15, 8))
    return ds


def check_Tm_func(Tmul_num=10, Ts_num=6, Toff_num=15):
    """ check and plot Tm function to understand which bounds to put on Tmul
    Toff optimization, found:Tmul (0,1), Toff (0,150)"""
    import xarray as xr
    Ts = np.linspace(-10, 50, Ts_num) + 273.15
    Toff = np.linspace(-300, 300, Toff_num)
    Tmul = np.linspace(-3, 3, Tmul_num)
    Tm = np.empty((Ts_num, Tmul_num, Toff_num))
    for i in range(Ts_num):
        for j in range(Tmul_num):
            for k in range(Toff_num):
                Tm[i, j, k] = Ts[i] * Tmul[j] + Toff[k]
    da = xr.DataArray(Tm, dims=['Ts', 'Tmul', 'Toff'])
    da['Ts'] = Ts
    da['Tmul'] = Tmul
    da['Toff'] = Toff
    da.plot.pcolormesh(col='Ts', col_wrap=3)
    return da


def kappa_ml(T, model=None, k2=22.1, k3=3.776e5, dk3=0.004e5, dk2=2.2,
             verbose=False, no_error=False, slope_err=None):
    """T in celsious, anton says k2=22.1 is better, """
    import numpy as np
    import xarray as xr
    # maybe implemment Tm= linear_fit(Ts_clim, Tm_clim) + linear_fit(Ts_anom, Tm_anom)
#    from sklearn.utils.estimator_checks import check_estimator
    # original k2=17.0 bevis 1992 etal.
    # [k2] = K / mbar, [k3] = K^2 / mbar
    # 100 Pa = 1 mbar
    dT = 0.5  # deg_C
    if model is None:
        if verbose:
            print('Bevis 1992-1994 model selected.')
        Tm = (273.15 + T) * 0.72 + 70.0  # K Bevis 1992 model
        dTm = 0.72 * dT
    elif isinstance(model, dict):
        if verbose:
            print(
                'using linear model of Tm = {} * Ts + {}'.format(model['coef'], model['intercept']))
        Tm = (273.15 + T) * model['coef'] + model['intercept']
        dTm = model['coef'] * dT
    elif isinstance(model, np.ndarray) and model.ndim == 2:
        print('using model arg as 2d np array with dims: [coef, intercept]')
        coef = model[0, :]
        intercept = model[1, :]
        tm = np.empty((T.values.shape[0], coef.shape[0], intercept.shape[0]))
        for i in range(coef.shape[0]):
            for j in range(intercept.shape[0]):
                tm[:, i, j] = (273.15 + T.values) * coef[i] + intercept[j]
        Tm = xr.DataArray(tm, dims=['time', 'coef', 'intercept'])
        Tm['time'] = T.time
        Tm['coef'] = coef
        Tm['intercept'] = intercept
    else:
        if verbose:
            print('Using sklearn model of: {}'.format(model))
            if hasattr(model, 'coef_'):
                print(
                        'with coef: {} and intercept: {}'.format(
                                model.coef_[0],
                                model.intercept_))
        # Tm = T.copy(deep=False)
        Tnp = T.dropna('time').values.reshape(-1, 1)
        # T = T.values.reshape(-1, 1)
        Tm = T.dropna('time').copy(deep=False,
                                   data=model.predict((273.15 + Tnp)))
        Tm = Tm.reindex(time=T['time'])
        if slope_err is not None:
            dTm = model.coef_[0] * dT + slope_err * Tm
        else:
            dTm = model.coef_[0] * dT
        # Tm = model.predict((273.15 + T))
    Rv = 461.52  # [Rv] = J / (kg * K) = (Pa * m^3) / (kg * K)
    # (1e-2 mbar * m^3) / (kg * K)
    k = 1e-6 * (k3 / Tm + k2) * Rv
    k = 1.0 / k  # [k] = 100 * kg / m^3 =  kg/ (m^2 * cm)
    # dk = (1e6 / Rv ) * (k3 / Tm + k2)**-2 * (dk3 / Tm + dTm * k3 / Tm**2.0 + dk2)
    # dk = k * np.sqrt(dk3Tm**2.0 + dk2**2.0)
    if no_error:
        return k, _
    else:
        dk = k * (k3 / Tm + k2)**-1 * np.sqrt((dk3 / Tm) **
                                              2.0 + (dTm * k3 / Tm**2.0)**2.0 + dk2**2.0)
        # 1 kg/m^2 IPW = 1 mm PW
        return k, dk


def kappa(T, Tmul=0.72, T_offset=70.2, k2=22.1, k3=3.776e5, Tm_input=False):
    """T in celsious, or in K when Tm_input is True"""
    # original k2=17.0 bevis 1992 etal.
    # [k2] = K / mbar, [k3] = K^2 / mbar
    # 100 Pa = 1 mbar
    if not Tm_input:
        Tm = (273.15 + T) * Tmul + T_offset  # K
    else:
        Tm = T
    Rv = 461.52  # [Rv] = J / (kg * K) = (Pa * m^3) / (kg * K)
    # (1e-2 mbar * m^3) / (kg * K)
    k = 1e-6 * (k3 / Tm + k2) * Rv
    k = 1.0 / k  # [k] = 100 * kg / m^3 =  kg/ (m^2 * cm)
    # 1 kg/m^2 IPW = 1 mm PW
    return k


def minimize_kappa_tela_sound(sound_path=sound_path, gps=garner_path,
                              ims_path=ims_path, station='TELA', bounds=None,
                              x0=None, times=None, season=None):
    from skopt import gp_minimize
    import xarray as xr
    from sklearn.metrics import mean_squared_error
    import numpy as np
    from aux_gps import dim_intersection

    def func_to_min(x):
        Tmul = x[0]
        Toff = x[1]
        # k2 = x[2]
        # Ta = Tmul * (Ts + 273.15) + Toff
        Ts_k = Ts + 273.15
        Ta = Tmul * (Ts_k) + Toff
        added_loss = np.mean((np.where(Ta > Ts_k, 1.0, 0.0))) * 100.0
        k = kappa(Ts, Tmul=Tmul, T_offset=Toff)  # , k2=k2)
        res = sound - k * zwd_gps
        rmse = np.sqrt(mean_squared_error(sound, k * zwd_gps))
        loss = np.abs(np.mean(res)) + rmse
        print('loss:{}, added_loss:{}'.format(loss, added_loss))
        loss += added_loss
        return loss

    # load gerner zwd data:
    zwd_gps = xr.open_dataset(gps / 'garner_israeli_stations_filtered.nc')
    zwd_gps = zwd_gps[station].sel(zwd='value')
    zwd_gps.load()
    # load bet dagan sounding data:
    sound = xr.open_dataarray(sound_path / 'PW_bet_dagan_soundings.nc')
    sound = sound.where(sound > 0, drop=True)
    sound.load()
    # load surface temperature data in C:
    Tds = xr.open_dataset(ims_path / 'IMS_TD_israeli_for_gps.nc')
    Ts = Tds[station.lower()]
    Ts.load()
    # intersect the datetimes:
    new_time = dim_intersection([zwd_gps, sound, Ts], 'time')
    zwd_gps = zwd_gps.sel(time=new_time)
    sound = sound.sel(time=new_time)
    Ts = Ts.sel(time=new_time)
    if times is not None:
        zwd_gps = zwd_gps.sel(time=slice(times[0], times[1]))
        sound = sound.sel(time=slice(times[0], times[1]))
        Ts = Ts.sel(time=slice(times[0], times[1]))
    if season is not None:
        print('Minimizing for season : {}'.format(season))
        zwd_gps = zwd_gps.sel(time=zwd_gps['time.season'] == season)
        sound = sound.sel(time=sound['time.season'] == season)
        Ts = Ts.sel(time=Ts['time.season'] == season)

    zwd_gps = zwd_gps.values
    sound = sound.values
    Ts = Ts.values
    if bounds is None:
        # default boundries:
        bounds = {}
        bounds['Tmul'] = (0.1, 1.0)
        bounds['Toff'] = (0.0, 110.0)
        # bounds['k2'] = (1.0, 150.0)
    if x0 is None:
        # default x0
        x0 = {}
        x0['Tmul'] = 0.5
        x0['Toff'] = 90.0
        # x0['k2'] = 17.0
    if isinstance(x0, dict):
        x0_list = [x0.get('Tmul'), x0.get('Toff')]  # , x0.get('k2')]
        print('Running minimization with initial X:')
        for k, v in x0.items():
            print(k + ': ', v)
    if not x0:
        x0_list = None
        print('Running minimization with NO initial X...')
    print('Running minimization with the following bounds:')
    for k, v in bounds.items():
        print(k + ': ', v)
    bounds_list = [bounds.get('Tmul'), bounds.get('Toff')]  # , bounds.get('k2')]
    res = gp_minimize(func_to_min, dimensions=bounds_list,
                      x0=x0_list, n_jobs=-1, random_state=42,
                      verbose=False)
    return res


def read_zwd_from_tdp_final(tdp_path, st_name='TELA', scatter_plot=True):
    import pandas as pd
    from pandas.errors import EmptyDataError
    from aux_gps import get_unique_index
    import matplotlib.pyplot as plt
    df_list = []
    for file in sorted(tdp_path.glob('*.txt')):
        just_date = file.as_posix().split('/')[-1].split('.')[0]
        dt = pd.to_datetime(just_date)
        try:
            df = pd.read_csv(file, index_col=0, delim_whitespace=True,
                             header=None)
            df.columns = ['zwd']
            df.index = dt + pd.to_timedelta(df.index * 60, unit='min')
            df_list.append(df)
        except EmptyDataError:
            print('found empty file...')
            continue
    df_all = pd.concat(df_list)
    df_all = df_all.sort_index()
    df_all.index.name = 'time'
    ds = df_all.to_xarray()
    ds = ds.rename({'zwd': st_name})
    ds = get_unique_index(ds)
    ds[st_name] = ds[st_name].where(ds[st_name] > 0, drop=True)
    if scatter_plot:
        ds[st_name].plot.line(marker='.', linewidth=0.)
        # plt.scatter(x=ds.time.values, y=ds.TELA.values, marker='.', s=10)
    return ds


def check_anton_tela_station(anton_path, ims_path=ims_path, plot=True):
    import pandas as pd
    from datetime import datetime, timedelta
    from pandas.errors import EmptyDataError
    import matplotlib.pyplot as plt
    import xarray as xr
    df_list = []
    for file in anton_path.glob('tela*.txt'):
        day = int(''.join([x for x in file.as_posix() if x.isdigit()]))
        year = 2015
        dt = pd.to_datetime(datetime(year, 1, 1) + timedelta(day - 1))
        try:
            df = pd.read_csv(file, index_col=0, delim_whitespace=True,
                             header=None)
            df.columns = ['zwd']
            df.index = dt + pd.to_timedelta(df.index * 60, unit='min')
            df_list.append(df)
        except EmptyDataError:
            print('found empty file...')
            continue
    df_all = pd.concat(df_list)
    df_all = df_all.sort_index()
    df_all.index.name = 'time'
    ds = df_all.to_xarray()
    ds = ds.rename({'zwd': 'TELA'})
    new_time = pd.date_range(pd.to_datetime(ds.time.min().values),
                             pd.to_datetime(ds.time.max().values), freq='5min')
    ds = ds.reindex(time=new_time)
    if plot:
        ds['TELA'].plot.line(marker='.', linewidth=0.)
        # plt.scatter(x=ds.time.values, y=ds.TELA.values, marker='.', s=10)
    # Tds = xr.open_dataset(ims_path / 'IMS_TD_israeli_for_gps.nc')
    # k = kappa(Tds.tela, k2=22.1)
    # ds = k * ds
    return ds


def from_opt_to_comparison(result=None, times=None, bounds=None, x0=None,
                           season=None, Tmul=None, T_offset=None):
    """ call optimization and comapring alltogather. can run optimization
    separetly and plugin the result to compare"""
    if result is None:
        print('minimizing the hell out of the function!...')
        result = minimize_kappa_tela_sound(times=times, bounds=bounds, x0=x0,
                                           season=season)
    geo_df = produce_geo_df()
    if result:
        Tmul = result.x[0]
        T_offset = result.x[1]
    if Tmul is not None and T_offset is not None:
        # k2 = result.x[2]
        ipw = produce_IPW_field(geo_df, Tmul=Tmul, T_offset=T_offset,
                                plot=False, hist=False, station='tela')
        pw = compare_to_sounding(gps=ipw, times=times, season=season)
        pw.attrs['result from fitted model'] = result.x
    return pw, result


def compare_to_sounding2(pw_from_gps, pw_from_sounding, station='TELA',
                         times=None, season=None, hour=None, title=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from aux_gps import get_unique_index
    from sklearn.metrics import mean_squared_error
    time_dim_gps = list(set(pw_from_gps.dims))[0]
    time_dim_sound = list(set(pw_from_sounding.dims))[0]
    sns.set_style('darkgrid')
    pw = pw_from_gps.to_dataset(name=station).reset_coords(drop=True)
    pw = pw.dropna(time_dim_gps)
    pw = get_unique_index(pw, time_dim_gps)
    pw_sound = pw_from_sounding.dropna(time_dim_sound)
    pw['sound'] = get_unique_index(pw_sound, time_dim_sound)
    pw['resid'] = pw['sound'] - pw[station]
    time_dim = list(set(pw.dims))[0]
    if time_dim != 'time':
        pw = pw.rename({time_dim: 'time'})
    if times is not None:
        pw = pw.sel(time=slice(times[0], times[1]))
    if season is not None:
        pw = pw.sel(time=pw['time.season'] == season)
    if hour is not None:
        pw = pw.sel(time=pw['time.hour'] == hour)
    if title is None:
        sup = 'TPW is created using Bevis Tm formulation'
    if title is not None:
        if title == 'hour':
            sup = 'TPW for {} is created using empirical hourly Tm segmentation and formulation'.format(station)
        elif title == 'season':
            sup = 'TPW for {} is created using empirical seasonly Tm segmentation and formulation'.format(station)
        elif title == 'whole':
            sup = 'TPW for {} is created using whole empirical Tm formulation'.format(station)
        elif title == 'hour_season':
            sup = 'TPW for {} is created using empirical seasonly and hourly Tm segmentation and formulation'.format(station)
    fig, ax = plt.subplots(1, 2, figsize=(20, 4),
                           gridspec_kw={'width_ratios': [3, 1]})
    ax[0].set_title(sup)
    pw[[station, 'sound']].to_dataframe().plot(ax=ax[0], style='.')
    sns.distplot(
        pw['resid'].values,
        bins=100,
        color='c',
        label='residuals',
        ax=ax[1])
    # pw['resid'].plot.hist(bins=100, color='c', edgecolor='k', alpha=0.65,
    #                      ax=ax[1])
    rmean = pw['resid'].mean().values
    rstd = pw['resid'].std().values
    rmedian = pw['resid'].median().values
    rmse = np.sqrt(mean_squared_error(pw['sound'], pw[station]))
    plt.axvline(rmean, color='r', linestyle='dashed', linewidth=1)
    # plt.axvline(rmedian, color='b', linestyle='dashed', linewidth=1)
    _, max_ = plt.ylim()
    plt.text(rmean + rmean / 10, max_ - max_ / 10,
             'Mean: {:.2f}, RMSE: {:.2f}'.format(rmean, rmse))
    fig.tight_layout()
    if season is None:
        pw['season'] = pw['time.season']
        pw['hour'] = pw['time.hour'].astype(str)
        pw['hour'] = pw.hour.where(pw.hour != '12', 'noon')
        pw['hour'] = pw.hour.where(pw.hour != '0', 'midnight')
        df = pw.to_dataframe()
    #    g = sns.relplot(
    #        data=df,
    #        x='sound',
    #        y='TELA',
    #        col='season',
    #        hue='hour',
    #        kind='scatter',
    #        style='season')
    #    if times is not None:
    #        plt.subplots_adjust(top=0.85)
    #        g.fig.suptitle('Time: ' + times[0] + ' to ' + times[1], y=0.98)
        h_order = ['noon', 'midnight']
        s_order = ['DJF', 'JJA', 'SON', 'MAM']
        g = sns.lmplot(
            data=df,
            x='sound',
            y='TELA',
            col='season',
            hue='season',
            row='hour',
            row_order=h_order,
            col_order=s_order)
        g.set(ylim=(0, 50), xlim=(0, 50))
        if times is not None:
            plt.subplots_adjust(top=0.9)
            g.fig.suptitle('Time: ' + times[0] + ' to ' + times[1], y=0.98)
        g = sns.FacetGrid(data=df, col='season', hue='season', row='hour',
                          row_order=h_order, col_order=s_order)
        g.fig.set_size_inches(15, 8)
        g = (g.map(sns.distplot, "resid"))
        rmeans = []
        rmses = []
        for hour in h_order:
            for season in s_order:
                sliced_pw = pw.sel(
                    time=pw['time.season'] == season).where(
                    pw.hour != hour).dropna('time')
                rmses.append(
                    np.sqrt(
                        mean_squared_error(
                            sliced_pw['sound'],
                            sliced_pw[station])))
                rmeans.append(sliced_pw['resid'].mean().values)
        for i, ax in enumerate(g.axes.flat):
            ax.axvline(rmeans[i], color='k', linestyle='dashed', linewidth=1)
            _, max_ = ax.get_ylim()
            ax.text(rmeans[i] + rmeans[i] / 10, max_ - max_ / 10,
                    'Mean: {:.2f}, RMSE: {:.2f}'.format(rmeans[i], rmses[i]))
        # g.set(xlim=(-5, 5))
        if times is not None:
            plt.subplots_adjust(top=0.9)
            g.fig.suptitle('Time: ' + times[0] + ' to ' + times[1], y=0.98)
    return pw


def compare_to_sounding(sound_path=sound_path, gps=garner_path, station='TELA',
                        times=None, season=None, hour=None, title=None):
    """ipw comparison to bet-dagan sounding, gps can be the ipw dataset"""
    import xarray as xr
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import mean_squared_error
    from pathlib import Path
    sns.set_style('darkgrid')
    if isinstance(gps, Path):
        pw_gps = xr.open_dataset(gps / 'IPW_israeli_from_gps.nc')
    else:
        pw_gps = gps
    if [x for x in pw_gps.coords if x == 'ipw']:
        pw_gps = pw_gps[station].sel(ipw='value')
    else:
        pw_gps = pw_gps[station]
    pw_gps.load()
    sound = xr.open_dataarray(sound_path / 'PW_bet_dagan_soundings.nc')
    # drop 0 pw - not physical
    sound = sound.where(sound > 0, drop=True)
    sound.load()
    new_time = list(set(pw_gps.dropna('time').time.values).intersection(
        set(sound.dropna('time').time.values)))
    new_dt = sorted(pd.to_datetime(new_time))
    # selecting requires time...
    print('selecting intersected datetime...')
    pw_gps = pw_gps.sel(time=new_dt)
    sound = sound.sel(time=new_dt)
    pw = pw_gps.to_dataset(name=station).reset_coords(drop=True)
    pw['sound'] = sound
    pw['resid'] = pw['sound'] - pw[station]
    pw.load()
    print('Done!')
    if times is not None:
        pw = pw.sel(time=slice(times[0], times[1]))
    if season is not None:
        pw = pw.sel(time=pw['time.season'] == season)
    if hour is not None:
        pw = pw.sel(time=pw['time.hour'] == hour)
    if title is None:
        sup = 'PW is created using Bevis Tm formulation'
    if title is not None:
        if title == 'hour':
            sup = 'PW is created using hourly Tm segmentation and formulation'
        elif title == 'season':
            sup = 'PW is created using seasonly Tm segmentation and formulation'
        elif title == 'whole':
            sup = 'PW is created using whole Tm formulation'
        elif title == 'hour_season':
            sup = 'PW is created using seasonly and hourly Tm segmentation and formulation'
    fig, ax = plt.subplots(1, 2, figsize=(20, 4),
                           gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(sup, fontweight='bold')
    pw[[station, 'sound']].to_dataframe().plot(ax=ax[0], style='.')
    sns.distplot(
        pw['resid'].values,
        bins=100,
        color='c',
        label='residuals',
        ax=ax[1])
    # pw['resid'].plot.hist(bins=100, color='c', edgecolor='k', alpha=0.65,
    #                      ax=ax[1])
    rmean = pw['resid'].mean().values
    rstd = pw['resid'].std().values
    rmedian = pw['resid'].median().values
    rmse = np.sqrt(mean_squared_error(pw['sound'], pw[station]))
    plt.axvline(rmean, color='r', linestyle='dashed', linewidth=1)
    # plt.axvline(rmedian, color='b', linestyle='dashed', linewidth=1)
    _, max_ = plt.ylim()
    plt.text(rmean + rmean / 10, max_ - max_ / 10,
             'Mean: {:.2f}, RMSE: {:.2f}'.format(rmean, rmse))
    fig.tight_layout()
    if season is None:
        pw['season'] = pw['time.season']
        pw['hour'] = pw['time.hour'].astype(str)
        pw['hour'] = pw.hour.where(pw.hour != '12', 'noon')
        pw['hour'] = pw.hour.where(pw.hour != '0', 'midnight')
        df = pw.to_dataframe()
    #    g = sns.relplot(
    #        data=df,
    #        x='sound',
    #        y='TELA',
    #        col='season',
    #        hue='hour',
    #        kind='scatter',
    #        style='season')
    #    if times is not None:
    #        plt.subplots_adjust(top=0.85)
    #        g.fig.suptitle('Time: ' + times[0] + ' to ' + times[1], y=0.98)
        h_order = ['noon', 'midnight']
        s_order = ['DJF', 'JJA', 'SON', 'MAM']
        g = sns.lmplot(
            data=df,
            x='sound',
            y='TELA',
            col='season',
            hue='season',
            row='hour',
            row_order=h_order,
            col_order=s_order)
        g.set(ylim=(0, 50), xlim=(0, 50))
        if times is not None:
            plt.subplots_adjust(top=0.9)
            g.fig.suptitle('Time: ' + times[0] + ' to ' + times[1], y=0.98)
        g = sns.FacetGrid(data=df, col='season', hue='season', row='hour',
                          row_order=h_order, col_order=s_order)
        g.fig.set_size_inches(15, 8)
        g = (g.map(sns.distplot, "resid"))
        rmeans = []
        rmses = []
        for hour in h_order:
            for season in s_order:
                sliced_pw = pw.sel(
                    time=pw['time.season'] == season).where(
                    pw.hour != hour).dropna('time')
                rmses.append(
                    np.sqrt(
                        mean_squared_error(
                            sliced_pw['sound'],
                            sliced_pw[station])))
                rmeans.append(sliced_pw['resid'].mean().values)
        for i, ax in enumerate(g.axes.flat):
            ax.axvline(rmeans[i], color='k', linestyle='dashed', linewidth=1)
            _, max_ = ax.get_ylim()
            ax.text(rmeans[i] + rmeans[i] / 10, max_ - max_ / 10,
                    'Mean: {:.2f}, RMSE: {:.2f}'.format(rmeans[i], rmses[i]))
        # g.set(xlim=(-5, 5))
        if times is not None:
            plt.subplots_adjust(top=0.9)
            g.fig.suptitle('Time: ' + times[0] + ' to ' + times[1], y=0.98)
    # maybe month ?
    # plt.text(rmedian + rmedian / 10, max_ - max_ / 10,
    #          'Mean: {:.2f}'.format(rmedian))
    return pw


def ml_models_T_from_sounding(sound_path=sound_path, categories=None,
                              models=['LR', 'TSEN'], physical_file=None,
                              times=['2005', '2019'], plot=True):
    """calls formulate_plot to analyse and model the ts-tm connection from
    radiosonde(bet-dagan). options for categories:season, hour, clouds
    you can choose some ,all or none categories"""
    import xarray as xr
    from aux_gps import get_unique_index
    from aux_gps import keep_iqr
    if isinstance(models, str):
        models = [models]
    if physical_file is not None:
        print('Overwriting ds input...')
        pds = xr.open_dataset(physical_file)
        pds = pds[['Tm', 'Ts']]
        pds = pds.rename({'Ts': 'ts', 'Tm': 'tm'})
        pds = pds.rename({'sound_time': 'time'})
        pds = get_unique_index(pds)
        pds = keep_iqr(pds, k=2.0)
        ds = pds.dropna('time')
    else:
        ds = xr.open_dataset(sound_path /
                             'bet_dagan_sounding_pw_Ts_Tk_with_clouds.nc')
        ds = ds.reset_coords(drop=True)
    if times is not None:
        ds = ds.sel(time=slice(*times))
    # define the possible categories and feed their dictionary:
    possible_cats = ['season', 'hour']
    pos_cats_dict = {}
    s_order = ['DJF', 'JJA', 'SON', 'MAM']
    h_order = [12, 0]
    cld_order = [0, 1]
    time_dim = list(set(ds.dims))[0]
    if 'season' in possible_cats:
        pos_cats_dict['{}.season'.format(time_dim)] = s_order
    if 'hour' in possible_cats:
        pos_cats_dict['{}.hour'.format(time_dim)] = h_order
    if categories is None:
        results = formulate_plot(ds, model_names=models, plot=plot)
    if categories is not None:
        if not isinstance(categories, list):
            categories = [categories]
        if set(categories + possible_cats) != set(possible_cats):
            raise ValueError(
                'choices for categories are: ' +
                ', '.join(possible_cats))
        categories = [x.replace(x, time_dim + '.' + x) if x ==
                      'season' or x == 'hour' else x for x in categories]
        results = formulate_plot(ds, pos_cats_dict=pos_cats_dict,
                                 chosen_cats=categories, model_names=models,
                                 plot=plot)
    results.attrs['time_dim'] = time_dim
    return results


#def linear_T_from_sounding(sound_path=sound_path, categories=None):
#    import xarray as xr
#    ds = xr.open_dataset(sound_path / 'bet_dagan_sounding_pw_Ts_Tk_with_clouds.nc')
#    ds = ds.reset_coords(drop=True)
#    s_order = ['DJF', 'JJA', 'SON', 'MAM']
#    h_order = ['noon', 'midnight']
#    cld_order = [0, 1]
#    if categories is None:
#        results = formulate_plot(ds)
#    if categories is not None:
#        if not isinstance(categories, list):
#            categories = [categories]
#        if set(categories + ['season', 'hour', 'clouds']) != set(['season',
#                                                                  'hour',
#                                                                  'clouds']):
#            raise ValueError('choices for categories are: season, hour, clouds')
#        if len(categories) == 1:
#            if 'season' in categories:
#                dd = {'season': s_order}
#            elif 'hour' in categories:
#                dd = {'hour': h_order}
#            elif 'clouds' in categories:
#                dd = {'any_cld': cld_order}
#        elif len(categories) == 2:
#            if 'season' in categories and 'hour' in categories:
#                dd = {'hour': h_order, 'season': s_order}
#            elif 'season' in categories and 'clouds' in categories:
#                dd = {'any_cld': cld_order, 'season': s_order}
#            elif 'clouds' in categories and 'hour' in categories:
#                dd = {'hour': h_order, 'any_cld': cld_order}
#        elif len(categories) == 3:
#            if 'season' in categories and 'hour' in categories and 'clouds' in categories:
#                dd = {'hour': h_order, 'any_cld': cld_order, 'season': s_order}
#        results = formulate_plot(ds, dd)
#    return results


def formulate_plot(ds, model_names=['LR', 'TSEN'],
                   pos_cats_dict=None, chosen_cats=None, plot=True):
    """accepts pos_cat (dict) with keys : hour, season ,and appropriate
    values, and chosen keys and returns trained sklearn models with
    the same slices.
    this function is called by 'ml_models_T_from_sounding' above."""
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import mean_squared_error
    from aux_gps import standard_error_slope
    time_dim = list(set(ds.dims))[0]
    print('time dim is: {}'.format(time_dim))
    sns.set_style('darkgrid')
    colors = ['red', 'green', 'magenta', 'cyan', 'orange', 'teal',
              'gray', 'purple']
    pos = np.linspace(0.95, 0.6, 8)
#    if res_save not in model_names:
#        raise KeyError('saved result should me in model names!')
    if len(model_names) > len(colors):
        raise ValueError(
            'Cannot support more than {} models simultenously!'.format(
                len(colors)))
    ml = ML_Switcher()
    models = [ml.pick_model(x) for x in model_names]
    if chosen_cats is None:
        print('no categories selected, using full data.')
        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(10, 7))
            fig.suptitle(
                'Bet Dagan WV weighted mean atmosphric temperature(Tm) vs. surface temperature(Ts)', fontweight='bold')
        X = ds.ts.values.reshape(-1, 1)
        y = ds.tm.values
        [model.fit(X, y) for model in models]
        predict = [model.predict(X) for model in models]
        coefs = [model.coef_[0] for model in models]
        inters = [model.intercept_ for model in models]
#        [a, b] = np.polyfit(ds.ts.values, ds.tm.values, 1)
#        result = np.empty((2))
#        result[0] = a
#        result[1] = b
        # sns.regplot(ds.ts.values, ds.tm.values, ax=axes[0])
        df = ds.ts.dropna('time').to_dataframe()
        df['tm'] = ds.tm.dropna('time')
        try:
            df['clouds'] = ds.any_cld.dropna('time')
            hue = 'clouds'
        except AttributeError:
            hue = None
            pass
        if plot:
            g = sns.scatterplot(
                data=df,
                x='ts',
                y='tm',
                hue=hue,
                marker='.',
                s=100,
                ax=axes[0])
            g.legend(loc='best')

        # axes[0].scatter(x=ds.ts.values, y=ds.tm.values, marker='.', s=10)
#        linex = np.array([ds.ts.min().item(), ds.ts.max().item()])
#        liney = a * linex + b
#        axes[0].plot(linex, liney, c='r')
        bevis_tm = ds.ts.values * 0.72 + 70.0
        if plot:
            axes[0].plot(ds.ts.values, bevis_tm, c='purple')
            min_, max_ = axes[0].get_ylim()
            [axes[0].plot(X, newy, c=colors[i]) for i, newy in enumerate(predict)]
            [axes[0].text(0.01, pos[i],
                          '{} a: {:.2f}, b: {:.2f}'.format(model_names[i],
                                                           coefs[i], inters[i]),
                          transform=axes[0].transAxes, color=colors[i],
                          fontsize=12) for i in range(len(coefs))]
            axes[0].text(0.01, 0.9,
                         'Bevis 1992 et al. a: 0.72, b: 70.0',
                         transform=axes[0].transAxes, color='purple',
                         fontsize=12)
    #        axes[0].text(0.01, 0.9, 'a: {:.2f}, b: {:.2f}'.format(a, b),
    #                     transform=axes[0].transAxes, color='black', fontsize=12)
            axes[0].text(0.1, 0.85, 'n={}'.format(ds.ts.size),
                         verticalalignment='top', horizontalalignment='center',
                         transform=axes[0].transAxes, color='blue', fontsize=12)
            axes[0].set_xlabel('Ts [K]')
            axes[0].set_ylabel('Tm [K]')
        # resid = ds.tm.values - ds.ts.values * a - b
        resid = predict[0] - y
        if plot:
            sns.distplot(resid, bins=25, color='c', label='residuals', ax=axes[1])
        rmean = np.mean(resid)
        rmse = np.sqrt(mean_squared_error(predict[0], y))
        if plot:
            _, max_ = axes[1].get_ylim()
            axes[1].text(rmean + rmean / 10, max_ - max_ / 10,
                         'Mean: {:.2f}, RMSE: {:.2f}'.format(rmean, rmse))
            axes[1].axvline(rmean, color='r', linestyle='dashed', linewidth=1)
            axes[1].set_xlabel('Residuals [K]')
            fig.tight_layout()
        da = xr.DataArray(models, dims=['name'])
        da['name'] = model_names
        da.name = 'all_data_trained_models'
        # results = xr.DataArray(result, dims=['parameter'])
        # results['parameter'] = ['slope', 'intercept']
    elif chosen_cats is not None:
        size = len(chosen_cats)
        if size == 1:
            key = chosen_cats[0]
            vals = pos_cats_dict[key]
            print('{} category selected.'.format(key))
#            other_keys = [
#                *set([x for x in pos_cats_dict.keys()]).difference([key])]
#            other_keys = [
#                *set(['any_cld', 'hour', 'season']).difference([key])]
#            result = np.empty((len(vals), 2))
#            residuals = []
#            rmses = []
            trained = []
            if plot:
                fig, axes = plt.subplots(1, len(vals), sharey=True, sharex=True,
                                         figsize=(15, 8))
                fig.suptitle(
                        'Bet Dagan WV weighted mean atmosphric temperature(Tm) vs. surface temperature(Ts) using {} selection criteria'.format(key.split('.')[-1]), fontweight='bold',x=0.5, y=1.0)

            for i, val in enumerate(vals):
                ts = ds.ts.where(ds[key] == val).dropna(time_dim)
                tm = ds.tm.where(ds[key] == val).dropna(time_dim)
#                other_val0 = ds[other_keys[0]].where(
#                    ds[key] == val).dropna(time_dim)
#                other_val1 = ds[other_keys[1]].where(
#                    ds[key] == val).dropna(time_dim)
                X = ts.values.reshape(-1, 1)
                y = tm.values
                models = [ml.pick_model(x) for x in model_names]
                [model.fit(X, y) for model in models]
                predict = [model.predict(X) for model in models]
                coefs = [model.coef_[0] for model in models]
                inters = [model.intercept_ for model in models]
                # [tmul, toff] = np.polyfit(x.values, y.values, 1)
                # result[i, 0] = tmul
                # result[i, 1] = toff
                # new_tm = tmul * x.values + toff
                # resid = new_tm - y.values
                # rmses.append(np.sqrt(mean_squared_error(y.values, new_tm)))
                # residuals.append(resid)
                if plot:
                    axes[i].text(0.15, 0.85, 'n={}'.format(ts.size),
                                 verticalalignment='top',
                                 horizontalalignment='center',
                                 transform=axes[i].transAxes, color='blue',
                                 fontsize=12)
                df = ts.to_dataframe()
                df['tm'] = tm
#                df[other_keys[0]] = other_val0
#                df[other_keys[1]] = other_val1
#                g = sns.scatterplot(data=df, x='ts', y='tm', marker='.', s=100,
#                                    ax=axes[i], hue=other_keys[0],
#                                    style=other_keys[1])
                if plot:
                    g = sns.scatterplot(data=df, x='ts', y='tm', marker='.', s=100,
                                        ax=axes[i])
                    g.legend(loc='upper right')
                    # axes[i, j].scatter(x=x.values, y=y.values, marker='.', s=10)
                    axes[i].set_title('{}:{}'.format(key, val))
    #                linex = np.array([x.min().item(), x.max().item()])
    #                liney = tmul * linex + toff
    #                axes[i].plot(linex, liney, c='r')
                    # unmark the following line to disable plotting y=x line:
    #                bevis_tm = ts.values * 0.72 + 70.0
    #                axes[i].plot(ts.values, bevis_tm, c='k')
                    min_, max_ = axes[i].get_ylim()
                    [axes[i].plot(X, newy, c=colors[j]) for j, newy in
                     enumerate(predict)]
                    [axes[i].text(0.01, pos[j],
                                  '{} a: {:.2f}, b: {:.2f}'.format(model_names[j],
                                                                   coefs[j],
                                                                   inters[j]),
                                  transform=axes[i].transAxes, color=colors[j],
                                  fontsize=12) for j in range(len(coefs))]
    #                axes[i].text(0.015, 0.9, 'a: {:.2f}, b: {:.2f}'.format(
    #                             tmul, toff), transform=axes[i].transAxes,
    #                             color='black', fontsize=12)
                    axes[i].set_xlabel('Ts [K]')
                    axes[i].set_ylabel('Tm [K]')
                    fig.tight_layout()
                trained.append(models)
            da = xr.DataArray(trained, dims=[key, 'name'])
            da['name'] = model_names
            da[key] = vals
        elif size == 2:
#            other_keys = [*set(['any_cld', 'hour', 'season']).difference(keys)]
#            other_keys = [*set(['hour', 'season']).difference(keys)]
            vals = [pos_cats_dict[key] for key in chosen_cats]
            keys = chosen_cats
#            result = np.empty((len(vals[0]), len(vals[1]), 2))
#            residuals = []
#            rmses = []
            trained = []
            if plot:
                fig, axes = plt.subplots(len(vals[0]), len(vals[1]), sharey=True,
                                         sharex=True, figsize=(15, 8))
                fig.suptitle(
                    'Bet Dagan WV weighted mean atmosphric temperature(Tm) vs. surface temperature(Ts) using {} and {} selection criteria'.format(keys[0].split('.')[-1], keys[1].split('.')[-1]), fontweight='bold',x=0.5, y=1.0)
            for i, val0 in enumerate(vals[0]):
                trained0 = []
                for j, val1 in enumerate(vals[1]):
                    ts = ds.ts.where(ds[keys[0]] == val0).dropna(
                        time_dim).where(ds[keys[1]] == val1).dropna(time_dim)
                    tm = ds.tm.where(ds[keys[0]] == val0).dropna(
                        time_dim).where(ds[keys[1]] == val1).dropna(time_dim)
#                    other_val = ds[other_keys[0]].where(ds[keys[0]] == val0).dropna(
#                        'time').where(ds[keys[1]] == val1).dropna('time')
                    X = ts.values.reshape(-1, 1)
                    y = tm.values
                    models = [ml.pick_model(x) for x in model_names]
                    [model.fit(X, y) for model in models]
                    predict = [model.predict(X) for model in models]
                    coefs = [model.coef_[0] for model in models]
                    inters = [model.intercept_ for model in models]
#                    [tmul, toff] = np.polyfit(x.values, y.values, 1)
#                    result[i, j, 0] = tmul
#                    result[i, j, 1] = toff
#                    new_tm = tmul * x.values + toff
#                    resid = new_tm - y.values
#                    rmses.append(np.sqrt(mean_squared_error(y.values, new_tm)))
#                    residuals.append(resid)
                    if plot:
                        axes[i, j].text(0.15, 0.85, 'n={}'.format(ts.size),
                                        verticalalignment='top',
                                        horizontalalignment='center',
                                        transform=axes[i, j].transAxes,
                                        color='blue', fontsize=12)
                    df = ts.to_dataframe()
                    df['tm'] = tm
                    # df[other_keys[0]] = other_val
#                    g = sns.scatterplot(data=df, x='ts', y='tm', marker='.',
#                                        s=100, ax=axes[i, j],
#                                        hue=other_keys[0])
                    if plot:
                        g = sns.scatterplot(data=df, x='ts', y='tm', marker='.',
                                            s=100, ax=axes[i, j])
                        g.legend(loc='upper right')
                        # axes[i, j].scatter(x=x.values, y=y.values, marker='.', s=10)
                        # axes[i, j].set_title('{}:{}'.format(key, val))
                        [axes[i, j].plot(X, newy, c=colors[k]) for k, newy in
                         enumerate(predict)]
                        # linex = np.array([x.min().item(), x.max().item()])
                        # liney = tmul * linex + toff
                        # axes[i, j].plot(linex, liney, c='r')
                        # axes[i, j].plot(ts.values, ts.values, c='k', alpha=0.2)
                        min_, max_ = axes[i, j].get_ylim()
    
                        [axes[i, j].text(0.01, pos[k],
                                         '{} a: {:.2f}, b: {:.2f}'.format(model_names[k],
                                                                          coefs[k],
                                                                          inters[k]),
                                         transform=axes[i, j].transAxes, color=colors[k],
                                         fontsize=12) for k in range(len(coefs))]
    #                    axes[i, j].text(0.015, 0.9, 'a: {:.2f}, b: {:.2f}'.format(
    #                                 tmul, toff), transform=axes[i, j].transAxes,
    #                                 color='black', fontsize=12)
                        axes[i, j].set_xlabel('Ts [K]')
                        axes[i, j].set_ylabel('Tm [K]')
                        axes[i, j].set_title('{}:{}, {}:{}'.format(keys[0], val0,
                                                                   keys[1], val1))
                        fig.tight_layout()
                    trained0.append(models)
                trained.append(trained0)
            da = xr.DataArray(trained, dims=keys + ['name'])
            da['name'] = model_names
            da[keys[0]] = vals[0]
            da[keys[1]] = vals[1]
        else:
            raise ValueError('size of categories must be <=2')
    X = ds.ts.values
    y = ds.tm.values
    std_err = standard_error_slope(X, y)
    da.attrs['LR_whole_stderr_slope'] = std_err
    return da


def israeli_gnss_stations_long_term_trend_analysis(gis_path=gis_path,
                                                   rel_plot='tela'):
    import pandas as pd
    from pathlib import Path
    import geopandas as gpd
    import matplotlib.pyplot as plt
    cwd = Path().cwd()
    filename = 'israeli_long_term_tectonics_trends.txt'
    if (cwd / filename).is_file():
        df = pd.read_csv(cwd / filename, delim_whitespace=True,
                         index_col='station')
    else:
        isr_stations = pd.read_csv(cwd / 'stations_approx_loc.txt',
                                   delim_whitespace=True)
        isr_stations = isr_stations.index.tolist()
        df_list = []
        for station in isr_stations:
            print('proccessing station: {}'.format(station))
            try:
                rds = get_long_trends_from_gnss_station(station, 'LR', False)
            except FileNotFoundError:
                print(
                    'didnt find {} in gipsyx solutions, skipping...'.format(station))
                continue
            df_list.append(rds.attrs)
        df = pd.DataFrame(df_list)
        df.set_index(df.station, inplace=True)
        df.drop('station', axis=1, inplace=True)
        rest = df.columns[3:].tolist()
        df.columns = [
            'north_cm_per_year',
            'east_cm_per_year',
            'up_mm_per_year'] + rest
        df['cm_per_year'] = np.sqrt(
            df['north_cm_per_year'] ** 2.0 +
            df['east_cm_per_year'] ** 2.0)
        # define angle from east : i.e., x axis is east
        df['angle_from_east'] = np.rad2deg(
            np.arctan2(df['north_cm_per_year'], df['east_cm_per_year']))
        for station in df.index:
            df['rel_mm_north_{}'.format(station)] = (df['north_cm_per_year'] \
                - df.loc[station, 'north_cm_per_year']) * 100.0
            df['rel_mm_east_{}'.format(station)] = (df['east_cm_per_year'] \
                - df.loc[station, 'east_cm_per_year']) * 100.0
            df['rel_mm_per_year_{}'.format(station)] = np.sqrt(
                df['rel_mm_north_{}'.format(station)] ** 2.0 +
                df['rel_mm_east_{}'.format(station)] ** 2.0)
            # define angle from east : i.e., x axis is east
            df['rel_angle_from_east_{}'.format(station)] = np.rad2deg(np.arctan2(
                df['rel_mm_north_{}'.format(station)], df['rel_mm_east_{}'.format(station)]))
        df.to_csv(cwd / filename, sep=' ')
        print('{} was saved to {}'.format(filename, cwd))
    isr_with_yosh = gpd.read_file(gis_path / 'Israel_demog_yosh.shp')
    isr_with_yosh.crs = {'init': 'epsg:4326'}
    stations = gpd.GeoDataFrame(df,
                                geometry=gpd.points_from_xy(df.lon,
                                                            df.lat),
                                crs=isr_with_yosh.crs)
    isr = gpd.sjoin(stations, isr_with_yosh, op='within')
    isr['X'] = isr.geometry.x
    isr['Y'] = isr.geometry.y
    isr['U'] = isr.east_cm_per_year
    isr['V'] = isr.north_cm_per_year
    if rel_plot is None:
        # isr.drop('dsea', axis=0, inplace=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        isr_with_yosh.plot(ax=ax)
        isr[(isr['years'] <= 10.0) & (isr['years'] >= 5.0)].plot(ax=ax, markersize=50, color='y', edgecolor='k', marker='o', label='5-10 yrs')
        isr[(isr['years'] <= 15.0) & (isr['years'] > 10.0)].plot(ax=ax, markersize=50, color='g', edgecolor='k', marker='o', label='10-15 yrs')
        isr[(isr['years'] <= 20.0) & (isr['years'] > 15.0)].plot(ax=ax, markersize=50, color='c', edgecolor='k', marker='o', label='15-20 yrs')
        isr[(isr['years'] <= 25.0) & (isr['years'] > 20.0)].plot(ax=ax, markersize=50, color='r', edgecolor='k', marker='o', label='20-25 yrs')
        plt.legend(prop={'size': 12}, bbox_to_anchor=(1.05, 1.0), title='number of data years')
        # isr.plot(ax=ax, column='cm_per_year', cmap='Greens',
        #          edgecolor='black', legend=True)
        cmap = plt.get_cmap('spring', 10)
        Q = ax.quiver(isr['X'], isr['Y'], isr['U'], isr['V'],
                      isr['cm_per_year'], cmap=cmap)
        fig.colorbar(Q, extend='max')
        qk = ax.quiverkey(Q, 0.8, 0.9, 1, r'$1 \frac{cm}{yr}$', labelpos='E',
                          coordinates='figure')
        for x, y, label in zip(isr.lon, isr.lat,
                               isr.index):
            ax.annotate(label, xy=(x, y), xytext=(3, 3),
                        textcoords="offset points")
    elif rel_plot is not None:
        # isr.drop('dsea', axis=0, inplace=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        isr_with_yosh.plot(ax=ax)
        isr[(isr['years'] <= 10.0) & (isr['years'] >= 5.0)].plot(ax=ax, markersize=50, color='y', edgecolor='k', marker='o', label='5-10 yrs')
        isr[(isr['years'] <= 15.0) & (isr['years'] > 10.0)].plot(ax=ax, markersize=50, color='g', edgecolor='k', marker='o', label='10-15 yrs')
        isr[(isr['years'] <= 20.0) & (isr['years'] > 15.0)].plot(ax=ax, markersize=50, color='c', edgecolor='k', marker='o', label='15-20 yrs')
        isr[(isr['years'] <= 25.0) & (isr['years'] > 20.0)].plot(ax=ax, markersize=50, color='r', edgecolor='k', marker='o', label='20-25 yrs')
        plt.legend(prop={'size': 12}, bbox_to_anchor=(-0.15, 1.0), title='number of data years')
        # isr.plot(ax=ax, column='cm_per_year', cmap='Greens',
        #          edgecolor='black', legend=True)
        isr['U'] = isr['rel_mm_east_{}'.format(rel_plot)]
        isr['V'] = isr['rel_mm_north_{}'.format(rel_plot)]
        cmap = plt.get_cmap('spring', 7)
        Q = ax.quiver(isr['X'], isr['Y'], isr['U'], isr['V'],
                      isr['rel_mm_per_year_{}'.format(rel_plot)],
                      cmap=cmap)
        qk = ax.quiverkey(Q, 0.8, 0.9, 1, r'$1 \frac{mm}{yr}$', labelpos='E',
                          coordinates='figure')
        fig.colorbar(Q, extend='max')
        plt.title('Relative to {} station'.format(rel_plot))
        for x, y, label in zip(isr.lon, isr.lat,
                               isr.index):
            ax.annotate(label, xy=(x, y), xytext=(3, 3),
                        textcoords="offset points")
        print(isr[['rel_mm_east_{}'.format(rel_plot),'rel_mm_north_{}'.format(rel_plot)]])
    return df


#def save_resampled_versions_gispyx_results(station='tela', sample_rate='H'):
#    from aux_gps import path_glob
#    import xarray as xr
#    """resample gipsyx results nc files and save them.options for
#    sample_rate are in sample dict"""
#    path = GNSS / station / 'gipsyx_solutions'
#    glob = '{}_PPP*.nc'.format(station.upper())
#    try:
#        file = path_glob(path, glob_str=glob)[0]
#    except FileNotFoundError:
#        print('did not find {} in gipsyx_solutions dir, skipping...'.format(station))
#        return
#    filename = file.as_posix().split('/')[-1].split('.')[0]
#    years = filename.split('_')[-1]
#    ds = xr.open_dataset(file)
#    time_dim = list(set(ds.dims))[0]
#    sample = {'H': 'hourly', 'W': 'weekly', 'MS': 'monthly'}
#    print('resampaling {} to {}'.format(station, sample[sample_rate]))
#    dsr = ds.resample({time_dim: sample_rate}, keep_attrs=True).mean(keep_attrs=True)
#    new_filename = '_'.join([station.upper(), sample[sample_rate], 'PPP',
#                             years])
#    new_filename = new_filename + '.nc'
#    print('saving resmapled station {} to {}'.format(station, path))
#    comp = dict(zlib=True, complevel=9)  # best compression
#    encoding = {var: comp for var in dsr.data_vars}
#    dsr.to_netcdf(path / new_filename, 'w', encoding=encoding)
#    print('Done!')
#    return dsr
def load_GNSS_TD(station='tela', sample_rate=None, plot=True):
    """load and plot temperature for station from IMS, to choose
    sample rate different than 5 mins choose: 'H', 'W' or 'MS'"""
    from aux_gps import path_glob
    from aux_gps import plot_tmseries_xarray
    import xarray as xr
    sample = {'1H': 'hourly', '3H': '3hourly', 'D': 'Daily', 'W': 'weekly',
              'MS': 'monthly'}
    path = ims_path
    if sample_rate is None:
        glob = 'GNSS_5mins_TD_ALL*.nc'
        try:
            file = path_glob(path, glob_str=glob)[0]
        except FileNotFoundError as e:
            print(e)
            return station
    else:
        glob = 'GNSS_{}_TD_ALL*.nc'.format(sample[sample_rate])
        try:
            file = path_glob(path, glob_str=glob)[0]
        except FileNotFoundError as e:
            print(e)
            return station
    ds = xr.open_dataset(file)
    da = ds[station]
    if plot:
        plot_tmseries_xarray(da)
    return da


def load_gipsyx_results(station='tela', sample_rate=None,
                        plot_fields=['WetZ'], field_all=None):
    """load and plot gipsyx solutions for station, to choose sample rate
    different than 5 mins choose: 'H', 'W' or 'MS', use field_all to select
    one field (e.g., WetZ) and get a dataset with all stations with
    the one field."""
    from aux_gps import path_glob
    from aux_gps import plot_tmseries_xarray
    import xarray as xr
    import pandas as pd
    from pathlib import Path

    def load_one_results_ds(station, sample_rate, plot_fields=None):
        path = GNSS / station / 'gipsyx_solutions'
        if sample_rate is None:
            glob = '{}_PPP*.nc'.format(station.upper())
            try:
                file = path_glob(path, glob_str=glob)[0]
                sample_rate = '5 mins'
            except FileNotFoundError as e:
                print(e)
                return None
        else:
            glob = '{}_{}_PPP*.nc'.format(station.upper(), sample[sample_rate])
            try:
                file = path_glob(path, glob_str=glob)[0]
            except FileNotFoundError as e:
                print(e)
                return None
        ds = xr.open_dataset(file)
        print('loaded {} station with a {} sample rate'.format(station,
                                                               sample_rate))
        if plot_fields is not None and plot_fields != 'all':
            plot_tmseries_xarray(ds, plot_fields)
        elif plot_fields == 'all':
            plot_tmseries_xarray(ds, ['GradNorth', 'GradEast', 'WetZ', 'lat',
                                      'lon', 'alt'])
        return ds

    sample = {'1H': 'hourly', '3H': '3hourly', 'D': 'Daily', 'W': 'weekly',
              'MS': 'monthly'}
    if field_all is None:
        ds = load_one_results_ds(station, sample_rate, plot_fields)
    else:
        print('Loading field {} for all stations'.format(field_all))
        cwd = Path().cwd()
        df = pd.read_csv(cwd / 'israeli_gnss_coords.txt', header=0,
                         delim_whitespace=True)
        stations = df.index.tolist()
        da_list = []
        stations_to_put = []
        for sta in stations:
            ds = load_one_results_ds(sta, sample_rate, plot_fields=None)
            if ds is not None:
                da_list.append(ds[field_all])
                stations_to_put.append(sta)
            else:
                print('skipping station {}'.format(sta))
                continue
        ds = xr.concat(da_list, dim='station')
        ds['station'] = stations_to_put
        ds = ds.to_dataset(dim='station')
    return ds


def get_long_trends_from_gnss_station(station='tela', modelname='LR',
                                      plot=True):
    import xarray as xr
    import numpy as np
    from aux_gps import plot_tmseries_xarray
    # dont try anonther model than LR except for lower-sampled data
    ds = load_gipsyx_results(station, sample_rate=None, plot_fields=None)
    if ds is None:
        raise FileNotFoundError
    # first do altitude [m]:
    da_alt = ML_fit_model_to_tmseries(ds['alt'], modelname=modelname,
                                      plot=False)
    years = da_alt.attrs['total_years']
    meters_per_year = da_alt.attrs['slope_per_year']
    da_alt.attrs['trend>mm_per_year'] = 1000.0 * meters_per_year
    # now do lat[deg]:
    one_degree_at_eq = 111.32  # km
    lat0 = ds['lat'].dropna('time')[0].values.item()
    factor = np.cos(np.deg2rad(lat0)) * one_degree_at_eq
    da_lat = ML_fit_model_to_tmseries(ds['lat'], modelname=modelname,
                                      plot=False)
    degs_per_year = da_lat.attrs['slope_per_year']
    da_lat.attrs['trend>cm_per_year'] = factor * 1e5 * degs_per_year
    da_lon = ML_fit_model_to_tmseries(ds['lon'], modelname=modelname,
                                      plot=False)
    degs_per_year = da_lon.attrs['slope_per_year']
    da_lon.attrs['trend>cm_per_year'] = factor * 1e5 * degs_per_year
    rds = xr.Dataset()
    # the following attrs are being read to israeli_gnss_stations procedure
    # above and used in its dataframe, so don't touch these attrs:
    rds['alt'] = ds['alt']
    rds['lat'] = ds['lat']
    rds['lon'] = ds['lon']
    rds['alt_trend'] = da_alt
    rds['lat_trend'] = da_lat
    rds['lon_trend'] = da_lon
    rds.attrs['lat_trend>cm_per_year'] = rds['lat_trend'].attrs['trend>cm_per_year']
    rds.attrs['lon_trend>cm_per_year'] = rds['lon_trend'].attrs['trend>cm_per_year']
    rds.attrs['alt_trend>mm_per_year'] = rds['alt_trend'].attrs['trend>mm_per_year']
    rds.attrs['years'] = years
    rds.attrs['station'] = station
    rds.attrs['lat'] = ds['lat'].dropna('time')[0].values.item()
    rds.attrs['lon'] = ds['lon'].dropna('time')[0].values.item()
    rds.attrs['alt'] = ds['alt'].dropna('time')[0].values.item()
    if plot:
        plot_tmseries_xarray(rds)
    return rds


def ML_fit_model_to_tmseries(tms_da, modelname='LR', plot=True, verbose=False):
    """fit a single time-series data-array with ML models specified in
    ML_Switcher"""
    import numpy as np
    import xarray as xr
    import pandas as pd
    # find the time dim:
    time_dim = list(set(tms_da.dims))[0]
    # pick a model:
    ml = ML_Switcher()
    model = ml.pick_model(modelname)
    # dropna for time-series:
    tms_da_no_nan = tms_da.dropna(time_dim)
    # df = tms_da_no_nan.to_dataframe()
    # ind = df.index.factorize()[0].reshape(-1, 1)
    # ind_with_nan = tms_da.to_dataframe().index.factorize()[0].reshape(-1, 1)
    # change datetime units to days:
    jul_with_nan = pd.to_datetime(tms_da[time_dim].values).to_julian_date()
    jul_with_nan -= jul_with_nan[0]
    jul_no_nan = pd.to_datetime(tms_da_no_nan[time_dim].values).to_julian_date()
    jul_no_nan -= jul_no_nan[0]
    jul_with_nan = np.array(jul_with_nan).reshape(-1, 1)
    jul_no_nan = np.array(jul_no_nan).reshape(-1, 1)
    model.fit(jul_no_nan, tms_da_no_nan.values)
    new_y = model.predict(jul_with_nan).squeeze()
    new_da = xr.DataArray(new_y, dims=[time_dim])
    new_da[time_dim] = tms_da[time_dim]
    if hasattr(model, 'coef_'):
        new_da.attrs['slope_per_day'] = model.coef_[0]
        days = pd.to_timedelta(tms_da.time.max(
                ).values - tms_da.time.min().values, unit='D')
        years = days / np.timedelta64(1, 'Y')
        slope_per_year = model.coef_[0] * days.days / years
        new_da.attrs['slope_per_year'] = slope_per_year
        new_da.attrs['total_years'] = years
        if verbose:
            print('slope_per_day: {}'.format(model.coef_[0]))
            print('slope_per_year: {}'.format(slope_per_year))
    if hasattr(model, 'intercept_'):
        new_da.attrs['intercept'] = model.intercept_
        if verbose:
            print('intercept: {}'.format(model.intercept_))
    if plot:
        tms_da.plot.line(marker='.', linewidth=0., color='b')
        new_da.plot(color='r')
    return new_da


#def analyze_sounding_and_formulate(sound_path=sound_path,
#                                   model_names = ['TSEN', 'LR'],
#                                   res_save='LR'):
#    import xarray as xr
#    import numpy as np
#    import matplotlib.pyplot as plt
#    import seaborn as sns
#    from sklearn.metrics import mean_squared_error
#    sns.set_style('darkgrid')
#    # colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#    colors = ['red', 'green', 'magenta', 'cyan', 'orange', 'teal',
#              'gray', 'purple']
#    if res_save not in model_names:
#        raise KeyError('saved result should me in model names!')
#    if len(model_names) > len(colors):
#        raise ValueError('Cannot support more than {} models simultenously!'.format(len(colors)))
#    ml = ML_Switcher()
#    models = [ml.pick_model(x) for x in model_names]
#    # md = dict(zip(model_names, models))
#    # ds = xr.open_dataset(sound_path / 'bet_dagan_sounding_pw_Ts_Tk1.nc')
#    ds = xr.open_dataset(sound_path / 'bet_dagan_sounding_pw_Ts_Tk_with_clouds.nc')
#    ds = ds.reset_coords(drop=True)
#    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
#    fig.suptitle(
#        'Water vapor weighted mean atmospheric temperature vs. bet dagan sounding station surface temperature')
#    X = ds.ts.values.reshape(-1, 1)
#    y = ds.tm.values
#    [model.fit(X, y) for model in models]
#    predict = [model.predict(X) for model in models]
#    coefs = [model.coef_[0] for model in models]
#    inters = [model.intercept_ for model in models]
#    # [a, b] = np.polyfit(ds.ts.values, ds.tm.values, 1)
#    # sns.regplot(ds.ts.values, ds.tm.values, ax=axes[0])
#    df = ds.ts.dropna('time').to_dataframe()
#    df['tm'] = ds.tm.dropna('time')
#    df['clouds'] = ds.any_cld.dropna('time')
#    g = sns.scatterplot(data=df, x='ts', y='tm', hue='clouds', marker='.', s=100,
#                        ax=axes[0])
#    g.legend(loc='best')
#    # axes[0].scatter(x=ds.ts.values, y=ds.tm.values, marker='.', s=10)
#    # linex = np.array([ds.ts.min().item(), ds.ts.max().item()])
#    # liney = a * linex + b
#    # axes[0].plot(linex, liney, c='r')
#    # [(i, j) for i, j in enumerate(mylist)]
#    [axes[0].plot(X, newy, c=colors[i]) for i, newy in enumerate(predict)]
#    min_, max_ = axes[0].get_ylim()
#    pos = np.linspace(0.95, 0.6, 8)
#    [axes[0].text(0.01,
#                  pos[i],
#                  '{} a: {:.2f}, b: {:.2f}'.format(model_names[i], coefs[i],
#                                                   inters[i]),
#                  transform=axes[0].transAxes,
#                  color=colors[i],
#                  fontsize=12) for i in range(len(coefs))]
##    axes[0].text(0.01, 0.9, 'a_lr: {:.2f}, b_lr: {:.2f}'.format(lr.coef_[0],lr.intercept_),
##        transform=axes[0].transAxes, color='red', fontsize=12)
##    axes[0].text(0.01, 0.85, 'a_tsen: {:.2f}, b_tsen: {:.2f}'.format(tsen.coef_[0],tsen.intercept_),
##        transform=axes[0].transAxes, color='green', fontsize=12)
#    axes[0].text(0.1,
#                 0.8,
#                 'n={}'.format(len(ds.ts.values)),
#                 verticalalignment='top',
#                 horizontalalignment='center',
#                 transform=axes[0].transAxes,
#                 color='blue',
#                 fontsize=12)
#    axes[0].set_xlabel('Ts [K]')
#    axes[0].set_ylabel('Tm [K]')
#    resid = predict[0] - y  # ds.tm.values - ds.ts.values * a - b
#    sns.distplot(resid, bins=25, color='c', label='residuals', ax=axes[1])
#    rmean = np.mean(resid)
#    # rmse = np.sqrt(mean_squared_error(ds.tm.values, ds.ts.values * a + b))
#    rmse = np.sqrt(mean_squared_error(predict[0], y))
#    _, max_ = axes[1].get_ylim()
#    axes[1].text(rmean + rmean / 10, max_ - max_ / 10,
#                 'Mean: {:.2f}, RMSE: {:.2f}'.format(rmean, rmse))
#    axes[1].axvline(rmean, color='r', linestyle='dashed', linewidth=1)
#    axes[1].set_xlabel('Residuals [K]')
#    fig.tight_layout()
#    da_all = xr.DataArray(models, dims=['name'])
#    da_all['name'] = model_names
#    da_all.name = 'all_data_trained_models'
#    # plot of just hours:
#    h_order = ['noon', 'midnight']
#    trained = []
#    # result = np.empty((len(h_order), 2))
#    # residuals = []
#    # rmses = []
#    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(8, 6))
#    for i, hour in enumerate(h_order):
#        ts = ds.ts.where(ds.hour == hour).dropna('time')
#        tm = ds.tm.where(ds.hour == hour).dropna('time')
#        X = ts.values.reshape(-1, 1)
#        y = tm.values
#        cld = ds.any_cld.where(ds.hour == hour).dropna('time')
#        models = [ml.pick_model(x) for x in model_names]
#        [model.fit(X, y) for model in models]
#        predict = [model.predict(X) for model in models]
#        coefs = [model.coef_[0] for model in models]
#        inters = [model.intercept_ for model in models]
#        # [tmul, toff] = np.polyfit(x.values, y.values, 1)
#        # result[i, 0] = tmul
#        # result[i, 1] = toff
#        # new_tm = tmul * x.values + toff
#        # resid = new_tm - y.values
#        # rmses.append(np.sqrt(mean_squared_error(y.values, new_tm)))
#        # residuals.append(resid)
#        axes[i].text(0.15, 0.7, 'n={}'.format(ts.size),
#                     verticalalignment='top', horizontalalignment='center',
#                     transform=axes[i].transAxes, color='blue', fontsize=12)
#        df = ts.to_dataframe()
#        df['tm'] = tm
#        df['clouds'] = cld
#        g = sns.scatterplot(data=df, x='ts', y='tm', hue='clouds',
#                            marker='.', s=100, ax=axes[i])
#        g.legend(loc='upper right')
#        # axes[i, j].scatter(x=x.values, y=y.values, marker='.', s=10)
#        axes[i].set_title('hour:{}'.format(hour))
#        # linex = np.array([x.min().item(), x.max().item()])
#        # liney = tmul * linex + toff
#        # axes[i].plot(linex, liney, c='r')
#        [axes[i].plot(X, newy, c=colors[j]) for j, newy in enumerate(predict)]
#        axes[i].plot(ts.values, ts.values, c='k', alpha=0.2)
#        min_, max_ = axes[i].get_ylim()
#        [axes[i].text(0.01,
#                      pos[j],
#                      '{} a: {:.2f}, b: {:.2f}'.format(model_names[j],
#                                                       coefs[j],
#                                                       inters[j]),
#                      transform=axes[i].transAxes,
#                      color=colors[j],
#                      fontsize=12) for j in range(len(coefs))]
#        axes[i].set_xlabel('Ts [K]')
#        axes[i].set_ylabel('Tm [K]')
#        trained.append(models)
#    da_hour = xr.DataArray(trained, dims=['hour', 'name'])
#    da_hour['name'] = model_names
#    da_hour['hour'] = h_order
#    da_hour.name = 'hour_data_trained_models'
#    s_order = ['DJF', 'JJA', 'SON', 'MAM']
#    # plot of hours and seasons:
##    Tmul = []
##    Toff = []
#    trained = []
##    residuals = []
##    rmses = []
##    result = np.empty((len(h_order), len(s_order), 2))
#    fig, axes = plt.subplots(2, 4, sharey=True, sharex=True, figsize=(20, 15))
#    for i, hour in enumerate(h_order):
#        for j, season in enumerate(s_order):
#            ts = ds.ts.sel(time=ds['time.season'] == season).where(
#                ds.hour == hour).dropna('time')
#            tm = ds.tm.sel(time=ds['time.season'] == season).where(
#                ds.hour == hour).dropna('time')
#            cld = ds.any_cld.sel(time=ds['time.season'] == season).where(
#                ds.hour == hour).dropna('time')
#            X = ts.values.reshape(-1, 1)
#            y = tm.values
#            models = [ml.pick_model(x) for x in model_names]
#            [model.fit(X, y) for model in models]
#            predict = [model.predict(X) for model in models]
#            coefs = [model.coef_[0] for model in models]
#            inters = [model.intercept_ for model in models]
#            # [tmul, toff] = np.polyfit(x.values, y.values, 1)
##            result[i, j, 0] = tmul
##            result[i, j, 1] = toff
##            new_tm = tmul * x.values + toff
##            resid = new_tm - y.values
##            rmses.append(np.sqrt(mean_squared_error(y.values, new_tm)))
##            residuals.append(resid)
#            axes[i, j].text(0.15, 0.7, 'n={}'.format(ts.size),
#                            verticalalignment='top', horizontalalignment='center',
#                            transform=axes[i, j].transAxes, color='blue',
#                            fontsize=12)
#            df = ts.to_dataframe()
#            df['tm'] = tm
#            df['clouds'] = cld
#            g = sns.scatterplot(data=df, x='ts', y='tm', hue='clouds',
#                                marker='.', s=100, ax=axes[i, j])
#            g.legend(loc='upper right')
#            # axes[i, j].scatter(x=x.values, y=y.values, marker='.', s=10)
#            axes[i, j].set_title('season:{} ,hour:{}'.format(season, hour))
#            # linex = np.array([x.min().item(), x.max().item()])
#            # liney = tmul * linex + toff
#            # axes[i, j].plot(linex, liney, c='r')
#            [axes[i, j].plot(X, newy, c=colors[k]) for k, newy
#             in enumerate(predict)]
#            axes[i, j].plot(ts.values, ts.values, c='k', alpha=0.2)
#            min_, max_ = axes[i, j].get_ylim()
##            axes[i, j].text(0.015, 0.9, 'a: {:.2f}, b: {:.2f}'.format(
##                tmul, toff), transform=axes[i, j].transAxes, color='black', fontsize=12)
#            [axes[i, j].text(0.01,
#             pos[k],
#              '{} a: {:.2f}, b: {:.2f}'.format(model_names[k],
#                                               coefs[k],
#                                               inters[k]),
#              transform=axes[i, j].transAxes,
#              color=colors[k],
#              fontsize=12) for k in range(len(coefs))]
#            axes[i, j].set_xlabel('Ts [K]')
#            axes[i, j].set_ylabel('Tm [K]')
#            trained.append(models)
##            Tmul.append(tmul)
##            Toff.append(toff)
#    da_hour_season = xr.DataArray(trained, dims=['hour', 'season', 'name'])
#    da_hour_season['name'] = model_names
#    da_hour_season['hour'] = h_order
#    da_hour_season['season'] = s_order
#    da_hour_season.name = 'hour_season_data_trained_models'
##    cnt = 0
##    fig, axes = plt.subplots(2, 4, sharey=True, sharex=True, figsize=(20, 15))
##    for i, hour in enumerate(h_order):
##        for j, season in enumerate(s_order):
##            sns.distplot(residuals[cnt], bins=25, color='c',
##                         label='residuals', ax=axes[i, j])
##            rmean = np.mean(residuals[cnt])
##            _, max_ = axes[i, j].get_ylim()
##            axes[i, j].text(rmean + rmean / 10, max_ - max_ / 10,
##                            'Mean: {:.2f}, RMSE: {:.2f}'.format(rmean,
##                                                                rmses[cnt]))
##            axes[i, j].axvline(rmean, color='r', linestyle='dashed',
##                               linewidth=1)
##            axes[i, j].set_xlabel('Residuals [K]')
##            axes[i, j].set_title('season:{} ,hour:{}'.format(season, hour))
##            cnt += 1
##    fig.tight_layout()
##    results = xr.DataArray(result, dims=['hour', 'season', 'parameter'])
##    results['hour'] = h_order
##    results['season'] = s_order
##    results['parameter'] = ['slope', 'intercept']
#    # results.attrs['all_data_slope'] = a
#    # results.attrs['all_data_intercept'] = b
#    return 


class ML_Switcher(object):
    def pick_model(self, model_name):
        """Dispatch method"""
        method_name = str(model_name)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid ML Model")
        # Call the method as we return it
        return method()

    def LR(self):
        from sklearn.linear_model import LinearRegression
        return LinearRegression(n_jobs=-1, copy_X=True)

    def GPSR(self):
        from gplearn.genetic import SymbolicRegressor
        return SymbolicRegressor(random_state=42, n_jobs=1, metric='mse')

    def TSEN(self):
        from sklearn.linear_model import TheilSenRegressor
        return TheilSenRegressor(random_state=42)

    def MTLASSOCV(self):
        from sklearn.linear_model import MultiTaskLassoCV
        import numpy as np
        return MultiTaskLassoCV(random_state=42, cv=10, n_jobs=-1,
                                alphas=np.logspace(-5, 2, 400))

    def MTLASSO(self):
        from sklearn.linear_model import MultiTaskLasso
        return MultiTaskLasso()

    def KRR(self):
        from sklearn.kernel_ridge import KernelRidge
        return KernelRidge(kernel='poly', degree=2)

    def GPR(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        return GaussianProcessRegressor(random_state=42)

    def MTENETCV(self):
        import numpy as np
        from sklearn.linear_model import MultiTaskElasticNetCV
        return MultiTaskElasticNetCV(random_state=42, cv=10, n_jobs=-1,
                                alphas=np.logspace(-5, 2, 400))