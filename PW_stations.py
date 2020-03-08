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
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy import stats
hydro_path = work_yuval / 'hydro'
garner_path = work_yuval / 'garner'
ims_path = work_yuval / 'IMS_T'
gis_path = work_yuval / 'gis'
sound_path = work_yuval / 'sounding'
phys_soundings = sound_path / 'bet_dagan_phys_sounding_2007-2019.nc'
tela_zwd = work_yuval / 'gipsyx_results/tela_newocean/TELA_PPP_1996-2019.nc'
jslm_zwd = work_yuval / 'gipsyx_results/jslm_newocean/JSLM_PPP_2001-2019.nc'
alon_zwd = work_yuval / 'gipsyx_results/alon_newocean/ALON_PPP_2005-2019.nc'
tela_zwd_aligned = work_yuval / 'tela_zwd_aligned_with_physical_bet_dagan.nc'
alon_zwd_aligned = work_yuval / 'ALON_zwd_aligned_with_physical_bet_dagan.nc'
jslm_zwd_aligned = work_yuval / 'JSLM_zwd_aligned_with_physical_bet_dagan.nc'
tela_ims = ims_path / '10mins/TEL-AVIV-COAST_178_TD_10mins_filled.nc'
alon_ims = ims_path / '10mins/ASHQELON-PORT_208_TD_10mins_filled.nc'
jslm_ims = ims_path / '10mins/JERUSALEM-CENTRE_23_TD_10mins_filled.nc'
station_on_geo = geo_path / 'Work_Files/PW_yuval/GNSS_stations'
era5_path = work_yuval / 'ERA5'
PW_stations_path = work_yuval / '1minute'
stations = pd.read_csv('All_gps_stations.txt', header=0, delim_whitespace=True,
                       index_col='name')
logs_path = geo_path / 'Python_Projects/PW_from_GPS/log_files'
GNSS = work_yuval / 'GNSS_stations'
cwd = Path().cwd()
gnss_sound_stations_dict = {'acor': '08001', 'mall': '08302'}

# TODO: do ZWD selection from post-proccesed GipsyX results
# TODO: re-do the ts-tm connection for the 2013-09 to 2019 radiosonde
# TODO: then assemble PW for all the stations.
# TODO: somwhow check if in winter the radiosonde drifts eastwards and dries out
class LinearRegression_with_stats(LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, *args, **kwargs):
        # if not "fit_intercept" in kwargs:
        #     kwargs['fit_intercept'] = False
        super().__init__(*args,**kwargs)

    def fit(self, X, y=None, verbose=True, **fit_params):
        from scipy import linalg
        """ A wrapper around the fitting function.
        Improved: adds the X_ and y_ and results_ attrs to class.
        Parameters
        ----------
        X : xarray DataArray, Dataset other other array-like
            The training input samples.

        y : xarray DataArray, Dataset other other array-like
            The target values.

        Returns
        -------
        Returns self.
        """
        self = super().fit(X, y, **fit_params)
        n, k = X.shape
        yHat = np.matrix(self.predict(X)).T

        # Change X and Y into numpy matricies. x also has a column of ones added to it.
        x = np.hstack((np.ones((n,1)),np.matrix(X)))
        y = np.matrix(y).T

        # Degrees of freedom.
        df = float(n-k-1)

        # Sample variance.     
        sse = np.sum(np.square(yHat - y),axis=0)
        self.sampleVariance = sse/df

        # Sample variance for x.
        self.sampleVarianceX = x.T*x

        # Covariance Matrix = [(s^2)(X'X)^-1]^0.5. (sqrtm = matrix square root.  ugly)
        self.covarianceMatrix = linalg.sqrtm(self.sampleVariance[0,0]*self.sampleVarianceX.I)

        # Standard erros for the difference coefficients: the diagonal elements of the covariance matrix.
        self.se = self.covarianceMatrix.diagonal()[1:]

        # T statistic for each beta.
        self.betasTStat = np.zeros(len(self.se))
        for i in range(len(self.se)):
            self.betasTStat[i] = self.coef_[i]/self.se[i]

        # P-value for each beta. This is a two sided t-test, since the betas can be 
        # positive or negative.
        self.betasPValue = 1 - stats.t.cdf(abs(self.betasTStat),df)
        return self


def PW_trend_analysis(path=work_yuval, anom=False, station='tela'):
    import xarray as xr
    pw = xr.open_dataset(path / 'GNSS_daily_PW.nc')[station]
    if anom:
        pw = pw.groupby('time.month') - pw.groupby('time.month').mean('time')
    pw_lr = ML_fit_model_to_tmseries(pw, modelname='LR', plot=False, verbose=True)
    pw_tsen = ML_fit_model_to_tmseries(pw, modelname='TSEN', plot=False, verbose=True)
    return pw_tsen


def produce_gnss_pw_from_era5(era5_path=era5_path,
                              glob_str='era5_PW_israel*.nc',
                              pw_path=work_yuval, savepath=None):
    from aux_gps import path_glob
    import xarray as xr
    filepath = path_glob(era5_path, glob_str)[0]
    era5_pw = xr.open_dataarray(filepath)
    gps = produce_geo_gps_stations(gis_path, plot=False)
    era5_pw_list = []
    for station in gps.index:
        slat = gps.loc[station, 'lat']
        slon = gps.loc[station, 'lon']
        da = era5_pw.sel(lat=slat, lon=slon, method='nearest')
        da.name = station
        da.attrs['era5_lat'] = da.lat.values.item()
        da.attrs['era5_lon'] = da.lon.values.item()
        da = da.reset_coords(drop=True)
        era5_pw_list.append(da)
    ds = xr.merge(era5_pw_list)
    if savepath is not None:
        filename = 'GNSS_era5_hourly_PW.nc'
        print('saving {} to {}'.format(filename, savepath))
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds}
        ds.to_netcdf(savepath / filename, 'w', encoding=encoding)
    print('Done!')
    return ds


def plug_in_approx_loc_gnss_stations(log_path=logs_path, file_path=cwd):
    from aux_gps import path_glob
    import pandas as pd
    def plug_loc_to_log_file(logfile, loc):

        def replace_field(content_list, string, replacment):
            pos = [(i, x) for i, x in enumerate(content_list)
                   if string in x][0][0]
            con = content_list[pos].split(':')
            con[-1] = ' {}'.format(replacment)
            con = ':'.join(con)
            content_list[pos] = con
            return content_list

        with open(logfile) as f:
            content = f.read().splitlines()
            repl = [
                    'X coordinate (m)',
                    'Y coordinate (m)',
                    'Z coordinate (m)',
                    'Latitude (deg)',
                    'Longitude (deg)',
                    'Elevation (m)']
            location = [loc['X'], loc['Y'], loc['Z'], '+' +
                        str(loc['lat']), '+' + str(loc['lon']), loc['alt']]
        for rep, loca in list(zip(repl, location)):
            try:
                content = replace_field(content, rep, loca)
            except IndexError:
                print('did not found {} field...'.format(rep))
                pass
        with open(logfile, 'w') as f:
            for item in content:
                f.write('{}\n'.format(item))
        print('writing {}'.format(logfile))
        return

    # load gnss accurate loc:
    acc_loc_df = pd.read_csv(file_path / 'israeli_gnss_coords.txt',
                             delim_whitespace=True)
    log_files = path_glob(log_path, '*updated_by_shlomi*.log')
    for logfile in log_files:
        st_log = logfile.as_posix().split('/')[-1].split('_')[0]
        try:
            loc = acc_loc_df.loc[st_log, :]
        except KeyError:
            print('station {} not found in accurate location df, skipping'.format(st_log))
            continue
        plug_loc_to_log_file(logfile, loc)
    print('Done!')
    return


def build_df_lat_lon_alt_gnss_stations(gnss_path=GNSS, savepath=None):
    from aux_gps import path_glob
    import pandas as pd
    import pyproj
    from pathlib import Path
    stations_in_gnss = [x.as_posix().split('/')[-1]
                        for x in path_glob(GNSS, '*')]
    dss = [
        load_gipsyx_results(
            x,
            sample_rate='MS',
            plot_fields=None) for x in stations_in_gnss]
    # stations_not_found = [x for x in dss if isinstance(x, str)]
    # [stations_in_gnss.remove(x) for x in stations_in_gnss if x is None]
    dss = [x for x in dss if not isinstance(x, str)]
    dss = [x for x in dss if x is not None]
    lats = [x.dropna('time').lat[0].values.item() for x in dss]
    lons = [x.dropna('time').lon[0].values.item() for x in dss]
    alts = [x.dropna('time').alt[0].values.item() for x in dss]
    df = pd.DataFrame(lats)
    df.index = [x.attrs['station'].lower() for x in dss]
    df['lon'] = lons
    df['alt'] = alts
    df.columns = ['lat', 'lon', 'alt']
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    X, Y, Z = pyproj.transform(lla, ecef, df['lon'].values, df['lat'].values,
                               df['alt'].values, radians=False)
    df['X'] = X
    df['Y'] = Y
    df['Z'] = Z
    # read station names from log files:
    stations_approx = pd.read_fwf(Path().cwd()/'stations_approx_loc.txt',
                              delim_whitespace=False, skiprows=1, header=None)
    stations_approx.columns=['index','X','Y','Z','name', 'extra']
    stations_approx['name'] = stations_approx['name'].fillna('') +' ' + stations_approx['extra'].fillna('')
    stations_approx.drop('extra', axis=1, inplace=True)
    stations_approx = stations_approx.set_index('index')
    df['name'] = stations_approx['name']
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


def compare_gipsyx_soundings(sound_path=sound_path, gps_station='acor',
                             times=['1996', '2019'], var='pw'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import matplotlib.dates as mdates
    import xarray as xr
    from aux_gps import path_glob
    # sns.set_style('whitegrid')
    # ds = mean_zwd_over_sound_time(
    #     physical_file, ims_path=ims_path, gps_station='tela',
    #     times=times)
    sound_station = gnss_sound_stations_dict.get(gps_station)
    gnss = load_gipsyx_results(plot_fields=None, station=gps_station)
    sound_file = path_glob(sound_path, 'station_{}_soundings_ts_tm_tpw*.nc'.format(sound_station))[0]
    sds = xr.open_dataset(sound_file)
    time_dim = list(set(sds.dims))[0]
    sds = sds.rename({time_dim: 'time'})
    sds[gps_station] = gnss.WetZ
    if var == 'zwd':
        k = kappa(sds['Tm'], Tm_input=True)
        sds['sound'] = sds.Tpw / k
        sds[gps_station] = gnss.WetZ
    elif var == 'pw':
        linear_model = ml_models_T_from_sounding(times=times,
                                                 station=sound_station,
                                                 plot=False, models=['LR'])
        linear_model = linear_model.sel(name='LR').values.item()
        k = kappa_ml(sds['Ts'] - 273.15, model=linear_model, no_error=True)
        sds[gps_station] = sds[gps_station] * k
        sds['sound'] = sds.Tpw
    sds = sds.dropna('time')
    sds = sds.sel(time=slice(*times))
    df = sds[['sound', gps_station]].to_dataframe()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    [x.set_xlim([pd.to_datetime(times[0]), pd.to_datetime(times[1])])
     for x in axes]
    df.columns = ['{} soundings'.format(sound_station), '{} GNSS station'.format(gps_station)]
    sns.scatterplot(
        data=df,
        s=20,
        ax=axes[0],
        style='x',
        linewidth=0,
        alpha=0.8)
    # axes[0].legend(['Bet_Dagan soundings', 'TELA GPS station'])
    df_r = df.iloc[:, 0] - df.iloc[:, 1]
    df_r.columns = ['Residual distribution']
    sns.scatterplot(
        data=df_r,
        color='k',
        s=20,
        ax=axes[1],
        linewidth=0,
        alpha=0.5)
    axes[0].grid(b=True, which='major')
    axes[1].grid(b=True, which='major')
    if var == 'zwd':
        axes[0].set_ylabel('Zenith Wet Delay [cm]')
        axes[1].set_ylabel('Residuals [cm]')
    elif var == 'pw':
        axes[0].set_ylabel('Precipitable Water [mm]')
        axes[1].set_ylabel('Residuals [mm]')
    # sonde_change_x = pd.to_datetime('2013-08-20')
    # axes[1].axvline(sonde_change_x, color='red')
    # axes[1].annotate(
    #     'changed sonde type from VIZ MK-II to PTU GPS',
    #     (mdates.date2num(sonde_change_x),
    #      10),
    #     xytext=(
    #         15,
    #         15),
    #     textcoords='offset points',
    #     arrowprops=dict(
    #         arrowstyle='fancy',
    #         color='red'),
    #     color='red')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.01)

    return sds

def produce_zwd_from_sounding_and_compare_to_gps(phys_sound_file=phys_soundings,
                                                 zwd_file=tela_zwd_aligned,
                                                 tm=None, plot=True):
    """compare zwd from any gps station (that first has to be aligned to
    Bet_dagan station) to that of Bet-Dagan radiosonde station using tm from
    either bet dagan or user inserted. by default, using zwd from pw by
    inversing Bevis 1992 et al. formula"""
    import xarray as xr
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.dates as mdates
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
    gps.name = ['WetZ_from_TELA']
    if plot:
        # sns.set_style("whitegrid")
        df = radio.to_dataframe()
        df[gps.name] = gps.to_dataframe()
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        [x.set_xlim([pd.to_datetime('2007-12-31'), pd.to_datetime('2019')]) for x in axes]
        # radio.plot.line(marker='.', linewidth=0., ax=axes[0])
        sns.scatterplot(data=df, s=20, ax=axes[0], style='x', linewidth=0, alpha=0.8)
        # gps.plot.line(marker='.', linewidth=0., ax=axes[0])
        #sns.scatterplot(data=df, y= 'tela_WetZ', s=10, ax=axes[0])
        # axes[0].legend('radiosonde', '{}_gnss_site'.format(station))
        df_r = df.iloc[:, 0] - df.iloc[:, 1]
        df_r.columns = ['Residuals']
        # (radio - gps).plot.line(marker='.', linewidth=0., ax=axes[1])
        sns.scatterplot(data=df_r, color = 'k', s=20, ax=axes[1], linewidth=0, alpha=0.5)
        axes[0].grid(b=True, which='major')
        axes[1].grid(b=True, which='major')
        axes[0].set_ylabel('Zenith Wet Delay [cm]')
        axes[1].set_ylabel('Residuals [cm]')
        axes[0].set_title('Zenith wet delay from Bet-Dagan radiosonde station and TELA GNSS satation')
        sonde_change_x = pd.to_datetime('2013-08-20')
        axes[1].axvline(sonde_change_x, color='red')
        axes[1].annotate('changed sonde type from VIZ MK-II to PTU GPS', (mdates.date2num(sonde_change_x), 15), xytext=(15, 15), 
            textcoords='offset points', arrowprops=dict(arrowstyle='fancy', color='red'), color='red')
        # axes[1].set_aspect(3)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
#        plt.figure()
#        (radio - gps).plot.hist(bins=100)
    return zwd_and_tpw


def fit_ts_tm_produce_ipw_and_compare_TELA(phys_sound_file=phys_soundings,
                                           zwd_file=tela_zwd_aligned,
                                           IMS_file=None,
                                           sound_path=sound_path,
                                           categories=None, model='LR',
                                           times=['2005', '2019'],
                                           **compare_kwargs):
    """categories can be :'bevis', None, 'season' and/or 'hour'. None means
    whole dataset ts-tm.
    models can be 'LR' or 'TSEN'. compare_kwargs is for
    compare_to_sounding2 i.e., times, season, hour, title"""
    import xarray as xr
    print(compare_kwargs)
    if categories == 'bevis':
        results = None
        compare_kwargs.update({'title': None})
    else:
        results = ml_models_T_from_sounding(sound_path, categories, model,
                                            physical_file=phys_sound_file,
                                            times=times)
    if categories is None:
        compare_kwargs.update({'title': 'whole'})
    elif categories is not None and categories != 'bevis':
        if isinstance(categories, str):
            compare_kwargs.update({'title': [categories][0]})
        elif isinstance(categories, list):
            compare_kwargs.update({'title': 'hour_season'})
    zwd_and_tpw = xr.open_dataset(zwd_file)
    if times is not None:
        zwd_and_tpw = zwd_and_tpw.sel(time=slice(*times))
    station = zwd_file.as_posix().split('/')[-1].split('_')[0]
    tpw = zwd_and_tpw['Tpw']
    if IMS_file is None:
        T = xr.open_dataset(ims_path / 'GNSS_5mins_TD_ALL_1996_2019.nc')
        T = T['tela']
    else:
        # load the 10 mins temperature data from IMS:
        T = xr.open_dataset(IMS_file)
        T = T.to_array(name='t').squeeze(drop=True)
    zwd_and_tpw = zwd_and_tpw.rename({'{}_WetZ'.format(
            station): 'WetZ', '{}_WetZ_error'.format(station): 'WetZ_error'})
    zwd = zwd_and_tpw[['WetZ', 'WetZ_error']]
    zwd.attrs['station'] = station
    pw_gps = produce_single_station_IPW(zwd, T, mda=results, model_name=model)
    compare_to_sounding2(pw_gps['PW'], tpw, station=station, **compare_kwargs)
    return pw_gps, tpw


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


def mean_zwd_over_sound_time(
        phys_sound_file=phys_soundings,
        ims_path=ims_path,
        gps_station='tela',
        times=['2007', '2019']):
    import xarray as xr
    from aux_gps import get_unique_index
    from aux_gps import keep_iqr
    from aux_gps import multi_time_coord_slice
    """mean the WetZ over the gps station soundings datetimes to get a more
        accurate realistic measurement comparison to soundings"""
    phys = xr.open_dataset(phys_sound_file)
    # clean and merge:
    p_list = [get_unique_index(phys[x], 'sound_time')
              for x in ['Ts', 'Tm', 'Tpw', 'dt_range']]
    phys_ds = xr.merge(p_list)
    phys_ds = keep_iqr(phys_ds, 'sound_time', k=2.0)
    phys_ds = phys_ds.rename({'Ts': 'ts', 'Tm': 'tm'})
    # load the zenith wet daley for GPS (e.g.,TELA) station:
    zwd = load_gipsyx_results(station=gps_station, plot_fields=None)
    # zwd = xr.open_dataset(zwd_file)
    zwd = zwd[['WetZ', 'WetZ_error']]
    min_time = phys_ds['dt_range'].sel(bnd='Min').values
    max_time = phys_ds['dt_range'].sel(bnd='Max').values
    min_time = zwd.time.sel(time=min_time, method='nearest').values
    max_time = zwd.time.sel(time=max_time, method='nearest').values
    freq = pd.infer_freq(zwd.time.values)
    da_group = multi_time_coord_slice(min_time, max_time, freq=freq,
                                      time_dim='time', name='sound_time')
    zwd[da_group.name] = da_group
    ds = zwd.WetZ.groupby(zwd[da_group.name]).mean(
        'time').to_dataset(name='{}_WetZ'.format(gps_station))
    ds['{}_WetZ_std'.format(gps_station)] = zwd.WetZ.groupby(
        zwd[da_group.name]).std('time')
    ds['{}_WetZ_error'.format(gps_station)] = zwd.WetZ_error.groupby(
        zwd[da_group.name]).mean('time')
    ds['sound_time'] = phys_ds.sound_time
    ds['tpw_bet_dagan'] = phys_ds['Tpw']
    wetz = ds['{}_WetZ'.format(gps_station)]
    wetz_error = ds['{}_WetZ_error'.format(gps_station)]
    # do the same for surface temperature:
    td = xr.open_dataset(ims_path / 'GNSS_5mins_TD_ALL_1996_2019.nc')
    td = td[gps_station].to_dataset(name='ts')
    min_time = phys_ds['dt_range'].sel(bnd='Min').values
    max_time = phys_ds['dt_range'].sel(bnd='Max').values
    min_time = td.time.sel(time=min_time, method='nearest').values
    max_time = td.time.sel(time=max_time, method='nearest').values
    freq = pd.infer_freq(td.time.values)
    da_group = multi_time_coord_slice(min_time, max_time, freq=freq,
                                      time_dim='time', name='sound_time')
    td[da_group.name] = da_group
    ts_sound = td.ts.groupby(td[da_group.name]).mean('time')
    ts_sound['sound_time'] = phys_ds.sound_time
    ds['{}_ts'.format(gps_station)] = ts_sound
    # select a model:
    mda = ml_models_T_from_sounding(categories=None, models=['LR'],
                                    physical_file=phys_sound_file, plot=False,
                                    times=times)
    model = mda.sel(name='LR').values.item()
    # compute the kappa function and multiply by ZWD to get PW(+error):
    k, dk = kappa_ml(ts_sound, model=model, verbose=True)
    ds['{}_pw'.format(gps_station)] = k * wetz
    ds['{}_pw_error'.format(gps_station)] = np.sqrt(
        wetz_error**2.0 + dk**2.0)
    # divide by kappa calculated from bet_dagan ts to get bet_dagan zwd:
    k = kappa(phys_ds['tm'], Tm_input=True)
    ds['zwd_bet_dagan'] = ds['tpw_bet_dagan'] / k
    return ds


#def align_physical_bet_dagan_soundings_pw_to_gps_station_zwd(
#        phys_sound_file, ims_path=ims_path, gps_station='tela',
#        savepath=work_yuval, model=None):
#    """compare the IPW of the physical soundings of bet dagan station to
#    the any gps station - using IMS temperature of that gps station"""
#    from aux_gps import get_unique_index
#    from aux_gps import keep_iqr
#    from aux_gps import dim_intersection
#    import xarray as xr
#    import numpy as np
#    filename = '{}_zwd_aligned_with_physical_bet_dagan.nc'.format(gps_station)
#    if not (savepath / filename).is_file():
#        print('saving {} to {}'.format(filename, savepath))
#        # first load physical bet_dagan Tpw, Ts, Tm and dt_range:
#        phys = xr.open_dataset(phys_sound_file)
#        # clean and merge:
#        p_list = [get_unique_index(phys[x], 'sound_time')
#                  for x in ['Ts', 'Tm', 'Tpw', 'dt_range']]
#        phys_ds = xr.merge(p_list)
#        phys_ds = keep_iqr(phys_ds, 'sound_time', k=2.0)
#        phys_ds = phys_ds.rename({'Ts': 'ts', 'Tm': 'tm'})
#        # load the zenith wet daley for GPS (e.g.,TELA) station:
#        zwd = load_gipsyx_results(station=gps_station, plot_fields=None)
#        # zwd = xr.open_dataset(zwd_file)
#        zwd = zwd[['WetZ', 'WetZ_error']]
#        # loop over dt_range and average the results on PW:
#        wz_list = []
#        wz_std = []
#        wz_error_list = []
#        for i in range(len(phys_ds['dt_range'].sound_time)):
#            min_time = phys_ds['dt_range'].isel(sound_time=i).sel(bnd='Min').values
#            max_time = phys_ds['dt_range'].isel(sound_time=i).sel(bnd='Max').values
#            wetz = zwd['WetZ'].sel(time=slice(min_time, max_time)).mean('time')
#            wetz_std = zwd['WetZ'].sel(time=slice(min_time, max_time)).std('time')
#            wetz_error = zwd['WetZ_error'].sel(time=slice(min_time, max_time)).mean('time')
#            wz_std.append(wetz_std)
#            wz_list.append(wetz)
#            wz_error_list.append(wetz_error)
#        wetz_gps = xr.DataArray(wz_list, dims='sound_time')
#        wetz_gps.name = '{}_WetZ'.format(gps_station)
#        wetz_gps_error = xr.DataArray(wz_error_list, dims='sound_time')
#        wetz_gps_error.name = '{}_WetZ_error'.format(gps_station)
#        wetz_gps_std = xr.DataArray(wz_list, dims='sound_time')
#        wetz_gps_std.name = '{}_WetZ_std'.format(gps_station)
#        wetz_gps['sound_time'] = phys_ds['sound_time']
#        wetz_gps_error['sound_time'] = phys_ds['sound_time']
#        new_time = dim_intersection([wetz_gps, phys_ds['Tpw']], 'sound_time')
#        wetz_gps = wetz_gps.sel(sound_time=new_time)
#        tpw_bet_dagan = phys_ds.Tpw.sel(sound_time=new_time)
#        zwd_and_tpw = xr.merge([wetz_gps, wetz_gps_error, wetz_gps_std,
#                                tpw_bet_dagan])
#        zwd_and_tpw = zwd_and_tpw.rename({'sound_time': 'time'})
#        comp = dict(zlib=True, complevel=9)  # best compression
#        encoding = {var: comp for var in zwd_and_tpw.data_vars}
#        zwd_and_tpw.to_netcdf(savepath / filename, 'w', encoding=encoding)
#        print('Done!')
#        return
#    else:
#        print('found file!')
#        zwd_and_tpw = xr.open_dataset(savepath / filename)
#        wetz = zwd_and_tpw['{}_WetZ'.format(gps_station)]
#        wetz_error = zwd_and_tpw['{}_WetZ_error'.format(gps_station)]
#        # load the 10 mins temperature data from IMS:
#        td = xr.open_dataset(ims_path/'GNSS_5mins_TD_ALL_1996_2019.nc')
#        td = td[gps_station]
#        td.name = 'Ts'
#        # tela_T = tela_T.resample(time='5min').ffill()
#        # compute the kappa function and multiply by ZWD to get PW(+error):
#        k, dk = kappa_ml(td, model=model, verbose=True)
#        kappa = k.to_dataset(name='{}_kappa'.format(gps_station))
#        kappa['{}_kappa_error'.format(gps_station)] = dk
#        PW = (
#            kappa['{}_kappa'.format(gps_station)] *
#            wetz).to_dataset(
#            name='{}_PW'.format(gps_station)).squeeze(
#                drop=True)
#        PW['{}_PW_error'.format(gps_station)] = np.sqrt(
#            wetz_error**2.0 +
#            kappa['{}_kappa_error'.format(gps_station)]**2.0)
#        PW['TPW_bet_dagan'] = zwd_and_tpw['Tpw']
#        PW = PW.dropna('time')
#    return PW


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
        print('reading station {} lLRog file'.format(station))
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
        text = [x for x in content if 'Site Name' in x][0]
        name = text.split(':')[-1]
        record[station] = pos_list
        pos_list.append(name)
    df = pd.DataFrame.from_dict(record, orient='index')
    posnames.append('name')
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
                                               figsize=(20, 10),
                                               cmap='seismic')
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
#    X = [x for x in allLines if 'XLR coordinate' in x][0].split()[-1]
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


def produce_pw_statistics(path=work_yuval, thresh=None):
    import xarray as xr
    from scipy.stats import kurtosis
    from scipy.stats import skew
    import pandas as pd
    pw = xr.load_dataset(path / 'GNSS_PW_{:.0f}.nc'.format(thresh))
    pd.options.display.float_format = '{:.1f}'.format
    mean = pw.mean('time').reset_coords().to_array(
        'index').to_dataframe('Mean')
    std = pw.std('time').reset_coords().to_array('index').to_dataframe('SD')
    median = pw.median('time').reset_coords().to_array(
        'index').to_dataframe('Median')
    q5 = pw.quantile(0.05, 'time').reset_coords(drop=True).to_array(
        'index').to_dataframe('5th')
    q95 = pw.quantile(0.95, 'time').reset_coords(drop=True).to_array(
        'index').to_dataframe('95th')
    maximum = pw.max('time').reset_coords().to_array(
        'index').to_dataframe('Maximum')
    minimum = pw.min('time').reset_coords().to_array(
        'index').to_dataframe('Minimum')
    sk = pw.map(skew, nan_policy='omit').to_array(
        'index').to_dataframe('Skewness')
    kurt = pw.map(kurtosis, nan_policy='omit').to_array(
        'index').to_dataframe('Kurtosis')
    df = pd.concat([mean, std, median, q5, q95,
                    maximum, minimum, sk, kurt], axis=1)
    cols=[]
    cols.append('Site ID')
    cols += [x for x in df.columns]
    df['Site ID'] = df.index.str.upper()
    df = df[cols]
    df.index.name = ''
    return df


def produce_geo_gnss_solved_stations(path=gis_path,
                                     file='israeli_gnss_coords.txt',
                                     plot=True):
    import geopandas as gpd
    import pandas as pd
    from pathlib import Path
    cwd = Path().cwd()
    df = pd.read_csv(cwd / file, delim_whitespace=True)
    df = df[['lat', 'lon', 'alt', 'name']]
    isr = gpd.read_file(path / 'Israel_and_Yosh.shp')
    isr.crs = {'init': 'epsg:4326'}
    stations = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon,
                                                                df.lat),
                                crs=isr.crs)
    if plot:
        ax = isr.plot()
        stations.plot(ax=ax, column='alt', cmap='Greens',
                      edgecolor='black', legend=True)
        for x, y, label in zip(stations.lon, stations.lat,
                               stations.index):
            ax.annotate(label, xy=(x, y), xytext=(3, 3),
                        textcoords="offset points")
    return stations


def produce_geo_gps_stations(path=gis_path, file='All_gps_stations.txt',
                             plot=True):
    import geopandas as gpd
    import xarray as xr
    from pathlib import Path
    from aux_gps import get_latlonalt_error_from_geocent_error
    stations_df = pd.read_csv(file, index_col='name',
                              delim_whitespace=True)
    isr_dem = xr.open_rasterio(path / 'israel_dem.tif')
    alt_list = []
    for index, row in stations_df.iterrows():
        lat = row['lat']
        lon = row['lon']
        alt = isr_dem.sel(band=1, x=lon, y=lat, method='nearest').values.item()
        alt_list.append(float(alt))
    stations_df['alt_dem'] = alt_list
    isr = gpd.read_file(path / 'israel_demog2012.shp')
    isr.crs = {'init': 'epsg:4326'}
    stations = gpd.GeoDataFrame(stations_df,
                                geometry=gpd.points_from_xy(stations_df.lon,
                                                            stations_df.lat),
                                crs=isr.crs)
    stations_isr = gpd.sjoin(stations, isr, op='within')
    stations_approx = pd.read_csv(Path().cwd()/'stations_approx_loc.txt',
                                  delim_whitespace=True)
    lon, lat, alt = get_latlonalt_error_from_geocent_error(
            stations_approx['X'].values, stations_approx['Y'].values,
            stations_approx['Z'].values)
    stations_approx.columns = ['approx_X', 'approx_Y', 'approx_Z']
    stations_approx['approx_lat'] = lat
    stations_approx['approx_lon'] = lon
    stations_approx['approx_alt'] = alt
    stations_isr_df = pd.DataFrame(stations_isr.drop(columns=['geometry',
                                                              'index_right']))
    compare_df = stations_isr_df.join(stations_approx)
    alt_list = []
    for index, row in compare_df.iterrows():
        lat = row['approx_lat']
        lon = row['approx_lon']
        alt = isr_dem.sel(band=1, x=lon, y=lat, method='nearest').values.item()
        alt_list.append(float(alt))
    compare_df['approx_alt_dem'] = alt_list
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


def produce_geo_df(gis_path=gis_path, plot=True):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from ims_procedures import read_ims_metadata_from_files
    print('getting IMS temperature stations metadata...')
    ims = read_ims_metadata_from_files(path=gis_path, freq='10mins')
    isr_with_yosh = gpd.read_file(gis_path / 'Israel_demog_yosh.shp')
    isr_with_yosh.crs = {'init': 'epsg:4326'}
    geo_ims = gpd.GeoDataFrame(ims, geometry=gpd.points_from_xy(ims.lon,
                                                                ims.lat),
                               crs=isr_with_yosh.crs)
    print('getting GPS stations ZWD from garner...')
    gps = produce_geo_gps_stations(gis_path, plot=False)
#    print('combining temperature and GPS stations into one dataframe...')
#    geo_df = get_minimum_distance(ims, gps, gis_path, plot=False)
    print('Done!')
    if plot:

        ax = isr_with_yosh.plot(figsize=(10, 8))
        geo_ims.plot(ax=ax, color='red', edgecolor='black', legend=True)
        gps.plot(ax=ax, color='green', edgecolor='black', legend=True)
        plt.legend(['IMS_stations', 'GNSS stations'])
#        for x, y, label in zip(gps.lon, gps.lat,
#                               gps.index):
#            ax.annotate(label, xy=(x, y), xytext=(3, 3),
#                        textcoords="offset points")
        plt.tight_layout()
    return ims, gps


def save_GNSS_PW_israeli_stations(savepath=work_yuval, phys=phys_soundings):
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
        pw = produce_GNSS_station_PW(sta, None, plot=False, phys=phys)
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
            pw = produce_GNSS_station_PW(sta, skey, plot=False, phys=phys)
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


def align_group_pw_and_T_to_long_term_monthly_means_and_save(
        load_path=work_yuval,
        ims_path=ims_path,
        thresh=50,
        grp='month',
        savepath=work_yuval):
    import xarray as xr
    from aux_gps import weighted_long_term_monthly_means_da
    pw = xr.load_dataset(load_path / 'GNSS_PW_thresh_{:.0f}.nc'.format(thresh))
    pw_attrs = pw.attrs
    attrs = {da: val.attrs for (da, val) in pw.data_vars.items()}
    # use upper() on names:
    da = pw.to_array('station')
    da['station'] = da['station'].str.upper()
    pw = da.to_dataset('station')
    for da in pw.data_vars.values():
        da.attrs = attrs.get(da.name.lower())
    T = xr.load_dataset(ims_path / 'GNSS_5mins_TD_ALL_1996_2019.nc')
    # align T and pw:
    for da in pw.data_vars:
        pw['{}_T'.format(da)] = T[da.lower()]
    # pw_grp = pw.map(, plot=False)
    pw_grp = pw.groupby('time.{}'.format(grp)).mean('time')
    pw_grp.attrs = pw_attrs
    # now do climatology also:
    pw_clim = pw.map(weighted_long_term_monthly_means_da, plot=False)
#    for sta in pw_clim.data_vars.keys():
#        pw_clim = pw_clim.rename({sta: sta + '_clim'})
    pw_clim.attrs = pw_attrs
    just_pw = [x for x in pw_clim if '_T' not in x]
    for da in just_pw:
        pw_clim[da].attrs = attrs.get(da.lower())
    if savepath is not None:
        filename = 'PW_T_{}ly_means_clim_thresh_{:.0f}.nc'.format(grp, thresh)
        pw_clim.to_netcdf(savepath / filename, 'w')
        print('saved {} to {}.'.format(filename, savepath))
    return pw_clim


def group_anoms_and_cluster(load_path=work_yuval, remove_grp='month', thresh=None, grp='hour',
                            season=None, n_clusters=4, load_mm=False):
    import xarray as xr
    from sklearn.cluster import KMeans
    # load data and save attrs in dict:
#    pw = xr.load_dataset(work_yuval/'GNSS_PW_anom_{:.0f}_hour_dayofyear.nc'.format(thresh))
    pw = xr.load_dataset(load_path / 'GNSS_PW_thresh_{:.0f}.nc'.format(thresh))
    attrs = {da: val.attrs for (da, val) in pw.data_vars.items()}
    # use upper() on names:
    da = pw.to_array('station')
    da['station'] = da['station'].str.upper()
    pw = da.to_dataset('station')
    for da in pw.data_vars.values():
        da.attrs = attrs.get(da.name.lower())
    # for now, drop hrmn and nizn station from analysis:
#    if to_drop is not None:
#        for sta in to_drop:
#            pw = pw[[x for x in pw.data_vars if sta not in x]]
    # extract weights from attrs:
    weights = [float(x.attrs['mean_years']) for x in pw.data_vars.values()]
    weights = np.array(weights) / np.max(np.array(weights))
    # select season:
    if season is not None and grp == 'hour':
        pw = pw.sel(time=pw['time.season'] == season)
    # groupby and create means:
    if not load_mm:
        if remove_grp is not None:
            print('removing long term {}ly means first'.format(remove_grp))
            pw = pw.groupby('time.{}'.format(remove_grp)) - pw.groupby('time.{}'.format(remove_grp)).mean('time')
            pw_anom = pw
        else:
            # pw_anom = pw_mean - pw_mean.mean('{}'.format(grp))
            pw_anom = pw.groupby('time.{}'.format('month')).mean('time')
    else:
        pw_mean = xr.load_dataset(load_path / 'PW_T_monthly_means_clim_thresh_{:.0f}.nc'.format(thresh))
        just_pw = [x for x in pw_mean.data_vars if '_T' not in x]                   
        pw_anom = pw_mean[just_pw]
    # to dataframe:
    df = pw_anom.to_dataframe()
    weights = pd.Series(weights, index=[x for x in pw.data_vars])
    if n_clusters is not None:
        # cluster the anomalies:
        clr = KMeans(n_clusters=n_clusters, random_state=0).fit(df.T, sample_weight=weights)
        # get the labels start with 1:
        clr.labels_ += 1
        # clustering = DBSCAN(eps=3, min_samples=2).fit(df)
        # clustering = OPTICS(min_samples=2).fit(df)
        labels = dict(zip(df.columns, clr.labels_))
        labels_sorted = {
            k: v for k,
            v in sorted(
                labels.items(),
                key=lambda item: item[1])}
        order = [x for x in labels_sorted.keys()]
        df = df[order]
    return df, labels_sorted, weights


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
    # use of damped Ts: ?
    Ts_daily = Ts.resample(time='1D').mean()
    upsampled_daily = Ts_daily.resample(time='1D').ffill()
    damped = Ts*0.25 + 0.75*upsampled_daily
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
        kappa_err_list = []
        h_key = [x for x in hours.keys()][0]
        for hr_num in [x for x in hours.values()][0]:
            print('working on hour {}'.format(hr_num))
            sliced = Tds.where(Tds[h_key] == hr_num).dropna(time_dim)
            m = mda.sel({'name': model_name, h_key: hr_num}).values.item()
            kappa_part, kappa_err = kappa_ml(sliced, model=m)
            kappa_list.append(kappa_part)
            kappa_err_list.append(kappa_err)
        des_attrs = 'hourly data Tm formulation using {} model'.format(
            model_name)
    elif len(mda.dims) == 2 and seasons is not None:
        print('Found season Ts-Tm relationship slice.')
        kappa_list = []
        kappa_err_list = []
        s_key = [x for x in seasons.keys()][0]
        for season in [x for x in seasons.values()][0]:
            print('working on season {}'.format(season))
            sliced = Tds.where(Tds[s_key] == season).dropna('time')
            m = mda.sel({'name': model_name, s_key: season}).values.item()
            kappa_part, kappa_err = kappa_ml(sliced, model=m)
            kappa_list.append(kappa_part)
            kappa_err_list.append(kappa_err)
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
        kappa_err_list = []
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
                kappa_part, kappa_err = kappa_ml(sliced, model=m)
                kappa_list.append(kappa_part)
                kappa_err_list.append(kappa_err)
        des_attrs = 'hourly and seasonly data Tm formulation using {} model'.format(model_name)
    kappa_ds = xr.concat(kappa_list, time_dim)
    kappa_err_ds = xr.concat(kappa_err_list, time_dim)
    ipw = kappa_ds * zwd
    ipw_error = kappa_ds * zwd_error + zwd * kappa_err_ds
    ipw_error.name = 'PW_error'
    ipw_error.attrs['long_name'] = 'Precipitable Water standard error'
    ipw_error.attrs['units'] = 'kg / m^2'
    ipw.name = 'PW'
    ipw.attrs['long_name'] = 'Precipitable Water'
    ipw.attrs['units'] = 'kg / m^2'
    ipw = ipw.to_dataset(name='PW')
    ipw['PW_error'] = ipw_error
    ipw.attrs['description'] = des_attrs
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
    time_dim = list(set(T.dims))[0]
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
        Tm = xr.DataArray(tm, dims=[time_dim, 'coef', 'intercept'])
        Tm['time'] = T[time_dim]
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
        Tnp = T.dropna(time_dim).values.reshape(-1, 1)
        # T = T.values.reshape(-1, 1)
        Tm = T.dropna(time_dim).copy(deep=False,
                                   data=model.predict((273.15 + Tnp)))
        Tm = Tm.reindex({time_dim: T[time_dim]})
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
        return k
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
    # sns.set_style('darkgrid')
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
            y=station,
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
    # sns.set_style('darkgrid')
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
                              times=['2005', '2019'], station=None, plot=True):
    """calls formulate_plot to analyse and model the ts-tm connection from
    radiosonde(bet-dagan). options for categories:season, hour, clouds
    you can choose some ,all or none categories"""
    import xarray as xr
    from aux_gps import get_unique_index
    from aux_gps import keep_iqr
    from aux_gps import path_glob
    if isinstance(models, str):
        models = [models]
    if physical_file is not None or station is not None:
        print('Overwriting ds input...')
        if station is not None:
            physical_file = path_glob(sound_path, 'station_{}_soundings_ts_tm_tpw*.nc'.format(station))[0]
            print('station {} selected and loaded.'.format(station))
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
    # sns.set_style('darkgrid')
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
                s=100, linewidth=0, alpha=0.5,
                ax=axes[0])
            g.legend(loc='best')

        # axes[0].scatter(x=ds.ts.values, y=ds.tm.values, marker='.', s=10)
#        linex = np.array([ds.ts.min().item(), ds.ts.max().item()])
#        liney = a * linex + b
#        axes[0].plot(linex, liney, c='r')
        bevis_tm = ds.ts.values * 0.72 + 70.0
        if plot:
            # plot bevis:
            axes[0].plot(ds.ts.values, bevis_tm, c='purple')
            min_, max_ = axes[0].get_ylim()
            [axes[0].plot(X, newy, c=colors[i]) for i, newy in enumerate(predict)]
            [axes[0].text(0.01, pos[i],
                          '{} a: {:.2f}, b: {:.2f}'.format(model_names[i],
                                                           coefs[i], inters[i]),
                          transform=axes[0].transAxes, color=colors[i],
                          fontsize=12) for i in range(len(coefs))]
            axes[0].text(0.01, 0.8,
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
                                        ax=axes[i], linewidth=0, alpha=0.5)
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
                                            s=100, ax=axes[i, j], linewidth=0,
                                            alpha=0.5)
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


station_continous_times = {
    # a station - continous data times dict:
    'alon': [None, None],
    'bshm': ['2010', '2017'],  # strong dis behaviour
    'csar': [None, '2017'],  # small glich in the end
    'drag': ['2004', None],  # small glich in the begining
    'dsea': [None, '2016'],
    'elat': ['2013', None],   # gliches in the begining, then up and then down
    'elro': ['2005', '2009'],  # up and down, chose up period
    'gilb': ['2005', None],
    'hrmn': [None, None],   # spikes(WetZ), positive spikes in alt due to snow
    'jslm': ['2006', None],
    'kabr': ['2013', None],  # strong dis behaviour
    'katz': ['2011', '2016'],  # dis behaviour
    'klhv': [None, None],
    'lhav': ['2004', '2006'],  # dis behaviour
    'mrav': [None, None],
    'nizn': ['2015-09', None],  # something in the begining
    'nrif': ['2012', None],
    'nzrt': [None, None],
    'ramo': ['2006', None],  # somethin in begining
    'slom': ['2015-07', '2017-07'],  # strong dis behaviour, ups and downs
    'spir': ['2015', '2018'],   # big glich in the end
    'tela': ['2005', None],   # gap in 2003-2004 , glich in 2004
    'yosh': [None, None],
    'yrcm': ['2011', None]  # small glich in begining
}






def israeli_gnss_stations_long_term_trend_analysis(
        gis_path=gis_path,
        rel_plot='tela', show_names=True,
        times_dict=station_continous_times):
    import pandas as pd
    from pathlib import Path
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from aux_gps import geo_annotate
    import contextily as ctx
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
                rds = get_long_trends_from_gnss_station(
                    station, 'LR', plot=False, times=times_dict[station])
            except KeyError:
                print(
                    'didnt find {} key in times_dict, skipping...'.format(station))
                continue
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
            df['rel_mm_north_{}'.format(station)] = (
                df['north_cm_per_year'] - df.loc[station, 'north_cm_per_year']) * 100.0
            df['rel_mm_east_{}'.format(station)] = (
                df['east_cm_per_year'] - df.loc[station, 'east_cm_per_year']) * 100.0
            df['rel_mm_per_year_{}'.format(station)] = np.sqrt(
                df['rel_mm_north_{}'.format(station)] ** 2.0 +
                df['rel_mm_east_{}'.format(station)] ** 2.0)
            # define angle from east : i.e., x axis is east
            df['rel_angle_from_east_{}'.format(station)] = np.rad2deg(np.arctan2(
                df['rel_mm_north_{}'.format(station)], df['rel_mm_east_{}'.format(station)]))
            df['rel_up_mm_per_year_{}'.format(
                station)] = df['up_mm_per_year'] - df.loc[station, 'up_mm_per_year']
        df.to_csv(cwd / filename, sep=' ')
        print('{} was saved to {}'.format(filename, cwd))
    isr_with_yosh = gpd.read_file(gis_path / 'Israel_demog_yosh.shp')
    isr_with_yosh.crs = {'init': 'epsg:4326'}
    stations = gpd.GeoDataFrame(df,
                                geometry=gpd.points_from_xy(df.lon,
                                                            df.lat),
                                crs=isr_with_yosh.crs)
    isr = gpd.sjoin(stations, isr_with_yosh, op='within')
    isr_with_yosh = isr_with_yosh.to_crs(epsg=3857)
    isr = isr.to_crs(epsg=3857)
    isr['X'] = isr.geometry.x
    isr['Y'] = isr.geometry.y
    isr['U'] = isr.east_cm_per_year
    isr['V'] = isr.north_cm_per_year
    if rel_plot is not None:
        isr['U'] = isr['rel_mm_east_{}'.format(rel_plot)]
        isr['V'] = isr['rel_mm_north_{}'.format(rel_plot)]
        title = 'Relative to {} station'.format(rel_plot)
        vertical_label = isr['rel_up_mm_per_year_{}'.format(rel_plot)]
        horizontal_label = isr['rel_mm_per_year_{}'.format(rel_plot)]
    else:
        title = ''
        vertical_label = isr['up_mm_per_year']
        horizontal_label = isr['cm_per_year']
        # isr.drop('dsea', axis=0, inplace=True)
    #fig, ax = plt.subplots(figsize=(20, 10))
    ax = isr_with_yosh.plot(alpha=0.0, figsize=(6, 15))
    ctx.add_basemap(ax, url=ctx.sources.ST_TERRAIN)
    ax.set_axis_off()
    if show_names:
        ax.plot(
            [],
            [],
            ' ',
            label=r'$^{\frac{mm}{year}}\bigcirc^{name}_{\frac{mm}{year}}$')
    else:
        ax.plot(
            [],
            [],
            ' ',
            label=r'$^{\frac{mm}{year}}\bigcirc_{\frac{mm}{year}}$')
    ax.plot([], [], ' ', label='station:')
    isr[(isr['years'] <= 5.0) & (isr['years'] >= 0.0)].plot(
        ax=ax, markersize=50, color='m', edgecolor='k', marker='o', label='0-5 yrs')
    isr[(isr['years'] <= 10.0) & (isr['years'] > 5.0)].plot(
        ax=ax, markersize=50, color='y', edgecolor='k', marker='o', label='5-10 yrs')
    isr[(isr['years'] <= 15.0) & (isr['years'] > 10.0)].plot(
        ax=ax, markersize=50, color='g', edgecolor='k', marker='o', label='10-15 yrs')
    isr[(isr['years'] <= 20.0) & (isr['years'] > 15.0)].plot(
        ax=ax, markersize=50, color='c', edgecolor='k', marker='o', label='15-20 yrs')
    isr[(isr['years'] <= 25.0) & (isr['years'] > 20.0)].plot(
        ax=ax, markersize=50, color='r', edgecolor='k', marker='o', label='20-25 yrs')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2, 3, 4, 5, 1, 0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
               prop={'size': 10}, bbox_to_anchor=(-0.15, 1.0),
               title='number of data years')
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='x-large')
    # isr.plot(ax=ax, column='cm_per_year', cmap='Greens',
    #          edgecolor='black', legend=True)
    cmap = plt.get_cmap('spring', 10)
#        Q = ax.quiver(isr['X'], isr['Y'], isr['U'], isr['V'],
#                      isr['cm_per_year'], cmap=cmap)
    Q = ax.quiver(isr['X'], isr['Y'], isr['U'], isr['V'], cmap=cmap)
#        fig.colorbar(Q, extend='max')
#        qk = ax.quiverkey(Q, 0.8, 0.9, 1, r'$1 \frac{cm}{yr}$', labelpos='E',
#                          coordinates='figure')
    if show_names:
        annot1 = geo_annotate(ax, isr.geometry.x, isr.geometry.y, isr.index,
                              xytext=(3, 3))
        annot2 = geo_annotate(ax, isr.geometry.x, isr.geometry.y,
                              vertical_label, xytext=(3, -10),
                              fmt='{:.1f}', fw='bold', colorupdown=True)
        annot3 = geo_annotate(ax, isr.geometry.x, isr.geometry.y,
                              horizontal_label, xytext=(-20, 3),
                              fmt='{:.1f}', c='k', fw='normal')
    # plt.legend(handles=[annot1, annot2, annot3])
    plt.title(title)
    plt.tight_layout()
#    elif rel_plot is not None:
#        # isr.drop('dsea', axis=0, inplace=True)
#        fig, ax = plt.subplots(figsize=(10, 8))
#        isr_with_yosh.plot(ax=ax)
#        isr[(isr['years'] <= 5.0) & (isr['years'] >= 0.0)].plot(ax=ax, markersize=50, color='m', edgecolor='k', marker='o', label='0-5 yrs')
#        isr[(isr['years'] <= 10.0) & (isr['years'] > 5.0)].plot(ax=ax, markersize=50, color='y', edgecolor='k', marker='o', label='5-10 yrs')
#        isr[(isr['years'] <= 15.0) & (isr['years'] > 10.0)].plot(ax=ax, markersize=50, color='g', edgecolor='k', marker='o', label='10-15 yrs')
#        isr[(isr['years'] <= 20.0) & (isr['years'] > 15.0)].plot(ax=ax, markersize=50, color='c', edgecolor='k', marker='o', label='15-20 yrs')
#        isr[(isr['years'] <= 25.0) & (isr['years'] > 20.0)].plot(ax=ax, markersize=50, color='r', edgecolor='k', marker='o', label='20-25 yrs')
#        plt.legend(prop={'size': 12}, bbox_to_anchor=(-0.15, 1.0), title='number of data years')
#        # isr.plot(ax=ax, column='cm_per_year', cmap='Greens',
#        #          edgecolor='black', legend=True)
#        isr['U'] = isr['rel_mm_east_{}'.format(rel_plot)]
#        isr['V'] = isr['rel_mm_north_{}'.format(rel_plot)]
#        cmap = plt.get_cmap('spring', 7)
##        Q = ax.quiver(isr['X'], isr['Y'], isr['U'], isr['V'],
##                      isr['rel_mm_per_year_{}'.format(rel_plot)],
##                      cmap=cmap)
#        Q = ax.quiver(isr['X'], isr['Y'], isr['U'], isr['V'], cmap=cmap)
##        qk = ax.quiverkey(Q, 0.8, 0.9, 1, r'$1 \frac{mm}{yr}$', labelpos='E',
##                          coordinates='figure')
##        fig.colorbar(Q, extend='max')
#        plt.title('Relative to {} station'.format(rel_plot))
#        geo_annotate(ax, isr.lon, isr.lat, isr.index, xytext=(3, 3))
#        geo_annotate(ax, isr.lon, isr.lat, isr['rel_up_mm_per_year_{}'.format(rel_plot)],
#                     xytext=(3, -6), fmt='{:.2f}', fw='bold',
#                     colorupdown=True)
#        geo_annotate(ax, isr.lon, isr.lat,
#                     isr['rel_mm_per_year_{}'.format(rel_plot)],
#                     xytext=(-21, 3), fmt='{:.2f}', c='k', fw='normal')
##        for x, y, label in zip(isr.lon, isr.lat,
##                               isr.index):
##            ax.annotate(label, xy=(x, y), xytext=(3, 3),
##                        textcoords="offset points")
#        # print(isr[['rel_mm_east_{}'.format(rel_plot),'rel_mm_north_{}'.format(rel_plot)]])
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
def classify_tide_events(gnss_path=work_yuval, hydro_path=hydro_path,
                         station='tela', window='1D', sample='hourly',
                         hydro_station=48130):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    kwargs = locals()
    [kwargs.pop(key) for key in ['LogisticRegression', 'confusion_matrix', 'train_test_split',
                'classification_report']]
    lr = LogisticRegression(n_jobs=-1)
    X, y = GNSS_pw_to_X_using_window(**kwargs)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(confusion_matrix)
    print(classification_report(y_test, y_pred))
    return lr


def GNSS_pw_to_X_using_window(gnss_path=work_yuval, hydro_path=hydro_path,
                              station='tela', window='1D', sample='hourly',
                              hydro_station=60190):
    """assemble n window length gnss_pw data array with datetimes and
    a boolean array of positive or negative tide events"""
    import xarray as xr
    from aux_gps import time_series_stack_with_window
    # read PW and select station:
    GNSS_pw = xr.open_dataset(gnss_path / 'GNSS_{}_PW.nc'.format(sample))
    pw = GNSS_pw[station]
    # create 1 day length data chunks from pw:
    ds_X = time_series_stack_with_window(pw, window='1D')
    # dropna:
    ds_X = ds_X.dropna('start_date')
    X = ds_X[station]
    # read tides and select station:
    tides = xr.open_dataset(hydro_path / 'hydro_tides.nc')
    # select station:
    tide = tides['TS_{}_max_flow'.format(hydro_station)]
    # dropna:
    tide = tide.dropna('tide_start')
    # resample to ds_X time:
    tide = tide.resample(tide_start=ds_X.attrs['freq']).mean()
    tide = tide.dropna('tide_start')
    # now build y:
    y = np.empty(X.values.shape[0], dtype=bool)
    start_date = X.start_date.values
    points = X.points.size
    tide_start = tide.tide_start.values
    for i in range(len(start_date) - points):
        st = start_date[i + points]
        if st in tide_start:
            y[i] = True
        else:
            y[i] = False
    y = xr.DataArray(y, dims=['start_date'])
    y['start_date'] = start_date
    y.name = 'tide_events'
    return X, y


def produce_all_GNSS_PW_anomalies(load_path=work_yuval, thresh=None,
                                  grp1='hour', grp2='dayofyear',
                                  savepath=work_yuval):
    import xarray as xr
    GNSS_pw = xr.open_dataset(load_path / 'GNSS_PW_{:.0f}.nc'.format(thresh))
    anom_list = []
    stations_only = [x for x in GNSS_pw.data_vars if '_error' not in x]
    for station in stations_only:
        pw = GNSS_pw[station]
        pw_anom = produce_PW_anomalies(pw, grp1, grp2, False)
        anom_list.append(pw_anom)
    GNSS_pw_anom = xr.merge(anom_list)
    if savepath is not None:
        filename = 'GNSS_PW_anom_{:.0f}_{}_{}.nc'.format(thresh, grp1, grp2)
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in GNSS_pw_anom.data_vars}
        GNSS_pw_anom.to_netcdf(savepath / filename, 'w', encoding=encoding)
        print('Done!')
    return GNSS_pw_anom


def produce_PW_anomalies(pw_da, grp1='hour', grp2='dayofyear', plot=True):
    from aux_gps import time_series_stack
    import xarray as xr
    from aux_gps import get_unique_index
    from aux_gps import xr_reindex_with_date_range
    from scipy import stats
    import matplotlib.pyplot as plt
    time_dim = list(set(pw_da.dims))[0]
    fname = pw_da.name
    print('computing anomalies for {}'.format(fname))
    stacked_pw = time_series_stack(pw_da, time_dim=time_dim, grp1=grp1,
                                   grp2=grp2, plot=False)
    pw_anom = stacked_pw.copy(deep=True)
    attrs = pw_anom.attrs
    rest_dim = [x for x in stacked_pw.dims if x != grp1 and x != grp2][0]
    # compute mean on rest dim and remove it from stacked_da:
    rest_mean = stacked_pw[fname].mean(rest_dim)
    for rest in stacked_pw[rest_dim].values:
        pw_anom[fname].loc[{rest_dim: rest}] -= rest_mean
    # now, flatten anomalies to restore the time-series structure:
    vals = pw_anom[fname].values.ravel()
    times = pw_anom[time_dim].values.ravel()
    pw_anom = xr.DataArray(vals, dims=[time_dim])
    pw_anom.attrs = attrs
    pw_anom[time_dim] = times
    pw_anom = get_unique_index(pw_anom)
    pw_anom = pw_anom.sortby(time_dim)
    pw_anom = xr_reindex_with_date_range(pw_anom, freq=pw_anom.attrs['freq'])
    pw_anom.name = fname
    pw_anom.attrs['description'] = 'anomalies are computed from {} and {} groupings'.format(grp1, grp2)
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]})
        pw = pw_anom.dropna(time_dim).values
        pw_anom.plot(ax=ax1)
        pw_anom.plot.hist(bins=100, density=True, ax=ax2)
        xt = ax2.get_xticks()
        xmin, xmax = min(xt), max(xt)
        lnspc = np.linspace(xmin, xmax, len(pw_anom.values))
        # lets try the normal distribution first
        m, s = stats.norm.fit(pw)  # get mean and standard deviation
        # now get theoretical values in our interval
        pdf_g = stats.norm.pdf(lnspc, m, s)
        ax2.plot(lnspc, pdf_g, label="Norm")  # plot it
        # exactly same as above
        ag, bg, cg = stats.gamma.fit(pw)
        pdf_gamma = stats.gamma.pdf(lnspc, ag, bg, cg)
        ax2.plot(lnspc, pdf_gamma, label="Gamma")
        # guess what :)
        ab, bb, cb, db = stats.beta.fit(pw)
        pdf_beta = stats.beta.pdf(lnspc, ab, bb, cb, db)
        ax2.plot(lnspc, pdf_beta, label="Beta")
    return pw_anom


def load_GNSS_TD(station='tela', sample_rate=None, plot=True):
    """load and plot temperature for station from IMS, to choose
    sample rate different than 5 mins choose: 'H', 'W' or 'MS'"""
    from aux_gps import path_glob
    from aux_gps import plot_tmseries_xarray
    import xarray as xr
    sample = {'1H': 'hourly', '3H': '3hourly', 'D': 'daily', 'W': 'weekly',
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


#def align_monthly_means_PW_and_T(path=work_yuval, ims_path=ims_path,
#                                 thresh=50.0):
#    """align monthly means PW and T for plots"""
#    import xarray as xr
#    pw = xr.load_dataset(path / 'GNSS_PW_thresh_{:.0f}.nc'.format(thresh))
#    # get attrs dict:
#    attrs = {}
#    for station in pw.data_vars:
#        attrs[station] = pw[station].attrs
#    stations = [x for x in pw.data_vars]
#    # resample to monthly means:
#    pw = pw.resample(time='MS').mean('time')
#    # copy attrs to each station:
#    for station in pw.data_vars:
#        pw[station].attrs = attrs[station]
#    T = xr.load_dataset(ims_path / 'GNSS_monthly_TD_ALL_1996_2019.nc')
#    T = T[stations]
#    # rename T stations to T:
#    for sta in T.data_vars.keys():
#        T = T.rename({sta: sta + '_T'})
#    combined = xr.merge([pw, T])
#    filename = 'PW_T_monthly_means_thresh_{:.0f}.nc'.format(thresh)
#    combined.to_netcdf(path / filename, 'w')
#    print('saved {} to {}'.format(filename, path))
#    return combined


def filter_month_year_data_heatmap_plot(da_ts, freq='5T', thresh=50.0,
                                        verbose=True, plot=True):
    """accepts dataarray time series(with freq <1D) and removes the daily data
    with less than thresh percent and then removes months with data less than
    thresh percent. data is saved to dataarray with some metadata"""
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    name = da_ts.name
    try:
        freq = da_ts.attrs['freq']
    except KeyError:
        pass
    # data points per day:
    if freq == '5T':
        points = 24 * 12
    elif freq == '1H':
        points = 24
    elif freq == '3H':
        points == 8
    elif freq == '1D' or freq == 'D':
        points = 1
    if verbose:
        print(
            'analysing {} station with {} data points per day:'.format(
                name, points))
    # dropna:
    df = da_ts.dropna('time').to_dataframe()
    # calculate daily data to drop (if points less than threshold):
    df['date'] = df.index.date
    points_in_day = df.groupby(['date']).count()[name].to_frame()
    # calculate daily data percentage (from maximum available):
    points_in_day['percent'] = (points_in_day[name] / points) * 100.0
    # get the number of days to drop and the dates themselves:
    number_of_days_to_drop = points_in_day[name][points_in_day['percent'] <= thresh].count()
    percent_of_days_to_drop = 100.0 * \
        number_of_days_to_drop / len(points_in_day)
    days_to_drop = points_in_day.index[points_in_day['percent'] <= thresh]
    if verbose:
        print('found {} ({:.2f} %) bad days with {:.0f} % drop thresh.'.format(
                number_of_days_to_drop, percent_of_days_to_drop, thresh))
    # now drop the days:
    for day_to_drop in days_to_drop:
        df = df[df['date'] != day_to_drop]
    # now calculate the number of months missing days with threshold:
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['max_points'] = df.index.days_in_month * points
    cnt = df.groupby(['month', 'year']).count()[name].to_frame()
    pivot = pd.pivot_table(cnt, index='year', columns='month')
    pivot_m = pd.pivot_table(
        df[['month', 'year', 'max_points']], index='year', columns='month')
    percent = 100 * pivot.values / pivot_m.values
    per_df = pd.DataFrame(
        percent,
        index=df.year.unique(),
        columns=sorted(
            df.month.unique()))
    stacked = per_df.stack().dropna().to_frame(name)
    months_and_years_to_drop = stacked[stacked < thresh].dropna()
    index = months_and_years_to_drop.index
    number_of_months_to_drop = index.size
    percent_of_months_to_drop = 100.0 * index.size / cnt.size
    if verbose:
        print('found {} ({:.2f} % ) bad months with {:.0f} % drop thresh'.format(
                number_of_months_to_drop, percent_of_months_to_drop, thresh))
        print('#months that are bigger then {:.0f} %:'.format(thresh))
    # now get the months to drop:
    dts = []
    for year, month in index.values:
        dts.append('{}-{}'.format(year, month))
    df['months'] = df['year'].astype(str) + '-' + df['month'].astype(str)
    for month_to_drop in dts:
        df = df[df['months'] != month_to_drop]
    # create some metadate to put in dataarray:
    month_dict = {}
    for month in per_df.columns:
        months = per_df[per_df >= thresh][month].dropna().count()
        month_dict[month] = months
        print('#{} months of months {}'.format(months, month))
    # transform to dataarray:
    da = df[name].to_xarray()
    da.attrs['threshold'] = '{:.0f}'.format(thresh)
    for month, value in month_dict.items():
        da.attrs['months_{}'.format(month)] = value
    # calculate the mean years of data:
    myears = np.mean(np.array([x for x in month_dict.values()]))
    da.attrs['mean_years'] = '{:.2f}'.format(myears)
    # add some more metadata:
    da.attrs['days_dropped'] = number_of_days_to_drop
    da.attrs['days_dropped_percent'] = '{:.1f}'.format(percent_of_days_to_drop)
    da.attrs['months_dropped'] = number_of_months_to_drop
    da.attrs['months_dropped_percent'] = '{:.1f}'.format(percent_of_months_to_drop)
    if plot:
        sns.heatmap(per_df, annot=True, fmt='.0f')
        plt.figure()
        per_df.stack().hist(bins=25)
    return da


def load_PW_with_drop_thresh(station='tela', thresh=50.0):
    import xarray as xr
    pw = xr.load_dataset(work_yuval / 'GNSS_PW.nc')[station]
    if thresh is not None:
        print('loading 5 mins GNSS PW {} station with threshold={}%.'.format(station, thresh))
        da = filter_month_year_data_heatmap_plot(pw, freq='5T', thresh=thresh,
                                                 plot=False, verbose=True)
    else:
        print('loading 5 mins GNSS PW {} station without threshold.'.format(station))
    return da


def drop_GNSS_PW_thresh_and_combine_save_all(
    path=work_yuval, thresh=None, to_drop=[
        'nizn', 'hrmn'], combine_dict={
            'klhv': [
                'klhv', 'lhav'], 'mrav': [
                    'gilb', 'mrav']}):
    import xarray as xr
    import numpy as np
    GNSS_pw = xr.open_dataset(path / 'GNSS_PW.nc')
    thresh_list = []
    stations_only = [x for x in GNSS_pw.data_vars if '_error' not in x]
    for station in stations_only:
        pw = load_PW_with_drop_thresh(station=station, thresh=thresh)
        thresh_list.append(pw)
    GNSS_pw_thresh = xr.merge(thresh_list)
    if combine_dict is not None:
        # combine stations:
        combined = []
        for new_sta, sta_to_merge in combine_dict.items():
            print('merging {} to {}'.format(sta_to_merge, new_sta))
            combined.append(combine_PW_stations(GNSS_pw_thresh, new_sta,
                                                sta_to_merge, thresh))
        # drop old stations:
        sta_to_drop = [item for sublist in combine_dict.values()
                       for item in sublist]
        for sta in sta_to_drop:
            GNSS_pw_thresh = GNSS_pw_thresh[[
                x for x in GNSS_pw_thresh.data_vars if sta not in x]]
        # plug them in GNSS dataset:
        for sta in combined:
            GNSS_pw_thresh[sta.name] = sta
    if to_drop is not None:
        for sta in to_drop:
            print('dropping {} station.'.format(sta))
            GNSS_pw_thresh = GNSS_pw_thresh[[
                x for x in GNSS_pw_thresh.data_vars if sta not in x]]
    mean_days_dropped_percent = np.mean(np.array([float(
        GNSS_pw_thresh[x].attrs['days_dropped_percent']) for x in
        GNSS_pw_thresh.data_vars]))
    mean_months_dropped_percent = np.mean(np.array([float(
        GNSS_pw_thresh[x].attrs['months_dropped_percent']) for x in GNSS_pw_thresh.data_vars]))
    GNSS_pw_thresh.attrs['thresh'] = '{:.0f}'.format(thresh)
    GNSS_pw_thresh.attrs['mean_days_dropped_percent'] = '{:.2f}'.format(mean_days_dropped_percent)
    GNSS_pw_thresh.attrs['mean_months_dropped_percent'] = '{:.2f}'.format(mean_months_dropped_percent)
    filename = 'GNSS_PW_thresh_{:.0f}.nc'.format(thresh)
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in GNSS_pw_thresh.data_vars}
    GNSS_pw_thresh.to_netcdf(path / filename, 'w', encoding=encoding)
    print('Done!')
    return GNSS_pw_thresh


def combine_PW_stations(pw, name, stations, thresh=None):
    import xarray as xr
    import numpy as np
    from aux_gps import get_unique_index
    time_dim = list(set(pw.dims))[0]
    slist = [pw[x] for x in stations]
    combined_station = xr.concat(slist, time_dim)
    # take care of months and attrs:
    if thresh is not None:
        months_add = np.zeros((12), dtype=int)
        for sta in stations:
            # extract month_dict from pw attrs:
            keys = ['months_{}'.format(x) for x in np.arange(1, 13)]
            vals = [pw[sta].attrs[x] for x in keys]
            month_dict = dict(zip(keys, vals))
            months = [(x, y) for x, y in month_dict.items()]
            months_add += np.array([x for x in dict(months).values()])
        months = dict(zip(dict(months).keys(), months_add))
        for month, val in months.items():
            combined_station.attrs[month] = val
        combined_station.attrs['mean_years'] = np.mean(
            [x for x in months.values()])
    # add attr of combined station:
    combined_station.attrs['combined_from'] = ', '.join(stations)
    combined_station.name = name
    # get unique times:
    combined_station = get_unique_index(combined_station)
    return combined_station


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
    import matplotlib.pyplot as plt

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
            fg = plot_tmseries_xarray(ds, plot_fields, points=points)
            try:
                # Axes object need to use .figure to access figure:
                fg.figure.suptitle(
                    'Station: {}'.format(
                        ds.attrs['station']),
                    fontweight='bold')
                fg.figure.subplots_adjust(top=0.90)
            except AttributeError:
                # while FacetGrid object needs .fig to access figure:
                fg.fig.suptitle(
                    'Station: {}'.format(
                        ds.attrs['station']),
                    fontweight='bold')
                fg.fig.subplots_adjust(top=0.93)
        elif plot_fields == 'all':
            fg = plot_tmseries_xarray(ds, ['GradNorth', 'GradEast', 'WetZ',
                                           'lat', 'lon', 'alt'], points=points)
            fg.fig.suptitle(
                'Station: {}'.format(
                    ds.attrs['station']),
                fontweight='bold')
            fg.fig.subplots_adjust(top=0.93)
        return ds

    sample = {'1H': 'hourly', '3H': '3hourly', 'D': 'Daily', 'W': 'weekly',
              'MS': 'monthly'}
    if sample_rate is None:
        points = False
    else:
        points = True
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
                                      plot=True, times=None):
    import xarray as xr
    import numpy as np
    from aux_gps import plot_tmseries_xarray
    # dont try anonther model than LR except for lower-sampled data
    ds = load_gipsyx_results(station, sample_rate='1H', plot_fields=None)
    if ds is None:
        raise FileNotFoundError
    # first do altitude [m]:
    if times is not None:
        time_dim = list(set(ds.dims))[0]
        ds = ds.sel({time_dim: slice(*times)})
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


def ML_fit_model_to_tmseries(tms_da, modelname='LR', plot=True,
                             verbose=False, ml_params=None, gridsearch=False):
    """fit a single time-series data-array with ML models specified in
    ML_Switcher"""
    import numpy as np
    import xarray as xr
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import GridSearchCV
    # find the time dim:
    time_dim = list(set(tms_da.dims))[0]
    # pick a model:
    ml = ML_Switcher()
    model = ml.pick_model(modelname)
    if ml_params is not None:
        model.set_params(**ml_params)
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
    if gridsearch:
        print('Grid Searching...')
        model = GridSearchCV(model, ml.param_grid, refit=True, n_jobs=-1)
    model.fit(jul_no_nan, tms_da_no_nan.values)
    new_y = model.predict(jul_with_nan).squeeze()
    new_da = xr.DataArray(new_y, dims=[time_dim])
    new_da[time_dim] = tms_da[time_dim]
    resid = tms_da.values - new_da.dropna('time').values
    if hasattr(model, 'coef_'):
        new_da.attrs['slope_per_day'] = model.coef_[0]
        days = pd.to_timedelta(tms_da.time.max(
                ).values - tms_da.time.min().values, unit='D')
        years = days / np.timedelta64(1, 'Y')
        per_year = days.days / years
        slope_per_year = model.coef_[0] * per_year
        new_da.attrs['slope_per_year'] = slope_per_year
        new_da.attrs['total_years'] = years
        if modelname == 'LR':
            se = calculate_linear_standard_error(jul_with_nan, resid)
            ci_95_per_year = [slope_per_year - 1.96 * se * per_year,
                              slope_per_year + 1.96 * se * per_year]
            new_da.attrs['ci_95_per_year'] = ci_95_per_year
        if verbose:
            print('slope_per_day: {}'.format(model.coef_[0]))
            print('slope_per_year: {}'.format(slope_per_year))
    if hasattr(model, 'intercept_'):
        new_da.attrs['intercept'] = model.intercept_
        if verbose:
            print('intercept: {}'.format(model.intercept_))
    new_da.attrs['model'] = modelname
    if verbose:
        print('model name: ', modelname)
        for atr, val in vars(model).items():
            print(atr, ': ', val)
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(20, 4),
                               gridspec_kw={'width_ratios': [3, 1]})
        tms_da.plot.line(ax=ax[0], marker='.', linewidth=0., color='b')
        new_da.plot(ax=ax[0], color='r')
        sns.distplot(
            resid,
            bins=25,
            color='c',
            label='residuals',
            ax=ax[1])
        ax[1].set_title('mean: {:.2f}'.format(np.mean(resid)))
        try:
            fig.suptitle('slope per decade: {:.2f}'.format(new_da.attrs['slope_per_year']*10))
        except KeyError:
            pass
    return model


def calculate_linear_standard_error(x, resid):
    import numpy as np
    n = len(resid)
    assert n == len(x)
    nume = np.nansum(resid**2.0) * 1.0/(n-2)
    denom = np.nansum((x-np.nanmean(x))**2.0)
    return np.sqrt(nume/denom)


def get_p_values(X, y):
    """produce p_values"""
    import numpy as np
    from sklearn.feature_selection import f_regression
    pval = np.empty((X.shape))
    f, pval[:] = f_regression(X, y)
    return pval
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
    
    def LRS(self):
        lr = LinearRegression(n_jobs=-1, copy_X=True)
        return LinearRegression_with_stats(lr)

    def GPSR(self):
        from gplearn.genetic import SymbolicRegressor
        return SymbolicRegressor(random_state=42, n_jobs=1, metric='mse')

    def TSEN(self):
        from sklearn.linear_model import TheilSenRegressor
        return TheilSenRegressor(random_state=42, n_jobs=-1)

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
        from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
        self.param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
                           "kernel": [ExpSineSquared(l, p)
                                      for l in np.arange(10, 70, 10)
                                      for p in np.arange(360, 370)]}
        return KernelRidge(kernel=ExpSineSquared(40.0, 365.0), alpha=0.001)

    def GPR(self):
        from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
        from sklearn.gaussian_process import GaussianProcessRegressor
        gp_kernel = ExpSineSquared(40.0, 365.0, periodicity_bounds=(340, 380)) \
            + WhiteKernel(10.0)
        return GaussianProcessRegressor(kernel=gp_kernel, random_state=42)

    def MTENETCV(self):
        import numpy as np
        from sklearn.linear_model import MultiTaskElasticNetCV
        return MultiTaskElasticNetCV(random_state=42, cv=10, n_jobs=-1,
                                alphas=np.logspace(-5, 2, 400))