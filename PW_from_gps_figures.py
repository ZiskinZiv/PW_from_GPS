#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 17:28:04 2020

@author: shlomi
"""
from PW_paths import work_yuval
tela_results_path = work_yuval / 'GNSS_stations/tela/rinex/30hr/results'
tela_solutions = work_yuval / 'GNSS_stations/tela/gipsyx_solutions'
sound_path = work_yuval / 'sounding'
phys_soundings = sound_path / 'bet_dagan_phys_sounding_2007-2019.nc'
ims_path = work_yuval / 'IMS_T'
gis_path = work_yuval / 'gis'


def plot_figure_1(path=work_yuval):
    from aux_gps import gantt_chart
    import xarray as xr
    ds = xr.open_dataset(path / 'GNSS_PW.nc')
    just_pw = [x for x in ds if 'error' not in x]
    ds = ds[just_pw]
    title = 'RINEX files availability for the Israeli GNSS stations'
    ax = gantt_chart(ds, title)
    return ax


def plot_figure_2(path=tela_results_path, plot='WetZ'):
    from aux_gps import path_glob
    import matplotlib.pyplot as plt
    from gipsyx_post_proc import process_one_day_gipsyx_output
    filepath = path_glob(path, 'tela*_smoothFinal.tdp')[3]
    if plot is None:
        df, meta = process_one_day_gipsyx_output(filepath, True)
        return df, meta
    else:
        df, meta = process_one_day_gipsyx_output(filepath, False)
        if not isinstance(plot, str):
            raise ValueError('pls pick only one field to plot., e.g., WetZ')
    error_plot = '{}_error'.format(plot)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    desc = meta['desc'][plot]
    unit = meta['units'][plot]
    df[plot].plot(ax=ax, legend=True, color='k')
    ax.fill_between(df.index, df[plot] - df[error_plot],
                    df[plot] + df[error_plot], alpha=0.5)
    ax.grid()
    ax.set_title('{} from station TELA'.format(desc))
    ax.set_ylabel(unit)
    ax.set_xlabel('')
    ax.grid('on')
    fig.tight_layout()
    return ax


def plot_figure_3(path=tela_solutions, year=2004, field='WetZ',
                  middle_date='11-25', zooms=[10, 3, 0.5]):
    from gipsyx_post_proc import analyse_results_ds_one_station
    import xarray as xr
    import matplotlib.pyplot as plt
    import pandas as pd
    dss = xr.open_dataset(path / 'TELA_ppp_raw_{}.nc'.format(year))
    nums = sorted(list(set([int(x.split('-')[1])
                            for x in dss if x.split('-')[0] == field])))
    ds = dss[['{}-{}'.format(field, i) for i in nums]]
    da = analyse_results_ds_one_station(dss, field=field, plot=False)
    fig, axes = plt.subplots(ncols=1, nrows=3, sharex=False, figsize=(16, 10))
    for j, ax in enumerate(axes):
        start = pd.to_datetime('{}-{}'.format(year, middle_date)
                               ) - pd.Timedelta(zooms[j], unit='D')
        end = pd.to_datetime('{}-{}'.format(year, middle_date)
                             ) + pd.Timedelta(zooms[j], unit='D')
        daa = da.sel(time=slice(start, end))
        for i, ppp in enumerate(ds):
            ds['{}-{}'.format(field, i)].plot(ax=ax, linewidth=3.0)
        daa.plot.line(marker='.', linewidth=0., ax=ax, color='k')
        axes[j].set_xlim(start, end)
        axes[j].set_ylim(daa.min() - 0.5, daa.max() + 0.5)
        try:
            axes[j - 1].axvline(x=start, color='r', alpha=0.7, linestyle='--')
            axes[j - 1].axvline(x=end, color='r', alpha=0.7, linestyle='--')
        except IndexError:
            pass
        units = ds.attrs['{}>units'.format(field)]
        sta = da.attrs['station']
        desc = da.attrs['{}>desc'.format(field)]
        ax.set_ylabel('{} [{}]'.format(field, units))
        ax.set_xlabel('')
        ax.grid()
    fig.suptitle(
        '30 hours stitched {} for GNSS station {}'.format(
            desc, sta), fontweight='bold')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    return axes


def plot_figure_4(physical_file=phys_soundings, model='LR',
                  times=['2007', '2019']):
    import xarray as xr
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PW_stations import ML_Switcher
    from sklearn.metrics import mean_squared_error
    from aux_gps import get_unique_index
    import numpy as np
    sns.set_style('whitegrid')
    pds = xr.open_dataset(phys_soundings)
    pds = pds[['Tm', 'Ts']]
    pds = get_unique_index(pds, 'sound_time')
    pds = pds.sel(sound_time=slice(*times))
    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    pds.plot.scatter(
        x='Ts',
        y='Tm',
        marker='.',
        s=100.,
        linewidth=0,
        alpha=0.5,
        ax=axes[0])
    ml = ML_Switcher()
    fit_model = ml.pick_model(model)
    X = pds.Ts.values.reshape(-1, 1)
    y = pds.Tm.values
    fit_model.fit(X, y)
    predict = fit_model.predict(X)
    coef = fit_model.coef_[0]
    inter = fit_model.intercept_
    axes[0].plot(X, predict, c='r')
    bevis_tm = pds.Ts.values * 0.72 + 70.0
    axes[0].plot(pds.Ts.values, bevis_tm, c='purple')
    axes[0].legend(['OLS ({:.2f}, {:.2f})'.format(
            coef, inter), 'Bevis 1992 et al. (0.72, 70.0)'])
    axes[0].set_xlabel('Surface Temperature [K]')
    axes[0].set_ylabel('Water Vapor Mean Atmospheric Temperature [K]')
    resid = predict - y
    sns.distplot(resid, bins=50, color='c', label='residuals', ax=axes[1],
                 kde=False,
                 hist_kws={"linewidth": 1, "alpha": 0.5, "color": "b"})
    rmean = np.mean(resid)
    rmse = np.sqrt(mean_squared_error(predict, y))
    axes[1].axvline(rmean, color='r', linestyle='dashed', linewidth=1)
    axes[1].set_xlabel('Residual distribution[K]')
    axes[1].text(0.1, 0.85, 'n={}'.format(pds.Ts.size),
                 verticalalignment='top', horizontalalignment='center',
                 transform=axes[1].transAxes, color='b', fontsize=12)
    axes[1].text(0.15, 0.8, 'RMSE: {:.2f} K'.format(rmse),
                 verticalalignment='top', horizontalalignment='center',
                 transform=axes[1].transAxes, color='b', fontsize=12)
    axes[1].set_xlim(-20, 20)
    fig.tight_layout()
    return


def plot_figure_5(physical_file=phys_soundings, station='tela',
                  times=['2007', '2019']):
    from PW_stations import mean_zwd_over_sound_time
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    sns.set_style('whitegrid')
    ds = mean_zwd_over_sound_time(
        physical_file, ims_path=ims_path, gps_station='tela',
        times=times)
    time_dim = list(set(ds.dims))[0]
    ds = ds.rename({time_dim: 'time'})
    ds = ds.dropna('time')
    ds = ds.sel(time=slice(*times))
    tpw = 'tpw_bet_dagan'
    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    ds.plot.scatter(x=tpw,
                    y='tela_pw',
                    marker='.',
                    s=100.,
                    linewidth=0,
                    alpha=0.5,
                    ax=axes[0])
    axes[0].plot(ds[tpw], ds[tpw], c='r')
    axes[0].legend(['y = x'])
    axes[0].set_xlabel('Total Precipitable Water from Bet-Dagan [mm]')
    axes[0].set_ylabel('Total Precipitable Water from TELA GPS station [mm]')
    resid = ds.tela_pw.values - ds[tpw].values
    sns.distplot(resid, bins=50, color='c', label='residuals', ax=axes[1],
                 kde=False,
                 hist_kws={"linewidth": 1, "alpha": 0.5, "color": "b"})
    rmean = np.mean(resid)
    rmse = np.sqrt(mean_squared_error(ds.tela_pw.values, ds[tpw].values))
    axes[1].axvline(rmean, color='r', linestyle='dashed', linewidth=1)
    axes[1].set_xlabel('Residual distribution[mm]')
    axes[1].text(0.1, 0.85, 'n={}'.format(ds[tpw].size),
                 verticalalignment='top', horizontalalignment='center',
                 transform=axes[1].transAxes, color='b', fontsize=12)
    axes[1].text(0.16, 0.80, 'mean: {:.2f} mm'.format(rmean),
                 verticalalignment='top', horizontalalignment='center',
                 transform=axes[1].transAxes, color='b', fontsize=12)
    axes[1].text(0.16, 0.75, 'RMSE: {:.2f} mm'.format(rmse),
                 verticalalignment='top', horizontalalignment='center',
                 transform=axes[1].transAxes, color='b', fontsize=12)
    fig.suptitle('Precipitable Water comparison for the years {} to {}'.format(*times))
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    return ds


def plot_figure_6(physical_file=phys_soundings, station='tela',
                  times=['2007', '2019']):
    from PW_stations import mean_zwd_over_sound_time
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import matplotlib.dates as mdates
    sns.set_style('whitegrid')
    ds = mean_zwd_over_sound_time(
        physical_file, ims_path=ims_path, gps_station='tela',
        times=times)
    time_dim = list(set(ds.dims))[0]
    ds = ds.rename({time_dim: 'time'})
    ds = ds.dropna('time')
    ds = ds.sel(time=slice(*times))
    df = ds[['zwd_bet_dagan', 'tela_WetZ']].to_dataframe()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    [x.set_xlim([pd.to_datetime(times[0]), pd.to_datetime(times[1])]) for x in axes]
    df.columns = ['Bet_Dagan soundings', 'TELA GPS station']
    sns.scatterplot(data=df, s=20, ax=axes[0], style='x', linewidth=0, alpha=0.8)
    # axes[0].legend(['Bet_Dagan soundings', 'TELA GPS station'])
    df_r = df.iloc[:, 0] - df.iloc[:, 1]
    df_r.columns = ['Residual distribution']
    sns.scatterplot(data=df_r, color = 'k', s=20, ax=axes[1], linewidth=0, alpha=0.5)
    axes[0].grid(b=True, which='major')
    axes[1].grid(b=True, which='major')
    axes[0].set_ylabel('Zenith Wet Delay [cm]')
    axes[1].set_ylabel('Residuals [cm]')
    # axes[0].set_title('Zenith wet delay from Bet-Dagan radiosonde station and TELA GNSS satation')
    sonde_change_x = pd.to_datetime('2013-08-20')
    axes[1].axvline(sonde_change_x, color='red')
    axes[1].annotate('changed sonde type from VIZ MK-II to PTU GPS', (mdates.date2num(sonde_change_x), 10), xytext=(15, 15), 
        textcoords='offset points', arrowprops=dict(arrowstyle='fancy', color='red'), color='red')
    # axes[1].set_aspect(3)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.01)
    return df


def plot_israel_map(gis_path=gis_path):
    """general nice map for israel, need that to plot stations, 
    and temperature field on top of it"""
    import geopandas as gpd
    import contextily as ctx
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("ticks")
    isr_with_yosh = gpd.read_file(gis_path / 'Israel_and_Yosh.shp')
    isr_with_yosh.crs = {'init': 'epsg:4326'}
    # isr_with_yosh = isr_with_yosh.to_crs(epsg=3857)
    ax = isr_with_yosh.plot(alpha=0.0, figsize=(6, 15))
    ctx.add_basemap(
            ax,
            url=ctx.sources.ST_TERRAIN_BACKGROUND,
            crs={
                    'init': 'epsg:4326'})
    return ax
