#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 17:28:04 2020

@author: shlomi
"""
from PW_paths import work_yuval
from matplotlib import rcParams
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from PW_paths import savefig_path
import matplotlib.ticker as ticker
tela_results_path = work_yuval / 'GNSS_stations/tela/rinex/30hr/results'
tela_solutions = work_yuval / 'GNSS_stations/tela/gipsyx_solutions'
sound_path = work_yuval / 'sounding'
phys_soundings = sound_path / 'bet_dagan_phys_sounding_2007-2019.nc'
ims_path = work_yuval / 'IMS_T'
gis_path = work_yuval / 'gis'
dem_path = work_yuval / 'AW3D30'
hydro_path = work_yuval / 'hydro'

rc = {
    'font.family': 'serif',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium'}
for key, val in rc.items():
    rcParams[key] = val
sns.set(rc=rc, style='white')


@ticker.FuncFormatter
def lon_formatter(x, pos):
    if x < 0:
        return r'{0:.1f}$\degree$W'.format(abs(x))
    elif x > 0:
        return r'{0:.1f}$\degree$E'.format(abs(x))
    elif x == 0:
        return r'0$\degree$'


@ticker.FuncFormatter
def lat_formatter(x, pos):
    if x < 0:
        return r'{0:.1f}$\degree$S'.format(abs(x))
    elif x > 0:
        return r'{0:.1f}$\degree$N'.format(abs(x))
    elif x == 0:
        return r'0$\degree$'


def caption(text, color='blue', **kwargs):
    from termcolor import colored
    print(colored('Caption:', color, attrs=['bold'], **kwargs))
    print(colored(text, color, attrs=['bold'], **kwargs))
    return


def plot_figure_rinex_with_map(path=work_yuval, gis_path=gis_path,
                               dem_path=dem_path, save=True):
    from aux_gps import gantt_chart
    import xarray as xr
    import pandas as pd
    import geopandas as gpd
    from PW_stations import produce_geo_gnss_solved_stations
    from aux_gps import geo_annotate
    from ims_procedures import produce_geo_ims
    from matplotlib.colors import ListedColormap
    fig = plt.figure(figsize=(20, 10))
    grid = plt.GridSpec(1, 2, width_ratios=[
        5, 2], wspace=0.1)
    ax_gantt = fig.add_subplot(grid[0, 0])  # plt.subplot(221)
    ax_map = fig.add_subplot(grid[0, 1])  # plt.subplot(122)
#    fig, ax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(20, 6))
    # RINEX gantt chart:
    ds = xr.open_dataset(path / 'GNSS_PW.nc')
    just_pw = [x for x in ds if 'error' not in x]
    ds = ds[just_pw]
    da = ds.to_array('station')
    da['station'] = [x.upper() for x in da.station.values]
    ds = da.to_dataset('station')
    title = 'RINEX files availability for the Israeli GNSS stations'
    ax_gantt = gantt_chart(ds, ax=ax_gantt, fw='normal', title='')
    # Israel gps ims map:
    ax_map = plot_israel_map(gis_path=gis_path, ax=ax_map)
    # overlay with dem data:
    cmap = plt.get_cmap('terrain', 41)
    dem = xr.open_dataarray(dem_path / 'israel_dem_250_500.nc')
    # dem = xr.open_dataarray(dem_path / 'israel_dem_500_1000.nc')
    fg = dem.plot.imshow(ax=ax_map, alpha=0.5, cmap=cmap,
                         vmin=dem.min(), vmax=dem.max(), add_colorbar=False)
    cbar_kwargs = {'fraction': 0.1, 'aspect': 50, 'pad': 0.03}
    cb = plt.colorbar(fg, **cbar_kwargs)
    cb.set_label(label='meters above sea level', size=8, weight='normal')
    cb.ax.tick_params(labelsize=8)
    ax_map.set_xlabel('')
    ax_map.set_ylabel('')
#    print('getting IMS temperature stations metadata...')
#    ims = produce_geo_ims(path=gis_path, freq='10mins', plot=False)
#    ims.plot(ax=ax_map, color='red', edgecolor='black', alpha=0.5)
    # ims, gps = produce_geo_df(gis_path=gis_path, plot=False)
    print('getting solved GNSS israeli stations metadata...')
    gps = produce_geo_gnss_solved_stations(path=gis_path, plot=False)
    gps.plot(ax=ax_map, color='black', edgecolor='black', marker='s',
             alpha=0.7, markersize=25)
    gps_stations = [x for x in gps.index]
    to_plot_offset = ['mrav', 'klhv', 'nzrt', 'katz', 'elro']
    [gps_stations.remove(x) for x in to_plot_offset]
    gps_normal_anno = gps.loc[gps_stations, :]
    gps_offset_anno = gps.loc[to_plot_offset, :]
    geo_annotate(ax_map, gps_normal_anno.lon, gps_normal_anno.lat,
                 gps_normal_anno.index.str.upper(), xytext=(3, 3), fmt=None,
                 c='k', fw='normal', fs=10, colorupdown=False)
    geo_annotate(ax_map, gps_offset_anno.lon, gps_offset_anno.lat,
                 gps_offset_anno.index.str.upper(), xytext=(4, -6), fmt=None,
                 c='k', fw='normal', fs=10, colorupdown=False)
    # plot bet-dagan:
    df = pd.Series([32.00, 34.81]).to_frame().T
    df.index = ['Beit Dagan']
    df.columns = ['lat', 'lon']
    bet_dagan = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon,
                                                                 df.lat),
                                 crs=gps.crs)
    bet_dagan.plot(ax=ax_map, color='black', edgecolor='black',
                   marker='+')
    geo_annotate(ax_map, bet_dagan.lon, bet_dagan.lat,
                 bet_dagan.index, xytext=(4, -6), fmt=None,
                 c='k', fw='normal', fs=10, colorupdown=False)
    plt.legend(['GNSS sites', 'radiosonde'], loc='upper left')
    fig.subplots_adjust(top=0.95,
                        bottom=0.11,
                        left=0.05,
                        right=0.95,
                        hspace=0.2,
                        wspace=0.2)
    # plt.legend(['IMS stations', 'GNSS stations'], loc='upper left')

    filename = 'rinex_israeli_gnss_map.png'
    caption('RINEX files availability for the Israeli GNSS station network at the SOPAC/GARNER website')
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fig


def plot_monthly_pw_and_T(load_path=work_yuval, thresh=50):
    import xarray as xr
    import numpy as np
    pw_and_T = xr.load_dataset(
        load_path /
        'PW_T_monthly_means_thresh_{:.0f}.nc'.format(thresh))
    pw_only = [x for x in pw_and_T if '_T' not in x]
    T_only = [x for x in pw_and_T if '_T' in x]
    pw = pw_and_T[pw_only]
    T = pw_and_T[T_only]
    cwrap = 5
    fg = pw.to_array('station').plot(col='station', col_wrap=cwrap,
                                     color='b', marker='o', alpha=0.7)
    fg.fig.subplots_adjust(wspace=0.0, hspace=0.0, right=0.974)
    col_arr = np.arange(0, len(pw))
    right_side = col_arr[cwrap-1::cwrap]
    for i, ax in enumerate(fg.axes.flatten()):
        title = ax.get_title().split('=')[-1].strip(' ')
        ax.set_title('')
        ax.text(.2, .9, title,
                horizontalalignment='center',
                transform=ax.transAxes)
        ax_t = ax.twinx()
        T['{}_T'.format(title)].plot(color='r', marker='o', alpha=0.7, ax=ax_t)
        ax_t.set_ylim(0, 30)
        fg.fig.canvas.draw()
        labels = [item.get_text() for item in ax_t.get_yticklabels()]
        ax_t.yaxis.set_ticklabels([])
        ax_t.tick_params(axis='y', color='r')
        if i in right_side:
            ax_t.yaxis.set_ticklabels(labels)
            ax_t.tick_params(axis='y', labelcolor='r', color='r')
        ax_t.set_ylabel('')
    return fg


def plot_figure_2(path=tela_results_path, plot='WetZ', save=True):
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
    df[plot].plot(ax=ax, legend=False, color='k')
    ax.fill_between(df.index, df[plot] - df[error_plot],
                    df[plot] + df[error_plot], alpha=0.5)
    ax.grid()
#    ax.set_title('{} from station TELA in {}'.format(
#            desc, df.index[100].strftime('%Y-%m-%d')))
    ax.set_ylabel('WetZ [{}]'.format(unit))
    ax.set_xlabel('')
    ax.grid('on')
    fig.tight_layout()
    filename = 'wetz_tela_daily.png'
    caption('{} from station TELA in {}. Note the error estimation from the GipsyX software(filled)'.format(
            desc, df.index[100].strftime('%Y-%m-%d')))
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return ax


def plot_figure_3(path=tela_solutions, year=2004, field='WetZ',
                  middle_date='11-25', zooms=[10, 3, 0.5], save=True):
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
            axes[j - 1].axvline(x=start, color='r', alpha=0.85, linestyle='--', linewidth=2.0)
            axes[j - 1].axvline(x=end, color='r', alpha=0.85, linestyle='--', linewidth=2.0)
        except IndexError:
            pass
        units = ds.attrs['{}>units'.format(field)]
        sta = da.attrs['station']
        desc = da.attrs['{}>desc'.format(field)]
        ax.set_ylabel('{} [{}]'.format(field, units))
        ax.set_xlabel('')
        ax.grid()
    # fig.suptitle(
    #     '30 hours stitched {} for GNSS station {}'.format(
    #         desc, sta), fontweight='bold')
    fig.tight_layout()
    caption('20, 6 and 1 days of zenith wet delay in 2004 from the TELA GNSS station for the top, middle and bottom figures respectively. The colored segments represent daily solutions while the black dots represent smoothed mean solutions.')
    filename = 'zwd_tela_discon_panel.png'
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    # fig.subplots_adjust(top=0.95)
    return axes


def plot_figure_3_1(path=work_yuval, data='zwd'):
    import xarray as xr
    from aux_gps import plot_tmseries_xarray
    from PW_stations import load_gipsyx_results
    if data == 'zwd':
        tela = load_gipsyx_results('tela', sample_rate='1H', plot_fields=None)
        label = 'ZWD [cm]'
        title = 'Zenith wet delay derived from GPS station TELA'
        ax = plot_tmseries_xarray(tela, 'WetZ')
    elif data == 'pw':
        ds = xr.open_dataset(path / 'GNSS_hourly_PW.nc')
        tela = ds['tela']
        label = 'PW [mm]'
        title = 'Precipitable water derived from GPS station TELA'
        ax = plot_tmseries_xarray(tela)
    ax.set_ylabel(label)
    ax.set_xlim('1996-02', '2019-07')
    ax.set_title(title)
    ax.set_xlabel('')
    ax.figure.tight_layout()
    return ax


def plot_figure_4(physical_file=phys_soundings, model='LR',
                  times=['2007', '2019'], save=True):
    """plot ts-tm relashonship"""
    import xarray as xr
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PW_stations import ML_Switcher
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from aux_gps import get_unique_index
    import numpy as np
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # sns.set_style('whitegrid')
    pds = xr.open_dataset(phys_soundings)
    pds = pds[['Tm', 'Ts']]
    pds = get_unique_index(pds, 'sound_time')
    pds = pds.sel(sound_time=slice(*times))
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    pds.plot.scatter(
        x='Ts',
        y='Tm',
        marker='.',
        s=100.,
        linewidth=0,
        alpha=0.5,
        ax=ax)
    ax.grid()
    ml = ML_Switcher()
    fit_model = ml.pick_model(model)
    X = pds.Ts.values.reshape(-1, 1)
    y = pds.Tm.values
    fit_model.fit(X, y)
    predict = fit_model.predict(X)
    coef = fit_model.coef_[0]
    inter = fit_model.intercept_
    ax.plot(X, predict, c='r')
    bevis_tm = pds.Ts.values * 0.72 + 70.0
    ax.plot(pds.Ts.values, bevis_tm, c='purple')
    ax.legend(['OLS ({:.2f}, {:.2f})'.format(
        coef, inter), 'Bevis 1992 et al. (0.72, 70.0)'])
#    ax.set_xlabel('Surface Temperature [K]')
#    ax.set_ylabel('Water Vapor Mean Atmospheric Temperature [K]')
    ax.set_xlabel('Ts [K]')
    ax.set_ylabel('Tm [K]')
    ax.set_ylim(265, 320)
    axin1 = inset_axes(ax, width="40%", height="40%", loc=2)
    resid = predict - y
    sns.distplot(resid, bins=50, color='k', label='residuals', ax=axin1,
                 kde=False,
                 hist_kws={"linewidth": 1, "alpha": 0.5, "color": "k"})
    axin1.yaxis.tick_right()
    rmean = np.mean(resid)
    rmse = np.sqrt(mean_squared_error(y, predict))
    r2 = r2_score(y, predict)
    axin1.axvline(rmean, color='r', linestyle='dashed', linewidth=1)
    # axin1.set_xlabel('Residual distribution[K]')
    textstr = '\n'.join(['n={}'.format(pds.Ts.size),
                         'RMSE: ', '{:.2f} K'.format(rmse),
                         r'R$^2$: {:.2f}'.format(r2)])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axin1.text(0.05, 0.95, textstr, transform=axin1.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
#    axin1.text(0.2, 0.9, 'n={}'.format(pds.Ts.size),
#               verticalalignment='top', horizontalalignment='center',
#               transform=axin1.transAxes, color='k', fontsize=10)
#    axin1.text(0.78, 0.9, 'RMSE: {:.2f} K'.format(rmse),
#               verticalalignment='top', horizontalalignment='center',
#               transform=axin1.transAxes, color='k', fontsize=10)
    axin1.set_xlim(-15, 15)
    fig.tight_layout()
    filename = 'Bet_dagan_ts_tm_fit.png'
    caption('Water vapor mean temperature (Tm) vs. surface temperature (Ts) of the Bet-dagan radiosonde station. Ordinary least squares linear fit(red) yields the residual distribution with RMSE of 4 K. Bevis(1992) model is plotted(purple) for comparison.')
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return


def plot_figure_5(physical_file=phys_soundings, station='tela',
                  times=['2007', '2019'], wv_name='pw', r2=False, save=True):
    """plot the PW of Bet-dagan vs. PW of gps station"""
    from PW_stations import mean_zwd_over_sound_time
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # sns.set_style('white')
    ds = mean_zwd_over_sound_time(
        physical_file, ims_path=ims_path, gps_station='tela',
        times=times)
    time_dim = list(set(ds.dims))[0]
    ds = ds.rename({time_dim: 'time'})
    ds = ds.dropna('time')
    ds = ds.sel(time=slice(*times))
    tpw = 'tpw_bet_dagan'
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ds.plot.scatter(x=tpw,
                    y='tela_pw',
                    marker='.',
                    s=100.,
                    linewidth=0,
                    alpha=0.5,
                    ax=ax)
    ax.plot(ds[tpw], ds[tpw], c='r')
    ax.legend(['y = x'], loc='upper right')
    if wv_name == 'pw':
        ax.set_xlabel('PW from Bet-Dagan [mm]')
        ax.set_ylabel('PW from TELA GPS station [mm]')
    elif wv_name == 'iwv':
        ax.set_xlabel(r'IWV from Bet-dagan station [kg$\cdot$m$^{-2}$]')
        ax.set_ylabel(r'IWV from TELA GPS station [kg$\cdot$m$^{-2}$]')
    ax.grid()
    axin1 = inset_axes(ax, width="40%", height="40%", loc=2)
    resid = ds.tela_pw.values - ds[tpw].values
    sns.distplot(resid, bins=50, color='k', label='residuals', ax=axin1,
                 kde=False,
                 hist_kws={"linewidth": 1, "alpha": 0.5, "color": "k"})
    axin1.yaxis.tick_right()
    rmean = np.mean(resid)
    rmse = np.sqrt(mean_squared_error(ds[tpw].values, ds.tela_pw.values))
    r2s = r2_score(ds[tpw].values, ds.tela_pw.values)
    axin1.axvline(rmean, color='r', linestyle='dashed', linewidth=1)
    # axin1.set_xlabel('Residual distribution[mm]')
    if wv_name == 'pw':
        if r2:
            textstr = '\n'.join(['n={}'.format(ds[tpw].size),
                                 'bias: {:.2f} mm'.format(rmean),
                                 'RMSE: {:.2f} mm'.format(rmse),
                                 r'R$^2$: {:.2f}'.format(r2s)])
        else:
            textstr = '\n'.join(['n={}'.format(ds[tpw].size),
                                 'bias: {:.2f} mm'.format(rmean),
                                 'RMSE: {:.2f} mm'.format(rmse)])
    elif wv_name == 'iwv':
        if r2:
            textstr = '\n'.join(['n={}'.format(ds[tpw].size),
                                 r'bias: {:.2f} kg$\cdot$m$^{{-2}}$'.format(rmean),
                                 r'RMSE: {:.2f} kg$\cdot$m$^{{-2}}$'.format(rmse),
                                 r'R$^2$: {:.2f}'.format(r2s)])
        else:
            textstr = '\n'.join(['n={}'.format(ds[tpw].size),
                                 r'bias: {:.2f} kg$\cdot$m$^{{-2}}$'.format(rmean),
                                 r'RMSE: {:.2f} kg$\cdot$m$^{{-2}}$'.format(rmse)])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axin1.text(0.05, 0.95, textstr, transform=axin1.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
#
#    axin1.text(0.2, 0.95, 'n={}'.format(ds[tpw].size),
#               verticalalignment='top', horizontalalignment='center',
#               transform=axin1.transAxes, color='k', fontsize=10)
#    axin1.text(0.3, 0.85, 'bias: {:.2f} mm'.format(rmean),
#               verticalalignment='top', horizontalalignment='center',
#               transform=axin1.transAxes, color='k', fontsize=10)
#    axin1.text(0.35, 0.75, 'RMSE: {:.2f} mm'.format(rmse),
#               verticalalignment='top', horizontalalignment='center',
#               transform=axin1.transAxes, color='k', fontsize=10)
    # fig.suptitle('Precipitable Water comparison for the years {} to {}'.format(*times))
    fig.tight_layout()
    caption('PW from TELA GNSS station vs. PW from Bet-dagan radiosonde station in {}-{}. A 45 degree line is plotted(red) for comparison. Note the skew in the residual distribution with an RMSE of 4.37 mm.'.format(times[0], times[1]))
    # fig.subplots_adjust(top=0.95)
    filename = 'Bet_dagan_tela_pw_compare_{}-{}.png'.format(times[0], times[1])
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return ds


def plot_figure_6(physical_file=phys_soundings, station='tela',
                  times=['2007', '2019'], save=True):
    from PW_stations import mean_zwd_over_sound_time
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import matplotlib.dates as mdates
    # sns.set_style('whitegrid')
    ds = mean_zwd_over_sound_time(
        physical_file, ims_path=ims_path, gps_station='tela',
        times=times)
    time_dim = list(set(ds.dims))[0]
    ds = ds.rename({time_dim: 'time'})
    ds = ds.dropna('time')
    ds = ds.sel(time=slice(*times))
    df = ds[['zwd_bet_dagan', 'tela_WetZ']].to_dataframe()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    [x.set_xlim([pd.to_datetime(times[0]), pd.to_datetime(times[1])])
     for x in axes]
    df.columns = ['Bet_Dagan soundings', 'TELA GPS station']
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
    axes[0].set_ylabel('Zenith Wet Delay [cm]')
    axes[1].set_ylabel('Residuals [cm]')
    # axes[0].set_title('Zenith wet delay from Bet-Dagan radiosonde station and TELA GNSS satation')
    sonde_change_x = pd.to_datetime('2013-08-20')
    axes[1].axvline(sonde_change_x, color='red')
    axes[1].annotate(
        'changed sonde type from VIZ MK-II to PTU GPS',
        (mdates.date2num(sonde_change_x),
         10),
        xytext=(
            15,
            15),
        textcoords='offset points',
        arrowprops=dict(
            arrowstyle='fancy',
            color='red'),
        color='red')
    # axes[1].set_aspect(3)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.01)
    filename = 'Bet_dagan_tela_zwd_compare.png'
    caption('Top: zenith wet delay from Bet-dagan radiosonde station(blue circles) and from TELA GNSS station(orange x) in 2007-2019. Bottom: residuals. Note the residuals become constrained from 08-2013 probebly due to an equipment change.')
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return df


def plot_israel_map(gis_path=gis_path, rc=rc, ax=None):
    """general nice map for israel, need that to plot stations,
    and temperature field on top of it"""
    import geopandas as gpd
    import contextily as ctx
    import seaborn as sns
    sns.set_style("ticks", rc=rc)
    isr_with_yosh = gpd.read_file(gis_path / 'Israel_and_Yosh.shp')
    isr_with_yosh.crs = {'init': 'epsg:4326'}
    # isr_with_yosh = isr_with_yosh.to_crs(epsg=3857)
    if ax is None:
        ax = isr_with_yosh.plot(alpha=0.0, figsize=(6, 15))
    else:
        isr_with_yosh.plot(alpha=0.0, ax=ax)
    ctx.add_basemap(
            ax,
            url=ctx.sources.ST_TERRAIN_BACKGROUND,
            crs={
                    'init': 'epsg:4326'})
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(top=True, bottom=True, left=True, right=True,
                   direction='out', labelsize=10)
    return ax


def plot_figure_7(gis_path=gis_path, save=True):
    from PW_stations import produce_geo_gnss_solved_stations
    from aux_gps import geo_annotate
    from ims_procedures import produce_geo_ims
    import matplotlib.pyplot as plt
    ax = plot_israel_map(gis_path)
    print('getting IMS temperature stations metadata...')
    ims = produce_geo_ims(path=gis_path, freq='10mins', plot=False)
    ims.plot(ax=ax, color='red', edgecolor='black', alpha=0.5)
    # ims, gps = produce_geo_df(gis_path=gis_path, plot=False)
    print('getting solved GNSS israeli stations metadata...')
    gps = produce_geo_gnss_solved_stations(path=gis_path, plot=False)
    gps.plot(ax=ax, color='green', edgecolor='black')
    gps_stations = [x for x in gps.index]
    to_plot_offset = ['mrav', 'klhv']
    [gps_stations.remove(x) for x in to_plot_offset]
    gps_normal_anno = gps.loc[gps_stations, :]
    gps_offset_anno = gps.loc[to_plot_offset, :]
    geo_annotate(ax, gps_normal_anno.lon, gps_normal_anno.lat,
                 gps_normal_anno.index, xytext=(3, 3), fmt=None,
                 c='k', fw='bold', fs=None, colorupdown=False)
    geo_annotate(ax, gps_offset_anno.lon, gps_offset_anno.lat,
                 gps_offset_anno.index, xytext=(4, -6), fmt=None,
                 c='k', fw='bold', fs=None, colorupdown=False)
#    plt.legend(['IMS stations', 'GNSS stations'],
#           prop={'size': 10}, bbox_to_anchor=(-0.15, 1.0),
#           title='Stations')
#    plt.legend(['IMS stations', 'GNSS stations'],
#               prop={'size': 10}, loc='upper left')
    plt.legend(['IMS stations', 'GNSS stations'], loc='upper left')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    caption('24 GNSS stations from the israeli network(green) and 88 IMS 10 mins automated stations(red).')
    filename = 'gnss_ims_stations_map.png'
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return ax


def plot_figure_8(ims_path=ims_path, dt='2013-10-19T22:00:00', save=True):
    from aux_gps import path_glob
    import xarray as xr
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    # from matplotlib import rc

    def choose_dt_and_lapse_rate(tdf, dt, T_alts, lapse_rate):
        ts = tdf.loc[dt, :]
        # dt_col = dt.strftime('%Y-%m-%d %H:%M')
        # ts.name = dt_col
        # Tloc_df = Tloc_df.join(ts, how='right')
        # Tloc_df = Tloc_df.dropna(axis=0)
        ts_vs_alt = pd.Series(ts.values, index=T_alts)
        ts_vs_alt_for_fit = ts_vs_alt.dropna()
        [a, b] = np.polyfit(ts_vs_alt_for_fit.index.values,
                            ts_vs_alt_for_fit.values, 1)
        if lapse_rate == 'auto':
            lapse_rate = np.abs(a) * 1000
            if lapse_rate < 5.0:
                lapse_rate = 5.0
            elif lapse_rate > 10.0:
                lapse_rate = 10.0
        return ts_vs_alt, lapse_rate

    # rc('text', usetex=False)
    # rc('text',latex.unicode=False)
    glob_str = 'IMS_TD_israeli_10mins*.nc'
    file = path_glob(ims_path, glob_str=glob_str)[0]
    ds = xr.open_dataset(file)
    time_dim = list(set(ds.dims))[0]
    # slice to a starting year(1996?):
    ds = ds.sel({time_dim: slice('1996', None)})
    # years = sorted(list(set(ds[time_dim].dt.year.values)))
    # get coords and alts of IMS stations:
    T_alts = np.array([ds[x].attrs['station_alt'] for x in ds])
#    T_lats = np.array([ds[x].attrs['station_lat'] for x in ds])
#    T_lons = np.array([ds[x].attrs['station_lon'] for x in ds])
    print('loading IMS_TD of israeli stations 10mins freq..')
    # transform to dataframe and add coords data to df:
    tdf = ds.to_dataframe()
    # dt_col = dt.strftime('%Y-%m-%d %H:%M')
    dt = pd.to_datetime(dt)
    # prepare the ims coords and temp df(Tloc_df) and the lapse rate:
    ts_vs_alt, lapse_rate = choose_dt_and_lapse_rate(tdf, dt, T_alts, 'auto')
    fig, ax_lapse = plt.subplots(figsize=(10, 6))
    sns.regplot(x=ts_vs_alt.index, y=ts_vs_alt.values, color='r',
                scatter_kws={'color': 'k'}, ax=ax_lapse)
    # suptitle = dt.strftime('%Y-%m-%d %H:%M')
    ax_lapse.set_xlabel('Altitude [m]')
    ax_lapse.set_ylabel(r'Temperature [$\degree$C]')
    ax_lapse.text(0.5, 0.95, r'Lapse rate: {:.2f} $\degree$C/km'.format(lapse_rate),
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax_lapse.transAxes, color='k')
    ax_lapse.grid()
    # ax_lapse.set_title(suptitle, fontsize=14, fontweight='bold')
    fig.tight_layout()
    filename = 'ims_lapse_rate_example.png'
    caption('Temperature vs. altitude for 10 PM in 2013-10-19 for all automated 10 mins IMS stations. The lapse rate is calculated using ordinary least squares linear fit.')
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return ax_lapse


def plot_figure_9(hydro_path=hydro_path, gis_path=gis_path, pw_anom=False,
                  max_flow_thresh=None, wv_name='pw', save=True):
    from hydro_procedures import get_hydro_near_GNSS
    from hydro_procedures import loop_over_gnss_hydro_and_aggregate
    import matplotlib.pyplot as plt
    df = get_hydro_near_GNSS(
        radius=5,
        hydro_path=hydro_path,
        gis_path=gis_path,
        plot=False)
    ds = loop_over_gnss_hydro_and_aggregate(df, pw_anom=pw_anom,
                                            max_flow_thresh=max_flow_thresh,
                                            hydro_path=hydro_path,
                                            work_yuval=work_yuval, ndays=3,
                                            plot=False, plot_all=False)
    names = [x for x in ds.data_vars]
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in names:
        ds.mean('station').mean('tide_start')[name].plot.line(
            marker='.', linewidth=0., ax=ax)
    ax.set_xlabel('Days before tide event')
    ax.grid()
    hstations = [ds[x].attrs['hydro_stations'] for x in ds.data_vars]
    events = [ds[x].attrs['total_events'] for x in ds.data_vars]
    fmt = list(zip(names, hstations, events))
    ax.legend(['{} with {} stations ({} total events)'.format(x, y, z)
               for x, y, z in fmt])
    fig.canvas.draw()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    xlabels = [x.replace('−', '') for x in labels]
    ax.set_xticklabels(xlabels)
    fig.canvas.draw()
    if wv_name == 'pw':
        if pw_anom:
            ax.set_ylabel('PW anomalies [mm]')
        else:
            ax.set_ylabel('PW [mm]')
    elif wv_name == 'iwv':
        if pw_anom:
            ax.set_ylabel(r'IWV anomalies [kg$\cdot$m$^{-2}$]')
        else:
            ax.set_ylabel(r'IWV [kg$\cdot$m$^{-2}$]')
    fig.tight_layout()
#    if pw_anom:
#        title = 'Mean PW anomalies for tide stations near all GNSS stations'
#    else:
#        title = 'Mean PW for tide stations near all GNSS stations'
#    if max_flow_thresh is not None:
#        title += ' (max_flow > {} m^3/sec)'.format(max_flow_thresh)
#    ax.set_title(title)
    if pw_anom:
        filename = 'hydro_tide_lag_pw_anom.png'
        if max_flow_thresh:
            filename = 'hydro_tide_lag_pw_anom_max{}.png'.format(max_flow_thresh)
    else:
        filename = 'hydro_tide_lag_pw.png'
        if max_flow_thresh:
            filename = 'hydro_tide_lag_pw_anom_max{}.png'.format(max_flow_thresh)
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return ax


def produce_table_1():
    from PW_stations import produce_geo_gnss_solved_stations
    import pandas as pd
    df_gnss = produce_geo_gnss_solved_stations(plot=False)
    df_gnss['ID'] = df_gnss.index.str.upper()
    pd.options.display.float_format = '{:.2f}'.format
    df = df_gnss[['name', 'ID', 'lat', 'lon', 'alt']]
    df['alt'] = df['alt'].astype(int)
    cols = ['GNSS Sites', 'Site ID', 'Latitude [N]',
            'Longitude [W]', 'Altitude [m a.s.l]']
    df.columns = cols
    df.loc['spir', 'GNSS Sites'] = 'Sapir'
    print(df.to_latex(index=False))
    return df


def produce_table_2(thresh=50):
    from PW_stations import produce_pw_statistics
    df = produce_pw_statistics(thresh=thresh)
    print(df.to_latex(index=False))
    return df


def plot_grp_anomlay_heatmap(load_path=work_yuval, gis_path=gis_path,
                             thresh=None, grp='hour', season=None,
                             n_clusters=4, save=True, title=False):
    import xarray as xr
    import seaborn as sns
    import numpy as np
    from PW_stations import group_anoms_and_cluster
    from aux_gps import geo_annotate
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import ListedColormap
    from palettable.scientific import diverging as divsci
    from PW_stations import produce_geo_gnss_solved_stations
    div_cmap = divsci.Vik_20.mpl_colormap
    dem_path = load_path / 'AW3D30'

    def weighted_average(grp_df, weights_col='weights'):
        return grp_df._get_numeric_data().multiply(
            grp_df[weights_col], axis=0).sum() / grp_df[weights_col].sum()

    df, labels_sorted, weights = group_anoms_and_cluster(
            load_path=load_path, thresh=thresh, grp=grp, season=season,
            n_clusters=n_clusters)
    # create figure and subplots axes:
    fig = plt.figure(figsize=(15, 10))
    if title:
        if season is not None:
            fig.suptitle('Precipitable water {}ly anomalies analysis for {} season'.format(grp, season))
        else:
            fig.suptitle('Precipitable water {}ly anomalies analysis'.format(grp))
    grid = plt.GridSpec(
        2, 2, width_ratios=[
            3, 2], height_ratios=[
            4, 1], wspace=0.1, hspace=0)
    ax_heat = fig.add_subplot(grid[0, 0])  # plt.subplot(221)
    ax_group = fig.add_subplot(grid[1, 0])  # plt.subplot(223)
    ax_map = fig.add_subplot(grid[0:, 1])  # plt.subplot(122)
    # get the camp and zip it to groups and produce dictionary:
    cmap = plt.get_cmap("Accent")
    # cmap = plt.get_cmap("Set2_r")
    # cmap = ListedColormap(cmap.colors[::-1])
    groups = list(set(labels_sorted.values()))
    palette = dict(zip(groups, [cmap(x) for x in range(len(groups))]))
    label_cmap_dict = dict(zip(labels_sorted.keys(),
                               [palette[x] for x in labels_sorted.values()]))
    cm = ListedColormap([x for x in palette.values()])
    # plot heatmap and colorbar:
    cbar_ax = fig.add_axes([0.57, 0.24, 0.01, 0.69])  #[left, bottom, width,
    # height]
    ax_heat = sns.heatmap(
            df.T,
            center=0.0,
            cmap=div_cmap,
            yticklabels=True,
            ax=ax_heat,
            cbar_ax=cbar_ax,
            cbar_kws={'label': '[mm]'})
    # activate top ticks and tickslabales:
    ax_heat.xaxis.set_tick_params(top='on', labeltop='on')
    # emphasize the yticklabels (stations):
    ax_heat.yaxis.set_tick_params(left='on')
    ax_heat.set_yticklabels(ax_heat.get_ymajorticklabels(),
        fontweight = 'bold', fontsize=10)
    # paint ytick labels with categorical cmap:
    boxes = [dict(facecolor=x, pad=0.05, alpha=0.6)
             for x in label_cmap_dict.values()]
    ylabels = [x for x in ax_heat.yaxis.get_ticklabels()]
    for label, box in zip(ylabels, boxes):
        label.set_bbox(box)
    # rotate xtick_labels:
#    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=0,
#                            fontsize=10)
    # plot summed groups (with weights):
    df_groups = df.T
    df_groups['groups'] = pd.Series(labels_sorted)
    df_groups['weights'] = weights
    df_groups = df_groups.groupby('groups').apply(weighted_average)
    df_groups.drop(['groups', 'weights'], axis=1, inplace=True)
    df_groups.T.plot(ax=ax_group, legend=False, cmap=cm)
    ax_group.grid()
    group_limit = ax_heat.get_xlim()
    ax_group.set_xlim(group_limit)
    ax_group.set_ylabel('[mm]')
    # set ticks and align with heatmap axis (move by 0.5):
    ax_group.set_xticks(df.index.values)
    offset = 1
    ax_group.xaxis.set(ticks=np.arange(offset / 2.,
                                       max(df.index.values) + 1 - min(df.index.values),
                                       offset),
                       ticklabels=df.index.values)
    # move the lines also by 0.5 to align with heatmap:
    lines = ax_group.lines  # get the lines
    [x.set_xdata(x.get_xdata() - min(df.index.values) + 0.5) for x in lines]
    # plot israel map:
    ax_map = plot_israel_map(gis_path=gis_path, ax=ax_map)
    # overlay with dem data:
    cmap = plt.get_cmap('terrain', 41)
    dem = xr.open_dataarray(dem_path / 'israel_dem_250_500.nc')
    # dem = xr.open_dataarray(dem_path / 'israel_dem_500_1000.nc')
    im = dem.plot.imshow(ax=ax_map, alpha=0.5, cmap=cmap,
                         vmin=dem.min(), vmax=dem.max(), add_colorbar=False)
    cbar_kwargs = {'fraction': 0.1, 'aspect': 50, 'pad': 0.03}
    cb = fig.colorbar(im, ax=ax_map, **cbar_kwargs)
    # cb = plt.colorbar(fg, **cbar_kwargs)
    cb.set_label(label='meters above sea level', size=8, weight='normal')
    cb.ax.tick_params(labelsize=8)
    ax_map.set_xlabel('')
    ax_map.set_ylabel('')
    print('getting solved GNSS israeli stations metadata...')
    gps = produce_geo_gnss_solved_stations(path=gis_path, plot=False)
    gps.index = gps.index.str.upper()
    gps = gps.loc[[x for x in df.columns], :]
    gps['group'] = pd.Series(labels_sorted)
    gps.plot(ax=ax_map, column='group', categorical=True, marker='o',
             edgecolor='black', cmap=cm, s=45, legend=True, alpha=1.0,
             legend_kwds={'prop': {'size': 10}, 'fontsize': 14,
                          'loc': 'upper left', 'title': 'Groups'})
    # ax_map.set_title('Groupings of {}ly anomalies'.format(grp))
    # annotate station names in map:
    geo_annotate(ax_map, gps.lon, gps.lat,
                 gps.index, xytext=(6, 6), fmt=None,
                 c='k', fw='bold', fs=10, colorupdown=False)
#    plt.legend(['IMS stations', 'GNSS stations'],
#           prop={'size': 10}, bbox_to_anchor=(-0.15, 1.0),
#           title='Stations')
#    plt.legend(prop={'size': 10}, loc='upper left')
    # plt.tight_layout()
    plt.subplots_adjust(top=0.92,
                        bottom=0.065,
                        left=0.065,
                        right=0.915,
                        hspace=0.19,
                        wspace=0.215)
    filename = 'pw_{}ly_anoms_{}_clusters_with_map.png'.format(grp, n_clusters)
    if save:
#        plt.savefig(savefig_path / filename, bbox_inches='tight')
        plt.savefig(savefig_path / filename, orientation='landscape')
    return df

