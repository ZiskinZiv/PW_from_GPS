#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 17:28:04 2020

@author: shlomi
"""
from PW_paths import work_yuval
from matplotlib import rcParams
import seaborn as sns
tela_results_path = work_yuval / 'GNSS_stations/tela/rinex/30hr/results'
tela_solutions = work_yuval / 'GNSS_stations/tela/gipsyx_solutions'
sound_path = work_yuval / 'sounding'
phys_soundings = sound_path / 'bet_dagan_phys_sounding_2007-2019.nc'
ims_path = work_yuval / 'IMS_T'
gis_path = work_yuval / 'gis'
hydro_path = work_yuval / 'hydro'
rc = {
    'font.family': 'serif',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium'}
for key, val in rc.items():
    rcParams[key] = val
sns.set(rc=rc, style='white')


def plot_figure_1(path=work_yuval):
    from aux_gps import gantt_chart
    import xarray as xr
    ds = xr.open_dataset(path / 'GNSS_PW.nc')
    just_pw = [x for x in ds if 'error' not in x]
    ds = ds[just_pw]
    title = 'RINEX files availability for the Israeli GNSS stations'
    ax = gantt_chart(ds, fw='normal', title=title)
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
    df[plot].plot(ax=ax, legend=False, color='k')
    ax.fill_between(df.index, df[plot] - df[error_plot],
                    df[plot] + df[error_plot], alpha=0.5)
    ax.grid()
    ax.set_title('{} from station TELA in {}'.format(
            desc, df.index[100].strftime('%Y-%m-%d')))
    ax.set_ylabel('WetZ [{}]'.format(unit))
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
                  times=['2007', '2019']):
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
    return


def plot_figure_5(physical_file=phys_soundings, station='tela',
                  times=['2007', '2019'], wv_name='pw'):
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
    r2 = r2_score(ds[tpw].values, ds.tela_pw.values)
    axin1.axvline(rmean, color='r', linestyle='dashed', linewidth=1)
    # axin1.set_xlabel('Residual distribution[mm]')
    if wv_name == 'pw':
        textstr = '\n'.join(['n={}'.format(ds[tpw].size),
                             'bias: {:.2f} mm'.format(rmean),
                             'RMSE: {:.2f} mm'.format(rmse),
                             r'R$^2$: {:.2f}'.format(r2)])
    elif wv_name == 'iwv':
        textstr = '\n'.join(['n={}'.format(ds[tpw].size),
                             r'bias: {:.2f} kg$\cdot$m$^{{-2}}$'.format(rmean),
                             r'RMSE: {:.2f} kg$\cdot$m$^{{-2}}$'.format(rmse),
                             r'R$^2$: {:.2f}'.format(r2)])
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
    # fig.subplots_adjust(top=0.95)
    return ds


def plot_figure_6(physical_file=phys_soundings, station='tela',
                  times=['2007', '2019']):
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
    return df


def plot_israel_map(gis_path=gis_path, rc=rc):
    """general nice map for israel, need that to plot stations,
    and temperature field on top of it"""
    import geopandas as gpd
    import contextily as ctx
    import seaborn as sns
    sns.set_style("ticks", rc=rc)
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


def plot_figure_7(gis_path=gis_path):
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
    return ax


def plot_figure_8(ims_path=ims_path, dt='2013-10-19T22:00:00'):
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
    return ax_lapse


def plot_figure_9(hydro_path=hydro_path, gis_path=gis_path, pw_anom=False,
                  max_flow_thresh=None, wv_name='pw'):
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
    return ax

