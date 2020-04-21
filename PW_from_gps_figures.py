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
# sns.set(rc=rc, style='white')


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


def qualitative_cmap(n=2):
    import matplotlib.colors as mcolors
    if n == 2:
        colorsList = [mcolors.BASE_COLORS['r'], mcolors.BASE_COLORS['g']]
        cmap = mcolors.ListedColormap(colorsList)
    elif n == 4:
        colorsList = [
                mcolors.BASE_COLORS['r'],
                mcolors.BASE_COLORS['g'],
                mcolors.BASE_COLORS['c'],
                mcolors.BASE_COLORS['m']]
        cmap = mcolors.ListedColormap(colorsList)
    elif n == 5:
        colorsList = [
                mcolors.BASE_COLORS['r'],
                mcolors.BASE_COLORS['g'],
                mcolors.BASE_COLORS['c'],
                mcolors.BASE_COLORS['m'],
                mcolors.BASE_COLORS['b']]
        cmap = mcolors.ListedColormap(colorsList)
    return cmap


def caption(text, color='blue', **kwargs):
    from termcolor import colored
    print(colored('Caption:', color, attrs=['bold'], **kwargs))
    print(colored(text, color, attrs=['bold'], **kwargs))
    return


def plot_figure_rinex_with_map(path=work_yuval, gis_path=gis_path,
                               dem_path=dem_path, save=True):
    # TODO: add box around merged stations and removed stations
    # TODO: add color map labels to stations removed and merged
    from aux_gps import gantt_chart
    import xarray as xr
    import pandas as pd
    import geopandas as gpd
    from PW_stations import produce_geo_gnss_solved_stations
    from aux_gps import geo_annotate
    from ims_procedures import produce_geo_ims
    from matplotlib.colors import ListedColormap
    from aux_gps import path_glob
    fig = plt.figure(figsize=(20, 10))
    grid = plt.GridSpec(1, 2, width_ratios=[
        5, 2], wspace=0.1)
    ax_gantt = fig.add_subplot(grid[0, 0])  # plt.subplot(221)
    ax_map = fig.add_subplot(grid[0, 1])  # plt.subplot(122)
#    fig, ax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(20, 6))
    # RINEX gantt chart:
    file = path_glob(path, 'ZWD_unselected_israel_*.nc')[-1]
    ds = xr.open_dataset(file)
    just_pw = [x for x in ds if 'error' not in x]
    ds = ds[just_pw]
    da = ds.to_array('station')
    da['station'] = [x.upper() for x in da.station.values]
    ds = da.to_dataset('station')
    title = 'Daily RINEX files availability for the Israeli GNSS stations'
    ax_gantt = gantt_chart(
        ds,
        ax=ax_gantt,
        fw='normal',
        title='',
        pe_dict=None)
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
    removed = ['hrmn', 'nizn', 'spir']
    merged = ['klhv', 'lhav', 'mrav', 'gilb']
    gps_list = [x for x in gps.index if x not in merged and x not in removed]
    gps.loc[gps_list, :].plot(ax=ax_map, color='black', edgecolor='black', marker='s',
             alpha=0.7, markersize=25)
    gps.loc[removed, :].plot(ax=ax_map, color='black', edgecolor='black', marker='s',
            alpha=1.0, markersize=25, facecolor='white')
    gps.loc[merged, :].plot(ax=ax_map, color='black', edgecolor='r', marker='s',
            alpha=0.7, markersize=25)
    gps_stations = [x for x in gps.index]
    to_plot_offset = ['mrav', 'klhv', 'nzrt', 'katz', 'elro']
#    [gps_stations.remove(x) for x in to_plot_offset]
#    gps_normal_anno = gps.loc[gps_stations, :]
#    gps_offset_anno = gps.loc[to_plot_offset, :]

    for x, y, label in zip(gps.loc[gps_stations, :].lon, gps.loc[gps_stations,
                                                                 :].lat, gps.loc[gps_stations, :].index.str.upper()):
        if label.lower() in to_plot_offset:
            ax_map.annotate(label, xy=(x, y), xytext=(4, -6),
                            textcoords="offset points", color='k',
                            fontweight='normal', fontsize=10)
        else:
            ax_map.annotate(label, xy=(x, y), xytext=(3, 3),
                            textcoords="offset points", color='k',
                            fontweight='normal', fontsize=10)
#    geo_annotate(ax_map, gps_normal_anno.lon, gps_normal_anno.lat,
#                 gps_normal_anno.index.str.upper(), xytext=(3, 3), fmt=None,
#                 c='k', fw='normal', fs=10, colorupdown=False)
#    geo_annotate(ax_map, gps_offset_anno.lon, gps_offset_anno.lat,
#                 gps_offset_anno.index.str.upper(), xytext=(4, -6), fmt=None,
#                 c='k', fw='normal', fs=10, colorupdown=False)
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
    plt.legend(['GNSS \nreceiver sites',
                'removed \nGNSS sites',
                'merged \nGNSS sites',
                'radiosonde\nstation'],
               loc='upper left', framealpha=0.7, fancybox=True,
               handletextpad=0.2, handlelength=1.5)
    fig.subplots_adjust(top=0.95,
                        bottom=0.11,
                        left=0.05,
                        right=0.95,
                        hspace=0.2,
                        wspace=0.2)
    # plt.legend(['IMS stations', 'GNSS stations'], loc='upper left')

    filename = 'rinex_israeli_gnss_map.png'
    caption('Daily RINEX files availability for the Israeli GNSS station network at the SOPAC/GARNER website')
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fig


def plot_means_box_plots(path=work_yuval, thresh=50, kind='box',
                         x='month', col_wrap=5, ylimits=None,
                         xlimits=None, anoms=True, bins=None,
                         season=None, attrs_plot=True, save=True, ds_input=None):
    import xarray as xr
    pw = xr.open_dataset(
                work_yuval /
                'GNSS_PW_thresh_{:.0f}_homogenized.nc'.format(thresh))
    pw = pw[[x for x in pw.data_vars if '_error' not in x]]
    attrs = [x.attrs for x in pw.data_vars.values()]
    if x == 'month':
        pw = xr.load_dataset(
                work_yuval /
                'GNSS_PW_monthly_thresh_{:.0f}_homogenized.nc'.format(thresh))
        # pw = pw.resample(time='MS').mean('time')
    elif x == 'hour':
        # pw = pw.resample(time='1H').mean('time')
        # pw = pw.groupby('time.hour').mean('time')
        pw = xr.load_dataset(work_yuval / 'GNSS_PW_hourly_thresh_{:.0f}_homogenized.nc'.format(thresh))
        pw = pw[[x for x in pw.data_vars if '_error' not in x]]
        # first remove long term monthly means:
        if anoms:
            pw = xr.load_dataset(work_yuval / 'GNSS_PW_hourly_anoms_thresh_{:.0f}_homogenized.nc'.format(thresh))
            # pw = pw.groupby('time.dayofyear') - pw.groupby('time.dayofyear').mean('time')
    elif x == 'day':
        # pw = pw.resample(time='1H').mean('time')
        # pw = pw.groupby('time.hour').mean('time')
        pw = xr.load_dataset(work_yuval / 'GNSS_PW_daily_thresh_{:.0f}_homogenized.nc'.format(thresh))
        pw = pw[[x for x in pw.data_vars if '_error' not in x]]
        # first remove long term monthly means:
        if anoms:
            # pw = pw.groupby('time.month') - pw.groupby('time.month').mean('time')
            pw = pw.groupby('time.dayofyear') - pw.groupby('time.dayodyear').mean('time')
    if season is not None:
        if season != 'all':
            print('{} season is selected'.format(season))
            pw = pw.sel(time=pw['time.season'] == season)
            all_seas = False
        else:
            print('all seasons selected')
            all_seas = True
    else:
        all_seas = False
    for i, da in enumerate(pw.data_vars):
        pw[da].attrs = attrs[i]
    if not attrs_plot:
        attrs = None
    if ds_input is not None:
        # be carful!:
        pw = ds_input
    fg = plot_multi_box_xr(pw, kind=kind, x=x, col_wrap=col_wrap,
                           ylimits=ylimits, xlimits=xlimits, attrs=attrs,
                           bins=bins, all_seasons=all_seas)
    attrs = [x.attrs for x in pw.data_vars.values()]
    for i, ax in enumerate(fg.axes.flatten()):
        try:
            mean_years = float(attrs[i]['mean_years'])
#            print(i)
            # print(mean_years)
        except IndexError:
            ax.set_axis_off()
            pass
    if kind != 'hist':
        [fg.axes[x, 0].set_ylabel('PW [mm]') for x in range(len(fg.axes[:, 0]))]
#    [fg.axes[-1, x].set_xlabel('month') for x in range(len(fg.axes[-1, :]))]
    fg.fig.subplots_adjust(top=0.98,
                           bottom=0.05,
                           left=0.025,
                           right=0.985,
                           hspace=0.27,
                           wspace=0.215)
    if season is not None:
        filename = 'pw_{}ly_means_{}_seas_{}.png'.format(x, kind, season)
    else:
        filename = 'pw_{}ly_means_{}.png'.format(x, kind)
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_multi_box_xr(pw, kind='violin', x='month', sharex=False, sharey=False,
                      col_wrap=5, ylimits=None, xlimits=None, attrs=None,
                      bins=None, all_seasons=False):
    import xarray as xr
    pw = pw.to_array('station')
    fg = xr.plot.FacetGrid(pw, col='station', col_wrap=col_wrap, sharex=sharex,
                           sharey=sharey)
    for i, (sta, ax) in enumerate(zip(pw['station'].values, fg.axes.flatten())):
        pw_sta = pw.sel(station=sta).reset_coords(drop=True)
        if all_seasons:
            pw_seas = pw_sta.sel(time=pw_sta['time.season']=='DJF')
            df = pw_seas.to_dataframe(sta)
            plot_box_df(df, ax=ax, x=x, title=sta, ylabel='', kind=kind,
                        ylimits=ylimits, xlimits=xlimits, attrs=None, bins=bins)
            pw_seas = pw_sta.sel(time=pw_sta['time.season']=='MAM')
            df = pw_seas.to_dataframe(sta)
            plot_box_df(df, ax=ax, x=x, title=sta, ylabel='', kind=kind,
                        ylimits=ylimits, xlimits=xlimits, attrs=None, bins=bins)
            pw_seas = pw_sta.sel(time=pw_sta['time.season']=='JJA')
            df = pw_seas.to_dataframe(sta)
            plot_box_df(df, ax=ax, x=x, title=sta, ylabel='', kind=kind,
                        ylimits=ylimits, xlimits=xlimits, attrs=None, bins=bins)
            pw_seas = pw_sta.sel(time=pw_sta['time.season']=='SON')
            df = pw_seas.to_dataframe(sta)
            plot_box_df(df, ax=ax, x=x, title=sta, ylabel='', kind=kind,
                        ylimits=ylimits, xlimits=xlimits, attrs=attrs[i], bins=bins)
            if sta == 'nrif' or sta == 'elat':
                ax.legend(['DJF', 'MAM', 'JJA', 'SON'],
                          prop={'size':8}, loc='upper center', framealpha=0.5, fancybox=True)
            else:
                ax.legend(['DJF', 'MAM', 'JJA', 'SON'],
                          prop={'size':8}, loc='best', framealpha=0.5, fancybox=True)
        else:
        # if x == 'hour':
        #     # remove seasonal signal:
        #     pw_sta = pw_sta.groupby('time.dayofyear') - pw_sta.groupby('time.dayofyear').mean('time')
        # elif x == 'month':
        #     # remove daily signal:
        #     pw_sta = pw_sta.groupby('time.hour') - pw_sta.groupby('time.hour').mean('time')            
            df = pw_sta.to_dataframe(sta)
            if attrs is not None:
                plot_box_df(df, ax=ax, x=x, title=sta, ylabel='', kind=kind,
                            ylimits=ylimits, xlimits=xlimits, attrs=attrs[i], bins=bins)
            else:
                plot_box_df(df, ax=ax, x=x, title=sta, ylabel='', kind=kind,
                            ylimits=ylimits, xlimits=xlimits, attrs=None, bins=bins)
    return fg


def plot_box_df(df, x='month', title='TELA',
                ylabel=r'IWV [kg$\cdot$m$^{-2}$]', ax=None, kind='violin',
                ylimits=(5, 40), xlimits=None, attrs=None, bins=None):
    # x=hour is experimental
    import seaborn as sns
    from matplotlib.ticker import MultipleLocator
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import kurtosis
    from scipy.stats import skew
    # df = da_ts.to_dataframe()
    if x == 'month':
        df[x] = df.index.month
        pal = sns.color_palette("Paired", 12)
    elif x == 'hour':
        df[x] = df.index.hour
        # df[x] = df.index
        pal = sns.color_palette("Paired", 12)
    y = df.columns[0]
    if ax is None:
        fig, ax = plt.subplots()
    if kind is None:
        df = df.groupby(x).mean()
        df.plot(ax=ax, legend=False, marker='o')
        ax.set_xlabel('Time of day [UTC]')
    elif kind == 'violin':
        sns.violinplot(ax=ax, data=df, x=x, y=y, palette=pal, fliersize=4,
                       gridsize=250, inner='quartile', scale='area')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlabel('')
    elif kind == 'box':
        kwargs = dict(markerfacecolor='r', marker='o')
        sns.boxplot(ax=ax, data=df, x=x, y=y, palette=pal, fliersize=4,
                    whis=1.0, flierprops=kwargs,showfliers=False)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlabel('')
    elif kind == 'hist':
        if bins is None:
            bins = 15
        a = df[y].dropna()
        sns.distplot(ax=ax, a=a, norm_hist=True, bins=bins, axlabel='PW [mm]')
        xmean = df[y].mean()
        xmedian = df[y].median()
        std = df[y].std()
        sk = skew(df[y].dropna().values)
        kurt = kurtosis(df[y].dropna().values)
        # xmode = df[y].mode().median()
        data_x, data_y = ax.lines[0].get_data()
        ymean = np.interp(xmean, data_x, data_y)
        ymed = np.interp(xmedian, data_x, data_y)
        # ymode = np.interp(xmode, data_x, data_y)
        ax.vlines(x=xmean, ymin=0, ymax=ymean, color='r', linestyle='--')
        ax.vlines(x=xmedian, ymin=0, ymax=ymed, color='g', linestyle='-')
        # ax.vlines(x=xmode, ymin=0, ymax=ymode, color='k', linestyle='-')
        # ax.legend(['Mean:{:.1f}'.format(xmean),'Median:{:.1f}'.format(xmedian),'Mode:{:.1f}'.format(xmode)])
        ax.legend(['Mean: {:.1f}'.format(xmean),'Median: {:.1f}'.format(xmedian)])
        ax.text(0.55, 0.45, "Std-Dev:    {:.1f}\nSkewness: {:.1f}\nKurtosis:   {:.1f}".format(std, sk, kurt),transform=ax.transAxes)
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.grid(True, which='minor', linestyle='--', linewidth=1, alpha=0.7)
    ax.yaxis.grid(True, linestyle='--', linewidth=1, alpha=0.7)
    title = ax.get_title().split('=')[-1].strip(' ')
    if attrs is not None:
        mean_years = float(attrs['mean_years'])
        ax.set_title('')
        ax.text(.2, .85, y.upper(),
                horizontalalignment='center', fontweight='bold',
                transform=ax.transAxes)
        if kind is not None:
            if kind != 'hist':
                ax.text(.22, .72, '{:.1f} years'.format(mean_years),
                        horizontalalignment='center',
                        transform=ax.transAxes)
    ax.yaxis.tick_left()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    if ylimits is not None:
        ax.set_ylim(*ylimits)
    if xlimits is not None:
        ax.set_xlim(*xlimits)
    return ax


def plot_means_pw(load_path=work_yuval, ims_path=ims_path, thresh=50,
                  col_wrap=5, means='hour', save=True):
    import xarray as xr
    import numpy as np
    pw = xr.load_dataset(
            work_yuval /
            'GNSS_PW_thresh_{:.0f}_homogenized.nc'.format(thresh))
    pw = pw[[x for x in pw.data_vars if '_error' not in x]]
    if means == 'hour':
        # remove long term monthly means:
        pw_clim = pw.groupby('time.month') - pw.groupby('time.month').mean('time')    
        pw_clim = pw_clim.groupby('time.{}'.format(means)).mean('time')
    else:
        pw_clim = pw.groupby('time.{}'.format(means)).mean('time')
#    T = xr.load_dataset(
#            ims_path /
#            'GNSS_5mins_TD_ALL_1996_2020.nc')
#    T_clim = T.groupby('time.month').mean('time')
    attrs = [x.attrs for x in pw.data_vars.values()]
    fg = pw_clim.to_array('station').plot(col='station', col_wrap=col_wrap,
                                          color='b', marker='o', alpha=0.7,
                                          sharex=False, sharey=True)
    col_arr = np.arange(0, len(pw_clim))
    right_side = col_arr[col_wrap-1::col_wrap]
    for i, ax in enumerate(fg.axes.flatten()):
        title = ax.get_title().split('=')[-1].strip(' ')
        try:
            mean_years = float(attrs[i]['mean_years'])
            ax.set_title('')
            ax.text(.2, .85, title.upper(),
                    horizontalalignment='center', fontweight='bold',
                    transform=ax.transAxes)
            ax.text(.2, .73, '{:.1f} years'.format(mean_years),
                    horizontalalignment='center',
                    transform=ax.transAxes)
#            ax_t = ax.twinx()
#            T_clim['{}'.format(title)].plot(
#                        color='r', linestyle='dashed', marker='s', alpha=0.7,
#                        ax=ax_t)
#            ax_t.set_ylim(0, 30)
            fg.fig.canvas.draw()

#            labels = [item.get_text() for item in ax_t.get_yticklabels()]
#            ax_t.yaxis.set_ticklabels([])
#            ax_t.tick_params(axis='y', color='r')
#            ax_t.set_ylabel('')
#            if i in right_side:
#                ax_t.set_ylabel(r'Surface temperature [$\degree$C]', fontsize=10)
#                ax_t.yaxis.set_ticklabels(labels)
#                ax_t.tick_params(axis='y', labelcolor='r', color='r')
            # show months ticks and grid lines for pw:
            ax.xaxis.tick_bottom()
            ax.yaxis.tick_left()
            ax.yaxis.grid()
#            ax.legend([ax.lines[0], ax_t.lines[0]], ['PW', 'T'],
#                      loc='upper right', fontsize=10, prop={'size': 8})
#            ax.legend([ax.lines[0]], ['PW'],
#                      loc='upper right', fontsize=10, prop={'size': 8})
        except IndexError:
            pass
    # change bottom xticks to 1-12 and show them:
    # fg.axes[-1, 0].xaxis.set_ticks(np.arange(1, 13))
    [fg.axes[x, 0].set_ylabel('PW [mm]') for x in range(len(fg.axes[:, 0]))]
    # adjust subplots:
    fg.fig.subplots_adjust(top=0.977,
                           bottom=0.039,
                           left=0.036,
                           right=0.959,
                           hspace=0.185,
                           wspace=0.125)
    filename = 'PW_{}_climatology.png'.format(means)
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_gnss_radiosonde_monthly_means(sound_path=sound_path, path=work_yuval,
                                       times=['2014', '2019'], sample='MS',
                                       gps_station='tela', east_height=5000):
    import xarray as xr
    from aux_gps import path_glob
    import pandas as pd
    file = path_glob(sound_path, 'bet_dagan_phys_PW_Tm_Ts_*.nc')
    phys = xr.load_dataset(file[0])['PW']
    if east_height is not None:
        file = path_glob(sound_path, 'bet_dagan_edt_sounding*.nc')
        east = xr.load_dataset(file[0])['east_distance']
        east = east.resample(sound_time=sample).mean().sel(Height=east_height, method='nearest')
        east_df = east.reset_coords(drop=True).to_dataframe()
    if times is not None:
        phys = phys.sel(sound_time=slice(*times))
    ds = phys.resample(sound_time=sample).mean().to_dataset(name='Bet-dagan-radiosonde')
    ds = ds.rename({'sound_time': 'time'})
    gps = xr.load_dataset(path / 'GNSS_PW_thresh_50_homogenized.nc')[gps_station]
    if times is not None:
        gps = gps.sel(time=slice(*times))
    ds[gps_station] = gps.resample(time=sample).mean()
    df = ds.to_dataframe()
    # now plot:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    # [x.set_xlim([pd.to_datetime(times[0]), pd.to_datetime(times[1])])
    #  for x in axes]
    df.columns = ['Bet dagan soundings', '{} GNSS station'.format(gps_station)]
    sns.lineplot(data=df, markers=['o','s'],linewidth=2.0, ax=axes[0])
    # axes[0].legend(['Bet_Dagan soundings', 'TELA GPS station'])
    df_r = df.iloc[:, 1] - df.iloc[:, 0]
    df_r.columns = ['Residual distribution']
    sns.lineplot(data=df_r, color='k', marker='o' ,linewidth=1.5, ax=axes[1])
    if east_height is not None:
        ax_east = axes[1].twinx()
        sns.lineplot(data=east_df, color='red', marker='x', linewidth=1.5, ax=ax_east)
        ax_east.set_ylabel('East drift at {} km altitude [km]'.format(east_height / 1000.0))
    axes[1].axhline(y=0, color='r')
    axes[0].grid(b=True, which='major')
    axes[1].grid(b=True, which='major')
    axes[0].set_ylabel('Precipitable Water [mm]')
    axes[1].set_ylabel('Residuals [mm]')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.01)
    return ds


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


def plot_ts_tm(path=sound_path, model='TSEN',
               times=['2007', '2019'], save=True):
    """plot ts-tm relashonship"""
    import xarray as xr
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PW_stations import ML_Switcher
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    import numpy as np
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from sounding_procedures import get_field_from_radiosonde
    models_dict = {'LR': 'Linear Regression',
                   'TSEN': 'Theil–Sen Regression'}
    # sns.set_style('whitegrid')
    pds = xr.Dataset()
    Ts = get_field_from_radiosonde(path=sound_path, field='Ts',
                                   data_type='phys', reduce=None, times=times,
                                   plot=False)
    Tm = get_field_from_radiosonde(path=sound_path, field='Tm',
                                   data_type='phys', reduce='min', times=times,
                                   plot=False)
    pds['Tm'] = Tm
    pds['Ts'] = Ts
    pds = pds.dropna('sound_time')
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
    ax.legend(['{} ({:.2f}, {:.2f})'.format(models_dict.get(model),
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
    print(rmean, rmse)
    r2 = r2_score(y, predict)
    axin1.axvline(rmean, color='r', linestyle='dashed', linewidth=1)
    # axin1.set_xlabel('Residual distribution[K]')
    textstr = '\n'.join(['n={}'.format(pds.Ts.size),
                         'RMSE: ', '{:.2f} K'.format(rmse)]) # ,
                         # r'R$^2$: {:.2f}'.format(r2)])
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
    filename = 'Bet_dagan_ts_tm_fit_{}-{}.png'.format(times[0], times[1])
    caption('Water vapor mean temperature (Tm) vs. surface temperature (Ts) of the Bet-dagan radiosonde station. Ordinary least squares linear fit(red) yields the residual distribution with RMSE of 4 K. Bevis(1992) model is plotted(purple) for comparison.')
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return


def plot_pw_tela_bet_dagan(path=work_yuval, sound_path=sound_path,
                           ims_path=ims_path, station='tela', cats=None,
                           times=['2007', '2019'], wv_name='pw', r2=False,
                           save=True):
    """plot the PW of Bet-dagan vs. PW of gps station"""
    from PW_stations import mean_ZWD_over_sound_time_and_fit_tstm
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # sns.set_style('white')
    ds, mda = mean_ZWD_over_sound_time_and_fit_tstm(path=path, sound_path=sound_path,
                                                    ims_path=ims_path,
                                                    data_type='phys',
                                                    gps_station=station,
                                                    times=times,
                                                    plot=False,
                                                    cats=cats)
    ds = ds.drop_dims('time')
    time_dim = list(set(ds.dims))[0]
    ds = ds.rename({time_dim: 'time'})
    tpw = 'tpw_bet_dagan'
    ds = ds[[tpw, 'tela_pw']].dropna('time')
    ds = ds.sel(time=slice(*times))
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
        ax.set_xlabel('PW from Beit-Dagan [mm]')
        ax.set_ylabel('PW from TELA GPS station [mm]')
    elif wv_name == 'iwv':
        ax.set_xlabel(r'IWV from Beit-dagan station [kg$\cdot$m$^{-2}$]')
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


def plot_zwd_tela_bet_dagan(path=work_yuval, sound_path=sound_path,
                            ims_path=ims_path, station='tela',
                            times=['2007', '2020'], cats=None,
                            save=True):
    from PW_stations import mean_ZWD_over_sound_time_and_fit_tstm
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import matplotlib.dates as mdates
    # sns.set_style('whitegrid')
    ds, mda = mean_ZWD_over_sound_time_and_fit_tstm(path=path,
                                                    sound_path=sound_path,
                                                    ims_path=ims_path,
                                                    data_type='phys',
                                                    gps_station=station,
                                                    times=times,
                                                    plot=False,
                                                    cats=cats)
    ds = ds.drop_dims('time')
    time_dim = list(set(ds.dims))[0]
    ds = ds.rename({time_dim: 'time'})
    ds = ds.dropna('time')
    ds = ds.sel(time=slice(*times))
    df = ds[['zwd_bet_dagan', 'tela']].to_dataframe()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    [x.set_xlim([pd.to_datetime(times[0]), pd.to_datetime(times[1])])
     for x in axes]
    df.columns = ['Beit Dagan soundings', 'TELA GPS station']
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


def plot_israel_with_stations(gis_path=gis_path, dem_path=dem_path, ims=True,
                              gps=True, radio=True, terrain=True, save=True):
    from PW_stations import produce_geo_gnss_solved_stations
    from aux_gps import geo_annotate
    from ims_procedures import produce_geo_ims
    import matplotlib.pyplot as plt
    import xarray as xr
    import pandas as pd
    import geopandas as gpd
    ax = plot_israel_map(gis_path)
    station_names = []
    legend = []
    if ims:
        print('getting IMS temperature stations metadata...')
        ims_t = produce_geo_ims(path=gis_path, freq='10mins', plot=False)
        ims_t.plot(ax=ax, color='red', edgecolor='black', alpha=0.5)
        station_names.append('ims')
        legend.append('IMS stations')
    # ims, gps = produce_geo_df(gis_path=gis_path, plot=False)
    if gps:
        print('getting solved GNSS israeli stations metadata...')
        gps = produce_geo_gnss_solved_stations(path=gis_path, plot=False)
        gps.plot(ax=ax, color='k', edgecolor='black', marker='s')
        gps_stations = [x for x in gps.index]
        to_plot_offset = ['mrav', 'klhv']
        [gps_stations.remove(x) for x in to_plot_offset]
        gps_normal_anno = gps.loc[gps_stations, :]
        gps_offset_anno = gps.loc[to_plot_offset, :]
        geo_annotate(ax, gps_normal_anno.lon, gps_normal_anno.lat,
                     gps_normal_anno.index.str.upper(), xytext=(3, 3), fmt=None,
                     c='k', fw='bold', fs=10, colorupdown=False)
        geo_annotate(ax, gps_offset_anno.lon, gps_offset_anno.lat,
                     gps_offset_anno.index.str.upper(), xytext=(4, -6), fmt=None,
                     c='k', fw='bold', fs=10, colorupdown=False)
        station_names.append('gps')
        legend.append('GNSS stations')
    if terrain:
        # overlay with dem data:
        cmap = plt.get_cmap('terrain', 41)
        dem = xr.open_dataarray(dem_path / 'israel_dem_250_500.nc')
        # dem = xr.open_dataarray(dem_path / 'israel_dem_500_1000.nc')
        fg = dem.plot.imshow(ax=ax, alpha=0.5, cmap=cmap,
                             vmin=dem.min(), vmax=dem.max(), add_colorbar=False)
        cbar_kwargs = {'fraction': 0.1, 'aspect': 50, 'pad': 0.03}
        cb = plt.colorbar(fg, **cbar_kwargs)
        cb.set_label(label='meters above sea level', size=8, weight='normal')
        cb.ax.tick_params(labelsize=8)
        ax.set_xlabel('')
        ax.set_ylabel('')
    if radio:   # plot bet-dagan:
        df = pd.Series([32.00, 34.81]).to_frame().T
        df.index = ['Beit Dagan']
        df.columns = ['lat', 'lon']
        bet_dagan = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon,
                                                                     df.lat),
                                     crs=gps.crs)
        bet_dagan.plot(ax=ax, color='black', edgecolor='black',
                       marker='+')
        geo_annotate(ax, bet_dagan.lon, bet_dagan.lat,
                     bet_dagan.index, xytext=(4, -6), fmt=None,
                     c='k', fw='bold', fs=10, colorupdown=False)
        station_names.append('radio')
        legend.append('radiosonde')
    if legend:
        plt.legend(legend, loc='upper left')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    if station_names:
        station_names = '_'.join(station_names)
    else:
        station_names = 'no_stations'
    filename = 'israel_map_{}.png'.format(station_names)
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


def produce_table_1(removed=['hrmn'], merged={'klhv': ['klhv', 'lhav'],
                    'mrav': ['gilb', 'mrav']}):
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
    if removed is not None:
        df = df.loc[[x for x in df.index if x not in removed], :]
    if merged is not None:
        return df
    print(df.to_latex(index=False))
    return df


def produce_table_stats(thresh=50):
    from PW_stations import produce_pw_statistics
    import xarray as xr
    pw_mm = xr.load_dataset(
            work_yuval /
             'GNSS_PW_monthly_thresh_{:.0f}_homogenized.nc'.format(thresh))

    df = produce_pw_statistics(thresh=thresh, resample_to_mm=False
                               , pw_input=pw_mm)
    print(df.to_latex(index=False))
    return df


def produce_table_mann_kendall(thresh=50):
    from PW_stations import mann_kendall_trend_analysis
    import xarray as xr
    import pandas as pd

    def process_mkt(ds_in, alpha=0.05, seasonal=False, factor=120):
        ds = ds_in.map(
            mann_kendall_trend_analysis,
            alpha=alpha,
            seasonal=seasonal,
            verbose=False)
        ds = ds.rename({'dim_0': 'mkt'})
        df = ds.to_dataframe().T
        df = df.drop(['test_name', 'trend', 'h', 'z', 's', 'var_s'], axis=1)
        df['id'] = df.index.str.upper()
        df = df[['id', 'Tau', 'p', 'slope']]
        df.index.name = ''
        df['slope'] = df['slope'] * factor
        df['slope'] = df['slope'][df['p'] < 0.05]
        df.loc[:, 'p'][df['p'] < 0.0001] = '<0.0001'
        df['p'][df['p'] != '<0.0001'] = df['p'][df['p'] !=
                                                '<0.0001'].astype(float).map('{:,.4f}'.format)
        df['Tau'] = df['Tau'].map('{:,.4f}'.format)
        df['slope'] = df['slope'].map('{:,.2f}'.format)
        df['slope'][df['slope'] == 'nan'] = '-'
        df.columns = ['Site ID', "Kendall's Tau", 'P-value', "Sen's slope"]
        return df

    anoms = xr.load_dataset(
            work_yuval /
             'GNSS_PW_monthly_anoms_thresh_{:.0f}_homogenized.nc'.format(thresh))
    mm = xr.load_dataset(
        work_yuval /
        'GNSS_PW_monthly_thresh_{:.0f}_homogenized.nc'.format(thresh))
    df_anoms = process_mkt(anoms)
    df_mm = process_mkt(mm, seasonal=True)
#    mkt_trends = [anoms[x].attrs['mkt_trend'] for x in anoms.data_vars]
#    mkt_bools = [anoms[x].attrs['mkt_h'] for x in anoms.data_vars]
#    mkt_slopes = [anoms[x].attrs['mkt_slope'] for x in anoms.data_vars]
#    mkt_pvalue = [anoms[x].attrs['mkt_p'] for x in anoms.data_vars]
#    mkt_95_lo = [anoms[x].attrs['mkt_trend_95'][0] for x in anoms.data_vars]
#    mkt_95_up = [anoms[x].attrs['mkt_trend_95'][1] for x in anoms.data_vars]
#    df = pd.DataFrame(mkt_trends, index=[x for x in anoms.data_vars], columns=['mkt_trend'])
#    df['mkt_h'] = mkt_bools
#    # transform into per decade:
#    df['mkt_slope'] = mkt_slopes
#    df['mkt_pvalue'] = mkt_pvalue
#    df['mkt_95_lo'] = mkt_95_lo
#    df['mkt_95_up'] = mkt_95_up
#    df[['mkt_slope', 'mkt_95_lo', 'mkt_95_up']] *= 120
#    df.index = df.index.str.upper()
#    df['Sen\'s slope'] = df['mkt_slope'].map('{:,.2f}'.format)
#    df.loc[:, 'Sen\'s slope'][~df['mkt_h']] = 'No trend'
#    con = ['({:.2f}, {:.2f})'.format(x, y) for (x, y) in list(
#        zip(df['mkt_95_lo'].values, df['mkt_95_up'].values))]
#    df['95% confidence intervals'] = con
#    df.loc[:, '95% confidence intervals'][~df['mkt_h']] = '-'
#    df = df[['Sen\'s slope', '95% confidence intervals']]
    print(df_anoms.to_latex(index=False))
    return df_anoms, df_mm


def plot_monthly_means_anomalies_with_station_mean(load_path=work_yuval,
                                                   thresh=50, save=True):
    import xarray as xr
    import seaborn as sns
    from palettable.scientific import diverging as divsci
    import numpy as np
    import matplotlib.dates as mdates
    import pandas as pd
    div_cmap = divsci.Vik_20.mpl_colormap
    anoms = xr.load_dataset(
            load_path /
             'GNSS_PW_monthly_anoms_thresh_{:.0f}.nc'.format(thresh))
    df = anoms.to_dataframe()
    df.columns = [x.upper() for x in df.columns]
    fig = plt.figure(figsize=(20, 10))
    grid = plt.GridSpec(
        2, 1, height_ratios=[
            4, 1], hspace=0)
    ax_heat = fig.add_subplot(grid[0, 0])  # plt.subplot(221)
    ax_group = fig.add_subplot(grid[1, 0])  # plt.subplot(223)
    cbar_ax = fig.add_axes([0.95, 0.24, 0.01, 0.745])  #[left, bottom, width,
    # height]
    ax_heat = sns.heatmap(
            df.T,
            center=0.0,
            cmap=div_cmap,
            yticklabels=True,
            ax=ax_heat,
            cbar_ax=cbar_ax,
            cbar_kws={'label': '[mm]'}, xticklabels=False)
    # activate top ticks and tickslabales:
    ax_heat.xaxis.set_tick_params(bottom='off', labelbottom='off')
    # emphasize the yticklabels (stations):
    ax_heat.yaxis.set_tick_params(left='on')
    ax_heat.set_yticklabels(ax_heat.get_ymajorticklabels(),
                            fontweight='bold', fontsize=10)
    ts = df.T.median().shift(periods=-1, freq='15D')
    ts.index.name = ''
    # dt_as_int = [x for x in range(len(ts.index))]
    # xticks_labels = ts.index.strftime('%Y-%m').values[::6]
    # xticks = dt_as_int[::6]
    # xticks = ts.index
    # ts.index = dt_as_int
    ts.plot(ax=ax_group, color='k')
    # group_limit = ax_heat.get_xlim()
    ax_group.set_xlim(ts.index.min(), ts.index.max() +
                      pd.Timedelta(15, unit='D'))
    ax_group.set_ylabel('[mm]')
    # set ticks and align with heatmap axis (move by 0.5):
    # ax_group.set_xticks(dt_as_int)
    # offset = 1
#    ax_group.xaxis.set(ticks=np.arange(offset / 2.,
#                                       max(dt_as_int) + 1 - min(dt_as_int),
#                                       offset),
#                       ticklabels=dt_as_int)
    # move the lines also by 0.5 to align with heatmap:
    # lines = ax_group.lines  # get the lines
    # [x.set_xdata(x.get_xdata() - min(dt_as_int) + 0.5) for x in lines]
    # ax_group.xaxis.set(ticks=xticks, ticklabels=xticks_labels)
    # ax_group.xaxis.set(ticks=xticks)
    years_fmt = mdates.DateFormatter('%Y')
    ax_group.xaxis.set_major_locator(mdates.YearLocator())
    ax_group.xaxis.set_major_formatter(years_fmt)
    ax_group.xaxis.set_minor_locator(mdates.MonthLocator())
    ax_group.grid()
    # ax_group.axvline('2015-09-15')
    # ax_group.axhline(2.5)
    # plt.setp(ax_group.xaxis.get_majorticklabels(), rotation=45 )
    fig.tight_layout()
    fig.subplots_adjust(right=0.946)
    if save:
        filename = 'pw_monthly_means_anomaly_heatmap.png'
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return ts


def plot_grp_anomlay_heatmap(load_path=work_yuval, gis_path=gis_path,
                             thresh=50, grp='hour', remove_grp=None, season=None,
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
            n_clusters=n_clusters, remove_grp=remove_grp)
    # create figure and subplots axes:
    fig = plt.figure(figsize=(15, 10))
    if title:
        if season is not None:
            fig.suptitle('Precipitable water {}ly anomalies analysis for {} season'.format(grp, season))
        else:
            fig.suptitle('Precipitable water {}ly anomalies analysis (Weighted KMeans {} clusters)'.format(grp, n_clusters))
    grid = plt.GridSpec(
        2, 2, width_ratios=[
            3, 2], height_ratios=[
            4, 1], wspace=0.1, hspace=0)
    ax_heat = fig.add_subplot(grid[0, 0])  # plt.subplot(221)
    ax_group = fig.add_subplot(grid[1, 0])  # plt.subplot(223)
    ax_map = fig.add_subplot(grid[0:, 1])  # plt.subplot(122)
    # get the camp and zip it to groups and produce dictionary:
    cmap = plt.get_cmap("Accent")
    cmap = qualitative_cmap(n_clusters)
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
    boxes = [dict(facecolor=x, boxstyle="square,pad=0.7", alpha=0.6)
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
    df_groups.T.plot(ax=ax_group, linewidth=2.0, legend=False, cmap=cm)
    if grp == 'hour':
        ax_group.set_xlabel('hour (UTC)')
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
             edgecolor='black', cmap=cm, s=100, legend=True, alpha=1.0,
             legend_kwds={'prop': {'size': 10}, 'fontsize': 14,
                          'loc': 'upper left', 'title': 'clusters'})
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


def plot_lomb_scargle(path=work_yuval, save=True):
    from aux_gps import lomb_scargle_xr
    import xarray as xr
    pw_mm = xr.load_dataset(path / 'GNSS_PW_monthly_thresh_50_homogenized.nc')
    pw_mm_median = pw_mm.to_array('station').median('station')
    da = lomb_scargle_xr(
        pw_mm_median.dropna('time'),
        user_freq='MS',
        kwargs={
            'nyquist_factor': 1,
            'samples_per_peak': 100})
    plt.ylabel('')
    plt.title('Lomb–Scargle periodogram')
    plt.xlim([0, 4])
    plt.grid()
    filename = 'Lomb_scargle_monthly_means.png'
    if save:
#        plt.savefig(savefig_path / filename, bbox_inches='tight')
        plt.savefig(savefig_path / filename, orientation='landscape')
    return da


def plot_vertical_climatology_months(path=sound_path, field='Rho_wv',
                                     center_month=7):
    from aux_gps import path_glob
    import xarray as xr
    ds = xr.open_dataset(
        path /
        'bet_dagan_phys_sounding_height_2007-2019.nc')[field]
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    day = ds.sel(sound_time=ds['sound_time.hour']==12).groupby('sound_time.month').mean('sound_time')
    night = ds.sel(sound_time=ds['sound_time.hour']==00).groupby('sound_time.month').mean('sound_time')
    next_month = center_month + 1
    last_month = center_month - 1
    day = day.sel(month=[last_month, center_month, next_month])
    night = night.sel(month=[last_month, center_month, next_month])
    for month in day.month:
        h=day.sel(month=month)['H-Msl'].values
        rh = day.sel(month=month).values
        ax[0].semilogy(rh, h)
    ax[0].set_title('noon')
    ax[0].set_ylabel('height [m]')
    ax[0].set_xlabel('{}, [{}]'.format(field, day.attrs['units']))
    plt.legend([x for x in ax.lines],[x for x in day.month.values])
    for month in night.month:
        h=night.sel(month=month)['H-Msl'].values
        rh = night.sel(month=month).values
        ax[1].semilogy(rh, h)
    ax[1].set_title('midnight')
    ax[1].set_ylabel('height [m]')
    ax[1].set_xlabel('{}, [{}]'.format(field, night.attrs['units']))
    plt.legend([x for x in ax.lines],[x for x in night.month.values])
    return day, night


def plot_pw_lapse_rate_fit(path=work_yuval, model='TSEN', plot=True):
    from PW_stations import produce_geo_gnss_solved_stations
    import xarray as xr
    from PW_stations import ML_Switcher
    import pandas as pd
    import matplotlib.pyplot as plt
    pw = xr.load_dataset(path / 'GNSS_PW_thresh_50.nc')
    pw = pw[[x for x in pw.data_vars if '_error' not in x]]
    df_gnss = produce_geo_gnss_solved_stations(plot=False)
    df_gnss = df_gnss.loc[[x for x in pw.data_vars], :]
    alt = df_gnss['alt'].values
    # add mean to anomalies:
    pw_new = pw.resample(time='MS').mean()
    pw_mean = pw_new.mean('time')
    # compute std:
#    pw_std = pw_new.std('time')
    pw_std = (pw_new.groupby('time.month') - pw_new.groupby('time.month').mean('time')).std('time')
    pw_vals = pw_mean.to_array().to_dataframe(name='pw')
    pw_vals = pd.Series(pw_vals.squeeze()).values
    pw_std_vals = pw_std.to_array().to_dataframe(name='pw')
    pw_std_vals = pd.Series(pw_std_vals.squeeze()).values
    ml = ML_Switcher()
    fit_model = ml.pick_model(model)
    y = pw_vals
    X = alt.reshape(-1, 1)
    fit_model.fit(X, y)
    predict = fit_model.predict(X)
    coef = fit_model.coef_[0]
    inter = fit_model.intercept_
    pw_lapse_rate = abs(coef)*1000
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(16, 4))
        ax.errorbar(x=alt, y=pw_vals, yerr=pw_std_vals,
                    marker='.', ls='', capsize=1.5, elinewidth=1.5,
                    markeredgewidth=1.5, color='k')
        ax.grid()
        ax.plot(X, predict, c='r')
        ax.set_xlabel('meters a.s.l')
        ax.set_ylabel('Precipitable Water [mm]')
        ax.legend(['{} ({:.2f} [mm/km], {:.2f} [mm])'.format(model,
                   pw_lapse_rate, inter)])
    return df_gnss['alt'], pw_lapse_rate


def plot_time_series_as_barplot(ts, anoms=False, ts_ontop=None):
    # plt.style.use('fast')
    time_dim = list(set(ts.dims))[0]
    fig, ax = plt.subplots(figsize=(20, 6), dpi=150)
    import matplotlib.dates as mdates
    import matplotlib.ticker
    from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
    import pandas as pd
    if not anoms:
        # sns.barplot(x=ts[time_dim].values, y=ts.values, ax=ax, linewidth=5)
        ax.bar(ts[time_dim].values, ts.values, linewidth=5, width=0.0,
               facecolor='black', edgecolor='black')
        # Series.plot.bar(ax=ax, linewidth=0, width=1)
    else:
        warm = 'tab:orange'
        cold = 'tab:blue'
        positive = ts.where(ts > 0).dropna(time_dim)
        negative = ts.where(ts < 0).dropna(time_dim)
        ax.bar(
            positive[time_dim].values,
            positive.values,
            linewidth=3.0,
            width=1.0,
            facecolor=warm, edgecolor=warm, alpha=1.0)
        ax.bar(
            negative[time_dim].values,
            negative.values,
            width=1.0,
            linewidth=3.0,
            facecolor=cold, edgecolor=cold, alpha=1.0)
    if ts_ontop is not None:
        ax_twin = ax.twinx()
        color = 'red'
        ts_ontop.plot.line(color=color, linewidth=2.0, ax=ax_twin)
        ax_twin.set_ylabel('PW [mm]', color=color)  # we already handled the x-label with ax1
        ax_twin.tick_params(axis='y', labelcolor=color)
        ax_twin.legend(['3-month running mean of PW anomalies'])
        title_add = ' and the median Precipitable Water anomalies from Israeli GNSS sites'
        l2 = ax_twin.get_ylim()
        ax.set_ylim(l2)
    else:
        title_add = ''
        
    ax.grid(None)
    ax.set_xlim([pd.to_datetime('1996'), pd.to_datetime('2020')])
    ax.set_title('Multivariate ENSO Index Version 2 {}'.format(title_add))
    ax.set_ylabel('MEI.v2')
    # ax.xaxis.set_major_locator(MultipleLocator(20))
    # Change minor ticks to show every 5. (20/4 = 5)
#    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    years_fmt = mdates.DateFormatter('%Y')
    # ax.figure.autofmt_xdate()
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1)) 
    ax.xaxis.set_major_formatter(years_fmt)

    # ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.figure.autofmt_xdate()
#     plt.tick_params(
#            axis='x',          # changes apply to the x-axis
#            which='both',      # both major and minor ticks are affected
#            bottom=True,      # ticks along the bottom edge are off
#            top=False,         # ticks along the top edge are off
#            labelbottom=True)
    # fig.tight_layout()
    plt.show()
    return


def plot_tide_pw_lags(path=hydro_path, pw_anom=False, rolling='1H', save=True):
    from aux_gps import path_glob
    import xarray as xr
    import numpy as np
    file = path_glob(path, 'PW_tide_sites_*.nc')[-1]
    if pw_anom:
        file = path_glob(path, 'PW_tide_sites_anom_*.nc')[-1]
    ds = xr.load_dataset(file)
    names = [x for x in ds.data_vars]
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in names:
        da = ds.mean('station').mean('tide_start')[name]
        ser = da.to_series()
        if rolling is not None:
            ser = ser.rolling(rolling).mean()
        time=(ser.index / np.timedelta64(1, 'D')).astype(float)
        # ser = ser.loc[pd.Timedelta(-2.2,unit='D'):pd.Timedelta(1, unit='D')]
        ser.index = time

        ser.plot(marker='.', linewidth=0., ax=ax)
    ax.set_xlabel('Days around tide event')
    ax.set_ylabel('PW [mm]')
    hstations = [ds[x].attrs['hydro_stations'] for x in ds.data_vars]
    events = [ds[x].attrs['total_events'] for x in ds.data_vars]
    fmt = list(zip(names, hstations, events))
    ax.legend(['{} with {} stations ({} total events)'.format(x.upper(), y, z)
               for x, y, z in fmt])
    ax.set_xlim([-3, 1])
    ax.axvline(0, color='k', linestyle='--')
    ax.grid()
    filename = 'pw_tide_sites.png'
    if pw_anom:
        filename = 'pw_tide_sites_anom.png'
    if save:
#        plt.savefig(savefig_path / filename, bbox_inches='tight')
        plt.savefig(savefig_path / filename, orientation='landscape')
#    ax.xaxis.set_major_locator(mdates.HourLocator(interval=24)) # tick every two hours
#    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
#    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
#    formatter = mdates.ConciseDateFormatter(locator)
#    ax.xaxis.set_major_locator(locator)
#    ax.xaxis.set_major_formatter(formatter)
    # title = 'Mean PW for tide stations near all GNSS stations'
    # ax.set_title(title)    
    return


def plot_hist_with_seasons(da_ts):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.kdeplot(da_ts.dropna('time'), ax=ax, color='k')
    sns.kdeplot(da_ts.sel(time=da_ts['time.season']=='DJF').dropna('time'), legend=False, ax=ax, shade=True)
    sns.kdeplot(da_ts.sel(time=da_ts['time.season']=='MAM').dropna('time'),  legend=False,ax=ax, shade=True)
    sns.kdeplot(da_ts.sel(time=da_ts['time.season']=='JJA').dropna('time'), legend=False ,ax=ax, shade=True)
    sns.kdeplot(da_ts.sel(time=da_ts['time.season']=='SON').dropna('time'), legend=False, ax=ax, shade=True)
    plt.legend(['ALL','MAM', 'DJF', 'SON', 'JJA'])
    return
