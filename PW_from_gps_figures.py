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
import matplotlib.dates as mdates
from PW_stations import produce_geo_gnss_solved_stations
tela_results_path = work_yuval / 'GNSS_stations/tela/rinex/30hr/results'
tela_solutions = work_yuval / 'GNSS_stations/tela/gipsyx_solutions'
sound_path = work_yuval / 'sounding'
phys_soundings = sound_path / 'bet_dagan_phys_sounding_2007-2019.nc'
ims_path = work_yuval / 'IMS_T'
gis_path = work_yuval / 'gis'
dem_path = work_yuval / 'AW3D30'
era5_path = work_yuval / 'ERA5'
hydro_path = work_yuval / 'hydro'
ceil_path = work_yuval / 'ceilometers'
aero_path = work_yuval / 'AERONET'
climate_path = work_yuval / 'climate'
df_gnss = produce_geo_gnss_solved_stations(
    plot=False, add_distance_to_coast=True)
st_order_climate = [x for x in df_gnss.dropna().sort_values(
    ['groups_climate', 'lat', 'lon'], ascending=[1, 0, 0]).index]
rc = {
    'font.family': 'serif',
    'xtick.labelsize': 'large',
    'ytick.labelsize': 'large'}
for key, val in rc.items():
    rcParams[key] = val
# sns.set(rc=rc, style='white')
seasonal_colors = {'DJF': 'tab:blue',
                   'SON': 'tab:red',
                   'JJA': 'tab:green',
                   'MAM': 'tab:orange',
                   'Annual': 'tab:purple'}


def get_twin(ax, axis):
    assert axis in ("x", "y")
    siblings = getattr(ax, f"get_shared_{axis}_axes")().get_siblings(ax)
    for sibling in siblings:
        if sibling.bbox.bounds == ax.bbox.bounds and sibling is not ax:
            return sibling
    return None


def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    from math import floor, log10
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)


def utm_from_lon(lon):
    """
    utm_from_lon - UTM zone for a longitude

    Not right for some polar regions (Norway, Svalbard, Antartica)

    :param float lon: longitude
    :return: UTM zone number
    :rtype: int
    """
    from math import floor
    return floor((lon + 180) / 6) + 1


def scale_bar(ax, proj, length, location=(0.5, 0.05), linewidth=3,
              units='km', m_per_unit=1000, bounds=None):
    """

    http://stackoverflow.com/a/35705477/1072212
    ax is the axes to draw the scalebar on.
    proj is the projection the axes are in
    location is center of the scalebar in axis coordinates ie. 0.5 is the middle of the plot
    length is the length of the scalebar in km.
    linewidth is the thickness of the scalebar.
    units is the name of the unit
    m_per_unit is the number of meters in a unit
    """
    import cartopy.crs as ccrs
    from matplotlib import patheffects
    # find lat/lon center to find best UTM zone
    try:
        x0, x1, y0, y1 = ax.get_extent(proj.as_geodetic())
    except AttributeError:
        if bounds is not None:
            x0, x1, y0, y1 = bounds
    # Projection in metres
    utm = ccrs.UTM(utm_from_lon((x0+x1)/2))
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(utm)
    # Turn the specified scalebar location into coordinates in metres
    sbcx, sbcy = x0 + (x1 - x0) * location[0], y0 + (y1 - y0) * location[1]
    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbcx - length * m_per_unit/2, sbcx + length * m_per_unit/2]
    # buffer for scalebar
    buffer = [patheffects.withStroke(linewidth=5, foreground="w")]
    # Plot the scalebar with buffer
    ax.plot(bar_xs, [sbcy, sbcy], transform=utm, color='k',
            linewidth=linewidth, path_effects=buffer)
    # buffer for text
    buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
    # Plot the scalebar label
    t0 = ax.text(sbcx, sbcy, str(length) + ' ' + units, transform=utm,
                 horizontalalignment='center', verticalalignment='bottom',
                 path_effects=buffer, zorder=2)
    left = x0+(x1-x0)*0.05
    # Plot the N arrow
    t1 = ax.text(left, sbcy, u'\u25B2\nN', transform=utm,
                 horizontalalignment='center', verticalalignment='bottom',
                 path_effects=buffer, zorder=2)
    # Plot the scalebar without buffer, in case covered by text buffer
    ax.plot(bar_xs, [sbcy, sbcy], transform=utm, color='k',
            linewidth=linewidth, zorder=3)
    return


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


def align_yaxis_np(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    import numpy as np
    axes = np.array([ax1, ax2])
    extrema = np.array([ax.get_ylim() for ax in axes])
    tops = extrema[:,1] / (extrema[:,1] - extrema[:,0])
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [a[::-1] for a in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    extrema[0,1] = extrema[0,0] + tot_span * (extrema[0,1] - extrema[0,0])
    extrema[1,0] = extrema[1,1] + tot_span * (extrema[1,0] - extrema[1,1])
    [axes[i].set_ylim(*extrema[i]) for i in range(2)]


# def align_yaxis(ax1, v1, ax2, v2):
#     """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
#     _, y1 = ax1.transData.transform((0, v1))
#     _, y2 = ax2.transData.transform((0, v2))
#     inv = ax2.transData.inverted()
#     _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
#     miny, maxy = ax2.get_ylim()
#     ax2.set_ylim(miny+dy, maxy+dy)


def alignYaxes(axes, align_values=None):
    '''Align the ticks of multiple y axes
    Args:
        axes (list): list of axes objects whose yaxis ticks are to be aligned.
    Keyword Args:
        align_values (None or list/tuple): if not None, should be a list/tuple
            of floats with same length as <axes>. Values in <align_values>
            define where the corresponding axes should be aligned up. E.g.
            [0, 100, -22.5] means the 0 in axes[0], 100 in axes[1] and -22.5
            in axes[2] would be aligned up. If None, align (approximately)
            the lowest ticks in all axes.
    Returns:
        new_ticks (list): a list of new ticks for each axis in <axes>.
        A new sets of ticks are computed for each axis in <axes> but with equal
        length.
    '''
    from matplotlib.pyplot import MaxNLocator
    import numpy as np
    nax = len(axes)
    ticks = [aii.get_yticks() for aii in axes]
    if align_values is None:
        aligns = [ticks[ii][0] for ii in range(nax)]
    else:
        if len(align_values) != nax:
            raise Exception(
                "Length of <axes> doesn't equal that of <align_values>.")
        aligns = align_values
    bounds = [aii.get_ylim() for aii in axes]
    # align at some points
    ticks_align = [ticks[ii]-aligns[ii] for ii in range(nax)]
    # scale the range to 1-100
    ranges = [tii[-1]-tii[0] for tii in ticks]
    lgs = [-np.log10(rii)+2. for rii in ranges]
    igs = [np.floor(ii) for ii in lgs]
    log_ticks = [ticks_align[ii]*(10.**igs[ii]) for ii in range(nax)]
    # put all axes ticks into a single array, then compute new ticks for all
    comb_ticks = np.concatenate(log_ticks)
    comb_ticks.sort()
    locator = MaxNLocator(nbins='auto', steps=[1, 2, 2.5, 3, 4, 5, 8, 10])
    new_ticks = locator.tick_values(comb_ticks[0], comb_ticks[-1])
    new_ticks = [new_ticks/10.**igs[ii] for ii in range(nax)]
    new_ticks = [new_ticks[ii]+aligns[ii] for ii in range(nax)]
    # find the lower bound
    idx_l = 0
    for i in range(len(new_ticks[0])):
        if any([new_ticks[jj][i] > bounds[jj][0] for jj in range(nax)]):
            idx_l = i-1
            break
    # find the upper bound
    idx_r = 0
    for i in range(len(new_ticks[0])):
        if all([new_ticks[jj][i] > bounds[jj][1] for jj in range(nax)]):
            idx_r = i
            break
    # trim tick lists by bounds
    new_ticks = [tii[idx_l:idx_r+1] for tii in new_ticks]
    # set ticks for each axis
    for axii, tii in zip(axes, new_ticks):
        axii.set_yticks(tii)
    return new_ticks


def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    adjust_yaxis(ax2, (y1 - y2) / 2, v2)
    adjust_yaxis(ax1, (y2 - y1) / 2, v1)


def adjust_yaxis(ax, ydif, v):
    """shift axis ax by ydiff, maintaining point v at the same location"""
    inv = ax.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, ydif))
    miny, maxy = ax.get_ylim()
    miny, maxy = miny - v, maxy - v
    if -miny > maxy or (-miny == maxy and dy > 0):
        nminy = miny
        nmaxy = miny * (maxy + dy) / (miny + dy)
    else:
        nmaxy = maxy
        nminy = maxy * (miny + dy) / (maxy + dy)
    ax.set_ylim(nminy + v, nmaxy + v)


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


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def produce_colors_for_pwv_station(scope='annual', zebra=False,
                                   as_dict=False, as_cat_dict=False):
    import pandas as pd
    stns = group_sites_to_xarray(scope=scope)
    cdict = {'coastal': 'tab:blue',
             'highland': 'tab:green',
             'eastern': 'tab:orange'}
    if as_cat_dict:
        return cdict
    # for grp, color in cdict.copy().items():
    #     cdict[grp] = to_rgba(get_named_colors_mapping()[
    #                         color], alpha=1)
    ds = stns.to_dataset('group')
    colors = []
    for group in ds:
        sts = ds[group].dropna('GNSS').values
        for i, st in enumerate(sts):
            color = cdict.get(group)
            if zebra:
                if i % 2 != 0:
                    # rgba = np.array(rgba)
                    # rgba[-1] = 0.5
                    color = adjust_lightness(color, 0.5)
            colors.append(color)
    # colors = [item for sublist in colors for item in sublist]
    stns = stns.T.values.ravel()
    stns = stns[~pd.isnull(stns)]
    if as_dict:
        colors = dict(zip(stns, colors))
    return colors


def fix_time_axis_ticks(ax, limits=None, margin=15):
    import pandas as pd
    import matplotlib.dates as mdates
    if limits is not None:
        ax.set_xlim(*pd.to_datetime(limits))
    years_fmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
#    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
#    formatter = mdates.ConciseDateFormatter(locator)
#    ax.xaxis.set_major_locator(locator)
#    ax.xaxis.set_major_formatter(formatter)
    return ax


def plot_qflux_climatotlogy_israel(path=era5_path, save=True, reduce='mean',
                                   plot_type='uv'):
    import xarray as xr
    import matplotlib.pyplot as plt
    import numpy as np
    ds = xr.load_dataset(path / 'ERA5_UVQ_mm_israel_1979-2020.nc')
    ds = ds.sel(expver=1).reset_coords(drop=True)
    if plot_type == 'uv':
        f1 = ds['q'] * ds['u']
        f2 = ds['q'] * ds['v']
    elif plot_type == 'md':
        qu = ds['q'] * ds['u']
        qv = ds['q'] * ds['v']
        f1 = np.sqrt(qu**2 + qv**2)
        f2 = np.rad2deg(np.arctan2(qv, qu))
    if reduce == 'mean':
        f1_clim = f1.groupby('time.month').mean().mean(
            'longitude').mean('latitude')
        f2_clim = f2.groupby('time.month').mean().mean(
            'longitude').mean('latitude')
        center = 0
        cmap = 'bwr'
    elif reduce == 'std':
        f1_clim = f1.groupby('time.month').std().mean(
            'longitude').mean('latitude')
        f2_clim = f2.groupby('time.month').std().mean(
            'longitude').mean('latitude')
        center = None
        cmap = 'viridis'
    ds_clim = xr.concat([f1_clim, f2_clim], 'direction')
    ds_clim['direction'] = ['zonal', 'meridional']
    if plot_type == 'md':
        fg, axes = plt.subplots(1, 2, figsize=(14, 7))
        f1_clim.sel(
            level=slice(
                300,
                1000)).T.plot.contourf(levels=41,
                                       yincrease=False,
                                       cmap=cmap,
                                       center=center, ax=axes[0])
        f2_clim.sel(
            level=slice(
                300,
                1000)).T.plot.contourf(levels=41,
                                       yincrease=False,
                                       cmap=cmap,
                                       center=center, ax=axes[1])
    else:
        fg = ds_clim.sel(
            level=slice(
                300,
                1000)).T.plot.contourf(
            levels=41,
            yincrease=False,
            cmap=cmap,
            center=center,
            col='direction',
            figsize=(
                15,
                6))
    fg.fig.suptitle('Moisture flux climatology over Israel')
#    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
#    qu_clim.sel(level=slice(300,1000)).T.plot.contourf(levels=41, yincrease=False, ax=axes[0], cmap='bwr', center=0)
#    qv_clim.sel(level=slice(300,1000)).T.plot.contourf(levels=41, yincrease=False, ax=axes[1], cmap='bwr', center=0)
    fg.fig.subplots_adjust(top=0.923,
                           bottom=0.102,
                           left=0.058,
                           right=0.818,
                           hspace=0.2,
                           wspace=0.045)
    if save:
        filename = 'moisture_clim_from_ERA5_over_israel.png'
#        plt.savefig(savefig_path / filename, bbox_inches='tight')
        plt.savefig(savefig_path / filename, orientation='landscape')
    return fg


def plot_mean_std_count(da_ts, time_reduce='hour', reduce='mean',
                        count_factor=1):
    import xarray as xr
    import seaborn as sns
    """plot mean, std and count of Xarray dataarray time-series"""
    cmap = sns.color_palette("colorblind", 2)
    time_dim = list(set(da_ts.dims))[0]
    grp = '{}.{}'.format(time_dim, time_reduce)
    if reduce == 'mean':
        mean = da_ts.groupby(grp).mean()
    elif reduce == 'median':
        mean = da_ts.groupby(grp).median()
    std = da_ts.groupby(grp).std()
    mean_plus_std = mean + std
    mean_minus_std = mean - std
    count = da_ts.groupby(grp).count()
    if isinstance(da_ts, xr.Dataset):
        dvars = [x for x in da_ts.data_vars.keys()]
        assert len(dvars) == 2
        secondary_y = dvars[1]
    else:
        secondary_y = None
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(15, 15))
    mean_df = mean.to_dataframe()
    if secondary_y is not None:
        axes[0] = mean_df[dvars[0]].plot(
            ax=axes[0], linewidth=2.0, marker='o', color=cmap[0])
        ax2mean = mean_df[secondary_y].plot(
            ax=axes[0],
            linewidth=2.0,
            marker='s',
            color=cmap[1],
            secondary_y=True)
        h1, l1 = axes[0].get_legend_handles_labels()
        h2, l2 = axes[0].right_ax.get_legend_handles_labels()
        handles = h1 + h2
        labels = l1 + l2
        axes[0].legend(handles, labels)
        axes[0].fill_between(mean_df.index.values,
                             mean_minus_std[dvars[0]].values,
                             mean_plus_std[dvars[0]].values,
                             color=cmap[0],
                             alpha=0.5)
        ax2mean.fill_between(
            mean_df.index.values,
            mean_minus_std[secondary_y].values,
            mean_plus_std[secondary_y].values,
            color=cmap[1],
            alpha=0.5)
        ax2mean.tick_params(axis='y', colors=cmap[1])
    else:
        mean_df.plot(ax=axes[0], linewidth=2.0, marker='o', color=cmap[0])
        axes[0].fill_between(
            mean_df.index.values,
            mean_minus_std.values,
            mean_plus_std.values,
            color=cmap[0],
            alpha=0.5)
    axes[0].grid()
    count_df = count.to_dataframe() / count_factor
    count_df.plot.bar(ax=axes[1], rot=0)
    axes[0].xaxis.set_tick_params(labelbottom=True)
    axes[0].tick_params(axis='y', colors=cmap[0])
    fig.tight_layout()
    if secondary_y is not None:
        return axes, ax2mean
    else:
        return axes


def plot_seasonal_histogram(da, dim='sound_time', xlim=None, xlabel=None,
                            suptitle=''):
    fig_hist, axs = plt.subplots(2, 2, sharex=False, sharey=True,
                                 figsize=(10, 8))
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    cmap = sns.color_palette("colorblind", 4)
    for i, ax in enumerate(axs.flatten()):
        da_season = da.sel(
            {dim: da['{}.season'.format(dim)] == seasons[i]}).dropna(dim)
        ax = sns.distplot(da_season, ax=ax, norm_hist=False,
                          color=cmap[i], hist_kws={'edgecolor': 'k'},
                          axlabel=xlabel,
                          label=seasons[i])
        ax.set_xlim(xlim)
        ax.legend()
    #            axes.set_xlabel('MLH [m]')
        ax.set_ylabel('Frequency')
    fig_hist.suptitle(suptitle)
    fig_hist.tight_layout()
    return axs


def plot_two_histograms_comparison(x, y, bins=None, labels=['x', 'y'],
                                   ax=None, colors=['b', 'r']):
    import numpy as np
    import matplotlib.pyplot as plt
    x_w = np.empty(x.shape)
    x_w.fill(1/x.shape[0])
    y_w = np.empty(y.shape)
    y_w.fill(1/y.shape[0])
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist([x, y], bins=bins, weights=[x_w, y_w], color=colors,
            label=labels)
    ax.legend()
    return ax


def plot_diurnal_wind_hodograph(path=ims_path, station='TEL-AVIV-COAST',
                                season=None, cmax=None, ax=None):
    import xarray as xr
    from metpy.plots import Hodograph
    # import matplotlib
    import numpy as np
    colorbar = False
    # from_list = matplotlib.colors.LinearSegmentedColormap.from_list
    cmap = plt.cm.get_cmap('hsv', 24)
    # cmap = from_list(None, plt.cm.jet(range(0,24)), 24)
    U = xr.open_dataset(path / 'IMS_U_israeli_10mins.nc')
    V = xr.open_dataset(path / 'IMS_V_israeli_10mins.nc')
    u_sta = U[station]
    v_sta = V[station]
    u_sta.load()
    v_sta.load()
    if season is not None:
        print('{} season selected'.format(season))
        u_sta = u_sta.sel(time=u_sta['time.season'] == season)
        v_sta = v_sta.sel(time=v_sta['time.season'] == season)
    u = u_sta.groupby('time.hour').mean()
    v = v_sta.groupby('time.hour').mean()
    if ax is None:
        colorbar = True
        fig, ax = plt.subplots()
    max_uv = max(max(u.values), max(v.values)) + 1
    if cmax is None:
        max_uv = max(max(u.values), max(v.values)) + 1
    else:
        max_uv = cmax
    h = Hodograph(component_range=max_uv, ax=ax)
    h.add_grid(increment=0.5)
    # hours = np.arange(0, 25)
    lc = h.plot_colormapped(u, v, u.hour, cmap=cmap,
                            linestyle='-', linewidth=2)
    #ticks = np.arange(np.min(hours), np.max(hours))
    # cb = fig.colorbar(lc, ticks=range(0,24), label='Time of Day [UTC]')
    if colorbar:
        cb = ax.figure.colorbar(lc, ticks=range(
            0, 24), label='Time of Day [UTC]')
    # cb.ax.tick_params(length=0)
    if season is None:
        ax.figure.suptitle('{} diurnal wind Hodograph'.format(station))
    else:
        ax.figure.suptitle(
            '{} diurnal wind Hodograph {}'.format(station, season))
    ax.set_xlabel('North')
    ax.set_ylabel('East')
    ax.set_title('South')
    ax2 = ax.twinx()
    ax2.tick_params(axis='y', right=False, labelright=False)
    ax2.set_ylabel('West')
    # axcb = fig.colorbar(lc)
    return ax


def plot_MLR_GNSS_PW_harmonics_facetgrid(path=work_yuval, season='JJA',
                                         n_max=2, ylim=None, scope='diurnal',
                                         save=True, era5=False, leg_size=15):
    """


    Parameters
    ----------
    path : TYPE, optional
        DESCRIPTION. The default is work_yuval.
    season : TYPE, optional
        DESCRIPTION. The default is 'JJA'.
    n_max : TYPE, optional
        DESCRIPTION. The default is 2.
    ylim : TYPE, optional
        the ylimits of each panel use [-6,8] for annual. The default is None.
    scope : TYPE, optional
        DESCRIPTION. The default is 'diurnal'.
    save : TYPE, optional
        DESCRIPTION. The default is True.
    era5 : TYPE, optional
        DESCRIPTION. The default is False.
    leg_size : TYPE, optional
        DESCRIPTION. The default is 15.

    Returns
    -------
    None.

    """
    import xarray as xr
    from aux_gps import run_MLR_harmonics
    from matplotlib.ticker import AutoMinorLocator
    from PW_stations import produce_geo_gnss_solved_stations
    import numpy as np
    sns.set_style('whitegrid')
    sns.set_style('ticks')
    geo = produce_geo_gnss_solved_stations(add_distance_to_coast=True, plot=False)
    if scope == 'diurnal':
        cunits = 'cpd'
        ticks = np.arange(0, 23, 3)
        xlabel = 'Hour of day [UTC]'
    elif scope == 'annual':
        cunits = 'cpy'
        ticks = np.arange(1, 13, 1)
        xlabel = 'month'
    print('producing {} harmonics plot.'.format(scope))
    if era5:
        harmonics = xr.load_dataset(path / 'GNSS_PW_ERA5_harmonics_{}.nc'.format('annual'))
    else:
        harmonics = xr.load_dataset(path / 'GNSS_PW_harmonics_{}.nc'.format(scope))
#    sites = sorted(list(set([x.split('_')[0] for x in harmonics])))
#    da = xr.DataArray([x for x in range(len(sites))], dims='GNSS')
#    da['GNSS'] = sites
    sites = group_sites_to_xarray(upper=False, scope=scope)
    sites_flat = [x for x in sites.values.flatten()]
    da = xr.DataArray([x for x in range(len(sites_flat))], dims='GNSS')
    da['GNSS'] = [x for x in range(len(da))]
    fg = xr.plot.FacetGrid(
        da,
        col='GNSS',
        col_wrap=3,
        sharex=False,
        sharey=False, figsize=(20, 20))

    for i in range(fg.axes.shape[0]):  # i is rows
        for j in range(fg.axes.shape[1]):  # j is cols
            site = sites.values[i, j]
            ax = fg.axes[i, j]
            try:
                harm_site = harmonics[[x for x in harmonics if site in x]]
                if site in ['nrif']:
                    leg_loc = 'upper center'
                elif site in ['yrcm', 'ramo']:
                    leg_loc = 'lower center'
#                elif site in ['katz']:
#                    leg_loc = 'upper right'
                else:
                    leg_loc = None
                if scope == 'annual':
                    leg_loc = 'upper left'
                ax, handles, labels = run_MLR_harmonics(harm_site, season=season,
                                                        cunits=cunits,
                                                        n_max=n_max, plot=True, ax=ax,
                                                        legend_loc=leg_loc, ncol=1,
                                                        legsize=leg_size, lw=2.5,
                                                        legend_S_only=True)
                ax.set_xlabel(xlabel, fontsize=16)
                if ylim is not None:
                    ax.set_ylim(*ylim)
                ax.tick_params(axis='x', which='major', labelsize=18)
                # if scope == 'diurnal':
                ax.yaxis.set_major_locator(plt.MaxNLocator(4))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.tick_params(axis='y', which='major', labelsize=18)
                ax.yaxis.tick_left()
                ax.xaxis.set_ticks(ticks)
                ax.grid()
                ax.set_title('')
                ax.set_ylabel('')
                ax.grid(axis='y', which='minor', linestyle='--')
                # get this for upper legend:
                # handles, labels = ax.get_legend_handles_labels()
                if scope == 'annual':
                    site_label = '{} ({:.0f})'.format(
                        site.upper(), geo.loc[site].alt)
                    label_coord = [0.52, 0.87]
                    fs = 18
                elif scope == 'diurnal':
                    site_label = site.upper()
                    label_coord = [0.1, 0.85]
                    fs = 20
                ax.text(*label_coord, site_label,
                        horizontalalignment='center', fontweight='bold',
                        transform=ax.transAxes, fontsize=fs)
                if j == 0:
                    ax.set_ylabel('PWV anomalies [mm]', fontsize=16)
#                if j == 0:
#                    ax.set_ylabel('PW anomalies [mm]', fontsize=12)
#                elif j == 1:
#                    if i>5:
#                        ax.set_ylabel('PW anomalies [mm]', fontsize=12)
            except TypeError:
                print('{}, {} axis off'.format(i, j))
                ax.set_axis_off()

#    for i, (site, ax) in enumerate(zip(da['GNSS'].values, fg.axes.flatten())):
#        harm_site = harmonics[[x for x in harmonics if sites[i] in x]]
#        if site in ['elat', 'nrif']:
#            loc = 'upper center'
#            text = 0.1
#        elif site in ['elro', 'yrcm', 'ramo', 'slom', 'jslm']:
#            loc = 'upper right'
#            text = 0.1
#        else:
#            loc = None
#            text = 0.1
#        ax = run_MLR_diurnal_harmonics(harm_site, season=season, n_max=n_max, plot=True, ax=ax, legend_loc=loc)
#        ax.set_title('')
#        ax.set_ylabel('PW anomalies [mm]')
#        if ylim is not None:
#            ax.set_ylim(ylim[0], ylim[1])
#        ax.text(text, .85, site.upper(),
#                horizontalalignment='center', fontweight='bold',
#                transform=ax.transAxes)
#    for i, ax in enumerate(fg.axes.flatten()):
#        if i > (da.GNSS.telasize-1):
#            ax.set_axis_off()
#            pass
    # add upper legend for all factes:
    S_labels = labels[:-2]
    S_labels = [x.split(' ')[0] for x in S_labels]
    last_label = 'Mean PWV anomalies'
    sum_label = labels[-2].split("'")[1]
    S_labels.append(sum_label)
    S_labels.append(last_label)
    fg.fig.legend(handles=handles, labels=S_labels, prop={'size': 20}, edgecolor='k',
                  framealpha=0.5, fancybox=True, facecolor='white',
                  ncol=5, fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.005),
                  bbox_transform=plt.gcf().transFigure)

    fg.fig.subplots_adjust(
        top=0.973,
        bottom=0.032,
        left=0.054,
        right=0.995,
        hspace=0.15,
        wspace=0.12)
    if save:
        filename = 'pw_{}_harmonics_{}_{}.png'.format(scope, n_max, season)
#        plt.savefig(savefig_path / filename, bbox_inches='tight')
        plt.savefig(savefig_path / filename, orientation='portrait')
    return fg


def plot_gustiness(path=work_yuval, ims_path=ims_path, site='tela',
                   ims_site='HAIFA-TECHNION', season='JJA', month=None, pts=7,
                   ax=None):
    import xarray as xr
    import numpy as np
    g = xr.open_dataset(
        ims_path / 'IMS_G{}_israeli_10mins_daily_anoms.nc'.format(pts))[ims_site]
    g.load()
    if season is not None:
        g = g.sel(time=g['time.season'] == season)
        label = 'Gustiness {} IMS station in {} season'.format(
            site, season)
    elif month is not None:
        g = g.sel(time=g['time.month'] == month)
        label = 'Gustiness {} IMS station in {} month'.format(
            site, month)
    elif season is not None and month is not None:
        raise('pls pick either season or month...')
#    date = groupby_date_xr(g)
#    # g_anoms = g.groupby('time.month') - g.groupby('time.month').mean('time')
#    g_anoms = g.groupby(date) - g.groupby(date).mean('time')
#    g_anoms = g_anoms.reset_coords(drop=True)
    G = g.groupby('time.hour').mean('time') * 100.0
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 8))
    Gline = G.plot(ax=ax, color='b', marker='o', label='Gustiness')
    ax.set_title(label)
    ax.axhline(0, color='b', linestyle='--')
    ax.set_ylabel('Gustiness anomalies [dimensionless]', color='b')
    ax.set_xlabel('Time of day [UTC]')
    # ax.set_xticks(np.arange(0, 24, step=1))
    ax.yaxis.label.set_color('b')
    ax.tick_params(axis='y', colors='b')
    ax.xaxis.set_ticks(np.arange(0, 23, 3))
    ax.grid()
    pw = xr.open_dataset(
        work_yuval /
        'GNSS_PW_hourly_anoms_thresh_50_homogenized.nc')[site]
    pw.load().dropna('time')
    if season is not None:
        pw = pw.sel(time=pw['time.season'] == season)
    elif month is not None:
        pw = pw.sel(time=pw['time.month'] == month)
#    date = groupby_date_xr(pw)
#    pw = pw.groupby(date) - pw.groupby(date).mean('time')
#    pw = pw.reset_coords(drop=True)
    pw = pw.groupby('time.hour').mean()
    axpw = ax.twinx()
    PWline = pw.plot.line(ax=axpw, color='tab:green',
                          marker='s', label='PW ({})'.format(season))
    axpw.axhline(0, color='k', linestyle='--')
    lns = Gline + PWline
    axpw.set_ylabel('PW anomalies [mm]')
    align_yaxis(ax, 0, axpw, 0)
    return lns


def plot_gustiness_facetgrid(path=work_yuval, ims_path=ims_path,
                             season='JJA', month=None, save=True):
    import xarray as xr
    gnss_ims_dict = {
        'alon': 'ASHQELON-PORT', 'bshm': 'HAIFA-TECHNION', 'csar': 'HADERA-PORT',
        'tela': 'TEL-AVIV-COAST', 'slom': 'BESOR-FARM', 'kabr': 'SHAVE-ZIYYON',
        'nzrt': 'DEIR-HANNA', 'katz': 'GAMLA', 'elro': 'MEROM-GOLAN-PICMAN',
        'mrav': 'MAALE-GILBOA', 'yosh': 'ARIEL', 'jslm': 'JERUSALEM-GIVAT-RAM',
        'drag': 'METZOKE-DRAGOT', 'dsea': 'SEDOM', 'ramo': 'MIZPE-RAMON-20120927',
        'nrif': 'NEOT-SMADAR', 'elat': 'ELAT', 'klhv': 'SHANI',
        'yrcm': 'ZOMET-HANEGEV', 'spir': 'PARAN-20060124'}
    da = xr.DataArray([x for x in gnss_ims_dict.values()], dims=['GNSS'])
    da['GNSS'] = [x for x in gnss_ims_dict.keys()]
    to_remove = ['kabr', 'nzrt', 'katz', 'elro', 'klhv', 'yrcm', 'slom']
    sites = [x for x in da['GNSS'].values if x not in to_remove]
    da = da.sel(GNSS=sites)
    gnss_order = ['bshm', 'mrav', 'drag', 'csar', 'yosh', 'dsea', 'tela', 'jslm',
                  'nrif', 'alon', 'ramo', 'elat']
    df = da.to_dataframe('gnss')
    da = df.reindex(gnss_order).to_xarray()['gnss']
    fg = xr.plot.FacetGrid(
        da,
        col='GNSS',
        col_wrap=3,
        sharex=False,
        sharey=False, figsize=(20, 20))
    for i, (site, ax) in enumerate(zip(da['GNSS'].values, fg.axes.flatten())):
        lns = plot_gustiness(path=path, ims_path=ims_path,
                             ims_site=gnss_ims_dict[site],
                             site=site, season=season, month=month, ax=ax)
        labs = [l.get_label() for l in lns]
        if site in ['tela', 'alon', 'dsea', 'csar', 'elat', 'nrif']:
            ax.legend(lns, labs, loc='upper center', prop={
                      'size': 8}, framealpha=0.5, fancybox=True, title=site.upper())
        elif site in ['drag']:
            ax.legend(lns, labs, loc='upper right', prop={
                      'size': 8}, framealpha=0.5, fancybox=True, title=site.upper())
        else:
            ax.legend(lns, labs, loc='best', prop={
                      'size': 8}, framealpha=0.5, fancybox=True, title=site.upper())
        ax.set_title('')
        ax.set_ylabel(r'G anomalies $\times$$10^{2}$')
#        ax.text(.8, .85, site.upper(),
#            horizontalalignment='center', fontweight='bold',
#            transform=ax.transAxes)
    for i, ax in enumerate(fg.axes.flatten()):
        if i > (da.GNSS.size-1):
            ax.set_axis_off()
            pass
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(top=0.974,
                           bottom=0.053,
                           left=0.041,
                           right=0.955,
                           hspace=0.15,
                           wspace=0.3)
    filename = 'gustiness_israeli_gnss_pw_diurnal_{}.png'.format(season)
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_fft_diurnal(path=work_yuval, save=True):
    import xarray as xr
    import numpy as np
    import matplotlib.ticker as tck
    sns.set_style("whitegrid",
                  {'axes.grid': True,
                   'xtick.bottom': True,
                   'font.family': 'serif',
                   'ytick.left': True})
    sns.set_context('paper')
    power = xr.load_dataset(path / 'GNSS_PW_power_spectrum_diurnal.nc')
    power = power.to_array('site')
    sites = [x for x in power.site.values]
    fg = power.plot.line(col='site', col_wrap=4,
                         sharex=False, figsize=(20, 18))
    fg.set_xlabels('Frequency [cpd]')
    fg.set_ylabels('PW PSD [dB]')
    ticklabels = np.arange(0, 7)
    for ax, site in zip(fg.axes.flatten(), sites):
        sns.despine()
        ax.set_title('')
        ax.set_xticklabels(ticklabels)
        # ax.tick_params(axis='y', which='minor')
        ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
        ax.set_xlim(0, 6.5)
        ax.set_ylim(70, 125)
        ax.grid(True)
        ax.grid(which='minor', axis='y')
        ax.text(.8, .85, site.upper(),
                horizontalalignment='center', fontweight='bold',
                transform=ax.transAxes)
    fg.fig.tight_layout()
    filename = 'power_pw_diurnal.png'
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_rinex_availability_with_map(path=work_yuval, gis_path=gis_path,
                                     scope='diurnal', ims=True,
                                     dem_path=dem_path, fontsize=18, save=True):
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
    sns.set_style('whitegrid')
    sns.set_style('ticks')
    print('{} scope selected.'.format(scope))
    fig = plt.figure(figsize=(20, 15))
#    grid = plt.GridSpec(1, 2, width_ratios=[
#        5, 2], wspace=0.1)
    grid = plt.GridSpec(1, 2, width_ratios=[
        5, 3], wspace=0.05)
    ax_gantt = fig.add_subplot(grid[0, 0])  # plt.subplot(221)
    ax_map = fig.add_subplot(grid[0, 1])  # plt.subplot(122)
#    fig, ax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(20, 6))
    # RINEX gantt chart:
    if scope == 'diurnal':
        file = path_glob(path, 'GNSS_PW_thresh_50_for_diurnal_analysis.nc')[-1]
    elif scope == 'annual':
        file = path / 'GNSS_PW_monthly_thresh_50.nc'
    ds = xr.open_dataset(file)
    just_pw = [x for x in ds if 'error' not in x]
    ds = ds[just_pw]
    da = ds.to_array('station').sel(time=slice(None,'2019'))
    da['station'] = [x.upper() for x in da.station.values]
    ds = da.to_dataset('station')
    # reorder for annual, coastal, highland and eastern:
    stns = group_sites_to_xarray(scope='annual', upper=True).T.values.ravel()
    stns = stns[~pd.isnull(stns)]
    ds = ds[stns]
    # colors:
    colors = produce_colors_for_pwv_station(scope=scope, zebra=False)
    title = 'Daily RINEX files availability for the Israeli GNSS stations'
    ax_gantt = gantt_chart(
        ds,
        ax=ax_gantt,
        fw='bold', grid=True,
        title='', colors=colors,
        pe_dict=None, fontsize=fontsize, linewidth=24, antialiased=False)

    years_fmt = mdates.DateFormatter('%Y')
    # ax_gantt.xaxis.set_major_locator(mdates.YearLocator())
    ax_gantt.xaxis.set_major_locator(mdates.YearLocator(4))
    ax_gantt.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax_gantt.xaxis.set_major_formatter(years_fmt)
    # ax_gantt.xaxis.set_minor_formatter(years_fmt)
    ax_gantt.tick_params(axis='x', labelrotation=0)
    # Israel gps ims map:
    ax_map = plot_israel_map(
        gis_path=gis_path, ax=ax_map, ticklabelsize=fontsize)
    # overlay with dem data:
    cmap = plt.get_cmap('terrain', 41)
    dem = xr.open_dataarray(dem_path / 'israel_dem_250_500.nc')
    # dem = xr.open_dataarray(dem_path / 'israel_dem_500_1000.nc')
    fg = dem.plot.imshow(ax=ax_map, alpha=0.5, cmap=cmap,
                         vmin=dem.min(), vmax=dem.max(), add_colorbar=False)
#    scale_bar(ax_map, 50)
    cbar_kwargs = {'fraction': 0.1, 'aspect': 50, 'pad': 0.03}
    cb = plt.colorbar(fg, **cbar_kwargs)
    cb.set_label(label='meters above sea level',
                 size=fontsize, weight='normal')
    cb.ax.tick_params(labelsize=fontsize)
    ax_map.set_xlabel('')
    ax_map.set_ylabel('')
    gps = produce_geo_gnss_solved_stations(path=gis_path, plot=False)
    # removed = ['hrmn', 'nizn', 'spir']
#    removed = ['hrmn']
    if scope == 'diurnal':
        removed = ['hrmn', 'gilb', 'lhav']
    elif scope == 'annual':
        removed = ['hrmn', 'gilb', 'lhav']
    print('removing {} stations from map.'.format(removed))
#    merged = ['klhv', 'lhav', 'mrav', 'gilb']
    merged = []
    gps_list = [x for x in gps.index if x not in merged and x not in removed]
    gps.loc[gps_list, :].plot(ax=ax_map, edgecolor='black', marker='s',
                              alpha=1.0, markersize=35, facecolor="None", linewidth=2, zorder=3)
#    gps.loc[removed, :].plot(ax=ax_map, color='black', edgecolor='black', marker='s',
#            alpha=1.0, markersize=25, facecolor='white')
#    gps.loc[merged, :].plot(ax=ax_map, color='black', edgecolor='r', marker='s',
#            alpha=0.7, markersize=25)
    gps_stations = gps_list  # [x for x in gps.index]
#    to_plot_offset = ['mrav', 'klhv', 'nzrt', 'katz', 'elro']
    to_plot_offset = []

    for x, y, label in zip(gps.loc[gps_stations, :].lon, gps.loc[gps_stations,
                                                                 :].lat, gps.loc[gps_stations, :].index.str.upper()):
        if label.lower() in to_plot_offset:
            ax_map.annotate(label, xy=(x, y), xytext=(4, -6),
                            textcoords="offset points", color='k',
                            fontweight='bold', fontsize=fontsize - 2)
        else:
            ax_map.annotate(label, xy=(x, y), xytext=(3, 3),
                            textcoords="offset points", color='k',
                            fontweight='bold', fontsize=fontsize - 2)
#    geo_annotate(ax_map, gps_normal_anno.lon, gps_normal_anno.lat,
#                 gps_normal_anno.index.str.upper(), xytext=(3, 3), fmt=None,
#                 c='k', fw='normal', fs=10, colorupdown=False)
#    geo_annotate(ax_map, gps_offset_anno.lon, gps_offset_anno.lat,
#                 gps_offset_anno.index.str.upper(), xytext=(4, -6), fmt=None,
#                 c='k', fw='normal', fs=10, colorupdown=False)
    # plot bet-dagan:
    df = pd.Series([32.00, 34.81]).to_frame().T
    df.index = ['Bet-Dagan']
    df.columns = ['lat', 'lon']
    bet_dagan = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon,
                                                                 df.lat),
                                 crs=gps.crs)
    bet_dagan.plot(ax=ax_map, color='black', edgecolor='black',
                   marker='x', linewidth=2, zorder=2)
    geo_annotate(ax_map, bet_dagan.lon, bet_dagan.lat,
                 bet_dagan.index, xytext=(4, -6), fmt=None,
                 c='k', fw='bold', fs=fontsize - 2, colorupdown=False)
#    plt.legend(['GNSS \nreceiver sites',
#                'removed \nGNSS sites',
#                'merged \nGNSS sites',
#                'radiosonde\nstation'],
#               loc='upper left', framealpha=0.7, fancybox=True,
#               handletextpad=0.2, handlelength=1.5)
    if ims:
        print('getting IMS temperature stations metadata...')
        ims = produce_geo_ims(path=gis_path, freq='10mins', plot=False)
        ims.plot(ax=ax_map, marker='o', edgecolor='tab:orange', alpha=1.0,
                 markersize=35, facecolor="tab:orange", zorder=1)
    # ims, gps = produce_geo_df(gis_path=gis_path, plot=False)
        print('getting solved GNSS israeli stations metadata...')
        plt.legend(['GNSS \nstations',
                    'radiosonde\nstation', 'IMS stations'],
                   loc='upper left', framealpha=0.7, fancybox=True,
                   handletextpad=0.2, handlelength=1.5, fontsize=fontsize - 2)
    else:
        plt.legend(['GNSS \nstations',
                    'radiosonde\nstation'],
                   loc='upper left', framealpha=0.7, fancybox=True,
                   handletextpad=0.2, handlelength=1.5, fontsize=fontsize - 2)
    fig.subplots_adjust(top=0.95,
                        bottom=0.11,
                        left=0.05,
                        right=0.95,
                        hspace=0.2,
                        wspace=0.2)
    # plt.legend(['IMS stations', 'GNSS stations'], loc='upper left')

    filename = 'rinex_israeli_gnss_map_{}.png'.format(scope)
#    caption('Daily RINEX files availability for the Israeli GNSS station network at the SOPAC/GARNER website')
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fig


def plot_means_box_plots(path=work_yuval, thresh=50, kind='box',
                         x='month', col_wrap=5, ylimits=None, twin=None,
                         twin_attrs=None,
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
        pw = xr.load_dataset(
            work_yuval / 'GNSS_PW_hourly_thresh_{:.0f}_homogenized.nc'.format(thresh))
        pw = pw[[x for x in pw.data_vars if '_error' not in x]]
        # first remove long term monthly means:
        if anoms:
            pw = xr.load_dataset(
                work_yuval / 'GNSS_PW_hourly_anoms_thresh_{:.0f}_homogenized.nc'.format(thresh))
            if twin is not None:
                twin = twin.groupby('time.month') - \
                    twin.groupby('time.month').mean('time')
                twin = twin.reset_coords(drop=True)
            # pw = pw.groupby('time.month') - pw.groupby('time.month').mean('time')
    elif x == 'day':
        # pw = pw.resample(time='1H').mean('time')
        # pw = pw.groupby('time.hour').mean('time')
        pw = xr.load_dataset(
            work_yuval / 'GNSS_PW_daily_thresh_{:.0f}_homogenized.nc'.format(thresh))
        pw = pw[[x for x in pw.data_vars if '_error' not in x]]
        # first remove long term monthly means:
        if anoms:
            # pw = pw.groupby('time.month') - pw.groupby('time.month').mean('time')
            pw = pw.groupby('time.dayofyear') - \
                pw.groupby('time.dayodyear').mean('time')
    if season is not None:
        if season != 'all':
            print('{} season is selected'.format(season))
            pw = pw.sel(time=pw['time.season'] == season)
            all_seas = False
            if twin is not None:
                twin = twin.sel(time=twin['time.season'] == season)
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
                           bins=bins, all_seasons=all_seas, twin=twin,
                           twin_attrs=twin_attrs)
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
        [fg.axes[x, 0].set_ylabel('PW [mm]')
         for x in range(len(fg.axes[:, 0]))]
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


def plot_interannual_MLR_results(path=climate_path, fontsize=16, save=True):
    import matplotlib.pyplot as plt
    from climate_works import run_best_MLR
#    rds = xr.load_dataset(path / 'best_MLR_interannual_gnss_pwv.nc')
    model_lci, rdf_lci = run_best_MLR(plot=False, heatmap=False, keep='lci',
                                      add_trend=True)
    rds_lci = model_lci.results_
    model_eofi, rdf_eofi = run_best_MLR(plot=False, heatmap=False, keep='eofi',
                                        add_trend=False)
    rds_eofi = model_eofi.results_
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(15, 7))
    origln = rds_lci['original'].plot.line('k-.', ax=axes[0], linewidth=1.5)
    predln_lci = rds_lci['predict'].plot.line('b-', ax=axes[0], linewidth=1.5)
    predln_eofi = rds_eofi['predict'].plot.line(
        'g-', ax=axes[0], linewidth=1.5)
    r2_lci = rds_lci['r2_adj'].item()
    r2_eofi = rds_eofi['r2_adj'].item()
    axes[0].legend(origln+predln_lci+predln_eofi, ['mean PWV (12m-mean)', 'MLR with LCI (Adj R$^2$:{:.2f})'.format(
        r2_lci), 'MLR with EOFs (Adj R$^2$:{:.2f})'.format(r2_eofi)], fontsize=fontsize-2)
    axes[0].grid()
    axes[0].set_xlabel('')
    axes[0].set_ylabel('PWV anomalies [mm]', fontsize=fontsize)
    axes[0].tick_params(labelsize=fontsize)
    axes[0].grid(which='minor', color='k', linestyle='--')
    residln_lci = rds_lci['resid'].plot.line('b-', ax=axes[1])
    residln_eofi = rds_eofi['resid'].plot.line('g-', ax=axes[1])
    axes[1].legend(residln_lci+residln_eofi, ['MLR with LCI',
                                              'MLR with EOFs'], fontsize=fontsize-2)
    axes[1].grid()
    axes[1].set_ylabel('Residuals [mm]', fontsize=fontsize)
    axes[1].tick_params(labelsize=fontsize)
    axes[1].set_xlabel('')
    years_fmt = mdates.DateFormatter('%Y')
    # ax.figure.autofmt_xdate()
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_minor_locator(mdates.YearLocator(1))
    axes[1].xaxis.set_major_formatter(years_fmt)
    axes[1].grid(which='minor', color='k', linestyle='--')
    # ax.xaxis.set_minor_locator(mdates.MonthLocator())
    axes[1].figure.autofmt_xdate()

    fig.tight_layout()
    fig.subplots_adjust()
    if save:
        filename = 'pw_interannual_MLR_comparison.png'
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fig


def plot_annual_pw(path=work_yuval, fontsize=20, labelsize=18, compare='uerra',
                   ylim=[7.5, 40], save=True, kind='violin', bins=None, ds=None,
                   add_temperature=False):
    """kind can be violin or hist, for violin choose ylim=7.5,40 and for hist
    choose ylim=0,0.3"""
    import xarray as xr
    import pandas as pd
    import numpy as np
    from synoptic_procedures import slice_xr_with_synoptic_class
    gnss_filename = 'GNSS_PW_monthly_thresh_50.nc'
#    gnss_filename = 'first_climatol_try.nc'
    pw = xr.load_dataset(path / gnss_filename)
    df_annual = pw.to_dataframe()
    hue = None
    if compare is not None:
        df_annual = prepare_reanalysis_monthly_pwv_to_dataframe(
            path, re=compare, ds=ds)
        hue = 'source'
    if not add_temperature:
        fg = plot_pw_geographical_segments(
            df_annual, scope='annual',
            kind=kind,
            fg=None,
            ylim=ylim,
            fontsize=fontsize,
            labelsize=labelsize, hue=hue,
            save=False, bins=bins)
        fg.fig.subplots_adjust(
            top=0.973,
            bottom=0.029,
            left=0.054,
            right=0.995,
            hspace=0.15,
            wspace=0.12)
        filename = 'pw_annual_means_{}.png'.format(kind)
    else:
        fg = plot_pw_geographical_segments(
            df_annual, scope='annual',
            kind='mean_month',
            fg=None, ticklabelcolor='tab:blue',
            ylim=[10, 31], color='tab:blue',
            fontsize=fontsize,
            labelsize=labelsize, hue=None,
            save=False, bins=None)
        # tmm = xr.load_dataset(path / 'GNSS_TD_monthly_1996_2020.nc')
        tmm = xr.load_dataset(path / 'IMS_T/GNSS_TD_daily.nc')
        tmm = tmm.groupby('time.month').mean()
        dftm = tmm.to_dataframe()
        # dftm.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        sites = group_sites_to_xarray(scope='annual')
        sites_flat = sites.values.ravel()
        # sites = sites[~pd.isnull(sites)]
        for i, ax in enumerate(fg.axes.flat):
            if pd.isnull(sites_flat[i]):
                continue
            twinax = ax.twinx()
            twinax.plot(dftm.index.values, dftm[sites_flat[i]].values, color='tab:red',
                        markersize=10, marker='s', lw=1, markerfacecolor="None",
                        label='Temperature')
            # dftm[sites[i]].plot(ax=twinax, color='r', markersize=10,
            #                     marker='s', lw=1, markerfacecolor="None")
            twinax.set_ylim(5, 37)
            twinax.set_yticks(np.arange(5, 40, 10))
            twinax.tick_params(axis='y', which='major', labelcolor='tab:red',
                               labelsize=labelsize)
            if sites_flat[i] in sites.sel(group='eastern'):
                twinax.set_ylabel(r'Temperature [$\degree$ C]',
                                  fontsize=labelsize)
            # fg.fig.canvas.draw()
            # twinax.xaxis.set_ticks(np.arange(1, 13))
            # twinax.tick_params(axis='x', which='major', labelsize=labelsize-2)
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = twinax.get_legend_handles_labels()
        labels = ['PWV', 'Surface Temperature']
        fg.fig.legend(handles=lines+lines2, labels=labels, prop={'size': 20}, edgecolor='k',
                      framealpha=0.5, fancybox=True, facecolor='white',
                      ncol=5, fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.005),
                      bbox_transform=plt.gcf().transFigure)
        fg.fig.subplots_adjust(
            top=0.97,
            bottom=0.029,
            left=0.049,
            right=0.96,
            hspace=0.15,
            wspace=0.17)
        filename = 'pw_annual_means_temperature.png'
    if save:
        if compare is not None:
            filename = 'pw_annual_means_{}_with_{}.png'.format(kind, compare)
        plt.savefig(savefig_path / filename, orientation='portrait')
    return fg


def plot_multi_box_xr(pw, kind='violin', x='month', sharex=False, sharey=False,
                      col_wrap=5, ylimits=None, xlimits=None, attrs=None,
                      bins=None, all_seasons=False, twin=None, twin_attrs=None):
    import xarray as xr
    pw = pw.to_array('station')
    if twin is not None:
        twin = twin.to_array('station')
    fg = xr.plot.FacetGrid(pw, col='station', col_wrap=col_wrap, sharex=sharex,
                           sharey=sharey)
    for i, (sta, ax) in enumerate(zip(pw['station'].values, fg.axes.flatten())):
        pw_sta = pw.sel(station=sta).reset_coords(drop=True)
        if all_seasons:
            pw_seas = pw_sta.sel(time=pw_sta['time.season'] == 'DJF')
            df = pw_seas.to_dataframe(sta)
            plot_box_df(df, ax=ax, x=x, title=sta, ylabel='', kind=kind,
                        ylimits=ylimits, xlimits=xlimits, attrs=None, bins=bins,
                        marker='o')
            pw_seas = pw_sta.sel(time=pw_sta['time.season'] == 'MAM')
            df = pw_seas.to_dataframe(sta)
            plot_box_df(df, ax=ax, x=x, title=sta, ylabel='', kind=kind,
                        ylimits=ylimits, xlimits=xlimits, attrs=None, bins=bins,
                        marker='^')
            pw_seas = pw_sta.sel(time=pw_sta['time.season'] == 'JJA')
            df = pw_seas.to_dataframe(sta)
            plot_box_df(df, ax=ax, x=x, title=sta, ylabel='', kind=kind,
                        ylimits=ylimits, xlimits=xlimits, attrs=None, bins=bins,
                        marker='s')
            pw_seas = pw_sta.sel(time=pw_sta['time.season'] == 'SON')
            df = pw_seas.to_dataframe(sta)
            plot_box_df(df, ax=ax, x=x, title=sta, ylabel='', kind=kind,
                        ylimits=ylimits, xlimits=xlimits, attrs=attrs[i], bins=bins,
                        marker='x')
            df = pw_sta.to_dataframe(sta)
            plot_box_df(df, ax=ax, x=x, title=sta, ylabel='', kind=kind,
                        ylimits=ylimits, xlimits=xlimits, attrs=attrs[i], bins=bins,
                        marker='d')
            if sta == 'nrif' or sta == 'elat':
                ax.legend(['DJF', 'MAM', 'JJA', 'SON', 'Annual'],
                          prop={'size': 8}, loc='upper center', framealpha=0.5, fancybox=True)
            elif sta == 'yrcm' or sta == 'ramo':
                ax.legend(['DJF', 'MAM', 'JJA', 'SON', 'Annual'],
                          prop={'size': 8}, loc='upper right', framealpha=0.5, fancybox=True)
            else:
                ax.legend(['DJF', 'MAM', 'JJA', 'SON', 'Annual'],
                          prop={'size': 8}, loc='best', framealpha=0.5, fancybox=True)
        else:
            # if x == 'hour':
            #     # remove seasonal signal:
            #     pw_sta = pw_sta.groupby('time.dayofyear') - pw_sta.groupby('time.dayofyear').mean('time')
            # elif x == 'month':
            #     # remove daily signal:
            #     pw_sta = pw_sta.groupby('time.hour') - pw_sta.groupby('time.hour').mean('time')
            df = pw_sta.to_dataframe(sta)
            if twin is not None:
                twin_sta = twin.sel(station=sta).reset_coords(drop=True)
                twin_df = twin_sta.to_dataframe(sta)
            else:
                twin_df = None
            if attrs is not None:
                plot_box_df(df, ax=ax, x=x, title=sta, ylabel='', kind=kind,
                            ylimits=ylimits, xlimits=xlimits, attrs=attrs[i],
                            bins=bins, twin_df=twin_df, twin_attrs=twin_attrs)
            else:
                plot_box_df(df, ax=ax, x=x, title=sta, ylabel='', kind=kind,
                            ylimits=ylimits, xlimits=xlimits, attrs=None,
                            bins=bins, twin_df=twin_df, twin_attrs=twin_attrs)
    return fg


def plot_box_df(df, x='month', title='TELA', marker='o',
                ylabel=r'IWV [kg$\cdot$m$^{-2}$]', ax=None, kind='violin',
                ylimits=(5, 40), xlimits=None, attrs=None, bins=None, twin_df=None,
                twin_attrs=None):
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
        if twin_df is not None:
            twin_df[x] = twin_df.index.hour
        # df[x] = df.index
        pal = sns.color_palette("Paired", 12)
    y = df.columns[0]
    if ax is None:
        fig, ax = plt.subplots()
    if kind is None:
        df = df.groupby(x).mean()
        df.plot(ax=ax, legend=False, marker=marker)
        if twin_df is not None:
            twin_df = twin_df.groupby(x).mean()
            twinx = ax.twinx()
            twin_df.plot.line(ax=twinx, color='r', marker='s')
            ax.axhline(0, color='k', linestyle='--')
            if twin_attrs is not None:
                twinx.set_ylabel(twin_attrs['ylabel'])
            align_yaxis(ax, 0, twinx, 0)
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
                    whis=1.0, flierprops=kwargs, showfliers=False)
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
        ax.legend(['Mean: {:.1f}'.format(xmean),
                   'Median: {:.1f}'.format(xmedian)])
        ax.text(0.55, 0.45, "Std-Dev:    {:.1f}\nSkewness: {:.1f}\nKurtosis:   {:.1f}".format(
            std, sk, kurt), transform=ax.transAxes)
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
        if twin_attrs is not None:
            twinx.set_ylim(*twin_attrs['ylimits'])
            align_yaxis(ax, 0, twinx, 0)
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
        pw_clim = pw.groupby('time.month') - \
            pw.groupby('time.month').mean('time')
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
        east = east.resample(sound_time=sample).mean().sel(
            Height=east_height, method='nearest')
        east_df = east.reset_coords(drop=True).to_dataframe()
    if times is not None:
        phys = phys.sel(sound_time=slice(*times))
    ds = phys.resample(sound_time=sample).mean(
    ).to_dataset(name='Bet-dagan-radiosonde')
    ds = ds.rename({'sound_time': 'time'})
    gps = xr.load_dataset(
        path / 'GNSS_PW_thresh_50_homogenized.nc')[gps_station]
    if times is not None:
        gps = gps.sel(time=slice(*times))
    ds[gps_station] = gps.resample(time=sample).mean()
    df = ds.to_dataframe()
    # now plot:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    # [x.set_xlim([pd.to_datetime(times[0]), pd.to_datetime(times[1])])
    #  for x in axes]
    df.columns = ['Bet dagan soundings', '{} GNSS station'.format(gps_station)]
    sns.lineplot(data=df, markers=['o', 's'], linewidth=2.0, ax=axes[0])
    # axes[0].legend(['Bet_Dagan soundings', 'TELA GPS station'])
    df_r = df.iloc[:, 1] - df.iloc[:, 0]
    df_r.columns = ['Residual distribution']
    sns.lineplot(data=df_r, color='k', marker='o', linewidth=1.5, ax=axes[1])
    if east_height is not None:
        ax_east = axes[1].twinx()
        sns.lineplot(data=east_df, color='red',
                     marker='x', linewidth=1.5, ax=ax_east)
        ax_east.set_ylabel(
            'East drift at {} km altitude [km]'.format(east_height / 1000.0))
    axes[1].axhline(y=0, color='r')
    axes[0].grid(b=True, which='major')
    axes[1].grid(b=True, which='major')
    axes[0].set_ylabel('Precipitable Water [mm]')
    axes[1].set_ylabel('Residuals [mm]')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.01)
    return ds


def plot_wetz_example(path=tela_results_path, plot='WetZ', fontsize=16,
                      save=True):
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
    ax.set_ylabel('WetZ [{}]'.format(unit), fontsize=fontsize)
    ax.set_xlabel('Time [UTC]', fontsize=fontsize)
    ax.tick_params(which='both', labelsize=fontsize)
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
            axes[j - 1].axvline(x=start, color='r', alpha=0.85,
                                linestyle='--', linewidth=2.0)
            axes[j - 1].axvline(x=end, color='r', alpha=0.85,
                                linestyle='--', linewidth=2.0)
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
               times=['2007', '2019'], fontsize=14, save=True):
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
                                            coef, inter), 'Bevis 1992 et al. (0.72, 70.0)'], fontsize=fontsize-4)
#    ax.set_xlabel('Surface Temperature [K]')
#    ax.set_ylabel('Water Vapor Mean Atmospheric Temperature [K]')
    ax.set_xlabel('Ts [K]', fontsize=fontsize)
    ax.set_ylabel('Tm [K]', fontsize=fontsize)
    ax.set_ylim(265, 320)
    ax.tick_params(labelsize=fontsize)
    axin1 = inset_axes(ax, width="40%", height="40%", loc=2)
    resid = predict - y
    sns.distplot(resid, bins=50, color='k', label='residuals', ax=axin1,
                 kde=False,
                 hist_kws={"linewidth": 1, "alpha": 0.5, "color": "k", 'edgecolor': 'k'})
    axin1.yaxis.tick_right()
    rmean = np.mean(resid)
    rmse = np.sqrt(mean_squared_error(y, predict))
    print(rmean, rmse)
    r2 = r2_score(y, predict)
    axin1.axvline(rmean, color='r', linestyle='dashed', linewidth=1)
    # axin1.set_xlabel('Residual distribution[K]')
    textstr = '\n'.join(['n={}'.format(pds.Ts.size),
                         'RMSE: ', '{:.2f} K'.format(rmse)])  # ,
    # r'R$^2$: {:.2f}'.format(r2)])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axin1.text(0.05, 0.95, textstr, transform=axin1.transAxes, fontsize=14,
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
    caption('Water vapor mean temperature (Tm) vs. surface temperature (Ts) of the Bet-Dagan radiosonde station. Ordinary least squares linear fit(red) yields the residual distribution with RMSE of 4 K. Bevis(1992) model is plotted(purple) for comparison.')
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return


def plot_pw_tela_bet_dagan_scatterplot(path=work_yuval, sound_path=sound_path,
                                       ims_path=ims_path, station='tela',
                                       cats=None,
                                       times=['2007', '2019'], wv_name='pw',
                                       r2=False, fontsize=14,
                                       save=True):
    """plot the PW of Bet-Dagan vs. PW of gps station"""
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
    ax.legend(['y = x'], loc='upper right', fontsize=fontsize)
    if wv_name == 'pw':
        ax.set_xlabel('PWV from Bet-Dagan [mm]', fontsize=fontsize)
        ax.set_ylabel('PWV from TELA GPS station [mm]', fontsize=fontsize)
    elif wv_name == 'iwv':
        ax.set_xlabel(
            r'IWV from Bet-Dagan station [kg$\cdot$m$^{-2}$]', fontsize=fontsize)
        ax.set_ylabel(
            r'IWV from TELA GPS station [kg$\cdot$m$^{-2}$]', fontsize=fontsize)
    ax.grid()
    axin1 = inset_axes(ax, width="40%", height="40%", loc=2)
    resid = ds.tela_pw.values - ds[tpw].values
    sns.distplot(resid, bins=50, color='k', label='residuals', ax=axin1,
                 kde=False,
                 hist_kws={"linewidth": 1, "alpha": 0.5, "color": "k", "edgecolor": 'k'})
    axin1.yaxis.tick_right()
    rmean = np.mean(resid)
    rmse = np.sqrt(mean_squared_error(ds[tpw].values, ds.tela_pw.values))
    r2s = r2_score(ds[tpw].values, ds.tela_pw.values)
    axin1.axvline(rmean, color='r', linestyle='dashed', linewidth=1)
    # axin1.set_xlabel('Residual distribution[mm]')
    ax.tick_params(labelsize=fontsize)
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
                                 r'bias: {:.2f} kg$\cdot$m$^{{-2}}$'.format(
                                     rmean),
                                 r'RMSE: {:.2f} kg$\cdot$m$^{{-2}}$'.format(
                                     rmse),
                                 r'R$^2$: {:.2f}'.format(r2s)])
        else:
            textstr = '\n'.join(['n={}'.format(ds[tpw].size),
                                 r'bias: {:.2f} kg$\cdot$m$^{{-2}}$'.format(
                                     rmean),
                                 r'RMSE: {:.2f} kg$\cdot$m$^{{-2}}$'.format(rmse)])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axin1.text(0.05, 0.95, textstr, transform=axin1.transAxes, fontsize=14,
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
    caption(
        'PW from TELA GNSS station vs. PW from Bet-Dagan radiosonde station in {}-{}. A 45 degree line is plotted(red) for comparison. Note the skew in the residual distribution with an RMSE of 4.37 mm.'.format(times[0], times[1]))
    # fig.subplots_adjust(top=0.95)
    filename = 'Bet_dagan_tela_pw_compare_{}-{}.png'.format(times[0], times[1])
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return ds


def plot_tela_bet_dagan_comparison(path=work_yuval, sound_path=sound_path,
                                   ims_path=ims_path, station='tela',
                                   times=['2007', '2020'], cats=None,
                                   compare='pwv',
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
    if compare == 'zwd':
        df = ds[['zwd_bet_dagan', 'tela']].to_dataframe()
    elif compare == 'pwv':
        df = ds[['tpw_bet_dagan', 'tela_pw']].to_dataframe()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    df.columns = ['Bet-Dagan soundings', 'TELA GNSS station']
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
    if compare == 'zwd':
        axes[0].set_ylabel('Zenith Wet Delay [cm]')
        axes[1].set_ylabel('Residuals [cm]')
    elif compare == 'pwv':
        axes[0].set_ylabel('Precipitable Water Vapor [mm]')
        axes[1].set_ylabel('Residuals [mm]')
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
    [x.set_xlim(*[pd.to_datetime(times[0]), pd.to_datetime(times[1])])
     for x in axes]
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.01)
    filename = 'Bet_dagan_tela_{}_compare.png'.format(compare)
    caption('Top: zenith wet delay from Bet-dagan radiosonde station(blue circles) and from TELA GNSS station(orange x) in 2007-2019. Bottom: residuals. Note the residuals become constrained from 08-2013 probebly due to an equipment change.')
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return df


def plot_israel_map(gis_path=gis_path, rc=rc, ticklabelsize=12, ax=None):
    """general nice map for israel, need that to plot stations,
    and temperature field on top of it"""
    import geopandas as gpd
    import contextily as ctx
    import seaborn as sns
    import cartopy.crs as ccrs
    sns.set_style("ticks", rc=rc)
    isr_with_yosh = gpd.read_file(gis_path / 'Israel_and_Yosh.shp')
    isr_with_yosh.crs = {'init': 'epsg:4326'}
#    isr_with_yosh = isr_with_yosh.to_crs(epsg=3857)
    crs_epsg = ccrs.epsg('3857')
#    crs_epsg = ccrs.epsg('2039')
    if ax is None:
        #        fig, ax = plt.subplots(subplot_kw={'projection': crs_epsg},
        #                               figsize=(6, 15))
        bounds = isr_with_yosh.geometry.total_bounds
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
        # ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], crs=crs_epsg)
        # ax.add_geometries(isr_with_yosh.geometry, crs=crs_epsg)
        ax = isr_with_yosh.plot(alpha=0.0, figsize=(6, 15))
    else:
        isr_with_yosh.plot(alpha=0.0, ax=ax)
    ctx.add_basemap(
        ax,
        url=ctx.sources.ST_TERRAIN_BACKGROUND,
        crs='epsg:4326')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(2))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(top=True, bottom=True, left=True, right=True,
                   direction='out', labelsize=ticklabelsize)
#    scale_bar(ax, ccrs.Mercator(), 50, bounds=bounds)
    return ax


def plot_israel_with_stations(gis_path=gis_path, dem_path=dem_path, ims=True,
                              gps=True, radio=True, terrain=True, alt=False,
                              ims_names=False, gps_final=False, save=True):
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
        if ims_names:
            geo_annotate(ax, ims_t.lon, ims_t.lat,
                         ims_t['name_english'], xytext=(3, 3), fmt=None,
                         c='k', fw='normal', fs=7, colorupdown=False)
    # ims, gps = produce_geo_df(gis_path=gis_path, plot=False)
    if gps:
        print('getting solved GNSS israeli stations metadata...')
        gps_df = produce_geo_gnss_solved_stations(path=gis_path, plot=False)
        if gps_final:
            to_drop = ['gilb', 'lhav', 'hrmn', 'nizn', 'spir']
            gps_final_stations = [x for x in gps_df.index if x not in to_drop]
            gps = gps_df.loc[gps_final_stations, :]
        gps.plot(ax=ax, color='k', edgecolor='black', marker='s')
        gps_stations = [x for x in gps.index]
        to_plot_offset = ['gilb', 'lhav']
        # [gps_stations.remove(x) for x in to_plot_offset]
        gps_normal_anno = gps.loc[gps_stations, :]
        # gps_offset_anno = gps.loc[to_plot_offset, :]
        geo_annotate(ax, gps_normal_anno.lon, gps_normal_anno.lat,
                     gps_normal_anno.index.str.upper(), xytext=(3, 3), fmt=None,
                     c='k', fw='bold', fs=10, colorupdown=False)
        if alt:
            geo_annotate(ax, gps_normal_anno.lon, gps_normal_anno.lat,
                         gps_normal_anno.alt, xytext=(4, -6), fmt='{:.0f}',
                         c='k', fw='bold', fs=9, colorupdown=False)
#        geo_annotate(ax, gps_offset_anno.lon, gps_offset_anno.lat,
#                     gps_offset_anno.index.str.upper(), xytext=(4, -6), fmt=None,
#                     c='k', fw='bold', fs=10, colorupdown=False)
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
        df.index = ['Bet-Dagan']
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


def plot_zwd_lapse_rate(path=work_yuval, fontsize=18, model='TSEN', save=True):
    from PW_stations import calculate_zwd_altitude_fit
    df, zwd_lapse_rate = calculate_zwd_altitude_fit(path=path, model=model,
                                                    plot=True, fontsize=fontsize)
    if save:
        filename = 'zwd_lapse_rate.png'
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return


def plot_ims_T_lapse_rate(ims_path=ims_path, dt='2013-10-19T22:00:00',
                          fontsize=16, save=True):
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
    ax_lapse.set_xlabel('Altitude [m]', fontsize=fontsize)
    ax_lapse.set_ylabel(r'Temperature [$\degree$C]', fontsize=fontsize)
    ax_lapse.text(0.5, 0.95, r'Lapse rate: {:.2f} $\degree$C/km'.format(lapse_rate),
                  horizontalalignment='center', verticalalignment='center',
                  fontsize=fontsize,
                  transform=ax_lapse.transAxes, color='k')
    ax_lapse.grid()
    ax_lapse.tick_params(labelsize=fontsize)
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
            filename = 'hydro_tide_lag_pw_anom_max{}.png'.format(
                max_flow_thresh)
    else:
        filename = 'hydro_tide_lag_pw.png'
        if max_flow_thresh:
            filename = 'hydro_tide_lag_pw_anom_max{}.png'.format(
                max_flow_thresh)
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return ax


def produce_table_1(removed=['hrmn', 'nizn', 'spir'], merged={'klhv': ['klhv', 'lhav'],
                                                              'mrav': ['gilb', 'mrav']}, add_location=False,
                    scope='annual', remove_distance=True):
    """for scope='diurnal' use removed=['hrmn'], add_location=True
    and remove_distance=False"""
    from PW_stations import produce_geo_gnss_solved_stations
    import pandas as pd
    sites = group_sites_to_xarray(upper=False, scope=scope)
    df_gnss = produce_geo_gnss_solved_stations(plot=False,
                                               add_distance_to_coast=True)
    new = sites.T.values.ravel()
    if scope == 'annual':
        new = [x for x in new.astype(str) if x != 'nan']
    df_gnss = df_gnss.reindex(new)
    df_gnss['ID'] = df_gnss.index.str.upper()
    pd.options.display.float_format = '{:.2f}'.format
    df = df_gnss[['name', 'ID', 'lat', 'lon', 'alt', 'distance']]
    df['alt'] = df['alt'].map('{:,.0f}'.format)
    df['distance'] = df['distance'].astype(int)
    cols = ['GNSS Station name', 'Station ID', 'Latitude [N]',
            'Longitude [E]', 'Altitude [m a.s.l]', 'Distance from shore [km]']
    df.columns = cols
    if scope != 'annual':
        df.loc['spir', 'GNSS Station name'] = 'Sapir'
    if remove_distance:
        df = df.iloc[:, 0:-1]
    if add_location:
        groups = group_sites_to_xarray(upper=False, scope=scope)
        coastal = groups.sel(group='coastal').values
        coastal = coastal[~pd.isnull(coastal)]
        highland = groups.sel(group='highland').values
        highland = highland[~pd.isnull(highland)]
        eastern = groups.sel(group='eastern').values
        eastern = eastern[~pd.isnull(eastern)]
        df.loc[coastal, 'Location'] = 'Coastal'
        df.loc[highland, 'Location'] = 'Highland'
        df.loc[eastern, 'Location'] = 'Eastern'
    if removed is not None:
        df = df.loc[[x for x in df.index if x not in removed], :]
    if merged is not None:
        return df
    print(df.to_latex(index=False))
    return df


def produce_table_stats(thresh=50, add_location=True, add_height=True):
    """add plot sd to height with se_sd errorbars"""
    from PW_stations import produce_pw_statistics
    from PW_stations import produce_geo_gnss_solved_stations
    import pandas as pd
    import xarray as xr
    sites = group_sites_to_xarray(upper=False, scope='annual')
    new = sites.T.values.ravel()
    sites = group_sites_to_xarray(upper=False, scope='annual')
    new = [x for x in new.astype(str) if x != 'nan']
    pw_mm = xr.load_dataset(
        work_yuval /
        'GNSS_PW_monthly_thresh_{:.0f}.nc'.format(thresh))

    pw_mm = pw_mm[new]
    df = produce_pw_statistics(
        thresh=thresh, resample_to_mm=False, pw_input=pw_mm)
    if add_location:
        cols = [x for x in df.columns]
        cols.insert(1, 'Location')
        gr_df = sites.to_dataframe('sites')
        location = [gr_df[gr_df == x].dropna().index.values.item()[
            1].title() for x in new]
        df['Location'] = location
        df = df[cols]
    if add_height:
        cols = [x for x in df.columns]
        if add_location:
            cols.insert(2, 'Height [m a.s.l]')
        else:
            cols.insert(1, 'Height [m a.s.l]')
        df_gnss = produce_geo_gnss_solved_stations(plot=False,
                                                   add_distance_to_coast=False)
        #    pd.options.display.float_format = '{:.2f}'.format
        df['Height [m a.s.l]'] = df_gnss['alt'].map('{:.0f}'.format)
        df = df[cols]
    print(df.to_latex(index=False))
    return df


def plot_pwv_longterm_trend(path=work_yuval, model_name='LR', save=True,
                            fontsize=16, add_era5=True):
    import matplotlib.pyplot as plt
    from aux_gps import linear_fit_using_scipy_da_ts
#    from PW_stations import ML_Switcher
    import xarray as xr
    from aux_gps import anomalize_xr
    """TSEN and LR for linear fit"""
    # load GNSS Israel:
#    pw = xr.load_dataset(path / 'GNSS_PW_monthly_thresh_50_homogenized.nc')
    pw = xr.load_dataset(
        path / 'GNSS_PW_monthly_thresh_50.nc').sel(time=slice('1998', None))
    pw_anoms = anomalize_xr(pw, 'MS', verbose=False)
    pw_mean = pw_anoms.to_array('station').mean('station')
    pw_std = pw_anoms.to_array('station').std('station')
    pw_weights = 1 / pw_anoms.to_array('station').count('station')
    # add ERA5:
    era5 = xr.load_dataset(work_yuval / 'GNSS_era5_monthly_PW.nc')
    era5_anoms = anomalize_xr(era5, 'MS', verbose=False)
    era5_anoms = era5_anoms.sel(time=slice(
        pw_mean.time.min(), pw_mean.time.max()))
    era5_mean = era5_anoms.to_array('station').mean('station')
    era5_std = era5_anoms.to_array('station').std('station')
    # init linear models
#    ml = ML_Switcher()
#    model = ml.pick_model(model_name)
    if add_era5:
        fig, ax = plt.subplots(2, 1, figsize=(15, 7.5))
        trend, trend_hi, trend_lo, slope, slope_hi, slope_lo = linear_fit_using_scipy_da_ts(pw_mean, model=model_name, slope_factor=3650.25,
                                                                                            plot=False, ax=None, units=None, method='curve_fit', weights=pw_weights)
        pwln = pw_mean.plot(ax=ax[0], color='k', marker='o', linewidth=1.5)
        trendln = trend.plot(ax=ax[0], color='r', linewidth=2)
        trend_hi.plot.line('r--', ax=ax[0], linewidth=1.5)
        trend_lo.plot.line('r--', ax=ax[0], linewidth=1.5)
        trend_label = '{} model, slope={:.2f} ({:.2f}, {:.2f}) mm/decade'.format(
            model_name, slope, slope_lo, slope_hi)
        handles = pwln+trendln
        labels = ['PWV-mean']
        labels.append(trend_label)
        ax[0].legend(handles=handles, labels=labels, loc='upper left',
                     fontsize=fontsize)
        ax[0].grid()
        ax[0].set_xlabel('')
        ax[0].set_ylabel('PWV mean anomalies [mm]', fontsize=fontsize)
        ax[0].tick_params(labelsize=fontsize)
        trend1, trend_hi1, trend_lo1, slope1, slope_hi1, slope_lo1 = linear_fit_using_scipy_da_ts(era5_mean, model=model_name, slope_factor=3650.25,
                                                                                                  plot=False, ax=None, units=None, method='curve_fit', weights=era5_std)

        era5ln = era5_mean.plot(ax=ax[1], color='k', marker='o', linewidth=1.5)
        trendln1 = trend1.plot(ax=ax[1], color='r', linewidth=2)
        trend_hi1.plot.line('r--', ax=ax[1], linewidth=1.5)
        trend_lo1.plot.line('r--', ax=ax[1], linewidth=1.5)
        trend_label = '{} model, slope={:.2f} ({:.2f}, {:.2f}) mm/decade'.format(
            model_name, slope1, slope_lo1, slope_hi1)
        handles = era5ln+trendln1
        labels = ['ERA5-mean']
        labels.append(trend_label)
        ax[1].legend(handles=handles, labels=labels, loc='upper left',
                     fontsize=fontsize)
        ax[1].grid()
        ax[1].set_xlabel('')
        ax[1].set_ylabel('PWV mean anomalies [mm]', fontsize=fontsize)
        ax[1].tick_params(labelsize=fontsize)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(15, 5.5))
        trend, trend_hi, trend_lo, slope, slope_hi, slope_lo = linear_fit_using_scipy_da_ts(pw_mean, model=model_name, slope_factor=3650.25,
                                                                                            plot=False, ax=None, units=None)
        pwln = pw_mean.plot(ax=ax, color='k', marker='o', linewidth=1.5)
        trendln = trend.plot(ax=ax, color='r', linewidth=2)
        trend_hi.plot.line('r--', ax=ax, linewidth=1.5)
        trend_lo.plot.line('r--', ax=ax, linewidth=1.5)
        trend_label = '{} model, slope={:.2f} ({:.2f}, {:.2f}) mm/decade'.format(
            model_name, slope, slope_lo, slope_hi)
        handles = pwln+trendln
        labels = ['PWV-mean']
        labels.append(trend_label)
        ax.legend(handles=handles, labels=labels, loc='upper left',
                  fontsize=fontsize)
        ax.grid()
        ax.set_xlabel('')
        ax.set_ylabel('PWV mean anomalies [mm]', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
    fig.suptitle('PWV mean anomalies and linear trend',
                 fontweight='bold', fontsize=fontsize)

    fig.tight_layout()
    if save:
        filename = 'pwv_mean_trend_{}.png'.format(model_name)
        plt.savefig(savefig_path / filename, orientation='portrait')
    return ax


def plot_trend_filled_pwv_and_era5_barh_plot(path=work_yuval):
    import xarray as xr
    from aux_gps import path_glob
    from PW_stations import process_mkt_from_dataset
    import pandas as pd
    import seaborn as sns
    file = sorted(
        path_glob(path, 'GNSS_PW_monthly_homogenized_filled_*.nc'))[0]
    gnss = xr.load_dataset(path / file)
    era5 = xr.load_dataset(path / 'GNSS_era5_monthly_PW.nc')
    era5 = era5.sel(time=slice(gnss.time.min(), gnss.time.max()))
    era5 = era5[[x for x in era5 if x in gnss]]
    df_gnss = process_mkt_from_dataset(
        gnss,
        alpha=0.95,
        season_selection=None,
        seasonal=False,
        factor=120,
        anomalize=True, CI=True)
    df_gnss = add_location_to_GNSS_stations_dataframe(df_gnss)
    df_gnss['sig'] = df_gnss['p'].astype(float) <= 0.05
    df_era5 = process_mkt_from_dataset(
        era5,
        alpha=0.95,
        season_selection=None,
        seasonal=False,
        factor=120,
        anomalize=True, CI=True)
    df_era5 = add_location_to_GNSS_stations_dataframe(df_era5)
    df_era5['sig'] = df_era5['p'].astype(float) <= 0.05
    df = pd.concat([df_gnss, df_era5], keys=['GNSS', 'ERA5'])
    df1 = df.unstack(level=0)
    df = df1.stack().reset_index()
    df.columns = ['station', '', 'p', 'Tau', 'slope', 'intercept', 'CI_5_low',
                  'CI_5_high', 'Location', 'sig']
    sns.barplot(x="slope", y='station', hue='', data=df[df['sig']])
    # df['slope'].unstack(level=0).plot(kind='barh', subplots=False, xerr=1)
    return df


def produce_filled_pwv_and_era5_mann_kendall_table(path=work_yuval):
    import xarray as xr
    from aux_gps import path_glob
    file = sorted(
        path_glob(path, 'GNSS_PW_monthly_homogenized_filled_*.nc'))[0]
    gnss = xr.load_dataset(path / file)
    era5 = xr.load_dataset(path / 'GNSS_era5_monthly_PW.nc')
    era5 = era5.sel(time=slice(gnss.time.min(), gnss.time.max()))
    df = add_comparison_to_mann_kendall_table(gnss, era5, 'GNSS', 'ERA5')
    print(df.to_latex(header=False, index=False))
    return df


def add_comparison_to_mann_kendall_table(ds1, ds2, name1='GNSS', name2='ERA5',
                                         alpha=0.05):
    df1 = produce_table_mann_kendall(ds1, alpha=alpha)
    df2 = produce_table_mann_kendall(ds2, alpha=alpha)
    df = df1['Site ID'].to_frame()
    df[name1+'1'] = df1["Kendall's Tau"]
    df[name2+'1'] = df2["Kendall's Tau"]
    df[name1+'2'] = df1['P-value']
    df[name2+'2'] = df2['P-value']
    df[name1+'3'] = df1["Sen's slope"]
    df[name2+'3'] = df2["Sen's slope"]
    df[name1+'4'] = df1["Percent change"]
    df[name2+'4'] = df2["Percent change"]
    return df


def produce_table_mann_kendall(pwv_ds, alpha=0.05,
                               sort_by=['groups_annual', 'lat']):
    from PW_stations import process_mkt_from_dataset
    from PW_stations import produce_geo_gnss_solved_stations
    from aux_gps import reduce_tail_xr
    import xarray as xr

    def table_process_df(df, means):
        df_sites = produce_geo_gnss_solved_stations(plot=False,
                                                    add_distance_to_coast=True)
        sites = df_sites.dropna()[['lat', 'alt', 'distance', 'groups_annual']].sort_values(
            by=sort_by, ascending=[1, 0]).index
        # calculate percent changes from last decade means:
        df['CI95'] = '(' + df['CI_95_low'].map('{:.2f}'.format).astype(
            str) + ', ' + df['CI_95_high'].map('{:.2f}'.format).astype(str) + ')'
        df['means'] = means
        df['Pct_change'] = 100 * df['slope'] / df['means']
        Pct_high = 100 * df['CI_95_high'] / df['means']
        Pct_low = 100 * df['CI_95_low'] / df['means']
        df['Pct_change_CI95'] = '(' + Pct_low.map('{:.2f}'.format).astype(
            str) + ', ' + Pct_high.map('{:.2f}'.format).astype(str) + ')'
#        df['Temperature change'] = df['Percent change'] / 7.0
        df.drop(['means', 'CI_95_low', 'CI_95_high'], axis=1, inplace=True)
        # station id is big:
        df['id'] = df.index.str.upper()
        # , 'Temperature change']]
        df = df[['id', 'Tau', 'p', 'slope', 'CI95',
                 'Pct_change', 'Pct_change_CI95']]
        # filter for non significant trends:
#        df['slope'] = df['slope'][df['p'] < 0.05]
#        df['Pct_change'] = df['Pct_change'][df['p'] < 0.05]
#        df['CI95'] = df['CI95'][df['p'] < 0.05]
#        df['Pct_change_CI95'] = df['Pct_change_CI95'][df['p'] < 0.05]
        # higher and better results:
        df.loc[:, 'p'][df['p'] < 0.001] = '<0.001'
        df['p'][df['p'] != '<0.001'] = df['p'][df['p'] !=
                                               '<0.001'].astype(float).map('{:,.3f}'.format)
        df['Tau'] = df['Tau'].map('{:,.3f}'.format)
        df['slope'] = df['slope'].map('{:,.2f}'.format)
        df['slope'][df['slope'] == 'nan'] = '-'
        df.columns = [
            'Site ID',
            "Kendall's Tau",
            'P-value',
            "Sen's slope", "Sen's slope CI 95%",
            'Percent change', 'Percent change CI 95%']  # , 'Temperature change']
        df['Percent change'] = df['Percent change'].map('{:,.1f}'.format)
        df['Percent change'] = df[df["Sen's slope"] != '-']['Percent change']
        df['Percent change'] = df['Percent change'].fillna('-')
        df["Sen's slope CI 95%"] = df["Sen's slope CI 95%"].fillna(' ')
        df['Percent change CI 95%'] = df['Percent change CI 95%'].fillna(' ')
        df["Sen's slope"] = df["Sen's slope"].astype(
            str) + ' ' + df["Sen's slope CI 95%"].astype(str)
        df['Percent change'] = df['Percent change'].astype(
            str) + ' ' + df['Percent change CI 95%'].astype(str)
        df.drop(['Percent change CI 95%', "Sen's slope CI 95%"],
                axis=1, inplace=True)
#        df['Temperature change'] = df['Temperature change'].map('{:,.1f}'.format)
#        df['Temperature change'] = df[df["Sen's slope"] != '-']['Temperature change']
#        df['Temperature change'] = df['Temperature change'].fillna('-')
        # last, reindex according to geography:
#        gr = group_sites_to_xarray(scope='annual')
#        new = [x for x in gr.T.values.ravel() if isinstance(x, str)]
        new = [x for x in sites if x in df.index]
        df = df.reindex(new)
        return df

#    if load_data == 'pwv-homo':
#        print('loading homogenized (RH) pwv dataset.')
#        data = xr.load_dataset(work_yuval /
#                               'GNSS_PW_monthly_thresh_{:.0f}_homogenized.nc'.format(thresh))
#    elif load_data == 'pwv-orig':
#        print('loading original pwv dataset.')
#        data = xr.load_dataset(work_yuval /
#                               'GNSS_PW_monthly_thresh_{:.0f}.nc'.format(thresh))
#    elif load_data == 'pwv-era5':
#        print('loading era5 pwv dataset.')
#        data = xr.load_dataset(work_yuval / 'GNSS_era5_monthly_PW.nc')
#    if pwv_ds is not None:
#        print('loading user-input pwv dataset.')
#     data = pwv_ds
    df = process_mkt_from_dataset(
        pwv_ds,
        alpha=alpha,
        season_selection=None,
        seasonal=False,
        factor=120,
        anomalize=True, CI=True)
    df_mean = reduce_tail_xr(pwv_ds, reduce='mean', records=120,
                             return_df=True)
    table = table_process_df(df, df_mean)
#    print(table.to_latex(index=False))
    return table


def plot_filled_and_unfilled_pwv_monthly_anomalies(pw_da, anomalize=True,
                                                   max_gap=6,
                                                   method='cubic',
                                                   ax=None):
    from aux_gps import anomalize_xr
    import matplotlib.pyplot as plt
    import numpy as np
    if anomalize:
        pw_da = anomalize_xr(pw_da, 'MS')
    max_gap_td = np.timedelta64(max_gap, 'M')
    filled = pw_da.interpolate_na('time', method=method, max_gap=max_gap_td)
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))
    filledln = filled.plot.line('b-', ax=ax)
    origln = pw_da.plot.line('r-', ax=ax)
    ax.legend(origln + filledln,
              ['original time series',
               'filled using {} interpolation with max gap of {} months'.format(method,
                                                                                max_gap)])
    ax.grid()
    ax.set_xlabel('')
    ax.set_ylabel('PWV [mm]')
    ax.set_title('PWV station {}'.format(pw_da.name.upper()))
    return ax


def plot_pwv_statistic_vs_height(pwv_ds, stat='mean', x='alt', season=None,
                                 ax=None, color='b'):
    from PW_stations import produce_geo_gnss_solved_stations
    import matplotlib.pyplot as plt
    from aux_gps import calculate_std_error
    import pandas as pd
    if season is not None:
        print('{} season selected'.format(season))
        pwv_ds = pwv_ds.sel(time=pwv_ds['time.season'] == season)
    df = produce_geo_gnss_solved_stations(plot=False,
                                          add_distance_to_coast=True)
    if stat == 'mean':
        pw_stat = pwv_ds.mean()
        pw_stat_error = pwv_ds.map(calculate_std_error, statistic=stat)
    elif stat == 'std':
        pw_stat = pwv_ds.std()
        pw_stat_error = pwv_ds.map(calculate_std_error, statistic=stat)
    df[stat] = pd.Series(
        pw_stat.to_array(
            dim='gnss'),
        index=pw_stat.to_array('gnss')['gnss'])
    df['{}_error'.format(stat)] = pd.Series(pw_stat_error.to_array(
        dim='gnss'), index=pw_stat_error.to_array('gnss')['gnss'])
    if ax is None:
        fig, ax = plt.subplots()
        if x == 'alt':
            ax.set_xlabel('Altitude [m a.s.l]')
        elif x == 'distance':
            ax.set_xlabel('Distance to sea shore [km]')
        ax.set_ylabel('{} [mm]'.format(stat))
    ax.errorbar(df[x],
                df[stat],
                df['{}_error'.format(stat)],
                marker='o',
                ls='',
                capsize=2.5,
                elinewidth=2.5,
                markeredgewidth=2.5,
                color=color)
    if season is not None:
        ax.set_title('{} season'.format(season))
    ax.grid()
    return ax


def add_location_to_GNSS_stations_dataframe(df, scope='annual'):
    import pandas as pd
    # load location data:
    gr = group_sites_to_xarray(scope=scope)
    gr_df = gr.to_dataframe('sites')
    new = gr.T.values.ravel()
    # remove nans form mixed nans and str numpy:
    new = new[~pd.isnull(new)]
    geo = [gr_df[gr_df == x].dropna().index.values.item()[1] for x in new]
    geo = [x.title() for x in geo]
    df = df.reindex(new)
    df['Location'] = geo
    return df


def plot_peak_amplitude_altitude_long_term_pwv(path=work_yuval, era5=False,
                                               add_a1a2=True, save=True, fontsize=16):
    import xarray as xr
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from fitting_routines import fit_poly_model_xr
    from aux_gps import remove_suffix_from_ds
    from PW_stations import produce_geo_gnss_solved_stations
    # load alt data, distance etc.,
    sns.set_style('whitegrid')
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    df_geo = produce_geo_gnss_solved_stations(
        plot=False, add_distance_to_coast=True)
    if era5:
        dss = xr.load_dataset(path / 'GNSS_PW_ERA5_harmonics_annual.nc')
    else:
        dss = xr.load_dataset(path / 'GNSS_PW_harmonics_annual.nc')
    dss = dss[[x for x in dss if '_params' in x]]
    dss = remove_suffix_from_ds(dss)
    df = dss.sel(cpy=1, params='ampl').reset_coords(drop=True).to_dataframe().T
    df.columns = ['A1', 'A1std']
    df = df.join(dss.sel(cpy=2, params='ampl').reset_coords(drop=True).to_dataframe().T)
    # abs bc sometimes the fit get a sine amp negative:
    df = np.abs(df)
    df.columns =['A1', 'A1std', 'A2', 'A2std']
    df['A2A1'] = df['A2'] / df['A1']
    a2a1std = np.sqrt((df['A2std']/df['A1'])**2 + (df['A2']*df['A1std']/df['A1']**2)**2)
    df['A2A1std'] = a2a1std
    # load location data:
    gr = group_sites_to_xarray(scope='annual')
    gr_df = gr.to_dataframe('sites')
    new = gr.T.values.ravel()
    # remove nans form mixed nans and str numpy:
    new = new[~pd.isnull(new)]
    geo = [gr_df[gr_df == x].dropna().index.values.item()[1] for x in new]
    geo = [x.title() for x in geo]
    df = df.reindex(new)
    df['Location'] = geo
    df['alt'] = df_geo['alt']
    df = df.set_index('alt')
    df = df.sort_index()
    cdict = produce_colors_for_pwv_station(scope='annual', as_cat_dict=True)
    cdict = dict(zip([x.capitalize() for x in cdict.keys()], cdict.values()))
    if add_a1a2:
        fig, axes=plt.subplots(2, 1, sharex=False, figsize=(8, 12))
        ax = axes[0]
    else:
        ax = None
    # colors=produce_colors_for_pwv_station(scope='annual')
    ax = sns.scatterplot(data=df, y='A1', x='alt', hue='Location',
                         palette=cdict, ax=ax, s=100, zorder=20)
    # ax.legend(prop={'size': fontsize})
    x_coords = []
    y_coords = []
    colors = []
    for point_pair in ax.collections:
        colors.append(point_pair.get_facecolor())
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)
    ax.errorbar(x_coords, y_coords,
                yerr=df['A1std'].values, ecolor=colors[0][:,0:-1],
                ls='', capsize=None, fmt=" ")#, zorder=-1)
    # linear fit:
    x = df.index.values
    y = df['A1'].values

    p = fit_poly_model_xr(x, y, 1, plot=None, ax=None, return_just_p=True)
    fit_label = r'Fitted line, slope: {:.2f} mm$\cdot$km$^{{-1}}$'.format(p[0] * -1000)
    fit_poly_model_xr(x,y,1,plot='manual', ax=ax, fit_label=fit_label)
    ax.set_ylabel('PWV annual amplitude [mm]', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.set_yticks(np.arange(1, 6, 1))
    if add_a1a2:
        ax.set_xlabel('')
    else:
        ax.set_xlabel('GNSS station height [m a.s.l]')
    ax.grid(True)
    ax.legend(prop={'size': fontsize-3})
    if add_a1a2:
        # convert to percent:
        df['A2A1'] = df['A2A1'].mul(100)
        df['A2A1std'] = df['A2A1std'].mul(100)
        ax = sns.scatterplot(data=df, y='A2A1', x='alt',
                             hue='Location', ax=axes[1],
                             legend=True, palette=cdict,
                             s=100, zorder=20)
        x_coords = []
        y_coords = []
        colors = []
        # ax.legend(prop={'size':fontsize+4}, fontsize=fontsize)
        for point_pair in ax.collections:
            colors.append(point_pair.get_facecolor())
            for x, y in point_pair.get_offsets():
                x_coords.append(x)
                y_coords.append(y)
        ax.errorbar(x_coords, y_coords,
                    yerr=df['A2A1std'].values, ecolor=colors[0][:,0:-1],
                    ls='', capsize=None, fmt=" ")#, zorder=-1)
        df_upper = df.iloc[9:]
        y = df_upper['A2A1'].values
        x = df_upper.index.values
        p = fit_poly_model_xr(x, y, 1, return_just_p=True)
        fit_label = r'Fitted line, slope: {:.1f} %$\cdot$km$^{{-1}}$'.format(p[0] * 1000)
        p = fit_poly_model_xr(x, y, 1, plot='manual', ax=ax,
                              return_just_p=False, color='r',
                              fit_label=fit_label)
        df_lower = df.iloc[:11]
        mean = df_lower['A2A1'].mean()
        std = df_lower['A2A1'].std()
        stderr = std / np.sqrt(len(df_lower))
        ci = 1.96 * stderr
        ax.hlines(xmin=df_lower.index.min(), xmax=df_lower.index.max(), y=mean,
                  color='k', label='Mean ratio: {:.1f} %'.format(mean))
        ax.fill_between(df_lower.index.values, mean + ci, mean - ci, color="#b9cfe7", edgecolor=None, alpha=0.6)
        # y = df_lower['A2A1'].values
        # x = df_lower.index.values
        # p = fit_poly_model_xr(x, y, 1, return_just_p=True)
        # fit_label = 'Linear Fit intercept: {:.2f} %'.format(p[1])
        # p = fit_poly_model_xr(x, y, 1, plot='manual', ax=ax,
        #                       return_just_p=False, color='k',
        #                       fit_label=fit_label)
        # arrange the legend a bit:
        handles, labels = ax.get_legend_handles_labels()
        h_stns = handles[1:4]
        l_stns = labels[1:4]
        h_fits = [handles[0] , handles[-1]]
        l_fits = [labels[0], labels[-1]]
        ax.legend(handles=h_fits+h_stns, labels=l_fits+l_stns, loc='upper left', prop={'size':fontsize-3})
        ax.set_ylabel('PWV semi-annual to annual amplitude ratio [%]', fontsize=fontsize)
        ax.set_xlabel('GNSS station height [m a.s.l]', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(True)
        ax.set_yticks(np.arange(0, 100, 20))
        fig.tight_layout()
    if save:
        filename = 'pwv_peak_amplitude_altitude.png'
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return ax


def plot_peak_hour_distance(path=work_yuval, season='JJA',
                            remove_station='dsea', fontsize=22, save=True):
    from PW_stations import produce_geo_gnss_solved_stations
    from aux_gps import groupby_half_hour_xr
    from aux_gps import xr_reindex_with_date_range
    import xarray as xr
    import pandas as pd
    import seaborn as sns
    import numpy as np
    from sklearn.metrics import r2_score
    pw = xr.open_dataset(path / 'GNSS_PW_thresh_50_for_diurnal_analysis.nc')
    pw = pw[[x for x in pw if '_error' not in x]]
    pw.load()
    pw = pw.sel(time=pw['time.season'] == season)
    pw = pw.map(xr_reindex_with_date_range)
    df = groupby_half_hour_xr(pw)
    halfs = [df.isel(half_hour=x)['half_hour'] for x in df.argmax().values()]
    names = [x for x in df]
    dfh = pd.DataFrame(halfs, index=names)
    geo = produce_geo_gnss_solved_stations(
        add_distance_to_coast=True, plot=False)
    geo['phase'] = dfh
    geo = geo.dropna()
    groups = group_sites_to_xarray(upper=False, scope='diurnal')
    geo.loc[groups.sel(group='coastal').values, 'group'] = 'coastal'
    geo.loc[groups.sel(group='highland').values, 'group'] = 'highland'
    geo.loc[groups.sel(group='eastern').values, 'group'] = 'eastern'
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.grid()
    if remove_station is not None:
        removed = geo.loc[remove_station].to_frame().T
        geo = geo.drop(remove_station, axis=0)
#     lnall = sns.scatterplot(data=geo.loc[only], x='distance', y='phase', ax=ax, hue='group', s=100)
#    geo['phase'] = pd.to_timedelta(geo['phase'], unit='H')
    coast = geo[geo['group'] == 'coastal']
    yerr = 1.0
    lncoast = ax.errorbar(x=coast.loc[:,
                                      'distance'],
                          y=coast.loc[:,
                                      'phase'],
                          yerr=yerr,
                          marker='o',
                          ls='',
                          capsize=2.5,
                          elinewidth=2.5,
                          markeredgewidth=2.5,
                          color='b')
    # lncoast = ax.scatter(coast.loc[:, 'distance'], coast.loc[:, 'phase'], color='b', s=50)
    highland = geo[geo['group'] == 'highland']
#    lnhighland = ax.scatter(highland.loc[:, 'distance'], highland.loc[:, 'phase'], color='brown', s=50)
    lnhighland = ax.errorbar(x=highland.loc[:,
                                            'distance'],
                             y=highland.loc[:,
                                            'phase'],
                             yerr=yerr,
                             marker='o',
                             ls='',
                             capsize=2.5,
                             elinewidth=2.5,
                             markeredgewidth=2.5,
                             color='brown')
    eastern = geo[geo['group'] == 'eastern']
#    lneastern = ax.scatter(eastern.loc[:, 'distance'], eastern.loc[:, 'phase'], color='green', s=50)
    lneastern = ax.errorbar(x=eastern.loc[:,
                                          'distance'],
                            y=eastern.loc[:,
                                          'phase'],
                            yerr=yerr,
                            marker='o',
                            ls='',
                            capsize=2.5,
                            elinewidth=2.5,
                            markeredgewidth=2.5,
                            color='green')
    lnremove = ax.scatter(
        removed.loc[:, 'distance'], removed.loc[:, 'phase'], marker='x', color='k', s=50)
    ax.legend([lncoast,
               lnhighland,
               lneastern,
               lnremove],
              ['Coastal stations',
               'Highland stations',
               'Eastern stations',
               'DSEA station'],
              fontsize=fontsize)
    params = np.polyfit(geo['distance'].values, geo.phase.values, 1)
    params2 = np.polyfit(geo['distance'].values, geo.phase.values, 2)
    x = np.linspace(0, 210, 100)
    y = np.polyval(params, x)
    y2 = np.polyval(params2, x)
    r2 = r2_score(geo.phase.values, np.polyval(params, geo['distance'].values))
    ax.plot(x, y, color='k')
    textstr = '\n'.join([r'R$^2$: {:.2f}'.format(r2)])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment='top', bbox=props)
    # ax.plot(x,y2, color='green')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel('Distance from shore [km]', fontsize=fontsize)
    ax.set_ylabel('Peak hour [UTC]', fontsize=fontsize)
    # add sunrise UTC hour
    ax.axhline(16.66, color='tab:orange', linewidth=2)
    # change yticks to hours minuets:
    fig.canvas.draw()
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels = [pd.to_timedelta(float(x), unit='H') for x in labels]
    labels = ['{}:{}'.format(x.components[1], x.components[2])
              if x.components[2] != 0 else '{}:00'.format(x.components[1]) for x in labels]
    ax.set_yticklabels(labels)
    fig.canvas.draw()
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    if save:
        filename = 'pw_peak_distance_shore.png'
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return ax


def plot_monthly_variability_heatmap_from_pwv_anomalies(load_path=work_yuval,
                                                        thresh=50, save=True,
                                                        fontsize=16,
                                                        sort_by=['groups_annual', 'alt']):
    """sort_by=['group_annual', 'lat'], ascending=[1,0]"""
    import xarray as xr
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    from calendar import month_abbr
    from PW_stations import produce_geo_gnss_solved_stations
    df = produce_geo_gnss_solved_stations(plot=False,
                                          add_distance_to_coast=True)
    sites = df.dropna()[['lat', 'alt', 'distance', 'groups_annual']].sort_values(
        by=sort_by, ascending=[1, 1]).index
#    anoms = xr.load_dataset(
#        load_path /
#        'GNSS_PW_monthly_anoms_thresh_{:.0f}_homogenized.nc'.format(thresh))
    anoms = xr.load_dataset(
        load_path /
        'GNSS_PW_monthly_anoms_thresh_{:.0f}.nc'.format(thresh))
    df = anoms.groupby('time.month').std().to_dataframe()
#    sites = group_sites_to_xarray(upper=True, scope='annual').T
#    sites_flat = [x.lower() for x in sites.values.flatten() if isinstance(x, str)]
#    df = df[sites_flat]

#    cols = [x for x in sites if x in df.columns]
    df = df[sites]
    df.columns = [x.upper() for x in df.columns]
    fig = plt.figure(figsize=(14, 10))
    grid = plt.GridSpec(
        2, 1, height_ratios=[
            2, 1], hspace=0)
    ax_heat = fig.add_subplot(grid[0, 0])  # plt.subplot(221)
    ax_group = fig.add_subplot(grid[1, 0])  # plt.subplot(223)
    cbar_ax = fig.add_axes([0.91, 0.37, 0.02, 0.62])  # [left, bottom, width,
    # height]
    ax_heat = sns.heatmap(
        df.T,
        cmap='Reds',
        vmin=df.min().min(),
        vmax=df.max().max(),
        annot=True,
        yticklabels=True,
        ax=ax_heat,
        cbar_ax=cbar_ax,
        cbar_kws={'label': 'PWV anomalies STD [mm]'},
        annot_kws={'fontsize': fontsize}, xticklabels=False)
    cbar_ax.set_ylabel('PWV anomalies STD [mm]', fontsize=fontsize)
    cbar_ax.tick_params(labelsize=fontsize)
    # activate top ticks and tickslabales:
    ax_heat.xaxis.set_tick_params(
        bottom='off',
        labelbottom='off',
        labelsize=fontsize)
    # emphasize the yticklabels (stations):
    ax_heat.yaxis.set_tick_params(left='on')
    ax_heat.set_yticklabels(ax_heat.get_ymajorticklabels(),
                            fontweight='bold', fontsize=fontsize)
    df_mean = df.T.mean()
    df_mean = df_mean.to_frame()
    df_mean[1] = [month_abbr[x] for x in range(1, 13)]
    df_mean.columns = ['std', 'month']
    g = sns.barplot(data=df_mean, x='month', y='std', ax=ax_group, palette='Reds',
                    hue='std', dodge=False, linewidth=2.5)
    g.legend_.remove()
    ax_group.set_ylabel('PWV anomalies STD [mm]', fontsize=fontsize)
    ax_group.grid(color='k', linestyle='--',
                  linewidth=1.5, alpha=0.5, axis='y')
    ax_group.xaxis.set_tick_params(labelsize=fontsize)
    ax_group.yaxis.set_tick_params(labelsize=fontsize)
    ax_group.set_xlabel('', fontsize=fontsize)
#    df.T.mean().plot(ax=ax_group, kind='bar', color='k', fontsize=fontsize, rot=0)
    fig.tight_layout()
    fig.subplots_adjust(right=0.906)
    if save:
        filename = 'pw_anoms_monthly_variability_heatmap.png'
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fig


def plot_monthly_means_anomalies_with_station_mean(load_path=work_yuval,
                                                   thresh=50, save=True,
                                                   anoms=None, agg='mean',
                                                   fontsize=16, units=None,
                                                   remove_stations=['nizn', 'spir'],
                                                   sort_by=['groups_annual', 'lat']):
    import xarray as xr
    import seaborn as sns
    from palettable.scientific import diverging as divsci
    import numpy as np
    import matplotlib.dates as mdates
    import pandas as pd
    from aux_gps import anomalize_xr
    from PW_stations import produce_geo_gnss_solved_stations
    sns.set_style('whitegrid')
    sns.set_style('ticks')
    div_cmap = divsci.Vik_20.mpl_colormap
    df = produce_geo_gnss_solved_stations(plot=False,
                                          add_distance_to_coast=True)
    sites = df.dropna()[['lat', 'alt', 'distance', 'groups_annual']].sort_values(
        by=sort_by, ascending=[1, 0]).index
    if anoms is None:
        #        anoms = xr.load_dataset(
        #                load_path /
        #                'GNSS_PW_monthly_anoms_thresh_{:.0f}_homogenized.nc'.format(thresh))
        anoms = xr.load_dataset(
            load_path /
            'GNSS_PW_monthly_thresh_{:.0f}.nc'.format(thresh))
        anoms = anomalize_xr(anoms, 'MS', units=units)
    if remove_stations is not None:
        anoms = anoms[[x for x in anoms if x not in remove_stations]]
    df = anoms.to_dataframe()[:'2019']
#    sites = group_sites_to_xarray(upper=True, scope='annual').T
#    sites_flat = [x.lower() for x in sites.values.flatten() if isinstance(x, str)]
#    df = df[sites_flat]
    cols = [x for x in sites if x in df.columns]
    df = df[cols]
    df.columns = [x.upper() for x in df.columns]
    weights = df.count(axis=1).shift(periods=-1, freq='15D').astype(int)
    fig = plt.figure(figsize=(20, 10))
    grid = plt.GridSpec(
        2, 1, height_ratios=[
            2, 1], hspace=0.0225)
    ax_heat = fig.add_subplot(grid[0, 0])  # plt.subplot(221)
    ax_group = fig.add_subplot(grid[1, 0])  # plt.subplot(223)
    cbar_ax = fig.add_axes([0.95, 0.43, 0.0125, 0.45])  # [left, bottom, width,
    # height]
    ax_heat = sns.heatmap(
        df.T,
        center=0.0,
        cmap=div_cmap,
        yticklabels=True,
        ax=ax_heat,
        cbar_ax=cbar_ax,
        cbar_kws={'label': 'PWV anomalies [mm]'}, xticklabels=False)
    cbar_ax.set_ylabel('PWV anomalies [mm]', fontsize=fontsize-4)
    cbar_ax.tick_params(labelsize=fontsize)
    # activate top ticks and tickslabales:
    ax_heat.xaxis.set_tick_params(
        bottom='off', labelbottom='off', labelsize=fontsize)
    # emphasize the yticklabels (stations):
    ax_heat.yaxis.set_tick_params(left='on')
    ax_heat.set_yticklabels(ax_heat.get_ymajorticklabels(),
                            fontweight='bold', fontsize=fontsize)
    ax_heat.set_xlabel('')
    if agg == 'mean':
        ts = df.T.mean().shift(periods=-1, freq='15D')
    elif agg == 'median':
        ts = df.T.median().shift(periods=-1, freq='15D')
    ts.index.name = ''
    # dt_as_int = [x for x in range(len(ts.index))]
    # xticks_labels = ts.index.strftime('%Y-%m').values[::6]
    # xticks = dt_as_int[::6]
    # xticks = ts.index
    # ts.index = dt_as_int
    ts.plot(ax=ax_group, color='k', fontsize=fontsize, lw=2)
    barax = ax_group.twinx()
    barax.bar(ts.index, weights.values, width=35, color='k', alpha=0.2)
    barax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    barax.set_ylabel('Stations [#]', fontsize=fontsize-4)
    barax.tick_params(labelsize=fontsize)
    ax_group.set_xlim(ts.index.min(), ts.index.max() +
                      pd.Timedelta(15, unit='D'))
    ax_group.set_ylabel('PWV {} anomalies [mm]'.format(agg), fontsize=fontsize-4)
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
    # ax_group.xaxis.tick_top()
    # ax_group.xaxis.set_ticks_position('both')
    # ax_group.tick_params(axis='x', labeltop='off', top='on',
                         # bottom='on', labelbottom='on')
    ax_group.grid()
    # ax_group.axvline('2015-09-15')
    # ax_group.axhline(2.5)
    # plt.setp(ax_group.xaxis.get_majorticklabels(), rotation=45 )
    fig.tight_layout()
    fig.subplots_adjust(right=0.946)
    if save:
        filename = 'pw_monthly_means_anomaly_heatmap.png'
        plt.savefig(savefig_path / filename, bbox_inches='tight', pad_inches=0.1)
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
            fig.suptitle(
                'Precipitable water {}ly anomalies analysis for {} season'.format(grp, season))
        else:
            fig.suptitle('Precipitable water {}ly anomalies analysis (Weighted KMeans {} clusters)'.format(
                grp, n_clusters))
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
    cbar_ax = fig.add_axes([0.57, 0.24, 0.01, 0.69])  # [left, bottom, width,
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
                            fontweight='bold', fontsize=10)
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
                                       max(df.index.values) + 1 -
                                       min(df.index.values),
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
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    day = ds.sel(sound_time=ds['sound_time.hour'] == 12).groupby(
        'sound_time.month').mean('sound_time')
    night = ds.sel(sound_time=ds['sound_time.hour'] == 00).groupby(
        'sound_time.month').mean('sound_time')
    next_month = center_month + 1
    last_month = center_month - 1
    day = day.sel(month=[last_month, center_month, next_month])
    night = night.sel(month=[last_month, center_month, next_month])
    for month in day.month:
        h = day.sel(month=month)['H-Msl'].values
        rh = day.sel(month=month).values
        ax[0].semilogy(rh, h)
    ax[0].set_title('noon')
    ax[0].set_ylabel('height [m]')
    ax[0].set_xlabel('{}, [{}]'.format(field, day.attrs['units']))
    plt.legend([x for x in ax.lines], [x for x in day.month.values])
    for month in night.month:
        h = night.sel(month=month)['H-Msl'].values
        rh = night.sel(month=month).values
        ax[1].semilogy(rh, h)
    ax[1].set_title('midnight')
    ax[1].set_ylabel('height [m]')
    ax[1].set_xlabel('{}, [{}]'.format(field, night.attrs['units']))
    plt.legend([x for x in ax.lines], [x for x in night.month.values])
    return day, night


def plot_global_warming_with_pwv_annual(climate_path=climate_path, work_path=work_yuval, fontsize=16):
    import pandas as pd
    import xarray as xr
    import numpy as np
    from aux_gps import anomalize_xr
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_style('whitegrid')
    sns.set_style('ticks')
    df = pd.read_csv(climate_path/'GLB.Ts+dSST_2007.csv',
                     header=1, na_values='*******')
    df = df.iloc[:19, :13]
    df = df.melt(id_vars='Year')
    df['time'] = pd.to_datetime(df['Year'].astype(
        str)+'-'+df['variable'].astype(str))
    df = df.set_index('time')
    df = df.drop(['Year', 'variable'], axis=1)
    df.columns = ['T']
    df['T'] = pd.to_numeric(df['T'])
    df = df.sort_index()
    df.columns = ['AIRS-ST-Global']
    # df = df.loc['2003':'2019']
    # df = df.resample('AS').mean()
    dss = xr.open_dataset(climate_path/'AIRS.2002-2021.L3.RetStd_IR031.v7.0.3.0.nc')
    dss = dss.sel(time=slice('2003','2019'), Longitude=slice(34,36), Latitude=slice(34,29))
    ds = xr.concat([dss['SurfAirTemp_A'], dss['SurfAirTemp_D']], 'dn')
    ds['dn'] = ['day', 'night']
    ds = ds.mean('dn')
    ds -= ds.sel(time=slice('2007','2016')).mean('time')
    anoms = anomalize_xr(ds, 'MS')
    anoms = anoms.mean('Latitude').mean('Longitude')
    df['AIRS-ST-Regional'] = anoms.to_dataframe('AIRS-ST-Regional')
    # else:
    #     df = pd.read_csv(climate_path/'GLB.Ts+dSST.csv',
    #                      header=1, na_values='***')
    #     df = df.iloc[:, :13]
    #     df = df.melt(id_vars='Year')
    #     df['time'] = pd.to_datetime(df['Year'].astype(
    #         str)+'-'+df['variable'].astype(str))
    #     df = df.set_index('time')
    #     df = df.drop(['Year', 'variable'], axis=1)
    #     df.columns = ['T']
    #     # df = df.resample('AS').mean()
    #     df = df.sort_index()
    pw = xr.load_dataset(work_path/'GNSS_PW_monthly_anoms_thresh_50.nc')
    # pw_2007_2016_mean = pw.sel(time=slice('2007','2016')).mean()
    # pw -= pw_2007_2016_mean
    pw = pw.to_array('s').mean('s')
    pw_df = pw.to_dataframe('PWV')
    # df['pwv'] = pw_df.resample('AS').mean()
    df['PWV'] = pw_df
    df = df.loc['2003': '2019']
    df = df.resample('AS').mean()

    fig, ax = plt.subplots(figsize=(15, 6))
    ax = df.plot(kind='bar', secondary_y='PWV',
                 color=['tab:red', 'tab:orange', 'tab:blue'],
                 ax=ax, legend=False, rot=45)
    twin = get_twin(ax, 'x')
    align_yaxis_np(ax, twin)
    # twin.set_yticks([-0.5, 0, 0.5, 1.0, 1.5])
    # locator = ticker.MaxNLocator(6)
    # ax.yaxis.set_major_locator(locator)
    twin.yaxis.set_major_locator(ticker.MaxNLocator(6))
    twin.set_ylabel('PWV anomalies [mm]', fontsize=fontsize)
    ax.set_ylabel(r'Surface Temperature anomalies [$\degree$C]', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    twin.tick_params(labelsize=fontsize)
    ax.set_xticklabels(np.arange(2003, 2020))
    ax.grid(True)
     # add legend:
    handles, labels = [], []
    for h, l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    for h, l in zip(*twin.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    ax.legend(handles, labels, prop={'size': fontsize-2}, loc='upper left')
    ax.set_xlabel('')
    fig.tight_layout()
    return df


def plot_SST_med(sst_path=work_yuval/'SST', fontsize=16, loop=True):
    import xarray as xr
    import seaborn as sns
    from aux_gps import lat_mean
    import numpy as np

    def clim_mean(med_sst):
        sst = med_sst - 273.15
        mean_sst = sst.mean('lon')
        mean_sst = lat_mean(mean_sst)
        mean_sst = mean_sst.groupby('time.dayofyear').mean()
        return mean_sst

    sns.set_style('whitegrid')
    sns.set_style('ticks')
    ds = xr.open_dataset(
        sst_path/'med1-1981_2020-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.0-fv02.0.nc')
    sst = ds['analysed_sst'].sel(time=slice('1997', '2019')).load()
    whole_med_lon = [-5, 37]
    whole_med_lat = [30, 40]
    sst_w = sst.copy().sel(lat=slice(*whole_med_lat), lon=slice(*whole_med_lon))
    sst_clim_w = clim_mean(sst_w)
    df = sst_clim_w.to_dataframe('SST_whole_Med')
    # now for emed:
    for i, min_lon in enumerate(np.arange(23, 34, 1)):
        e_med_lon = [min_lon, 37]
        e_med_lat = [30, 40]
        sst_e = sst.copy().sel(lat=slice(*e_med_lat), lon=slice(*e_med_lon))
        sst_clim_e = clim_mean(sst_e)
        df['SST_EMed_{}'.format(min_lon)] = sst_clim_e.to_dataframe()

    # df['SST_EMed'] = sst_clim_e.to_dataframe()
    if loop:
        ax = df.idxmax().plot(kind='barh')
        ax.set_xticks(np.linspace(0, 365, 13)[:-1])
        ax.set_xticklabels(np.arange(1, 13))
        ax.grid(True)
        ax.set_xlabel('month')
    else:
        ax = df.plot(lw=2, legend=True)
        ax.set_xticks(np.linspace(0, 365, 13)[:-1])
        ax.set_xticklabels(np.arange(1, 13))
        ax.grid(True)
        ax.tick_params(labelsize=fontsize)
        ax.set_ylabel(r'Temperature [$^{\circ}$C]', fontsize=fontsize)
        ax.set_xlabel('month')
    return df


def plot_SST_med_with_PWV_S1_panel(path=work_yuval,
                                   sst_path=work_yuval/'SST',
                                   ims_path=ims_path,
                                   stations=['tela', 'jslm'], fontsize=16, save=True):
    from ims_procedures import gnss_ims_dict
    import matplotlib.pyplot as plt
    ims_stations = [gnss_ims_dict.get(x) for x in stations]
    fig, axes = plt.subplots(1, len(stations), figsize=(15, 6))
    for i, (pwv, ims) in enumerate(zip(stations, ims_stations)):
        plot_SST_med_with_PWV_first_annual_harmonic(path=work_yuval,
                                                    sst_path=sst_path,
                                                    ims_path=ims_path,
                                                    station=pwv, ims_station=ims,
                                                    fontsize=16, ax=axes[i],
                                                    save=False)
        twin = get_twin(axes[i], 'x')
        twin.set_ylim(-4.5, 4.5)
        axes[i].set_ylim(8, 30)


    fig.tight_layout()
    if save:
        filename = 'Med_SST_surface_temp_PWV_harmonic_annual_{}_{}.png'.format(
            *stations)
        plt.savefig(savefig_path / filename, orientation='portrait')
    return


def plot_SST_med_with_PWV_first_annual_harmonic(path=work_yuval,
                                                sst_path=work_yuval/'SST',
                                                ims_path=ims_path,
                                                station='tela', ims_station='TEL-AVIV-COAST',
                                                fontsize=16, ax=None,
                                                save=True):
    import xarray as xr
    from aux_gps import month_to_doy_dict
    import pandas as pd
    import numpy as np
    from aux_gps import lat_mean
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')
    sns.set_style('ticks')
    # load harmonics:
    ds = xr.load_dataset(path/'GNSS_PW_harmonics_annual.nc')
    # stns = group_sites_to_xarray(scope='annual').sel(group='coastal').values
    # harms = []
    # for stn in stns:
    #     da = ds['{}_mean'.format(stn)].sel(cpy=1)
    #     harms.append(da)
    # harm_da = xr.concat(harms, 'station')
    # harm_da['station'] = stns
    harm_da = ds['{}_mean'.format(station)].sel(cpy=1).reset_coords(drop=True)
    # harm_da = harm_da.reset_coords(drop=True)
    harm_da['month'] = [month_to_doy_dict.get(
        x) for x in harm_da['month'].values]
    harm_da = harm_da.rename({'month': 'dayofyear'})
    # df = harm_da.to_dataset('station').to_dataframe()
    df = harm_da.to_dataframe(station)
    # load surface temperature data:
    # da = xr.open_dataset(ims_path/'GNSS_5mins_TD_ALL_1996_2020.nc')[station]
    da = xr.open_dataset(ims_path / 'IMS_TD_israeli_10mins.nc')[ims_station]
    da.load()
    print(da.groupby('time.year').count())
    # da += 273.15
    da_mean = da.groupby('time.dayofyear').mean()
    df['{}_ST'.format(station)] = da_mean.to_dataframe()
    # add 366 dayofyear for visualization:
    df366 = pd.DataFrame(df.iloc[0].values+0.01).T
    df366.index = [366]
    df366.columns = df.columns
    df = df.append(df366)
    ind = np.arange(1, 367)
    df = df.reindex(ind)
    df = df.interpolate('cubic')
    # now load sst for MED
    ds = xr.open_dataset(
        sst_path/'med1-1981_2020-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.0-fv02.0.nc')
    sst = ds['analysed_sst'].sel(time=slice('1997', '2019')).load()
    # sst_mean = sst.sel(lon=slice(25,35)).mean('lon')
    sst -= 273.15
    sst_mean = sst.mean('lon')
    sst_mean = lat_mean(sst_mean)
    sst_clim = sst_mean.groupby('time.dayofyear').mean()
    df['Med-SST'] = sst_clim.to_dataframe()
    pwv_name = '{} PWV-S1'.format(station.upper())
    ims_name = '{} IMS-ST'.format(station.upper())
    df.columns = [pwv_name, ims_name, 'Med-SST']
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    # first plot temp:
    df[[ims_name, 'Med-SST']].plot(ax=ax, color=['tab:red', 'tab:blue'],
                                   style=['-', '-'], lw=2, legend=False)
    ax.set_xticks(np.linspace(0, 365, 13)[:-1])
    ax.set_xticklabels(np.arange(1, 13))
    ax.grid(True)
    ax.tick_params(labelsize=fontsize)
    ax.set_ylabel(r'Temperature [$^{\circ}$C]', fontsize=fontsize)
    vl = df[[ims_name, 'Med-SST']].idxmax().to_frame('x')
    vl['colors'] = ['tab:red', 'tab:blue']
    vl['ymin'] = df[[ims_name, 'Med-SST']].min()
    vl['ymax'] = df[[ims_name, 'Med-SST']].max()
    print(vl)
    ax.vlines(x=vl['x'], ymin=vl['ymin'], ymax=vl['ymax'],
              colors=vl['colors'], zorder=0)
    ax.plot(vl.iloc[0]['x'], vl.iloc[0]['ymax'], color=vl.iloc[0]['colors'],
            linewidth=0, marker='o', zorder=15)
    ax.plot(vl.iloc[1]['x'], vl.iloc[1]['ymax'], color=vl.iloc[1]['colors'],
            linewidth=0, marker='o', zorder=15)

    # ax.annotate(text='', xy=(213,15), xytext=(235,15), arrowprops=dict(arrowstyle='<->'), color='k')
    # ax.arrow(213, 15, dx=21, dy=0, shape='full', color='k', width=0.25)
    #p1 = patches.FancyArrowPatch((213, 15), (235, 15), arrowstyle='<->', mutation_scale=20)

    # ax.arrow(217, 15, 16, 0, head_width=0.14, head_length=2,
    #          linewidth=2, color='k', length_includes_head=True)
    # ax.arrow(231, 15, -16, 0, head_width=0.14, head_length=2,
    #          linewidth=2, color='k', length_includes_head=True)
    start = vl.iloc[0]['x'] + 4
    end = vl.iloc[1]['x'] - 4
    mid = vl['x'].mean()
    dy = vl.iloc[1]['x'] - vl.iloc[0]['x'] - 8
    days = dy + 8
    ax.arrow(start, 15, dy, 0, head_width=0.14, head_length=2,
              linewidth=1.5, color='k', length_includes_head=True, zorder=20)
    ax.arrow(end, 15, -dy, 0, head_width=0.14, head_length=2,
             linewidth=1.5, color='k', length_includes_head=True, zorder=20)
    t = ax.text(
        mid, 15.8, "{} days".format(days), ha="center", va="center", rotation=0, size=12,
        bbox=dict(boxstyle="round4,pad=0.15", fc="white", ec="k", lw=1), zorder=21)
    twin = ax.twinx()
    df[pwv_name].plot(ax=twin, color='tab:cyan', style='--', lw=2, zorder=0)
    twin.set_ylabel('PWV annual anomalies [mm]', fontsize=fontsize)
    ax.set_xlabel('month', fontsize=fontsize)
    locator = ticker.MaxNLocator(7)
    ax.yaxis.set_major_locator(locator)
    twin.yaxis.set_major_locator(ticker.MaxNLocator(7))
    # add legend:
    handles, labels = [], []
    for h, l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    for h, l in zip(*twin.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

    ax.legend(handles, labels, prop={'size': fontsize-2}, loc='upper left')
    # ax.right_ax.set_yticks(np.linspace(ax.right_ax.get_yticks()[0], ax.right_ax.get_yticks()[-1], 7))
    twin.vlines(x=df[pwv_name].idxmax(), ymin=df[pwv_name].min(),
                ymax=df[pwv_name].max(), colors=['tab:cyan'], ls=['--'], zorder=0)
    twin.tick_params(labelsize=fontsize)
    # plot points:
    twin.plot(df[pwv_name].idxmax(), df[pwv_name].max(),
              color='tab:cyan', linewidth=0, marker='o')
    # fig.tight_layout()
    if save:
        filename = 'Med_SST_surface_temp_PWV_harmonic_annual_{}.png'.format(
            station)
        plt.savefig(savefig_path / filename, orientation='portrait')
    return df


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
    pw_std = (pw_new.groupby('time.month') -
              pw_new.groupby('time.month').mean('time')).std('time')
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
        # we already handled the x-label with ax1
        ax_twin.set_ylabel('PW [mm]', color=color)
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
        time = (ser.index / np.timedelta64(1, 'D')).astype(float)
        # ser = ser.loc[pd.Timedelta(-2.2,unit='D'):pd.Timedelta(1, unit='D')]
        ser.index = time

        ser.plot(marker='.', linewidth=0., ax=ax)
    ax.set_xlabel('Days around tide event')
    if pw_anom:
        ax.set_ylabel('PWV anomalies [mm]')
    else:
        ax.set_ylabel('PWV [mm]')
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


def plot_profiler(path=work_yuval, ceil_path=ceil_path, title=False,
                  field='maxsnr', save=True):
    import xarray as xr
    from ceilometers import read_coastal_BL_levi_2011
    from aux_gps import groupby_half_hour_xr
    from calendar import month_abbr
    df = read_coastal_BL_levi_2011(path=ceil_path)
    ds = df.to_xarray()
    pw = xr.open_dataset(path / 'GNSS_PW_thresh_50_for_diurnal_analysis.nc')
    pw = pw['csar']
    pw.load()
    pw = pw.sel(time=pw['time.month'] == 7).dropna('time')
    pw_size = pw.dropna('time').size
    pwyears = [pw.time.dt.year.min().item(), pw.time.dt.year.max().item()]
    pw_std = groupby_half_hour_xr(pw, reduce='std')['csar']
    pw_hour = groupby_half_hour_xr(pw, reduce='mean')['csar']
    pw_hour_plus = (pw_hour + pw_std).values
    pw_hour_minus = (pw_hour - pw_std).values
    if field == 'maxsnr':
        mlh_hour = ds['maxsnr']
        mlh_std = ds['std_maxsnr']
        label = 'Max SNR'
    elif field == 'tv_inversion':
        mlh_hour = ds['tv_inversion']
        mlh_std = ds['std_tv200']
        label = 'Tv inversion'
    mlh_hour_minus = (mlh_hour - mlh_std).values
    mlh_hour_plus = (mlh_hour + mlh_std).values
    half_hours = pw_hour.half_hour.values
    fig, ax = plt.subplots(figsize=(10, 8))
    red = 'tab:red'
    blue = 'tab:blue'
    pwln = pw_hour.plot(color=blue, marker='s', ax=ax)
    ax.fill_between(half_hours, pw_hour_minus,
                    pw_hour_plus, color=blue, alpha=0.5)
    twin = ax.twinx()
    mlhln = mlh_hour.plot(color=red, marker='o', ax=twin)
    twin.fill_between(half_hours, mlh_hour_minus,
                      mlh_hour_plus, color=red, alpha=0.5)
    pw_label = 'PW: {}-{}, {} ({} pts)'.format(
        pwyears[0], pwyears[1], month_abbr[7], pw_size)
    mlh_label = 'MLH: {}-{}, {} ({} pts)'.format(1997, 1999, month_abbr[7], 90)
#    if month is not None:
#        pwmln = pw_m_hour.plot(color='tab:orange', marker='^', ax=ax)
#        pwm_label = 'PW: {}-{}, {} ({} pts)'.format(pw_years[0], pw_years[1], month_abbr[month], pw_month.dropna('time').size)
#        ax.legend(pwln + mlhln + pwmln, [pw_label, mlh_label, pwm_label], loc=leg_loc)
#    else:
    ax.legend([pwln[0], mlhln[0]], [pw_label, mlh_label], loc='best')
#    plt.legend([pw_label, mlh_label])
    ax.tick_params(axis='y', colors=blue)
    twin.tick_params(axis='y', colors=red)
    ax.set_ylabel('PW [mm]', color=blue)
    twin.set_ylabel('MLH [m]', color=red)
    twin.set_ylim(400, 1250)
    ax.set_xticks([x for x in range(24)])
    ax.set_xlabel('Hour of day [UTC]')
    ax.grid()
    mlh_name = 'Hadera'
    textstr = '{}, {}'.format(mlh_name, pw.name.upper())
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    if title:
        ax.set_title('The diurnal cycle of {} Mixing Layer Height ({}) and {} GNSS site PW'.format(
            mlh_name, label, pw.name.upper()))
    fig.tight_layout()
    if save:
        filename = 'PW_diurnal_with_MLH_csar_{}.png'.format(field)
        plt.savefig(savefig_path / filename, orientation='landscape')
    return ax


def plot_ceilometers(path=work_yuval, ceil_path=ceil_path, interpolate='6H',
                     fontsize=14, save=True):
    import xarray as xr
    from ceilometers import twin_hourly_mean_plot
    from ceilometers import read_all_ceilometer_stations
    import numpy as np
    pw = xr.open_dataset(path / 'GNSS_PW_thresh_50_for_diurnal_analysis.nc')
    pw = pw[['tela', 'jslm', 'yrcm', 'nzrt', 'klhv', 'csar']]
    pw.load()
    ds = read_all_ceilometer_stations(path=ceil_path)
    if interpolate is not None:
        attrs = [x.attrs for x in ds.data_vars.values()]
        ds = ds.interpolate_na('time', max_gap=interpolate, method='cubic')
        for i, da in enumerate(ds):
            ds[da].attrs.update(attrs[i])
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 6))
    couples = [['tela', 'TLV'], ['jslm', 'JR']]
    twins = []
    for i, ax in enumerate(axes.flatten()):
        ax, twin = twin_hourly_mean_plot(pw[couples[i][0]],
                                         ds[couples[i][1]],
                                         month=None,
                                         ax=ax,
                                         title=False,
                                         leg_loc='best', fontsize=fontsize)
        twins.append(twin)
        ax.xaxis.set_ticks(np.arange(0, 23, 3))
        ax.grid()
    twin_ylim_min = min(min([x.get_ylim() for x in twins]))
    twin_ylim_max = max(max([x.get_ylim() for x in twins]))
    for twin in twins:
        twin.set_ylim(twin_ylim_min, twin_ylim_max)
    fig.tight_layout()
    filename = 'PW_diurnal_with_MLH_tela_jslm.png'
    if save:
        #        plt.savefig(savefig_path / filename, bbox_inches='tight')
        plt.savefig(savefig_path / filename, orientation='landscape')
    return fig


def plot_field_with_fill_between(da, dim='hour', mean_dim=None, ax=None,
                                 color='b', marker='s'):
    if dim not in da.dims:
        raise KeyError('{} not in {}'.format(dim, da.name))
    if mean_dim is None:
        mean_dim = [x for x in da.dims if dim not in x][0]
    da_mean = da.mean(mean_dim)
    da_std = da.std(mean_dim)
    da_minus = da_mean - da_std
    da_plus = da_mean + da_std
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    line = da_mean.plot(color=color, marker=marker, ax=ax)
    ax.fill_between(da_mean[dim], da_minus, da_plus, color=color, alpha=0.5)
    return line


def plot_mean_with_fill_between_std(da, grp='hour', mean_dim='time', ax=None,
                                    color='b', marker='s', alpha=0.5):
    da_mean = da.groupby('{}.{}'.format(mean_dim, grp)
                         ).mean('{}'.format(mean_dim))
    da_std = da.groupby('{}.{}'.format(mean_dim, grp)
                        ).std('{}'.format(mean_dim))
    da_minus = da_mean - da_std
    da_plus = da_mean + da_std
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    line = da_mean.plot(color=color, marker=marker, ax=ax)
    ax.fill_between(da_mean[grp], da_minus, da_plus, color=color, alpha=alpha)
    return line


def plot_hist_with_seasons(da_ts):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.kdeplot(da_ts.dropna('time'), ax=ax, color='k')
    sns.kdeplot(
        da_ts.sel(
            time=da_ts['time.season'] == 'DJF').dropna('time'),
        legend=False,
        ax=ax,
        shade=True)
    sns.kdeplot(
        da_ts.sel(
            time=da_ts['time.season'] == 'MAM').dropna('time'),
        legend=False,
        ax=ax,
        shade=True)
    sns.kdeplot(
        da_ts.sel(
            time=da_ts['time.season'] == 'JJA').dropna('time'),
        legend=False,
        ax=ax,
        shade=True)
    sns.kdeplot(
        da_ts.sel(
            time=da_ts['time.season'] == 'SON').dropna('time'),
        legend=False,
        ax=ax,
        shade=True)
    plt.legend(['ALL', 'MAM', 'DJF', 'SON', 'JJA'])
    return


def plot_diurnal_pw_all_seasons(path=work_yuval, season='ALL', synoptic=None,
                                fontsize=20, labelsize=18,
                                ylim=[-2.7, 3.3], save=True):
    import xarray as xr
    from synoptic_procedures import slice_xr_with_synoptic_class
    gnss_filename = 'GNSS_PW_thresh_50_for_diurnal_analysis_removed_daily.nc'
    pw = xr.load_dataset(path / gnss_filename)
    df_annual = pw.groupby('time.hour').mean().to_dataframe()
    if season is None and synoptic is None:
        # plot annual diurnal cycle only:
        fg = plot_pw_geographical_segments(df_annual, fg=None, marker='o', color='b',
                                           ylim=ylim)
        legend = ['Annual']
    elif season == 'ALL' and synoptic is None:
        df_jja = pw.sel(time=pw['time.season'] == 'JJA').groupby(
            'time.hour').mean().to_dataframe()
        df_son = pw.sel(time=pw['time.season'] == 'SON').groupby(
            'time.hour').mean().to_dataframe()
        df_djf = pw.sel(time=pw['time.season'] == 'DJF').groupby(
            'time.hour').mean().to_dataframe()
        df_mam = pw.sel(time=pw['time.season'] == 'MAM').groupby(
            'time.hour').mean().to_dataframe()
        fg = plot_pw_geographical_segments(
            df_jja,
            fg=None,
            marker='s',
            color='tab:green',
            ylim=ylim,
            fontsize=fontsize,
            labelsize=labelsize, zorder=0, label='JJA')
        fg = plot_pw_geographical_segments(
            df_son,
            fg=fg,
            marker='^',
            color='tab:red',
            ylim=ylim,
            fontsize=fontsize,
            labelsize=labelsize, zorder=1, label='SON')
        fg = plot_pw_geographical_segments(
            df_djf,
            fg=fg,
            marker='x',
            color='tab:blue',
            fontsize=fontsize,
            labelsize=labelsize, zorder=2, label='DJF')
        fg = plot_pw_geographical_segments(
            df_mam,
            fg=fg,
            marker='+',
            color='tab:orange',
            ylim=ylim,
            fontsize=fontsize,
            labelsize=labelsize, zorder=4, label='MAM')
        fg = plot_pw_geographical_segments(df_annual, fg=fg, marker='d',
                                           color='tab:purple', ylim=ylim,
                                           fontsize=fontsize,
                                           labelsize=labelsize, zorder=3,
                                           label='Annual')
    elif season is None and synoptic == 'ALL':
        df_pt = slice_xr_with_synoptic_class(
            pw, path=path, syn_class='PT').groupby('time.hour').mean().to_dataframe()
        df_rst = slice_xr_with_synoptic_class(
            pw, path=path, syn_class='RST').groupby('time.hour').mean().to_dataframe()
        df_cl = slice_xr_with_synoptic_class(
            pw, path=path, syn_class='CL').groupby('time.hour').mean().to_dataframe()
        df_h = slice_xr_with_synoptic_class(
            pw, path=path, syn_class='H').groupby('time.hour').mean().to_dataframe()
        fg = plot_pw_geographical_segments(
            df_pt,
            fg=None,
            marker='s',
            color='tab:green',
            ylim=ylim,
            fontsize=fontsize,
            labelsize=labelsize, zorder=0, label='PT')
        fg = plot_pw_geographical_segments(
            df_rst,
            fg=fg,
            marker='^',
            color='tab:red',
            ylim=ylim,
            fontsize=fontsize,
            labelsize=labelsize, zorder=1, label='RST')
        fg = plot_pw_geographical_segments(
            df_cl,
            fg=fg,
            marker='x',
            color='tab:blue',
            fontsize=fontsize,
            labelsize=labelsize, zorder=2, label='CL')
        fg = plot_pw_geographical_segments(
            df_h,
            fg=fg,
            marker='+',
            color='tab:orange',
            ylim=ylim,
            fontsize=fontsize,
            labelsize=labelsize, zorder=4, label='H')
        fg = plot_pw_geographical_segments(df_annual, fg=fg, marker='d',
                                           color='tab:purple', ylim=ylim,
                                           fontsize=fontsize,
                                           labelsize=labelsize, zorder=3,
                                           label='Annual')
    sites = group_sites_to_xarray(False, scope='diurnal')
    for i, (ax, site) in enumerate(zip(fg.axes.flatten(), sites.values.flatten())):
        lns = ax.get_lines()
        if site in ['yrcm']:
            leg_loc = 'upper right'
        elif site in ['nrif', 'elat']:
            leg_loc = 'upper center'
        elif site in ['ramo']:
            leg_loc = 'lower center'
        else:
            leg_loc = None
        # do legend for each panel:
#        ax.legend(
#            lns,
#            legend,
#            prop={
#                'size': 12},
#            framealpha=0.5,
#            fancybox=True,
#            ncol=2,
#            loc=leg_loc, fontsize=12)
    lines_labels = [ax.get_legend_handles_labels() for ax in fg.fig.axes][0]
#    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fg.fig.legend(lines_labels[0], lines_labels[1], prop={'size': 20}, edgecolor='k',
                  framealpha=0.5, fancybox=True, facecolor='white',
                  ncol=5, fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.005),
                  bbox_transform=plt.gcf().transFigure)
    fg.fig.subplots_adjust(
        top=0.973,
        bottom=0.029,
        left=0.054,
        right=0.995,
        hspace=0.15,
        wspace=0.12)
    if save:
        filename = 'pw_diurnal_geo_{}.png'.format(season)
#        plt.savefig(savefig_path / filename, bbox_inches='tight')
        plt.savefig(savefig_path / filename, orientation='portrait')
    return fg


def plot_climate_classification(path=climate_path, gis_path=gis_path,
                                fontsize=16):
    import xarray as xr
    from climate_works import read_climate_classification_legend
    from PW_stations import produce_geo_gnss_solved_stations
    import numpy as np
    from matplotlib import colors

    ras = xr.open_rasterio(path / 'Beck_KG_V1_present_0p0083.tif')
    ds = ras.isel(band=0)
    minx = 34.0
    miny = 29.0
    maxx = 36.5
    maxy = 34.0
    ds = ds.sortby('y')
    ds = ds.sel(x=slice(minx, maxx), y=slice(miny, maxy))
    ds = ds.astype(int)
    ds = ds.reset_coords(drop=True)
    ax_map = plot_israel_map(
        gis_path=gis_path,
        ax=None,
        ticklabelsize=fontsize)
    df = read_climate_classification_legend(path)
    # get color pixels to dict:
    d = df['color'].to_dict()
    sort_idx = np.argsort([x for x in d.keys()])
    idx = np.searchsorted([x for x in d.keys()], ds.values, sorter=sort_idx)
    out = np.asarray([x for x in d.values()])[sort_idx][idx]
    ds_as_color = xr.DataArray(out, dims=['y', 'x', 'c'])
    ds_as_color['y'] = ds['y']
    ds_as_color['x'] = ds['x']
    ds_as_color['c'] = ['R', 'G', 'B']
    # overlay with dem data:
#    cmap = plt.get_cmap('terrain', 41)
#    df_gnss = produce_geo_gnss_solved_stations(plot=False)
#    c_colors = df.set_index('class_code').loc[df_gnss['code'].unique()]['color'].values
    c_colors = df['color'].values
    c_li = [c for c in c_colors]
    c_colors = np.asarray(c_li)
    c_colors = np.unique(ds_as_color.stack(coor=['x', 'y']).T.values, axis=0)
    # remove black:
#    c_colors = c_colors[:-1]
    int_code = np.unique(ds.stack(coor=['x', 'y']).T.values, axis=0)
    ticks = [df.loc[x]['class_code'] for x in int_code[1:]]
    cc = [df.set_index('class_code').loc[x]['color'] for x in ticks]
    cc_as_hex = [colors.rgb2hex(x) for x in cc]
    tickd = dict(zip(cc_as_hex, ticks))
#    ticks.append('Water')
#    ticks.reverse()
    bounds = [x for x in range(len(c_colors) + 1)]
    chex = [colors.rgb2hex(x) for x in c_colors]
    ticks = [tickd.get(x, 'Water') for x in chex]
    cmap = colors.ListedColormap(chex)
    norm = colors.BoundaryNorm(bounds, cmap.N)
#    vmin = ds_as_color.min().item()
#    vmax = ds_as_color.max().item()
    im = ds_as_color.plot.imshow(
        ax=ax_map,
        alpha=.7,
        add_colorbar=False,
        cmap=cmap,
        interpolation='antialiased',
        origin='lower',
        norm=norm)
#    colours = im.cmap(im.norm(np.unique(ds_as_color)))
#    chex = [colors.rgb2hex(x) for x in colours]
#    cmap = colors.ListedColormap(chex)
#    bounds=[x for x in range(len(colours))]
    cbar_kwargs = {'fraction': 0.1, 'aspect': 50, 'pad': 0.03}
    cb = plt.colorbar(
        im,
        boundaries=bounds,
        ticks=None,
        ax=ax_map,
        **cbar_kwargs)
    cb.set_label(
        label='climate classification',
        size=fontsize,
        weight='normal')
    n = len(c_colors)
    tick_locs = (np.arange(n) + 0.5) * (n) / n
    cb.set_ticks(tick_locs)
    # set tick labels (as before)
    cb.set_ticklabels(ticks)
    cb.ax.tick_params(labelsize=fontsize)
    ax_map.set_xlabel('')
    ax_map.set_ylabel('')
    # now for the gps stations:
    gps = produce_geo_gnss_solved_stations(plot=False)
    removed = ['hrmn', 'gilb', 'lhav', 'nizn', 'spir']
    removed = []
    print('removing {} stations from map.'.format(removed))
#    merged = ['klhv', 'lhav', 'mrav', 'gilb']
    merged = []
    gps_list = [x for x in gps.index if x not in merged and x not in removed]
    gps.loc[gps_list, :].plot(ax=ax_map, edgecolor='black', marker='s',
                              alpha=1.0, markersize=35, facecolor="None", linewidth=2, zorder=3)
    gps_stations = gps_list
    to_plot_offset = []
    for x, y, label in zip(gps.loc[gps_stations, :].lon, gps.loc[gps_stations,
                                                                 :].lat, gps.loc[gps_stations, :].index.str.upper()):
        if label.lower() in to_plot_offset:
            ax_map.annotate(label, xy=(x, y), xytext=(4, -6),
                            textcoords="offset points", color='k',
                            fontweight='bold', fontsize=fontsize - 2)
        else:
            ax_map.annotate(label, xy=(x, y), xytext=(3, 3),
                            textcoords="offset points", color='k',
                            fontweight='bold', fontsize=fontsize - 2)
    return


def group_sites_to_xarray(upper=False, scope='diurnal'):
    import xarray as xr
    import numpy as np
    if scope == 'diurnal':
        group1 = ['KABR', 'BSHM', 'CSAR', 'TELA', 'ALON', 'SLOM', 'NIZN']
        group2 = ['NZRT', 'MRAV', 'YOSH', 'JSLM', 'KLHV', 'YRCM', 'RAMO']
        group3 = ['ELRO', 'KATZ', 'DRAG', 'DSEA', 'SPIR', 'NRIF', 'ELAT']
    elif scope == 'annual':
        group1 = ['KABR', 'BSHM', 'CSAR', 'TELA', 'ALON', 'SLOM', 'NIZN']
        group2 = ['NZRT', 'MRAV', 'YOSH', 'JSLM', 'KLHV', 'YRCM', 'RAMO']
        group3 = ['ELRO', 'KATZ', 'DRAG', 'DSEA', 'SPIR', 'NRIF', 'ELAT']
    if not upper:
        group1 = [x.lower() for x in group1]
        group2 = [x.lower() for x in group2]
        group3 = [x.lower() for x in group3]
    gr1 = xr.DataArray(group1, dims='GNSS')
    gr2 = xr.DataArray(group2, dims='GNSS')
    gr3 = xr.DataArray(group3, dims='GNSS')
    gr1['GNSS'] = np.arange(0, len(gr1))
    gr2['GNSS'] = np.arange(0, len(gr2))
    gr3['GNSS'] = np.arange(0, len(gr3))
    sites = xr.concat([gr1, gr2, gr3], 'group').T
    sites['group'] = ['coastal', 'highland', 'eastern']
    return sites


# def plot_diurnal_pw_geographical_segments(df, fg=None, marker='o', color='b',
#                                          ylim=[-2, 3]):
#    import xarray as xr
#    import numpy as np
#    from matplotlib.ticker import MultipleLocator
#    from PW_stations import produce_geo_gnss_solved_stations
#    geo = produce_geo_gnss_solved_stations(plot=False)
#    sites = group_sites_to_xarray(upper=False, scope='diurnal')
#    sites_flat = [x for x in sites.values.flatten() if isinstance(x, str)]
#    da = xr.DataArray([x for x in range(len(sites_flat))], dims='GNSS')
#    da['GNSS'] = [x for x in range(len(da))]
#    if fg is None:
#        fg = xr.plot.FacetGrid(
#            da,
#            col='GNSS',
#            col_wrap=3,
#            sharex=False,
#            sharey=False, figsize=(20, 20))
#    for i in range(fg.axes.shape[0]):  # i is rows
#        for j in range(fg.axes.shape[1]):  # j is cols
#            try:
#                site = sites.values[i, j]
#                ax = fg.axes[i, j]
#                df.loc[:, site].plot(ax=ax, marker=marker, color=color)
#                ax.set_xlabel('Hour of day [UTC]')
#                ax.yaxis.tick_left()
#                ax.grid()
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.spines["bottom"].set_visible(False)
#                ax.xaxis.set_ticks(np.arange(0, 23, 3))
#                if j == 0:
#                    ax.set_ylabel('PW anomalies [mm]', fontsize=12)
# elif j == 1:
# if i>5:
##                        ax.set_ylabel('PW anomalies [mm]', fontsize=12)
#                site_label = '{} ({:.0f})'.format(site.upper(), geo.loc[site].alt)
#                ax.text(.12, .85, site_label,
#                        horizontalalignment='center', fontweight='bold',
#                        transform=ax.transAxes)
# ax.yaxis.set_minor_locator(MultipleLocator(3))
# ax.yaxis.grid(
# True,
# which='minor',
# linestyle='--',
# linewidth=1,
# alpha=0.7)
##                ax.yaxis.grid(True, linestyle='--', linewidth=1, alpha=0.7)
#                if ylim is not None:
#                    ax.set_ylim(*ylim)
#            except KeyError:
#                ax.set_axis_off()
# for i, ax in enumerate(fg.axes[:, 0]):
# try:
##            df[gr1].iloc[:, i].plot(ax=ax)
# except IndexError:
# ax.set_axis_off()
# for i, ax in enumerate(fg.axes[:, 1]):
# try:
##            df[gr2].iloc[:, i].plot(ax=ax)
# except IndexError:
# ax.set_axis_off()
# for i, ax in enumerate(fg.axes[:, 2]):
# try:
##            df[gr3].iloc[:, i].plot(ax=ax)
# except IndexError:
# ax.set_axis_off()
#
#    fg.fig.tight_layout()
#    fg.fig.subplots_adjust()
#    return fg


def prepare_reanalysis_monthly_pwv_to_dataframe(path=work_yuval, re='era5',
                                                ds=None):
    import xarray as xr
    import pandas as pd
    if re == 'era5':
        reanalysis = xr.load_dataset(work_yuval / 'GNSS_era5_monthly_PW.nc')
        re_name = 'ERA5'
    elif re == 'uerra':
        reanalysis = xr.load_dataset(work_yuval / 'GNSS_uerra_monthly_PW.nc')
        re_name = 'UERRA-HARMONIE'
    elif re is not None and ds is not None:
        reanalysis = ds
        re_name = re
    df_re = reanalysis.to_dataframe()
    df_re['month'] = df_re.index.month
    pw_mm = xr.load_dataset(
        work_yuval /
        'GNSS_PW_monthly_thresh_50_homogenized.nc')
    df = pw_mm.to_dataframe()
    df['month'] = df.index.month
    # concat:
    dff = pd.concat([df, df_re], keys=['GNSS', re_name])
    dff['source'] = dff.index.get_level_values(0)
    dff = dff.reset_index()
    return dff



def plot_long_term_era5_comparison(path=work_yuval, era5_path=era5_path,
                                   fontsize=16,
                                   remove_stations=['nizn', 'spir'], save=True):
    import xarray as xr
    from aux_gps import anomalize_xr

#    from aeronet_analysis import prepare_station_to_pw_comparison
    # from PW_stations import ML_Switcher
    # from aux_gps import get_julian_dates_from_da
    # from scipy.stats.mstats import theilslopes
    # TODO: add merra2, 3 panel plot and trend
    # load GNSS Israel:
    sns.set_style('whitegrid')
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    pw = xr.load_dataset(
        path / 'GNSS_PW_monthly_thresh_50.nc').sel(time=slice('1998', None))
    if remove_stations is not None:
        pw = pw[[x for x in pw if x not in remove_stations]]
    pw_anoms = anomalize_xr(pw, 'MS', verbose=False)
    pw_percent = anomalize_xr(pw, 'MS', verbose=False, units='%')
    pw_percent = pw_percent.to_array('station').mean('station')
    pw_mean = pw_anoms.to_array('station').mean('station')
    pw_mean = pw_mean.sel(time=slice('1998', '2019'))
    # load ERA5:
    era5 = xr.load_dataset(path / 'GNSS_era5_monthly_PW.nc')
    era5_anoms = anomalize_xr(era5, 'MS', verbose=False)
    era5_mean = era5_anoms.to_array('station').mean('station')
    df = pw_mean.to_dataframe(name='GNSS')
    # load MERRA2:
    # merra2 = xr.load_dataset(
    #     path / 'MERRA2/MERRA2_TQV_israel_area_1995-2019.nc')['TQV']
    # merra2_mm = merra2.resample(time='MS').mean()
    # merra2_anoms = anomalize_xr(
    #     merra2_mm, time_dim='time', freq='MS', verbose=False)
    # merra2_mean = merra2_anoms.mean('lat').mean('lon')
    # load AERONET:
#    if aero_path is not None:
#        aero = prepare_station_to_pw_comparison(path=aero_path, gis_path=gis_path,
#                                                station='boker', mm_anoms=True)
#        df['AERONET'] = aero.to_dataframe()
    era5_to_plot = era5_mean - 5
    # merra2_to_plot = merra2_mean - 10
    df['ERA5'] = era5_mean.to_dataframe(name='ERA5')
    # df['MERRA2'] = merra2_mean.to_dataframe('MERRA2')
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
#    df['GNSS'].plot(ax=ax, color='k')
#    df['ERA5'].plot(ax=ax, color='r')
#    df['AERONET'].plot(ax=ax, color='b')
    pwln = pw_mean.plot.line('k-', marker='o', ax=ax,
                             linewidth=2, markersize=3.5)
    era5ln = era5_to_plot.plot.line(
        'k--', marker='s', ax=ax, linewidth=2, markersize=3.5)
    # merra2ln = merra2_to_plot.plot.line(
    #     'g-', marker='d', ax=ax, linewidth=2, markersize=2.5)
    era5corr = df.corr().loc['GNSS', 'ERA5']
    # merra2corr = df.corr().loc['GNSS', 'MERRA2']
    handles = pwln + era5ln # + merra2ln
    # labels = ['GNSS', 'ERA5, r={:.2f}'.format(
    #     era5corr), 'MERRA2, r={:.2f}'.format(merra2corr)]
    labels = ['GNSS station average', 'ERA5 regional mean, r={:.2f}'.format(
        era5corr)]
    ax.legend(handles=handles, labels=labels, loc='upper left',
                 prop={'size': fontsize-2})
#    if aero_path is not None:
#        aeroln = aero.plot.line('b-.', ax=ax, alpha=0.8)
#        aerocorr = df.corr().loc['GNSS', 'AERONET']
#        aero_label = 'AERONET, r={:.2f}'.format(aerocorr)
#        handles += aeroln
    ax.set_ylabel('PWV anomalies [mm]', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel('')
    ax.grid()
    ax = fix_time_axis_ticks(ax, limits=['1998-01', '2020-01'])
    fig.tight_layout()
    if save:
        filename = 'pwv_long_term_anomalies_era5_comparison.png'
        plt.savefig(savefig_path / filename, orientation='portrait')
    return fig


def plot_long_term_anomalies_with_trends(path=work_yuval,
                                         model_name='TSEN',
                                         fontsize=16,
                                         remove_stations=['nizn', 'spir'],
                                         save=True,
                                         add_percent=False):  # ,aero_path=aero_path):
    import xarray as xr
    from aux_gps import anomalize_xr
    from PW_stations import mann_kendall_trend_analysis
    from aux_gps import linear_fit_using_scipy_da_ts
    sns.set_style('whitegrid')
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    pw = xr.load_dataset(
        path / 'GNSS_PW_monthly_thresh_50.nc').sel(time=slice('1998', None))
    if remove_stations is not None:
        pw = pw[[x for x in pw if x not in remove_stations]]
    pw_anoms = anomalize_xr(pw, 'MS', verbose=False)
    pw_percent = anomalize_xr(pw, 'MS', verbose=False, units='%')
    pw_percent = pw_percent.to_array('station').mean('station')
    pw_mean = pw_anoms.to_array('station').mean('station')
    pw_mean = pw_mean.sel(time=slice('1998', '2019'))
    if add_percent:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(16, 5))
        axes = [ax, ax]
    pwln = pw_mean.plot.line('k-', marker='o', ax=axes[0],
                             linewidth=2, markersize=5.5)
    handles = pwln
    labels = ['GNSS station average']
    pwv_trends, trend_dict = linear_fit_using_scipy_da_ts(
        pw_mean, model=model_name, slope_factor=3652.5, plot=False)
    trend = pwv_trends['trend']
    trend_hi = pwv_trends['trend_hi']
    trend_lo = pwv_trends['trend_lo']
    slope_hi = trend_dict['slope_hi']
    slope_lo = trend_dict['slope_lo']
    slope = trend_dict['slope']
    mann_pval = mann_kendall_trend_analysis(pw_mean).loc['p']
    trend_label = r'{} model, slope={:.2f} ({:.2f}, {:.2f}) mm$\cdot$decade$^{{-1}}$, pvalue={:.4f}'.format(
        model_name, slope, slope_lo, slope_hi, mann_pval)
    labels.append(trend_label)
    trendln = trend.plot(ax=axes[0], color='b', linewidth=2, alpha=1)
    handles += trendln
    trend_hi.plot.line('b--', ax=axes[0], linewidth=1.5, alpha=0.8)
    trend_lo.plot.line('b--', ax=axes[0], linewidth=1.5, alpha=0.8)
    pwv_trends, trend_dict = linear_fit_using_scipy_da_ts(
        pw_mean.sel(time=slice('2010', '2019')), model=model_name, slope_factor=3652.5, plot=False)
    mann_pval = mann_kendall_trend_analysis(pw_mean.sel(time=slice('2010','2019'))).loc['p']
    trend = pwv_trends['trend']
    trend_hi = pwv_trends['trend_hi']
    trend_lo = pwv_trends['trend_lo']
    slope_hi = trend_dict['slope_hi']
    slope_lo = trend_dict['slope_lo']
    slope = trend_dict['slope']
    trendln = trend.plot(ax=axes[0], color='r', linewidth=2, alpha=1)
    handles += trendln
    trend_label = r'{} model, slope={:.2f} ({:.2f}, {:.2f}) mm$\cdot$decade$^{{-1}}$, pvalue={:.4f}'.format(
            model_name, slope, slope_lo, slope_hi, mann_pval)
    labels.append(trend_label)
    trend_hi.plot.line('r--', ax=axes[0], linewidth=1.5, alpha=0.8)
    trend_lo.plot.line('r--', ax=axes[0], linewidth=1.5, alpha=0.8)
    # ax.grid()
    # ax.set_xlabel('')
    # ax.set_ylabel('PWV mean anomalies [mm]')
    # ax.legend(labels=[],handles=[trendln[0]])
    # fig.tight_layout()
    axes[0].legend(handles=handles, labels=labels, loc='upper left',
              prop={'size': fontsize-2})
    axes[0].set_ylabel('PWV anomalies [mm]', fontsize=fontsize)
    axes[0].tick_params(labelsize=fontsize)
    axes[0].set_xlabel('')
    axes[0].grid(True)
    axes[0] = fix_time_axis_ticks(axes[0], limits=['1998-01', '2020-01'])
    if add_percent:
        pwln = pw_percent.plot.line('k-', marker='o', ax=axes[1],
                                    linewidth=2, markersize=5.5)
        handles = pwln
        labels = ['GNSS station average']
        pwv_trends, trend_dict = linear_fit_using_scipy_da_ts(
            pw_percent, model=model_name, slope_factor=3652.5, plot=False)
        trend = pwv_trends['trend']
        trend_hi = pwv_trends['trend_hi']
        trend_lo = pwv_trends['trend_lo']
        slope_hi = trend_dict['slope_hi']
        slope_lo = trend_dict['slope_lo']
        slope = trend_dict['slope']
        mann_pval = mann_kendall_trend_analysis(pw_percent).loc['p']
        trend_label = r'{} model, slope={:.2f} ({:.2f}, {:.2f}) %$\cdot$decade$^{{-1}}$, pvalue={:.4f}'.format(
            model_name, slope, slope_lo, slope_hi, mann_pval)
        labels.append(trend_label)
        trendln = trend.plot(ax=axes[1], color='b', linewidth=2, alpha=1)
        handles += trendln
        trend_hi.plot.line('b--', ax=axes[1], linewidth=1.5, alpha=0.8)
        trend_lo.plot.line('b--', ax=axes[1], linewidth=1.5, alpha=0.8)
        pwv_trends, trend_dict = linear_fit_using_scipy_da_ts(
            pw_percent.sel(time=slice('2010', '2019')), model=model_name, slope_factor=3652.5, plot=False)
        mann_pval = mann_kendall_trend_analysis(pw_percent.sel(time=slice('2010','2019'))).loc['p']
        trend = pwv_trends['trend']
        trend_hi = pwv_trends['trend_hi']
        trend_lo = pwv_trends['trend_lo']
        slope_hi = trend_dict['slope_hi']
        slope_lo = trend_dict['slope_lo']
        slope = trend_dict['slope']
        trendln = trend.plot(ax=axes[1], color='r', linewidth=2, alpha=1)
        handles += trendln
        trend_label = r'{} model, slope={:.2f} ({:.2f}, {:.2f}) %$\cdot$decade$^{{-1}}$, pvalue={:.4f}'.format(
                model_name, slope, slope_lo, slope_hi, mann_pval)
        labels.append(trend_label)
        trend_hi.plot.line('r--', ax=axes[1], linewidth=1.5, alpha=0.8)
        trend_lo.plot.line('r--', ax=axes[1], linewidth=1.5, alpha=0.8)
        # ax.grid()
        # ax.set_xlabel('')
        # ax.set_ylabel('PWV mean anomalies [mm]')
        # ax.legend(labels=[],handles=[trendln[0]])
        # fig.tight_layout()
        axes[1].legend(handles=handles, labels=labels, loc='upper left',
                       prop={'size': fontsize-2})
        axes[1].set_ylabel('PWV anomalies [%]', fontsize=fontsize)
        axes[1].tick_params(labelsize=fontsize)
        axes[1].set_xlabel('')
        axes[1].grid()
        axes[1] = fix_time_axis_ticks(axes[1], limits=['1998-01', '2020-01'])
    fig.tight_layout()
    if save:
        filename = 'pwv_station_averaged_trends.png'
        plt.savefig(savefig_path / filename, orientation='portrait')
    return fig


def plot_day_night_pwv_monthly_mean_std_heatmap(
        path=work_yuval, day_time=['09:00', '15:00'], night_time=['17:00', '21:00'], compare=['day', 'std']):
    import xarray as xr
    import seaborn as sns
    import matplotlib.pyplot as plt
    pw = xr.load_dataset(work_yuval / 'GNSS_PW_thresh_50_homogenized.nc')
    pw = pw[[x for x in pw if 'error' not in x]]
    df = pw.to_dataframe()
    sites = group_sites_to_xarray(upper=False, scope='annual')
    coast = [x for x in sites.sel(group='coastal').dropna('GNSS').values]
    high = [x for x in sites.sel(group='highland').dropna('GNSS').values]
    east = [x for x in sites.sel(group='eastern').dropna('GNSS').values]
    box_coast = dict(facecolor='cyan', pad=0.05, alpha=0.4)
    box_high = dict(facecolor='green', pad=0.05, alpha=0.4)
    box_east = dict(facecolor='yellow', pad=0.05, alpha=0.4)
    color_dict = [{x: box_coast} for x in coast]
    color_dict += [{x: box_high} for x in high]
    color_dict += [{x: box_east} for x in east]
    color_dict = dict((key, d[key]) for d in color_dict for key in d)
    sites = sites.T.values.ravel()
    sites_flat = [x for x in sites if isinstance(x, str)]
    df = df[sites_flat]
    df_mm = df.resample('MS').mean()
    df_mm_mean = df_mm.groupby(df_mm.index.month).mean()
    df_mm_std = df_mm.groupby(df_mm.index.month).std()
    df_day = df.between_time(*day_time)
    df_night = df.between_time(*night_time)
    df_day_mm = df_day.resample('MS').mean()
    df_night_mm = df_night.resample('MS').mean()
    day_std = df_day_mm.groupby(df_day_mm.index.month).std()
    night_std = df_night_mm.groupby(df_night_mm.index.month).std()
    day_mean = df_day_mm.groupby(df_day_mm.index.month).mean()
    night_mean = df_night_mm.groupby(df_night_mm.index.month).mean()
    per_day_std = 100 * (day_std - df_mm_std) / df_mm_std
    per_day_mean = 100 * (day_mean - df_mm_mean) / df_mm_mean
    per_night_std = 100 * (night_std - df_mm_std) / df_mm_std
    per_night_mean = 100 * (night_mean - df_mm_mean) / df_mm_mean
    day_night = compare[0]
    mean_std = compare[1]
    fig, axes = plt.subplots(
        1, 2, sharex=False, sharey=False, figsize=(17, 10))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    if compare[1] == 'std':
        all_heat = df_mm_std.T
        day_heat = day_std.T
        title = 'STD'
    elif compare[1] == 'mean':
        all_heat = df_mm_mean.T
        day_heat = day_mean.T
        title = 'MEAN'
    vmax = max(day_heat.max().max(), all_heat.max().max())
    vmin = min(day_heat.min().min(), all_heat.min().min())
    sns.heatmap(all_heat, ax=axes[0], cbar=False, vmin=vmin, vmax=vmax,
                annot=True, cbar_ax=None, cmap='Reds')
    sns.heatmap(day_heat, ax=axes[1], cbar=True, vmin=vmin, vmax=vmax,
                annot=True, cbar_ax=cbar_ax, cmap='Reds')
    labels_1 = [x for x in axes[0].yaxis.get_ticklabels()]
    [label.set_bbox(color_dict[label.get_text()]) for label in labels_1]
    labels_2 = [x for x in axes[1].yaxis.get_ticklabels()]
    [label.set_bbox(color_dict[label.get_text()]) for label in labels_2]
    axes[0].set_title('All {} in mm'.format(title))
    axes[1].set_title('Day only ({}-{}) {} in mm'.format(*day_time, title))
    [ax.set_xlabel('month') for ax in axes]
    fig.tight_layout(rect=[0, 0, .9, 1])

#    fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(17, 10))
#    ax_mean = sns.heatmap(df_mm_mean.T, annot=True, ax=axes[0])
#    ax_mean.set_title('All mean in mm')
#    ax_std = sns.heatmap(df_mm_std.T, annot=True, ax=axes[1])
#    ax_std.set_title('All std in mm')
#    labels_mean = [x for x in ax_mean.yaxis.get_ticklabels()]
#    [label.set_bbox(color_dict[label.get_text()]) for label in labels_mean]
#    labels_std = [x for x in ax_std.yaxis.get_ticklabels()]
#    [label.set_bbox(color_dict[label.get_text()]) for label in labels_std]
#    [ax.set_xlabel('month') for ax in axes]
#    fig.tight_layout()
#    fig_day, axes_day = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(17, 10))
#    ax_mean = sns.heatmap(per_day_mean.T, annot=True, cmap='bwr', center=0, ax=axes_day[0])
#    ax_mean.set_title('Day mean - All mean in % from All mean')
#    ax_std = sns.heatmap(per_day_std.T, annot=True, cmap='bwr', center=0, ax=axes_day[1])
#    ax_std.set_title('Day std - All std in % from All std')
#    labels_mean = [x for x in ax_mean.yaxis.get_ticklabels()]
#    [label.set_bbox(color_dict[label.get_text()]) for label in labels_mean]
#    labels_std = [x for x in ax_std.yaxis.get_ticklabels()]
#    [label.set_bbox(color_dict[label.get_text()]) for label in labels_std]
#    [ax.set_xlabel('month') for ax in axes_day]
#    fig_day.tight_layout()
#    fig_night, axes_night = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(17, 10))
#    ax_mean = sns.heatmap(per_night_mean.T, annot=True, cmap='bwr', center=0, ax=axes_night[0])
#    ax_mean.set_title('Night mean - All mean in % from All mean')
#    ax_std = sns.heatmap(per_night_std.T, annot=True, cmap='bwr', center=0, ax=axes_night[1])
#    ax_std.set_title('Night std - All std in % from All std')
#    labels_mean = [x for x in ax_mean.yaxis.get_ticklabels()]
#    [label.set_bbox(color_dict[label.get_text()]) for label in labels_mean]
#    labels_std = [x for x in ax_std.yaxis.get_ticklabels()]
#    [label.set_bbox(color_dict[label.get_text()]) for label in labels_std]
#    [ax.set_xlabel('month') for ax in axes_night]
#    fig_night.tight_layout()
    return fig


def plot_pw_geographical_segments(df, scope='diurnal', kind=None, fg=None,
                                  marker='o', color='b', ylim=[-2, 3],
                                  hue=None, fontsize=14, labelsize=10,
                                  ticklabelcolor=None,
                                  zorder=0, label=None, save=False, bins=None):
    import xarray as xr
    import numpy as np
    from scipy.stats import kurtosis
    from scipy.stats import skew
    from matplotlib.ticker import MultipleLocator
    from PW_stations import produce_geo_gnss_solved_stations
    from matplotlib.ticker import AutoMinorLocator
    from matplotlib.ticker import FormatStrFormatter
    import seaborn as sns
    scope_dict = {'diurnal': {'xticks': np.arange(0, 23, 3),
                              'xlabel': 'Hour of day [UTC]',
                              'ylabel': 'PWV anomalies [mm]',
                              'colwrap': 3},
                  'annual': {'xticks': np.arange(1, 13),
                             'xlabel': 'month',
                             'ylabel': 'PWV [mm]',
                             'colwrap': 3}
                  }
    sns.set_style('whitegrid')
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    color_dict = produce_colors_for_pwv_station(scope=scope, zebra=False, as_dict=True)
    geo = produce_geo_gnss_solved_stations(plot=False)
    sites = group_sites_to_xarray(upper=False, scope=scope)
#    if scope == 'annual':
#        sites = sites.T
    sites_flat = [x for x in sites.values.flatten() if isinstance(x, str)]
    da = xr.DataArray([x for x in range(len(sites_flat))], dims='GNSS')
    da['GNSS'] = [x for x in range(len(da))]
    if fg is None:
        fg = xr.plot.FacetGrid(
            da,
            col='GNSS',
            col_wrap=scope_dict[scope]['colwrap'],
            sharex=False,
            sharey=False, figsize=(20, 20))
    for i in range(fg.axes.shape[0]):  # i is rows
        for j in range(fg.axes.shape[1]):  # j is cols
            site = sites.values[i, j]
            ax = fg.axes[i, j]
            if not isinstance(site, str):
                ax.set_axis_off()
                continue
            else:
                if kind is None:
                    df[site].plot(ax=ax, marker=marker, color=color,
                                  zorder=zorder, label=label)
                    ax.xaxis.set_ticks(scope_dict[scope]['xticks'])
                    ax.grid(True, which='major')
                    ax.grid(True, axis='y', which='minor', linestyle='--')
                elif kind == 'violin':
                    if not 'month' in df.columns:
                        df['month'] = df.index.month
                    pal = sns.color_palette("Paired", 12)
                    sns.violinplot(ax=ax, data=df, x='month', y=df[site],
                                   hue=hue,
                                   fliersize=4, gridsize=250, inner='quartile',
                                   scale='area')
                    ax.set_ylabel('')
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["bottom"].set_visible(False)
                    ax.grid(True, axis='y', which='major')
                    ax.grid(True, axis='y', which='minor', linestyle='--')
                elif kind == 'violin+swarm':
                    if not 'month' in df.columns:
                        df['month'] = df.index.month
                    pal = sns.color_palette("Paired", 12)
                    pal = sns.color_palette("tab20")
                    sns.violinplot(ax=ax, data=df, x='month', y=df[site],
                                   hue=None, color=color_dict.get(site),                                   fliersize=4, gridsize=250, inner=None,
                                   scale='width')
                    sns.swarmplot(ax=ax, data=df, x='month', y=df[site],
                                  color="k", edgecolor="gray",
                                  size=2.8)
                    ax.set_ylabel('')
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["bottom"].set_visible(False)
                    ax.grid(True, axis='y', which='major')
                    ax.grid(True, axis='y', which='minor', linestyle='--')
                elif kind == 'mean_month':
                    if not 'month' in df.columns:
                        df['month'] = df.index.month
                    df_mean = df.groupby('month').mean()
                    df_mean[site].plot(ax=ax, color=color, marker='o', markersize=10, markerfacecolor="None")
                    ax.set_ylabel('')
                    ax.xaxis.set_ticks(scope_dict[scope]['xticks'])
                    ax.set_xlabel('')
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["bottom"].set_visible(False)
                    ax.grid(True, axis='y', which='major')
                    ax.grid(True, axis='y', which='minor', linestyle='--')
                elif kind == 'hist':
                    if bins is None:
                        bins = 15
                    sns.histplot(ax=ax, data=df[site].dropna(),
                                 line_kws={'linewidth': 3}, stat='density', kde=True, bins=bins)
                    ax.set_xlabel('PWV [mm]', fontsize=fontsize)
                    ax.grid(True)
                    ax.set_ylabel('')
                    xmean = df[site].mean()
                    xmedian = df[site].median()
                    std = df[site].std()
                    sk = skew(df[site].dropna().values)
                    kurt = kurtosis(df[site].dropna().values)
                    # xmode = df[y].mode().median()
                    data_x, data_y = ax.lines[0].get_data()
                    ymean = np.interp(xmean, data_x, data_y)
                    ymed = np.interp(xmedian, data_x, data_y)
                    # ymode = np.interp(xmode, data_x, data_y)
                    ax.vlines(x=xmean, ymin=0, ymax=ymean,
                              color='r', linestyle='--', linewidth=3)
                    ax.vlines(x=xmedian, ymin=0, ymax=ymed,
                              color='g', linestyle='-', linewidth=3)
                    # ax.vlines(x=xmode, ymin=0, ymax=ymode, color='k', linestyle='-')
                    ax.legend(['Mean: {:.1f}'.format(
                        xmean), 'Median: {:.1f}'.format(xmedian)], fontsize=fontsize)
#                    ax.text(0.55, 0.45, "Std-Dev:    {:.1f}\nSkewness: {:.1f}\nKurtosis:   {:.1f}".format(std, sk, kurt),transform=ax.transAxes, fontsize=fontsize)
                ax.tick_params(axis='x', which='major', labelsize=labelsize)
                if kind != 'hist':
                    ax.set_xlabel(scope_dict[scope]['xlabel'], fontsize=16)
                ax.yaxis.set_major_locator(plt.MaxNLocator(4))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.tick_params(axis='y', which='major', labelsize=labelsize)
                # set minor y tick labels:
#                ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
#                ax.tick_params(axis='y', which='minor', labelsize=labelsize-8)
                ax.yaxis.tick_left()
                if j == 0:
                    if kind != 'hist':
                        ax.set_ylabel(scope_dict[scope]['ylabel'], fontsize=16)
                    else:
                        ax.set_ylabel('Frequency', fontsize=16)
#                elif j == 1:
#                    if i>5:
#                        ax.set_ylabel(scope_dict[scope]['ylabel'], fontsize=12)
                site_label = '{} ({:.0f})'.format(
                    site.upper(), geo.loc[site].alt)
                ax.text(.17, .87, site_label, fontsize=fontsize,
                        horizontalalignment='center', fontweight='bold',
                        transform=ax.transAxes)
                if ticklabelcolor is not None:
                    ax.tick_params(axis='y', labelcolor=ticklabelcolor)
#                ax.yaxis.grid(
#                    True,
#                    which='minor',
#                    linestyle='--',
#                    linewidth=1,
#                    alpha=0.7)
#                ax.yaxis.grid(True, linestyle='--', linewidth=1, alpha=0.7)
                if ylim is not None:
                    ax.set_ylim(*ylim)
#            except KeyError:
#                ax.set_axis_off()
#    for i, ax in enumerate(fg.axes[:, 0]):
#        try:
#            df[gr1].iloc[:, i].plot(ax=ax)
#        except IndexError:
#            ax.set_axis_off()
#    for i, ax in enumerate(fg.axes[:, 1]):
#        try:
#            df[gr2].iloc[:, i].plot(ax=ax)
#        except IndexError:
#            ax.set_axis_off()
#    for i, ax in enumerate(fg.axes[:, 2]):
#        try:
#            df[gr3].iloc[:, i].plot(ax=ax)
#        except IndexError:
#            ax.set_axis_off()

    fg.fig.tight_layout()
    fg.fig.subplots_adjust()
    if save:
        filename = 'pw_{}_means_{}.png'.format(scope, kind)
        plt.savefig(savefig_path / filename, orientation='portrait')
#        plt.savefig(savefig_path / filename, orientation='landscape')
    return fg


def plot_PWV_comparison_GNSS_radiosonde(path=work_yuval, sound_path=sound_path,
                                        save=True, fontsize=16):
    import xarray as xr
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.patches as mpatches
    import matplotlib
    matplotlib.rcParams['lines.markeredgewidth'] = 1
    sns.set_style('whitegrid')
    sns.set_style('ticks')
    pal = sns.color_palette("tab10", 2)
    # load radiosonde:
    radio = xr.load_dataarray(sound_path / 'bet_dagan_2s_sounding_PWV_2014-2019.nc')
    radio = radio.rename({'sound_time': 'time'})
    radio = radio.resample(time='MS').mean()
    radio.name = 'radio'
    dfr = radio.to_dataframe()
    dfr['month'] = dfr.index.month
    # load tela:
    tela = xr.load_dataset(path / 'GNSS_PW_monthly_thresh_50.nc')['tela']
    dfm = tela.to_dataframe(name='tela-pwv')
    dfm = dfm.loc[dfr.index]
    dfm['month'] = dfm.index.month
    dff = pd.concat([dfm, dfr], keys=['GNSS-TELA', 'Radiosonde'])
    dff['source'] = dff.index.get_level_values(0)
    # dff['month'] = dfm.index.month
    dff = dff.reset_index()
    dff['pwv'] = dff['tela-pwv'].fillna(0)+dff['radio'].fillna(0)
    dff = dff[dff['pwv'] != 0]
    fig = plt.figure(figsize=(20, 6))
    sns.set_style("whitegrid")
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    grid = plt.GridSpec(
        1, 2, width_ratios=[
            2, 1], wspace=0.1, hspace=0)
    ax_ts = fig.add_subplot(grid[0])  # plt.subplot(221)
    ax_v = fig.add_subplot(grid[1])
    # fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    ax_v = sns.violinplot(data=dff, x='month', y='pwv',
                          fliersize=10, gridsize=250, ax=ax_v,
                          inner=None, scale='width', palette=pal,
                          hue='source', split=True, zorder=20)
    [x.set_alpha(0.5) for x in ax_v.collections]
    ax_v = sns.pointplot(x='month', y='pwv', data=dff, estimator=np.mean,
                         dodge=True, ax=ax_v, hue='source', color=None,
                         linestyles='-', markers=['s', 'o'], scale=0.7,
                         ci=None, alpha=0.5, zorder=0, style='source',edgecolor='k', edgewidth=0.4)
    ax_v.get_legend().set_title('')
    p1 = (mpatches.Patch(facecolor=pal[0], edgecolor='k', alpha=0.5))
    p2 = (mpatches.Patch(facecolor=pal[1], edgecolor='k', alpha=0.5))
    handles = [p1, p2]
    ax_v.legend(handles=handles, labels=['GNSS-TELA', 'Radiosonde'],
                loc='upper left', prop={'size': fontsize-2})
    # ax_v.legend(loc='upper left', prop={'size': fontsize-2})
    ax_v.tick_params(labelsize=fontsize)
    ax_v.set_ylabel('')
    ax_v.grid(True, axis='both')
    ax_v.set_xlabel('month', fontsize=fontsize)
    df = dfm['tela-pwv'].to_frame()
    df.columns = ['GNSS-TELA']
    df['Radiosonde'] = dfr['radio']
    cmap = sns.color_palette("tab10", as_cmap=True)
    df.plot(ax=ax_ts, style=['s-', 'o-'], cmap=cmap)
    # df['GNSS-TELA'].plot(ax=ax_ts, style='s-', cmap=cmap)
    # df['Radiosonde'].plot(ax=ax_ts, style='o-', cmap=cmap)
    ax_ts.grid(True, axis='both')
    ylim = ax_v.get_ylim()
    ax_ts.set_ylim(*ylim)
    ax_ts.set_ylabel('PWV [mm]', fontsize=fontsize)
    ax_ts.set_xlabel('')
    ax_ts.legend(loc='upper left', prop={'size': fontsize-2})
    ax_ts.tick_params(labelsize=fontsize)
    fig.tight_layout()
    if save:
        filename = 'pwv_radio_comparison_violin+ts.png'
        plt.savefig(savefig_path / filename, orientation='landscape',bbox_inches='tight')
    return fig


def prepare_diurnal_variability_table(path=work_yuval, rename_cols=True):
    from PW_stations import calculate_diurnal_variability
    df = calculate_diurnal_variability()
    gr = group_sites_to_xarray(scope='diurnal')
    gr_df = gr.to_dataframe('sites')
    new = gr.T.values.ravel()
    geo = [gr_df[gr_df == x].dropna().index.values.item()[1] for x in new]
    geo = [x.title() for x in geo]
    df = df.reindex(new)
    if rename_cols:
        df.columns = ['Annual [%]', 'JJA [%]', 'SON [%]', 'DJF [%]', 'MAM [%]']
    cols = [x for x in df.columns]
    df['Location'] = geo
    cols = ['Location'] + cols
    df = df[cols]
    df.index = df.index.str.upper()
    print(df.to_latex())
    print('')
    print(df.groupby('Location').mean().to_latex())
    return df


def prepare_harmonics_table(path=work_yuval, season='ALL',
                            scope='diurnal', era5=False, add_third=False):
    import xarray as xr
    from aux_gps import run_MLR_harmonics
    import pandas as pd
    import numpy as np
    from calendar import month_abbr
    if scope == 'diurnal':
        cunits = 'cpd'
        grp = 'hour'
        grp_slice = [0, 12]
        tunits = 'UTC'
    elif scope == 'annual':
        cunits = 'cpy'
        grp = 'month'
        grp_slice = [7, 12]
        tunits = 'month'
    if era5:
        ds = xr.load_dataset(work_yuval / 'GNSS_PW_ERA5_harmonics_annual.nc')
    else:
        ds = xr.load_dataset(work_yuval / 'GNSS_PW_harmonics_{}.nc'.format(scope))
    stations = list(set([x.split('_')[0] for x in ds]))
    records = []
    for station in stations:
        if season in ds.dims:
            diu_ph = ds[station + '_mean'].sel({season: season, cunits: 1}).idxmax()
            diu_amp = ds[station + '_mean'].sel({season: season, cunits: 1}).max()
            semidiu_ph = ds[station +
                        '_mean'].sel({season: season, cunits: 2, grp: slice(*grp_slice)}).idxmax()
            semidiu_amp = ds[station +
                        '_mean'].sel({season: season, cunits: 2, grp: slice(*grp_slice)}).max()
        else:
            diu_ph = ds[station + '_mean'].sel({cunits: 1}).idxmax()
            diu_amp = ds[station + '_mean'].sel({cunits: 1}).max()
            semidiu_ph = ds[station +
                        '_mean'].sel({cunits: 2, grp: slice(*grp_slice)}).idxmax()
            semidiu_amp = ds[station +
                         '_mean'].sel({cunits: 2, grp: slice(*grp_slice)}).max()
            if add_third:
                third_ph = ds[station +
                            '_mean'].sel({cunits: 3, grp: slice(*grp_slice)}).idxmax()
                third_amp = ds[station +
                             '_mean'].sel({cunits: 3, grp: slice(*grp_slice)}).max()

        ds_for_MLR = ds[['{}'.format(station), '{}_mean'.format(station)]]
        if add_third:
            harm_di = run_MLR_harmonics(
                ds_for_MLR, season=season, cunits=cunits, plot=False)
            record = [station, diu_amp.item(), diu_ph.item(), harm_di[1],
                      semidiu_amp.item(), semidiu_ph.item(), harm_di[2],
                      third_amp.item(), third_ph.item(), harm_di[3],
                      harm_di[1] + harm_di[2] + harm_di[3]]
        else:
            harm_di = run_MLR_harmonics(
                ds_for_MLR, season=season, cunits=cunits, plot=False)
            record = [station, diu_amp.item(), diu_ph.item(), harm_di[1],
                      semidiu_amp.item(), semidiu_ph.item(), harm_di[2],
                      harm_di[1] + harm_di[2]]

        records.append(record)
    df = pd.DataFrame(records)
    if add_third:
        df.columns = ['Station', 'A1 [mm]', 'P1 [{}]'.format(tunits), 'V1 [%]', 'A2 [mm]',
                  'P2 [{}]'.format(tunits), 'V2 [%]', 'A3 [mm]', 'P3 [{}]'.format(tunits), 'V3 [%]', 'VT [%]']
    else:
        df.columns = ['Station', 'A1 [mm]', 'P1 [{}]'.format(tunits), 'V1 [%]', 'A2 [mm]',
                      'P2 [{}]'.format(tunits), 'V2 [%]', 'VT [%]']
    df = df.set_index('Station')
    gr = group_sites_to_xarray(scope=scope)
    gr_df = gr.to_dataframe('sites')
    new = gr.T.values.ravel()
    # remove nans form mixed nans and str numpy:
    new = new[~pd.isnull(new)]
    geo = [gr_df[gr_df == x].dropna().index.values.item()[1] for x in new]
    geo = [x.title() for x in geo]
    df = df.reindex(new)
    df['Location'] = geo
    df.index = df.index.str.upper()
    pd.options.display.float_format = '{:.1f}'.format
    if scope == 'annual':
        df['P1 [Month]'] = df['P1 [Month]'].astype(int).apply(lambda x: month_abbr[x])
        df['P2 [Month]'] = df['P2 [Month]'].astype(int).apply(lambda x: month_abbr[x])
        if add_third:
            df['P3 [Month]'] = df['P3 [Month]'].astype(int).apply(lambda x: month_abbr[x])
    if add_third:
        df = df[['Location', 'A1 [mm]', 'A2 [mm]', 'A3 [mm]', 'P1 [{}]'.format(tunits),
                 'P2 [{}]'.format(tunits),'P3 [{}]'.format(tunits), 'V1 [%]', 'V2 [%]', 'V3 [%]', 'VT [%]']]
    else:
        df = df[['Location', 'A1 [mm]', 'A2 [mm]', 'P1 [{}]'.format(tunits),
                 'P2 [{}]'.format(tunits), 'V1 [%]', 'V2 [%]', 'VT [%]']]
    print(df.to_latex())
    return df


def plot_station_mean_violin_plot(path=work_yuval,
                                  remove_stations=['nizn','spir'],
                                  fontsize=16, save=True):
    import xarray as xr
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    sns.set_style('whitegrid')
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    pw = xr.load_dataset(path / 'GNSS_PW_monthly_anoms_thresh_50.nc')
    if remove_stations is not None:
        pw = pw[[x for x in pw if x not in remove_stations]]
    pw_mean = pw.to_array('s').mean('s')
    df = pw_mean.to_dataframe(name='pwv')
    df['month'] = df.index.month
    df['last_decade'] = df.index.year >= 2010
    df['years'] = '1997-2009'
    df['years'].loc[df['last_decade']] = '2010-2019'
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    # sns.histplot(pw_mean, bins=25, ax=axes[0], kde=True, stat='count')
    # axes[0].set_xlabel('PWV anomalies [mm]')
    # df = pw_mean.groupby('time.month').std().to_dataframe(name='PWV-SD')
    # df.plot.bar(ax=axes[1], rot=0)
    # axes[1].set_ylabel('PWV anomalies SD [mm]')
    axes[0]= sns.violinplot(ax=axes[0], data=df, x='month', y='pwv', color='tab:purple',
                   fliersize=10, gridsize=250, inner=None, scale='width',
                   hue=None)
    [x.set_alpha(0.8) for x in axes[0].collections]
    sns.swarmplot(ax=axes[0], x="month", y='pwv', data=df,
                  color="k", edgecolor="gray",
                  hue=None, dodge=False)
    colors = ["tab:blue", "tab:red"]  # Set your custom color palette
    blue_red = sns.set_palette(sns.color_palette(colors))
    axes[1] = sns.violinplot(ax=axes[1], data=df, x='month', y='pwv',
                             palette=blue_red, fliersize=10, gridsize=250,
                             inner=None, scale='width',
                             hue='years', split=True)
    sns.swarmplot(ax=axes[1], x="month", y='pwv', data=df,
                  size=4.5, color='k', edgecolor="gray", palette=None,
                  hue='years', dodge=True)
    [x.set_alpha(0.8) for x in axes[1].collections]
    # remove legend, reorder and re-plot:
    axes[1].get_legend().remove()
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles=handles[0:2], labels=labels[0:2],
                   loc='upper left', prop={'size': 16})
    # upper legend:
    color = axes[0].collections[0].get_facecolor()[0]
    handle = (mpatches.Patch(facecolor=color, edgecolor='k', alpha=0.8))
    axes[0].legend(handles=[handle], labels=['1997-2019'],
                   loc='upper left', prop={'size': 16})
    axes[0].grid()
    axes[1].grid()
    axes[0].set_ylabel('PWV anomalies [mm]', fontsize=fontsize)
    axes[1].set_ylabel('PWV anomalies [mm]', fontsize=fontsize)
    axes[0].tick_params(labelsize=fontsize)
    axes[1].tick_params(labelsize=fontsize)
    axes[1].set_xlabel('month', fontsize=fontsize)
    # draw 0 line:
    axes[0].axhline(0, color='k', lw=2, zorder=0)
    axes[1].axhline(0, color='k', lw=2, zorder=0)
    # annotate extreme events :
    axes[0].annotate('2015', xy=(9, 5.58),  xycoords='data',
                     xytext=(8, 7), textcoords='data',
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='right', verticalalignment='center',
                     fontsize=fontsize, fontweight='bold')
    axes[0].annotate('2013', xy=(9, -5.8),  xycoords='data',
                     xytext=(8, -7), textcoords='data',
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='right', verticalalignment='center',
                     fontsize=fontsize, fontweight='bold')
    axes[0].set_ylim(-10, 10)
    axes[1].set_ylim(-10, 10)
    fig.tight_layout()
    fig.subplots_adjust(top=0.984,
                        bottom=0.078,
                        left=0.099,
                        right=0.988,
                        hspace=0.092,
                        wspace=0.175)
    if save:
        filename = 'pwv_inter-annual_violin+swarm.png'
        plt.savefig(savefig_path / filename, orientation='portrait')
    return fig



def plot_october_2015(path=work_yuval):
    import xarray as xr
    pw_daily = xr.load_dataset(work_yuval /
                               'GNSS_PW_daily_thresh_50_homogenized.nc')
    pw = xr.load_dataset(work_yuval / 'GNSS_PW_thresh_50_homogenized.nc')
    pw = pw[[x for x in pw if '_error' not in x]]
    pw_daily = pw_daily[[x for x in pw if '_error' not in x]]
    fig, ax = plt.subplots(figsize=(20, 12))
    ln1 = pw['tela'].sel(time=slice('2015-07', '2015-12')
                         ).plot(linewidth=0.5, ax=ax)
    ln2 = pw['jslm'].sel(time=slice('2015-07', '2015-12')
                         ).plot(linewidth=0.5, ax=ax)
    ln3 = pw_daily['tela'].sel(time=slice(
        '2015-07', '2015-12')).plot(color=ln1[0].get_color(), linewidth=2.0, ax=ax)
    ln4 = pw_daily['jslm'].sel(time=slice(
        '2015-07', '2015-12')).plot(color=ln2[0].get_color(), linewidth=2.0, ax=ax)
    ax.grid()
    ax.legend(ln1+ln2+ln3+ln4, ['TELA-5mins',
                                'JSLM-5mins', 'TELA-daily', 'JSLM-daily'])
    fig, ax = plt.subplots(figsize=(20, 12))
    ln1 = pw['tela'].sel(time='2015-10').plot(ax=ax)
    ln2 = pw['jslm'].sel(time='2015-10').plot(ax=ax)
    ax.grid()
    ax.legend(ln1+ln2, ['TELA-5mins', 'JSLM-5mins'])
    fig, ax = plt.subplots(figsize=(20, 12))
    ln1 = pw['tela'].sel(time=slice('2015-10-22', '2015-10-27')).plot(ax=ax)
    ln2 = pw['jslm'].sel(time=slice('2015-10-22', '2015-10-27')).plot(ax=ax)
    ax.grid()
    ax.legend(ln1+ln2, ['TELA-5mins', 'JSLM-5mins'])
    return ax


def plot_correlation_pwv_mean_anoms_and_qflux_anoms(era5_path=era5_path,
                                                    work_path=work_yuval,
                                                    anoms=None, pwv_mm=None,
                                                    all_months=False, mf='qf',
                                                    add_hline=None, title=None,
                                                    save=True,
                                                    remove_stations=['nizn', 'spir']):
    import xarray as xr
    from aux_gps import anomalize_xr
    import matplotlib.pyplot as plt
    from aux_gps import get_season_for_pandas_dtindex
    from aux_gps import calculate_pressure_integral
    import seaborn as sns
    # first load pw and produce mean anomalies:
    pw = xr.load_dataset(work_path/'GNSS_PW_monthly_thresh_50.nc')
    if remove_stations is not None:
        pw = pw[[x for x in pw if x not in remove_stations]]
    if anoms is None:
        pw_anoms = anomalize_xr(pw, 'MS')
        pw_anoms_mean = pw_anoms.to_array('s').mean('s')
    else:
        pw_anoms_mean = pw[anoms]
    if pwv_mm is not None:
        pw_anoms_mean = pwv_mm
    # now load qflux and resmaple to mm:
    if anoms is None:
        ds = xr.load_dataset(
                era5_path/'ERA5_MF_anomalies_4xdaily_israel_mean_1996-2019.nc')
    else:
        ds = xr.load_dataset(work_path / 'GNSS_ERA5_qf_1996-2019.nc')
        mf = anoms
    qf_mm = ds[mf].resample(time='MS').mean()
    # add pressure integral:
    iqf = calculate_pressure_integral(qf_mm)/9.79
    iqf = iqf.expand_dims('level')
    iqf['level'] = ['integrated']
    qf_mm = xr.concat([qf_mm.sortby('level'), iqf], 'level')
    # now produce corr for each level:
    dsl = [xr.corr(qf_mm.sel(level=x), pw_anoms_mean) for x in ds['level']][::-1]
    dsl.append(xr.corr(qf_mm.sel(level='integrated'), pw_anoms_mean))
    dsl = xr.concat(dsl, 'level')
    # corr = xr.concat(dsl + [iqf], 'level')
    corr_annual = xr.concat(dsl, 'level')
    df = pw_anoms_mean.to_dataframe('pwv')
    df = df.join(qf_mm.to_dataset('level').to_dataframe())
    season = get_season_for_pandas_dtindex(df)
    # corr = df.groupby(df.index.month).corr()['pwv'].unstack()
    corr = df.groupby(season).corr()['pwv'].unstack()
    corr = corr.drop('pwv', axis=1).T
    corr = corr[['DJF','MAM','JJA','SON']]
    corr['Annual'] = corr_annual.to_dataframe('Annual')
    if all_months:
        corr.index.name = 'season'
        fig, ax = plt.subplots(figsize=(6, 9))
        sns.heatmap(corr, annot=True, center=0, cmap='coolwarm', ax=ax, cbar_kws={
                    'label': 'pearson correlation coefficient ', 'aspect': 40})
        ax.set_ylabel('pressure level [hPa]')
        ax.set_xlabel('')
        # add line to separate integrated from level
        ax.hlines([37], *ax.get_xlim(), color='k')
        # add boxes around maximal values:
        ax.hlines([26], [1], [5], color='w', lw=0.5)
        ax.hlines([27], [1], [5], color='w', lw=0.5)
        ax.vlines([1, 2, 3, 4], 26, 27, color='w', lw=0.5)
        ax.hlines([28], [0], [1], color='w', lw=0.5)
        ax.hlines([29], [0], [1], color='w', lw=0.5)
        ax.vlines([0, 1], 28, 29, color='w', lw=0.5)
        fig.tight_layout()
        filename = 'pwv_qflux_levels_correlations_months.png'
    else:
        # fig = plt.figure(figsize=(20, 6))
        # gridax = plt.GridSpec(1, 2, width_ratios=[
        #     10, 2], wspace=0.05)
        # ax_level = fig.add_subplot(gridax[0, 1])  # plt.subplot(221)
        # ax_ts = fig.add_subplot(gridax[0, 0])  # plt.subplot(122)
        fig, ax = plt.subplots(figsize=(8, 6))
        corr_annual = corr_annual.to_dataframe('Annual')
        corr_annual.plot(ax=ax, lw=2, label='Annual', color=seasonal_colors['Annual'])
        colors = [seasonal_colors[x] for x in corr.columns]
        corr.iloc[0:37].plot(ax=ax, lw=2, color=colors)
        # ax_level.yaxis.set_ticks_position("right")
        # ax_level.yaxis.set_label_position("right")
        ax.grid()
        ax.set_ylabel('pearson correlation coefficient')
        ax.set_xlabel('pressure level [hPa]')
        if add_hline is not None:
            ax.axvline(add_hline, color='k', lw=2)
        int_corr = df[['pwv','integrated']].corr()['integrated']['pwv']
        # ax.axhline(int_corr, color='r', linestyle='--', lw=2)
        # df[['pwv', add_hline]].loc['1997':'2019'].plot(ax=ax_ts, secondary_y=add_hline)
        filename = 'pwv_qflux_levels_correlations.png'
    if title is not None:
        fig.suptitle(title)
    if save:
        plt.savefig(savefig_path / filename, orientation='portrait')
    return fig


def plot_pwv_anomalies_histogram(path=work_yuval):
    import xarray as xr
    import numpy as np
    import seaborn as sns
    from scipy.stats import norm
    pw = xr.load_dataset(
        path / 'GNSS_PW_monthly_anoms_thresh_50_homogenized.nc')
    arr = pw.to_array('station').to_dataframe('pw').values.ravel()
    arr_no_nans = arr[~np.isnan(arr)]
    mu, std = norm.fit(arr_no_nans)
    ax = sns.histplot(
        arr_no_nans,
        stat='density',
        color='tab:orange',
        alpha=0.5)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ln = ax.plot(x, p, 'k', linewidth=2)
#    x_std = x[(x>=-std) & (x<=std)]
#    y_std = norm.pdf(x_std, mu, std)
#    x_std2 = x[(x>=-2*std) & (x<=-std) | (x>=std) & (x<=2*std)]
#    y_std2 = norm.pdf(x_std2, mu, std)
#    ax.fill_between(x_std,y_std,0, alpha=0.7, color='b')
#    ax.fill_between(x_std2,y_std2,0, alpha=0.7, color='r')
    y_std = [norm.pdf(std, mu, std), norm.pdf(-std, mu, std)]
    y_std2 = [norm.pdf(std * 2, mu, std), norm.pdf(-std * 2, mu, std)]
    ln_std = ax.vlines([-std, std], ymin=[0, 0], ymax=y_std,
                       color='tab:blue', linewidth=2)
    ln_std2 = ax.vlines([-std * 2, std * 2], ymin=[0, 0],
                        ymax=y_std2, color='tab:red', linewidth=2)
    leg_labels = ['Normal distribution fit',
                  '1-Sigma: {:.2f} mm'.format(std),
                  '2-Sigma: {:.2f} mm'.format(2 * std)]
    ax.legend([ln[0], ln_std, ln_std2], leg_labels)
    ax.set_xlabel('PWV anomalies [mm]')
    return ax

    return ax


# def plot_quiver_panels(u, v, tcwv,
#                        times=['2015-10', '2013-10'], level=750):
#     import matplotlib.pyplot as plt
#     from matplotlib.colors import Normalize
#     from mpl_toolkits.axes_grid1 import AxesGrid
#     import matplotlib.cm as cm
#     import pandas as pd
#     from palettable.colorbrewer import sequential as seq_cmap
#     from palettable.colorbrewer import diverging as div_cmap
#     from aux_gps import anomalize_xr
#     cmap_yl = seq_cmap.YlOrRd_9.mpl_colormap
#     cmap_rb = div_cmap.PuOr_11.mpl_colormap
#     cmap = cmap_rb
#     times = pd.to_datetime(times)
#     tcwv = slice_time_level_geo_field(tcwv, level=None, time=times,
#                                       anoms=True,
#                                       lats=[17, 47], lons=[17, 47])
#     qu = slice_time_level_geo_field(u, level=750, time=times,
#                                     anoms=True,
#                                     lats=[17, 47], lons=[17, 47])
#     qv = slice_time_level_geo_field(v, level=750, time=times,
#                                     anoms=True,
#                                     lats=[17, 47], lons=[17, 47])
#     fig = plt.figure(figsize=(15, 5))
#     # fig, axes = plt.subplots(1, 2, figsize=(15, 5))
#     grid = AxesGrid(fig, 111,          # as in plt.subplot(111)
#                     nrows_ncols=(1, 2),
#                     axes_pad=0.15,
#                     share_all=True,
#                     cbar_location="right",
#                     cbar_mode="single",
#                     cbar_size="7%",
#                     cbar_pad=0.15,
#                     )
#     # normalizer=Normalize(-6,6)
#     vmax= abs(max(abs(tcwv.min().values), abs(tcwv.max().values)))
#     vmin = -vmax
#     print(vmin, vmax)
#     # vmax = tcwv.max().item()
#     cs1 = plot_contourf_field_with_map_overlay(tcwv.sel(time=times[0]), ax=grid[0],
#                                                vmin=vmin, vmax=vmax, cmap=cmap,
#                                                colorbar=False, title='2015-10',
#                                                cbar_label='', extend=None,
#                                                alpha=0.5, levels=21)
#     cs2 = plot_contourf_field_with_map_overlay(tcwv.sel(time=times[1]), ax=grid[1],
#                                                vmin=vmin, vmax=vmax, cmap=cmap,
#                                                colorbar=False, title='2013-10',
#                                                cbar_label='', extend=None,
#                                                alpha=0.5, levels=21)
#     cbar = grid[0].cax.colorbar(cs2)
#     # cbar = grid.cbar_axes[0].colorbar(cs2)
#     label = 'PWV anomalies [mm]'
#     cbar.set_label_text(label)
#     # for cax in grid.cbar_axes:
#     #     cax.toggle_label(False)
#     # im=cm.ScalarMappable(norm=normalizer)
#     return fig

# TODO:  calculate long term monthly mean from slice and incorporate it easily:

def plot_quiver_panels(u, v, sf,
                       times=['2013-10', '2015-10'], level=750,
                       anoms=False, suptitle='', labelsize=12):
    import matplotlib.pyplot as plt
    import pandas as pd
    # from palettable.colorbrewer import sequential as seq_cmap
    from palettable.colorbrewer import sequential as colorbrewer_seq
    from palettable.scientific import sequential as scientific_seq
    from palettable.cmocean import sequential as cmocean_seq
    from palettable.cartocolors import sequential as seq_cmap
    from palettable.cartocolors import diverging as div_cmap
    import cartopy.crs as ccrs
    import xarray as xr
    cmap_seq = seq_cmap.BluYl_7.mpl_colormap
    cmap_seq = colorbrewer_seq.Blues_9.mpl_colormap
    cmap_div = div_cmap.Tropic_7.mpl_colormap
    cmap_quiver = seq_cmap.SunsetDark_7.mpl_colormap
    # cmap_quiver = colorbrewer_seq.YlOrRd_9.mpl_colormap
    # cmap_quiver = scientific_seq.LaJolla_20.mpl_colormap
    # cmap_quiver = cmocean_seq.Solar_20.mpl_colormap
    cmap = cmap_seq
    if anoms:
        cmap = cmap_div
    times_dt = pd.to_datetime(times)
    cb_label = 'PWV [mm]'
    tcwv = slice_time_level_geo_field(sf, level=None, time=times_dt,
                                      anoms=anoms, clim_month=10,
                                      lats=[17, 47], lons=[17, 47])
    qu = slice_time_level_geo_field(u, level=750, time=times_dt,
                                    anoms=anoms, clim_month=10,
                                    lats=[17, 47], lons=[17, 47])
    qv = slice_time_level_geo_field(v, level=750, time=times_dt,
                                    anoms=anoms, clim_month=10,
                                    lats=[17, 47], lons=[17, 47])
    fg = plot_scaler_field_ontop_map_cartopy(tcwv, col='time', levels=21,
                                             cmap=cmap, alpha=0.8, cbar_label=cb_label,
                                             labelsize=labelsize, figsize=(18, 6))
    fg = plot_vector_arrows_ontop_map_cartopy(qu, qv, lon_dim='longitude',
                                              lat_dim='latitude', fg=fg,
                                              qp=5, col='time', qkey=True,
                                              cmap=cmap_quiver, zorder=20)
    gdf = box_lat_lon_polygon_as_gpd(lat_bounds=[29, 34], lon_bounds=[34, 36])
    for i, ax in enumerate(fg.axes.flat):
        # add the box over Israel:
        ax.add_geometries(gdf['geometry'].values, crs=ccrs.PlateCarree(),
                          edgecolor='k', linestyle='--', alpha=1, linewidth=2)
        # add gridlines:
        gl = ax.gridlines(alpha=0.5, color='k', linestyle='--', draw_labels=True,
                          dms=True, x_inline=False, y_inline=False, linewidth=1)
        gl.top_labels = False
        # gl.left_labels = False
        gl.xlabel_style = {'size': labelsize, 'color': 'k'}
        gl.ylabel_style = {'size': labelsize, 'color': 'k'}
        if i == 0:
            gl.right_labels = False
        elif i == 1:
            gl.right_labels = False
            gl.left_labels = False
        elif i == 2:
            gl.right_labels = False
            gl.left_labels = False
        if i <= 1:
            ax.set_title(times_dt[i].strftime('%b %Y'))
        else:
            ax.set_title('Mean Oct')
    fg.fig.suptitle(suptitle)
    fg.fig.subplots_adjust(top=0.899,
                           bottom=0.111,
                           left=0.03,
                           right=0.94,
                           hspace=0.17,
                           wspace=0.0)

    return fg


def slice_time_level_geo_field(field, level=750, lat_dim='latitude',
                               lon_dim='longitude', time='2012-10',
                               level_dim='level', time_dim='time',
                               lats=[None, None], lons=[None, None],
                               anoms=False, clim_month=None):
    from aux_gps import anomalize_xr
    import pandas as pd
    import xarray as xr

    if level is not None:
        field = field.sel({level_dim: level})
    if field[lat_dim].diff(lat_dim).median() < 0:
        lats = lats[::-1]
    field = field.sel({lat_dim: slice(*lats), lon_dim: slice(*lons)}).load()
    if time is not None and anoms and clim_month is None:
        field = field.load()
        field = anomalize_xr(field, freq='MS', time_dim=time_dim)
    if time is not None and clim_month is None:
        field = field.sel({time_dim: time})
    elif time is None and clim_month is not None:
        field = field.load()
        field = field.groupby('{}.month'.format(
            time_dim)).mean().sel(month=clim_month)
    elif time is not None and clim_month is not None:
        clim = field.groupby('{}.month'.format(time_dim)
                             ).mean().sel(month=clim_month)
        clim = clim.rename({'month': time_dim})
        clim[time_dim] = pd.to_datetime(
            '2200-{}'.format(clim_month), format='%Y-%m')
        field = field.sel({time_dim: time})
        field = xr.concat([field, clim], time_dim)
    field = field.sortby(lat_dim).squeeze()
    return field


# def plot_contourf_field_with_map_overlay(field, lat_dim='latitude',
#                                          lon_dim='longitude', ax=None,
#                                          vmin=None, vmax=None, cmap='viridis',
#                                          colorbar=False, title=None,
#                                          cbar_label='', extend=None,
#                                          alpha=0.5, levels=11):
#     import salem
#     import matplotlib.pyplot as plt
#     field = field.transpose(lon_dim, lat_dim, ...)
#     if ax is None:
#         f, ax = plt.subplots(figsize=(10, 8))
#     # plot the salem map background, make countries in grey
#     smap = field.salem.get_map(countries=False)
#     smap.set_shapefile(countries=False, oceans=True, lakes=True, color='grey')
#     smap.plot(ax=ax)
#     # transform the coordinates to the map reference system and contour the data
#     xx, yy = smap.grid.transform(field[lat_dim].values, field[lon_dim].values,
#                                  crs=field.salem.grid.proj)

#     cs = ax.contourf(xx, yy, field, cmap=cmap, levels=levels,
#                      alpha=alpha, vmin=vmin, vmax=vmax, extend=extend)
#     if colorbar:
#         f.colorbar(cs, ax=ax, aspect=40, label=cbar_label)
#     if title is not None:
#         ax.set_title(title)
#     return cs


# def plot_quiver_ontop_map(u, v, ax=None, lat_dim='latitude',
#                           lon_dim='longitude', plot_map=False,
#                           qp=5, qkey=True):
#     import salem
#     import matplotlib.pyplot as plt
#     import numpy as np
#     u = u.transpose(lon_dim, lat_dim, ...)
#     v = v.transpose(lon_dim, lat_dim, ...)
#     if ax is None:
#         f, ax = plt.subplots(figsize=(10, 8))
#     # plot the salem map background, make countries in grey
#     smap = u.salem.get_map(countries=False)
#     smap.set_shapefile(countries=False, oceans=True, lakes=True, color='grey')
#     if plot_map:
#         smap.plot(ax=ax)
#     # transform the coordinates to the map reference system and contour the data
#     xx, yy = smap.grid.transform(u[lat_dim].values, u[lon_dim].values,
#                                  crs=u.salem.grid.proj)
#     # Quiver only every 7th grid point
#     u = u[4::qp, 4::qp]
#     v = v[4::qp, 4::qp]
#     # transform their coordinates to the map reference system and plot the arrows
#     xx, yy = smap.grid.transform(u[lat_dim].values, u[lon_dim].values,
#                                  crs=u.salem.grid.proj)
#     xx, yy = np.meshgrid(xx, yy)
#     qu = ax.quiver(xx, yy, u.values, v.values)
#     if qkey:
#         qk = ax.quiverkey(qu, 0.7, 1.05, 2, '2 msec',
#                           labelpos='E', coordinates='axes')
#     return ax


def plot_scaler_field_ontop_map_cartopy(field, col='time', levels=21,
                                        cmap='bwr', alpha=1,
                                        labelsize=14, figsize=(15, 6),
                                        cbar_label=''):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    fg = field.plot.contourf(levels=levels, col=col, transform=ccrs.PlateCarree(),
                             cmap=cmap, alpha=alpha, figsize=figsize, add_colorbar=False,
                             subplot_kws={"projection": ccrs.PlateCarree()})
    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
    cbar_ax = fg.fig.add_axes([0.94, 0.1, 0.01, 0.8])
    fg.add_colorbar(cax=cbar_ax, label=cbar_label)
    for ax in fg.axes.flat:
        # land_50m = cfeature.NaturalEarthFeature('physical', 'lakes', '10m',
        #                                 edgecolor='face',
        #                                 facecolor='b', alpha=0.3)
        # ax.add_feature(land_50m, zorder=30)
        # ax.add_feature(cfeature.LAKES.with_scale('110m'), facecolor='b')
        # ax.add_image(tiler, 6)
        ax.coastlines('50m')
        # ax.background_img(extent=[17, 47, 17, 47])
        ax.tick_params(axis="y", direction="out", length=8)
    return fg


def plot_vector_arrows_ontop_map_cartopy(u, v, lon_dim='longitude',
                                         lat_dim='latitude', fg=None,
                                         qp=5, col='time', qkey=True,
                                         cmap=None, zorder=None):
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import cartopy.feature as cfeature
    import numpy as np
    scale = np.sqrt(u**2+v**2).max().item()
    import numpy as np
    if fg is None:
        fg = plt.figure(figsize=(8, 10))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale('110m'))
        # ax.add_image(tiler, 6)
        ax.coastlines('50m')
        gl = ax.gridlines(alpha=0.5, color='k', linestyle='--', draw_labels=True,
                          dms=True, x_inline=False, y_inline=False, linewidth=1)
        # Quiver only every 7th grid point
        u = u[4::qp, 4::qp]
        v = v[4::qp, 4::qp]
        x = u[lon_dim].values
        y = u[lat_dim].values
        # set displayed arrow length for longest arrow
        displayed_arrow_length = 2
        scale_factor = scale / displayed_arrow_length

        ax.quiver(x, y, u, v, units='xy',
                  width=0.1, zorder=zorder,
                  scale=scale_factor, scale_units='xy',
                  transform=ccrs.PlateCarree())
        return fg
    for i, ax in enumerate(fg.axes.flat):
        # set displayed arrow length for longest arrow
        displayed_arrow_length = 2
        scale_factor = scale / displayed_arrow_length
        u1 = u.isel({col: i})
        v1 = v.isel({col: i})
        # colors1 = colors.isel({col: i})
        # Quiver only every 7th grid point
        u1 = u1[4::qp, 4::qp]
        v1 = v1[4::qp, 4::qp]
        colors = np.sqrt(u1**2 + v1**2) / scale
        x = u1[lon_dim].values
        y = u1[lat_dim].values
        if cmap is not None:
            q = ax.quiver(x, y, u1, v1, colors, units='xy',
                          width=0.1, cmap=cmap,
                          scale=scale_factor, scale_units='xy',
                          transform=ccrs.PlateCarree(),
                          zorder=zorder)
        else:
            q = ax.quiver(x, y, u1, v1, units='xy',
                          width=0.1, zorder=zorder,
                          scale=scale_factor, scale_units='xy',
                          transform=ccrs.PlateCarree())
    if qkey:
        qk = ax.quiverkey(q, 0.7, 1.05, 0.03, r'0.03 m$\cdot$sec$^{-1}$',
                          labelpos='E', coordinates='axes')

    return fg


def box_lat_lon_polygon_as_gpd(lat_bounds=[29, 34], lon_bounds=[34, 36.5]):
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    point1 = [lon_bounds[0], lat_bounds[0]]
    point2 = [lon_bounds[0], lat_bounds[1]]
    point3 = [lon_bounds[1], lat_bounds[1]]
    point4 = [lon_bounds[1], lat_bounds[0]]
    line1 = LineString([Point(*point1), Point(*point2)])
    line2 = LineString([Point(*point2), Point(*point3)])
    line3 = LineString([Point(*point3), Point(*point4)])
    line4 = LineString([Point(*point4), Point(*point1)])
    geo_df = gpd.GeoDataFrame(geometry=[line1, line2, line3, line4])
    return geo_df


def plot_relative_wind_direction_frequency(station='tela', ims_path=ims_path,
                                           clim=True):
    import xarray as xr
    import pandas as pd
    wd_daily = xr.load_dataset(ims_path / 'GNSS_WD_daily.nc')[station]
    bins = [0, 45, 90, 135, 180, 215, 270, 315, 360]
    bin_labels = ['N-NE', 'NE-E', 'E-SE',
                  'SE-S', 'S-SW', 'SW-W', 'W-NW', 'NW-N']
    wd_daily = wd_daily.dropna('time')
    cats = pd.cut(wd_daily.values, bins=bins, labels=bin_labels)
    df = wd_daily.dropna('time').to_dataframe(name='WD')
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['months'] = df['year'].astype(str) + '-' + df['month'].astype(str)
    cats = pd.Series(cats, index=df.index)
    df['direction'] = cats
    ndf = df.groupby([df['months'], df['direction']]).size().to_frame()
    ndf = ndf.unstack()
    ndf.columns = ndf.columns.droplevel()
    ndf.index.name = 'time'
    ndf.index = pd.to_datetime(ndf.index)
    da = ndf.to_xarray()
    return da


def plot_multiparams_daily_pwv_single_time(station='tela', ims_path=ims_path,
                                           climate_path=climate_path,
                                           ts1='2013-09-15', days=47,
                                           ts2='2015-09-15',
                                           pwv_lim=[10, 45], dtr_lim=[6, 14.5],
                                           wd_lim=[50, 320],
                                           add_synoptics=['CL', 'RST', 'PT'],
                                           save=True, fontsize=16):
    import matplotlib.pyplot as plt
    import pandas as pd
    import xarray as xr
    import numpy as np
    from calendar import month_abbr
    from aux_gps import replace_time_series_with_its_group
    from synoptic_procedures import read_synoptic_classification
    sns.set_style('whitegrid')
    sns.set_style('ticks')
    dt1 = pd.date_range(ts1, periods=days)
    # dt2 = pd.date_range(ts2, periods=days)
    months = list(set(dt1.month))
    year = list(set(dt1.year))[0]  # just one year
    dt1_str = ', '.join([month_abbr[x] for x in months]) + ' {}'.format(year)
    # months = list(set(dt2.month))
    # year = list(set(dt2.year))[0]  # just one year
    # dt2_str = ', '.join([month_abbr[x] for x in months]) + ' {}'.format(year)
    pw_daily_all = xr.open_dataset(
            work_yuval/'GNSS_PW_daily_thresh_50.nc')[station].load()
    # pw_daily2 = pw_daily_all.sel(time=dt2)
    pw_daily = pw_daily_all.sel(time=dt1)
    dtr_daily_all = xr.load_dataset(ims_path /'GNSS_IMS_DTR_mm_israel_1996-2020.nc')[station]
    dtr_daily = dtr_daily_all.sel(time=dt1)
    # dtr_daily2 = dtr_daily_all.sel(time=dt2)
    wd_daily_all = xr.load_dataset(ims_path /'GNSS_WD_daily.nc')[station]
    wd_daily = wd_daily_all.sel(time=dt1)
    # wd_daily2 = wd_daily_all.sel(time=dt2)
    # wind directions:
    # 0 north
    # 45 northeast
    # 90 east
    # 135 southeast
    # 180 south
    # 225 southwest
    # 270 west
    # 315 northwest
    # 360 north

    fig, axes = plt.subplots(3, 1, figsize=(20, 10))
    # twins = [ax.twiny() for ax in axes]
    pwv_mm = replace_time_series_with_its_group(pw_daily, 'month')
    # pwv_mm2 = replace_time_series_with_its_group(pw_daily2, 'month')
    blue = 'k'
    red = 'tab:red'
    pwv1 = dt1_str + ' PWV'
    # pwv2 = dt2_str + ' PWV'
    pwv1_mm = pwv1 + ' monthly mean'
    # pwv2_mm = pwv2 + ' monthly mean'
    pw_daily.plot.line('-', color=blue, lw=2, ax=axes[0], label=pwv1)
    # pw_daily2.plot.line('-', lw=2, color=red, ax=twins[0], label=pwv2)
    pwv_mm.plot.line('--', lw=2, color=blue, ax=axes[0], label=pwv1_mm)
    # pwv_mm2.plot.line('--', lw=2, color=red, ax=twins[0], label=pwv2_mm)
    axes[0].set_ylabel('PWV [mm]', fontsize=fontsize)
    hand, labl = axes[0].get_legend_handles_labels()
    # hand2, labl2 = twins[0].get_legend_handles_labels()
    # axes[0].legend(handles=hand+hand2, labels=labl+labl2)
    axes[0].set_ylim(*pwv_lim)
    wd_daily.plot.line('-', lw=2, color=blue, ax=axes[1])
    # wd_daily2.plot.line('-', lw=2,color=red, ax=twins[1])
    axes[1].set_ylabel(r'Wind Direction [$^{\circ}$]', fontsize=fontsize)
    axes[1].set_ylabel('Wind Direction', fontsize=fontsize)
    #  axes[1].set_ylim(*wd_lim)
    dtr_daily.plot.line('-', lw=2, color=blue, ax=axes[2])
    # dtr_daily2.plot.line('-', lw=2, color=red, ax=twins[2])
    axes[2].set_ylabel('Diurnal Temperature Range [K]', fontsize=fontsize)
    axes[2].set_ylim(*dtr_lim)
    [ax.xaxis.set_major_locator(mdates.DayLocator(interval=1)) for ax in axes]
    # set formatter
    [ax.xaxis.set_major_formatter(mdates.DateFormatter('%d')) for ax in axes]
    [ax.grid(True) for ax in axes]
    [ax.set_xlabel('') for ax in axes]
    [ax.tick_params(labelsize=fontsize) for ax in axes]
    xlim = [dt1[0]- pd.Timedelta(1, unit='d'), dt1[-1]+ pd.Timedelta(1, unit='d')]
    [ax.set_xlim(*xlim) for ax in axes]
    [ax.set_xticks(ax.get_xticks()[1:-1]) for ax in axes]
    # for ax, twin in zip(axes, twins):
    #     ylims_low = min(min(ax.get_ylim()), min(twin.get_ylim()))
    #     ylims_high = max(max(ax.get_ylim()), max(twin.get_ylim()))
    #     ax.set_ylim(ylims_low, ylims_high)
    wd_ticks = np.arange(45, 360, 45)
    wind_labels = ['NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    lbl = []
    for tick, label  in zip(wd_ticks, wind_labels):
        if len(label) == 1:
            lbl.append(label + '   ' + str(tick))
        elif len(label) == 2:
            lbl.append(label + ' ' + str(tick))
    # wind_label = [y + ' ' + str(x) for x,y in zip(wd_ticks, wind_labels)]
    axes[1].set_yticks(wd_ticks)
    axes[1].set_yticklabels(wind_labels, ha='left')
    fig.canvas.draw()
    yax = axes[1].get_yaxis()
    # find the maximum width of the label on the major ticks
    pad = max(T.label.get_window_extent().width for T in yax.majorTicks)
    yax.set_tick_params(pad=pad-10)
    if add_synoptics is not None:
        df = read_synoptic_classification(climate_path, report=False)
        ind = pw_daily.to_dataframe().index
        df = df.loc[ind]
        grp_dict = df.groupby('upper_class').groups
        [grp_dict.pop(x) for x in grp_dict.copy().keys()
         if x not in add_synoptics]
        # add_ARSTs:
        grp_dict['ARST'] = pd.DatetimeIndex(['2013-10-30', '2015-10-05',
                                             '2015-10-19', '2015-10-20',
                                             '2015-10-25', '2015-10-29'])
        grp_dict['RST'] = grp_dict['RST'].difference(grp_dict['ARST'])
        color_dict = {'CL': 'tab:green', 'ARST': 'tab:orange',
                      'RST': 'tab:orange', 'PT': 'tab:purple'}
        alpha_dict = {'CL': 0.3, 'ARST': 0.6,
                      'RST': 0.3, 'PT': 0.3}
        ylim0 = axes[0].get_ylim()
        ylim1 = axes[1].get_ylim()
        ylim2 = axes[2].get_ylim()
        for key_class, key_ind in grp_dict.items():
            color = color_dict[key_class]
            alpha = alpha_dict[key_class]
            # ecolor='k'
    #         edge_color = edge_dict[key_class]
    #         abbr = add_class_abbr(key_class)
    #         # abbr_count = month_counts.sel(syn_cls=key_class).sum().item()
    #         abbr_count = df[df['class'] == key_class].count().values[0]
    #         abbr_label = r'${{{}}}$: {}'.format(abbr, int(abbr_count))
    # #    for ind, row in df.iterrows():
    #         da_ts[da_ts['syn_class'] == key_class].plot.line(
    #             'k-', lw=0, ax=ax, marker='o', markersize=20,
    #             markerfacecolor=color, markeredgewidth=2,
    #             markeredgecolor=edge_color, label=abbr_label)

            axes[0].vlines(key_ind, ylim0[0], ylim0[1],
                           color=color, alpha=alpha, lw=20,
                           label=key_class)
            axes[1].vlines(key_ind, ylim1[0], ylim1[1],
                           color=color, alpha=alpha, lw=20,
                           label=key_class)
            axes[2].vlines(key_ind, ylim2[0], ylim2[1],
                           color=color, alpha=alpha, lw=20,
                           label=key_class)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, prop={'size': 16}, edgecolor='k',
               framealpha=0.5, fancybox=False, facecolor='white',
               ncol=4, fontsize=fontsize, loc='upper left', bbox_to_anchor=(0.05, 1.005),
               bbox_transform=plt.gcf().transFigure)

    # [twin.tick_params(axis='x',which='both', top=False,         # ticks along the top edge are off
    #                   labeltop=False) for twin in twins]
    # [twin.set_xlabel('') for twin in twins]
    # months = list(set(times_dt.month))
    # year = list(set(times_dt.year))[0]  # just one year
    dt_str = ', '.join([month_abbr[x] for x in months]) + ' {}'.format(year)
    # axes[2].set_xlabel(dt_str)
    fig.suptitle('{} {}'.format(station.upper(),dt_str), fontsize=fontsize)
    fig.tight_layout()
    if save:
        filename = '{}_multiparam_{}-{}.png'.format(station, '-'.join([str(x) for x in months]), year)
#        plt.savefig(savefig_path / filename, bbox_inches='tight')
        plt.savefig(savefig_path / filename, orientation='landscape')
    return fig


def plot_synoptic_daily_on_pwv_daily_with_colors(climate_path=climate_path,
                                                 station='tela',ims_path=ims_path,
                                                 times=['2013-09-15',
                                                        '2015-09-15'],
                                                 days=47, add_era5=True,
                                                 add_dtr=True,
                                                 twin_ylims=None):
    from synoptic_procedures import visualize_synoptic_class_on_time_series
    import matplotlib.pyplot as plt
    import xarray as xr
    import pandas as pd
    import matplotlib.dates as mdates
    from calendar import month_abbr
    # TODO: add option of plotting 3 stations and/without ERA5
    times_dt = [pd.date_range(x, periods=days) for x in times]
    if isinstance(station, list):
        pw_daily = [xr.open_dataset(
            work_yuval/'GNSS_PW_daily_thresh_50_homogenized.nc')[x].load() for x in station]
        pw_daily = xr.merge(pw_daily)
        add_mm = False
        label = ', '.join([x.upper() for x in station])
        ncol = 6
    else:
        pw_daily = xr.open_dataset(
            work_yuval/'GNSS_PW_daily_thresh_50.nc')[station].load()
        add_mm = True
        label = station.upper()
        ncol = 4
    era5_hourly = xr.open_dataset(work_yuval/'GNSS_era5_hourly_PW.nc')[station]
    era5_daily = era5_hourly.resample(time='D').mean().load()
    dtr_daily = xr.load_dataset(work_yuval/'GNSS_ERA5_DTR_daily_1996-2020.nc')[station]
    dtr_daily = xr.load_dataset(ims_path /'GNSS_IMS_DTR_mm_israel_1996-2020.nc')[station]
    fig, axes = plt.subplots(len(times), 1, figsize=(20, 10))
    leg_locs = ['upper right', 'lower right']
    for i, ax in enumerate(axes.flat):
        if add_era5:
            second_da_ts = era5_daily.sel(time=times_dt[i])
        elif add_dtr:
            second_da_ts = dtr_daily.sel(time=times_dt[i])
        else:
            second_da_ts = None
        visualize_synoptic_class_on_time_series(pw_daily.sel(time=times_dt[i]),
                                                path=climate_path, ax=ax,
                                                second_da_ts=second_da_ts,
                                                leg_ncol=ncol,
                                                leg_loc=leg_locs[i],
                                                add_mm=add_mm,
                                                twin=twin_ylims)
        ax.set_ylabel('PWV [mm]')
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        # set formatter
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        # set font and rotation for date tick labels
        months = list(set(times_dt[i].month))
        year = list(set(times_dt[i].year))[0]  # just one year
        dt_str = ', '.join([month_abbr[x] for x in months]) + ' {}'.format(year)
        ax.set_title(dt_str, fontweight='bold', fontsize=14)
        ax.set_xlabel('')
    # set ylims :
    ylims_low = [ax.get_ylim()[0] for ax in axes]
    ylims_high = [ax.get_ylim()[1] for ax in axes]
    [ax.set_ylim(min(ylims_low), max(ylims_high)) for ax in axes]
    # set ylims in right_axes:
    # ylims_low = [ax.right_ax.get_ylim()[0] for ax in axes]
    # ylims_high = [ax.right_ax.get_ylim()[1] for ax in axes]
    # [ax.right_ax.set_ylim(min(ylims_low), max(ylims_high)) for ax in axes]
    # axes[0].right_ax.set_ylim(0,100)
    if add_era5:
        fig.suptitle(
            'Daily PWV and synoptic class for {} station using GNSS(solid - monthly means in dot-dashed) and ERA5(dashed)'.format(label))
    elif add_dtr:
        fig.suptitle(
            'Daily PWV and synoptic class for {} station using GNSS(solid - monthly means in dot-dashed) and DTR(dashed)'.format(label))
    else:
        fig.suptitle(
            'Daily PWV and synoptic class for {} station using GNSS(solid)'.format(label))
    fig.tight_layout()
    return axes


def create_enhanced_qualitative_color_map(plot=True, alevels=[1, 0.75, 0.5, 0.25]):
    import matplotlib.colors as cm
    import seaborn as sns
    colors = sns.color_palette('colorblind')
    colors_with_alpha = [cm.to_rgba(colors[x]) for x in range(len(colors))]
    new = []
    for color in colors_with_alpha:
        r = color[0]
        g = color[1]
        b = color[2]
        for alev in alevels:
            alpha = alev
            new.append(tuple([r, g, b, alpha]))
    if plot:
        sns.palplot(new)
    return new


def plot_IMS_wind_speed_direction_violins(ims_path=ims_path,
                                          station='tela', save=True,
                                          fontsize=16):
    from ims_procedures import gnss_ims_dict
    import seaborn as sns
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    sns.set_style('whitegrid')
    sns.set_style('ticks')
    pal = sns.color_palette(n_colors=4)
    green = pal[2]
    red = pal[3]
    ims_station = gnss_ims_dict.get(station)
    WS = xr.open_dataset(ims_path / 'IMS_WS_israeli_10mins.nc')[ims_station]
    WD = xr.open_dataset(ims_path / 'IMS_WD_israeli_10mins.nc')[ims_station]
    ws_mm = WS.resample(time='MS').mean().sel(time=slice('2014', '2019'))
    wd_mm = WD.resample(time='MS').mean().sel(time=slice('2014', '2019'))
    df = ws_mm.to_dataframe(name='Wind Speed')
    df['Wind Direction'] = wd_mm.to_dataframe(name='Wind Direction')
    df['month'] = df.index.month
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    axes[0] = sns.violinplot(data=df, x='month', y='Wind Speed',
                             fliersize=10, gridsize=250, ax=axes[0],
                             inner=None, scale='width', color=green,
                             hue=None, split=False, zorder=20)
    axes[1] = sns.violinplot(data=df, x='month', y='Wind Direction',
                             fliersize=10, gridsize=250, ax=axes[1],
                             inner=None, scale='width', color=red,
                             hue=None, split=False, zorder=20)
    [x.set_alpha(0.5) for x in axes[0].collections]
    [x.set_alpha(0.5) for x in axes[1].collections]
    axes[0] = sns.pointplot(x='month', y='Wind Speed', data=df,
                            estimator=np.mean,
                            dodge=False, ax=axes[0], hue=None, color=green,
                            linestyles="None", markers=['s'], scale=0.7,
                            ci=None, alpha=0.5, zorder=0, style=None)
    axes[1] = sns.pointplot(x='month', y='Wind Direction', data=df,
                            estimator=np.mean,
                            dodge=False, ax=axes[1], hue=None, color=red,
                            linestyles="None", markers=['o'], scale=0.7,
                            ci=None, alpha=0.5, zorder=0, style=None)
    [ax.grid(True) for ax in axes]
    wind_labels = ['SE', 'S', 'SW', 'W', 'NW']
    wd_ticks = np.arange(135, 360, 45)
    axes[1].set_yticks(wd_ticks)
    axes[1].set_yticklabels(wind_labels, ha='left')
    fig.canvas.draw()
    yax = axes[1].get_yaxis()
    # find the maximum width of the label on the major ticks
    pad = max(T.label.get_window_extent().width for T in yax.majorTicks)
    yax.set_tick_params(pad=pad-10)
    axes[0].set_ylabel(r'Wind Speed [m$\cdot$sec$^{-1}$]')
    fig.tight_layout()
    return


def plot_ERA5_wind_speed_direction_profiles_at_bet_dagan(ear5_path=era5_path,
                                                         save=True, fontsize=16):
    import seaborn as sns
    import xarray as xr
    from aux_gps import convert_wind_direction
    import numpy as np
    import matplotlib.pyplot as plt
    sns.set_style('whitegrid')
    sns.set_style('ticks')
    pal = sns.color_palette(n_colors=4)
    bd_lat = 32.01
    bd_lon = 34.81
    v = xr.open_dataset(era5_path/'ERA5_V_mm_EM_area_1979-2020.nc')
    u = xr.open_dataset(era5_path/'ERA5_U_mm_EM_area_1979-2020.nc')
    u = u.sel(expver=1)
    v = v.sel(expver=1)
    u1 = u.sel(latitude=bd_lat, longitude=bd_lon, method='nearest')
    v1 = v.sel(latitude=bd_lat, longitude=bd_lon, method='nearest')
    u1.load().dropna('time')
    v1.load().dropna('time')
    ws1, wd1 = convert_wind_direction(u=u1['u'], v=v1['v'])
    ws1 = ws1.reset_coords(drop=True)
    wd1 = wd1.reset_coords(drop=True)
    levels = [1000, 900, 800, 700]
    df_ws = ws1.sel(level=levels).to_dataframe('ws')
    df_ws['level'] = df_ws.index.get_level_values(1)
    df_ws['month'] = df_ws.index.get_level_values(0).month
    df_wd = wd1.sel(level=levels).to_dataframe('wd')
    df_wd['level'] = df_wd.index.get_level_values(1)
    df_wd['month'] = df_wd.index.get_level_values(0).month
    fig, axes = plt.subplots(2, 1, figsize=(8, 15))
    axes[0] = sns.lineplot(data=df_ws, x='month', y='ws',
                           hue='level', markers=True,
                           style='level', markersize=10,
                           ax=axes[0], palette=pal)
    axes[1] = sns.lineplot(data=df_wd, x='month', y='wd',
                           hue='level', markers=True,
                           style='level', markersize=10,
                           ax=axes[1], palette=pal)
    axes[0].legend(title='pressure level [hPa]', prop={'size': fontsize-2}, loc='upper center')
    axes[1].legend(title='pressure level [hPa]', prop={'size': fontsize-2}, loc='lower center')
    [ax.grid(True) for ax in axes]
    wind_labels = ['SE', 'S', 'SW', 'W', 'NW']
    wd_ticks = np.arange(135, 360, 45)
    axes[1].set_yticks(wd_ticks)
    axes[1].set_yticklabels(wind_labels, ha='left')
    fig.canvas.draw()
    yax = axes[1].get_yaxis()
    # find the maximum width of the label on the major ticks
    pad = max(T.label.get_window_extent().width for T in yax.majorTicks)
    yax.set_tick_params(pad=pad)
    axes[0].set_ylabel(r'Wind Speed [m$\cdot$sec$^{-1}$]', fontsize=fontsize)
    axes[1].set_ylabel('Wind Direction', fontsize=fontsize)
    axes[1].set_xlabel('month', fontsize=fontsize)
    mticks = np.arange(1, 13)
    [ax.set_xticks(mticks) for ax in axes]
    [ax.tick_params(labelsize=fontsize) for ax in axes]
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.051)
    if save:
        filename = 'ERA5_wind_speed_dir_bet-dagan_profiles.png'
        plt.savefig(savefig_path / filename, orientation='potrait')
    return fig


def plot_PWV_anomalies_groups_maps(work_path=work_yuval, station='drag',
                                   fontsize=16, save=True):
    import xarray as xr
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import gaussian_filter
    from PW_stations import produce_geo_gnss_solved_stations
    sns.set_style('whitegrid')
    sns.set_style('ticks')
    cmap = 'jet' # sns.color_palette('terrain', as_cmap=True)
    df = produce_geo_gnss_solved_stations(plot=False)
    file = work_path/'GNSS_PW_thresh_0_hour_dayofyear_rest.nc'
    pw = xr.open_dataset(file)
    if isinstance(station, str):
        st_mean = pw[station].mean('rest').expand_dims('station')
        st_mean['station'] = [station.upper()]
        data = gaussian_filter(st_mean, 5)
        st_mean = st_mean.copy(data=data)
    elif isinstance(station, list):
        pws = [pw[x].mean('rest') for x in pw if x in station]
        pws = [x.copy(data=gaussian_filter(x, 5)) for x in pws]
        st_mean = xr.merge(pws)
        st_mean = st_mean[station].to_array('station')
        st_mean['station'] = [x.upper() for x in st_mean['station'].values]
        alts = df.loc[station,'alt'].values
    # drag = pw['drag'].mean('rest')
    # elat = pw['elat'].mean('rest')
    # dsea = pw['dsea'].mean('rest')
    # da = xr.concat([drag, dsea, elat], 'station')
    # da['station'] = ['DRAG', 'DSEA', 'ELAT']
    n = st_mean['station'].size
    fg = st_mean.plot.contourf(levels=41, row='station', add_colorbar=False,
                               figsize=(6.5, 13), cmap=cmap)
    for i, ax in enumerate(fg.fig.axes):
        ax.set_xticks(np.arange(50, 400, 50))
        ax.tick_params(labelsize=fontsize)
        ax.set_ylabel('Hour of day [UTC]', fontsize=fontsize)
        title = ax.get_title()
        title = title + ' ({:.0f} m a.s.l)'.format(alts[i])
        ax.set_title(title, fontsize=fontsize)
    fg.fig.axes[-1].set_xlabel('Day of Year', fontsize=fontsize)
    cbar_ax = fg.fig.add_axes([0.87, 0.05, 0.025, 0.917])
    fg.add_colorbar(cax=cbar_ax)
    cb = fg.cbar
    cb.ax.tick_params(labelsize=fontsize-2)
    cb.set_label('PWV [mm]', size=fontsize-2)
    fg.fig.subplots_adjust(top=0.967,
                           bottom=0.05,
                           left=0.125,
                           right=0.85,
                           hspace=0.105,
                           wspace=0.2)
    if save:
        filename = 'PWV_climatology_{}_stacked_groups.png'.format('_'.join(station))
        plt.savefig(savefig_path / filename, orientation='potrait')
    return fg


def plot_hydro_pwv_before_event_motivation(work_path=work_yuval,
                                           hydro_path=hydro_path,
                                           days_prior=3, fontsize=16,
                                           save=True, smoothed=False):
    import xarray as xr
    from hydro_procedures import hydro_pw_dict
    from hydro_procedures import produce_pwv_days_before_tide_events
    from hydro_procedures import read_station_from_tide_database
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    def smooth_df(df):
        import numpy as np
        dfs = df.copy()
        dfs.index = pd.to_timedelta(dfs.index, unit='d')
        dfs = dfs.resample('15S').interpolate(method='cubic')
        dfs = dfs.resample('5T').mean()
        dfs = dfs.reset_index(drop=True)
        dfs.index = np.linspace(df.index[0], df.index[-1], dfs.index.size)
        return dfs
    sns.set_style('whitegrid')
    sns.set_style('ticks')
    pw = xr.open_dataset(work_path / 'GNSS_PW_thresh_0_hour_dayofyear_anoms.nc')
    pws = [pw[x].load() for x in hydro_pw_dict.keys()]
    dfs = [read_station_from_tide_database(hydro_pw_dict.get(x), hydro_path=hydro_path) for x in hydro_pw_dict.keys()]
    df_list = []
    for pw_da, df_da in zip(pws, dfs):
        df, _, _ = produce_pwv_days_before_tide_events(pw_da, df_da,
                                                       plot=False,
                                                       days_prior=days_prior,
                                                       drop_thresh=0.5,
                                                       max_gap='12H')
        df_list.append(df)
    n_events = [len(x.columns) for x in df_list]
    if smoothed:
        df_list = [smooth_df(x) for x in df_list]
    df_mean = pd.concat([x.T.mean().to_frame(x.columns[0].split('_')[0]) for x in df_list], axis=1)
    fig, ax = plt.subplots(figsize=(8, 10))
    labels = ['{}: mean from {} events'.format(x.upper(), y) for x,y in zip(df_mean.columns, n_events)]
    for i, station in enumerate(df_mean.columns):
        sns.lineplot(data=df_mean, y=station, x=df.index, ax=ax, label=labels[i], lw=4)
    ax.grid(True)
    ax.axvline(0, color='k', linestyle='--')
    ax.set_xlabel('Days before/after tide event', fontsize=fontsize)
    ax.set_ylabel('PWV anomalies [mm]', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.legend(prop={'size': fontsize-2})
    fig.tight_layout()
    if save:
        filename = 'PWV_anoms_dsea_drag_elat_{}_prior_tides.png'.format(days_prior)
        plt.savefig(savefig_path / filename, orientation='potrait')
    return fig


def plot_typical_tide_event_with_PWV(work_path=work_yuval,
                                     hydro_path=hydro_path,
                                     station='yrcm',
                                     days_prior=3, days_after=1, fontsize=16,
                                     date='2018-04-27',
                                     save=True, smoothed=True):
    # possible dates: 2014-11-16T13:50, 2018-04-26T18:55
    # best to show 2018-04-24-27,
    # TODO: x-axis time hours, ylabels in color
    import xarray as xr
    import pandas as pd
    from hydro_procedures import hydro_pw_dict
    from matplotlib.ticker import FormatStrFormatter
    import numpy as np

    def smooth_df(df):
        dfs = df.copy()
        # dfs.index = pd.to_timedelta(dfs.index, unit='d')
        dfs = dfs.resample('15S').interpolate(method='cubic')
        dfs = dfs.resample('5T').mean()
        # dfs = dfs.reset_index(drop=True)
        # dfs.index = np.linspace(df.index[0], df.index[-1], dfs.index.size)
        return dfs

    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    colors = sns.color_palette('tab10', n_colors=2)
    # sns.set_style('whitegrid')
    sns.set_style('ticks')
    # load hydro graphs:
    hgs = xr.open_dataset(hydro_path/'hydro_graphs.nc')
    # select times:
    dt_start = pd.to_datetime(date) - pd.Timedelta(days_prior, unit='d')
    dt_end = pd.to_datetime(date) + pd.Timedelta(days_after, unit='d')
    hs_id = hydro_pw_dict.get(station)
    hg_da = hgs['HS_{}_flow'.format(hs_id)].sel(time=slice(dt_start, dt_end)).dropna('time')
    hg_da = hg_da.resample(time='15T').mean().interpolate_na('time', method='spline', max_gap='12H')
    # load pwv:
    pw = xr.open_dataset(work_path / 'GNSS_PW_thresh_0_for_hydro_analysis.nc')[station]
    pw = pw.sel(time=slice(dt_start, dt_end))
    df = pw.to_dataframe(name='pwv')
    df['flow'] = hg_da.to_dataframe()
    if smoothed:
        df = smooth_df(df)
    fig, ax = plt.subplots(figsize=(15, 4))
    flow_label = r'Flow [m$^3\cdot$sec$^{-1}$]'
    # df['time'] = df.index
    # sns.lineplot(data=df, y='flow', x=df.index, ax=ax, label=48125, lw=2, color=colors[0])
    # twin = ax.twinx()
    # sns.lineplot(data=df, y='pwv', x=df.index, ax=twin, label='DRAG', lw=2, color=colors[1])
    df.index.name=''
    ax = df['flow'].plot(color=colors[0], ax=ax, lw=2)
    twin = ax.twinx()
    df['pwv'].plot(color=colors[1], ax=twin, lw=2)
    ax.set_ylim(0, 100)
    ax.set_ylabel(flow_label, fontsize=fontsize, color=colors[0])
    twin.set_ylabel('PWV [mm]', fontsize=fontsize, color=colors[1])
    ax.tick_params(axis='y', labelsize=fontsize, labelcolor=colors[0])
    # ax.tick_params(axis='x', labelsize=fontsize, bottom=True, which='both')
    twin.tick_params(axis='y', labelsize=fontsize, labelcolor=colors[1])
    twin.yaxis.set_ticks(np.arange(10, 35, 5))
    # twin.yaxis.set_major_locator(ticker.FixedLocator(locs=np.arange(0,35,5)))
    twin.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # align_yaxis_np(ax, twin)
    # alignYaxes([ax, twin], [0, 10])
    # lim = ax.get_ylim()
    # l2 = twin.get_ylim()
    # fun = lambda x: l2[0]+(x-lim[0])/(lim[1]-lim[0])*(l2[1]-l2[0])
    # ticks = fun(ax.get_yticks())
    sns.set(rc={"xtick.bottom" : True, "ytick.left" : True})
    xticks=df.resample('12H').mean().index
    ax.xaxis.set_ticks(xticks)
    strDates = [x.strftime('%d-%H') for x in xticks]
    ax.set_xticklabels(strDates)
    xticks=df.resample('4H').mean().index
    ax.xaxis.set_ticks(xticks, minor=True)

    # locator = mdates.AutoDateLocator(minticks = 15,
    #                                  maxticks = 20)
    # # formatter = mdates.ConciseDateFormatter(locator)

    # ax.xaxis.set_major_locator(locator)
    # ax.xaxis.set_major_formatter(formatter)
    # loc = mdates.AutoDateLocator()
    # ax.xaxis.set_major_locator(loc)
    # ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    # ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    # ax.xaxis.set_major_locator(mdates.DayLocator())
    # minorLocator = ticker.AutoMinorLocator()
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H'))
    # ax.xaxis.set_major_locator(mdates.DayLocator())
    # ax.xaxis.set_minor_locator(minorLocator)
    # ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    ax.grid(True, which='major', axis='y',color='k', ls='--')
    # ax.set_xticklabels([x.strftime("%d-%H") for x in df.index], rotation=45)
    ax.grid(True, which='minor', axis='x')

    # twin.yaxis.set_major_locator(ticker.FixedLocator(ticks))
    # twin.grid(True, axis='y',color='k', ls='--')
    # twin.xaxis.set_major_locator(mdates.DayLocator())
    # twin.xaxis.set_minor_locator(mdates.HourLocator())
    # Fmt = mdates.AutoDateFormatter(mdates.DayLocator())
    # twin.xaxis.set_major_formatter(Fmt)
    # ax.set_ylim(0, 20)
    fig.autofmt_xdate()
    fig.tight_layout()
    if save:
        filename = 'typical_tide_event_with_pwv'
        plt.savefig(savefig_path / filename, orientation='potrait')
    return df


def plot_hydro_pressure_anomalies(hydro_path=hydro_path,
                                  fontsize=16, save=True):
    import xarray as xr
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')
    sns.set_style('ticks')
    feats = xr.load_dataset(hydro_path/'hydro_tides_hourly_features_with_positives.nc')
    dts = pd.DatetimeIndex(feats['X_pos']['positive_sample'].values)
    bd = feats['bet-dagan']
    dts_ranges = []
    for dt in dts:
        prior = dt - pd.Timedelta(3, unit='D')
        after = dt + pd.Timedelta(1, unit='D')
        dt_range = pd.date_range(start=prior, end=after, freq='H')
        bd_range = bd.sel(time=dt_range)
        dts_ranges.append(bd_range.values)
    df = pd.DataFrame(dts_ranges).T
    df.index = np.linspace(-3, 1, len(df))
    fig, ax = plt.subplots(figsize=(8, 6))
    ts = df.T.mean() #.shift(periods=-1, freq='15D')
    ts_std = df.T.std()
    ts.index.name = ''
    ts.plot(ax=ax, color='k', fontsize=fontsize, lw=2)
    ax.fill_between(x=ts.index, y1=ts-ts_std, y2=ts+ts_std, color='k', alpha=0.4)
    ax.set_xlim(ts.index.min(), ts.index.max())  #+
                      # pd.Timedelta(15, unit='D'))
    ax.set_ylabel('Pressure mean anomalies [hPa]', fontsize=fontsize-2)
    ax.set_xlabel('Days before/after a tide event', fontsize=fontsize-2)
    ax.axvline(0, color='r', ls='--')
    ax.grid(True)
    fig.tight_layout()
    fig.subplots_adjust(right=0.946)
    if save:
        filename = 'Pressure_anoms_3_prior_tides.png'
        plt.savefig(savefig_path / filename, bbox_inches='tight', pad_inches=0.1)
    return df


def plot_hydro_pwv_anomalies_with_station_mean(work_path=work_yuval,
                                               hydro_path=hydro_path,
                                               days_prior=3, fontsize=14,
                                               save=True, smoothed=False):
    import xarray as xr
    from hydro_procedures import hydro_pw_dict
    from hydro_procedures import produce_pwv_days_before_tide_events
    from hydro_procedures import read_station_from_tide_database
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    def smooth_df(df):
        import numpy as np
        dfs = df.copy()
        dfs.index = pd.to_timedelta(dfs.index, unit='d')
        dfs = dfs.resample('15S').interpolate(method='cubic')
        dfs = dfs.resample('5T').mean()
        dfs = dfs.reset_index(drop=True)
        dfs.index = np.linspace(df.index[0], df.index[-1], dfs.index.size)
        return dfs
    sns.set_style('whitegrid')
    sns.set_style('ticks')
    cmap = 'jet' #sns.color_palette('gist_rainbow_r', as_cmap=True)
    pw = xr.open_dataset(work_path / 'GNSS_PW_thresh_0_hour_dayofyear_anoms.nc')
    pws = [pw[x].load() for x in hydro_pw_dict.keys()]
    dfs = [read_station_from_tide_database(hydro_pw_dict.get(x), hydro_path=hydro_path) for x in hydro_pw_dict.keys()]
    df_list = []
    for pw_da, df_da in zip(pws, dfs):
        df, _, _ = produce_pwv_days_before_tide_events(pw_da, df_da,
                                                       plot=False,
                                                       days_prior=days_prior,
                                                       drop_thresh=0.75,
                                                       max_gap='6H')
        df_list.append(df)
    n_events = [len(x.columns) for x in df_list]
    if smoothed:
        df_list = [smooth_df(x) for x in df_list]
    df_mean = pd.concat([x.T.mean().to_frame(x.columns[0].split('_')[0]) for x in df_list], axis=1)

    df_mean.columns = [x.upper() for x in df_mean.columns]
    df_mean.index = pd.to_timedelta(df_mean.index, unit='D')
    df_mean = df_mean.resample('30T').mean()
    df_mean.index = np.linspace(-3, 1, len(df_mean.index))
    # weights = df.count(axis=1).shift(periods=-1, freq='15D').astype(int)
    fig = plt.figure(figsize=(5, 8))
    grid = plt.GridSpec(
        2, 1, height_ratios=[
            1, 1], hspace=0.0225)
    ax_heat = fig.add_subplot(grid[0, 0])  # plt.subplot(221)
    ax_group = fig.add_subplot(grid[1, 0])  # plt.subplot(223)
    cbar_ax = fig.add_axes([0.95, 0.50, 0.02, 0.38])  # [left, bottom, width,
    # height]
    ax_heat = sns.heatmap(
        df_mean.T,
        cmap=cmap,
        yticklabels=True,
        ax=ax_heat,
        cbar_ax=cbar_ax,
        cbar_kws={'label': 'PWV anomalies [mm]'}, xticklabels=False)
    cbar_ax.set_ylabel('PWV anomalies [mm]', fontsize=fontsize-2)
    cbar_ax.tick_params(labelsize=fontsize)
    zero_in_heat = df_mean.index.get_loc(0, method='nearest') + 1
    ax_heat.vlines([zero_in_heat], *ax_heat.get_ylim(), color='k',
                   linestyle='--', linewidth=1.5, zorder=20)
    # activate top ticks and tickslabales:
    ax_heat.xaxis.set_tick_params(
        bottom='off', labelbottom='off', labelsize=fontsize)
    # emphasize the yticklabels (stations):
    ax_heat.yaxis.set_tick_params(left='on')
    # labels = ['{} ({})'.format(x.get_text(), y) for x, y in zip(ax_heat.get_ymajorticklabels(), n_events)]
    # ax_heat.set_yticklabels(labels,
    #                         fontweight='bold', fontsize=fontsize,
    #                         rotation='horizontal')
    ax_heat.set_xlabel('')
    ts = df_mean.T.mean() #.shift(periods=-1, freq='15D')
    ts_std = df_mean.T.std()
    # ts.index= pd.to_timedelta(ts.index, unit='D')
    ts.index.name = ''
    ts.plot(ax=ax_group, color='k', fontsize=fontsize, lw=2)
    ax_group.fill_between(x=ts.index, y1=ts-ts_std, y2=ts+ts_std, color='k', alpha=0.4)
    # barax = ax_group.twinx()
    # barax.bar(ts.index, weights.values, width=35, color='k', alpha=0.2)
    # barax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    # barax.set_ylabel('Stations [#]', fontsize=fontsize-4)
    # barax.tick_params(labelsize=fontsize)
    ax_group.set_xlim(ts.index.min(), ts.index.max())  #+
                      # pd.Timedelta(15, unit='D'))
    ax_group.set_ylabel('PWV mean anomalies [mm]', fontsize=fontsize-2)
    ax_group.set_xlabel('Days before/after a tide event', fontsize=fontsize-2)
    # set ticks and align with heatmap axis (move by 0.5):
    # ax_group.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
    # ax_group.set_xticks(np.arange(-3, 1, 0.25))
    # offset = 1
    # ax_group.xaxis.set(ticks=np.arange(offset / 2.,
    #                                   max(ts.index) + 1 - min(ts.index),
    #                                   offset),
    #                   ticklabels=ts.index)
    # # move the lines also by 0.5 to align with heatmap:
    # lines = ax_group.lines  # get the lines
    # [x.set_xdata(x.get_xdata() - min(ts.index) + 0.5) for x in lines]
    # ax_group.xaxis.set(ticks=xticks, ticklabels=xticks_labels)
    # ax_group.xaxis.set(ticks=xticks)
    # mytime = mdates.DateFormatter('%D-%H')
    # ax_group.xaxis.set_major_formatter(mytime)
    # ax_group.xaxis.set_major_locator(mdates.DayLocator(interval=0.5))
    # xticks = pd.timedelta_range(pd.Timedelta(-3, unit='D'), pd.Timedelta(1, unit='D'), freq='3H')
    # ax_group.set_xticks(xticks)
    ax_group.axvline(0, color='k', ls='--', lw=1.5)
    # ax_heat.axvline(0, color='r', ls='--')
    ax_group.grid(True)
    fig.tight_layout()
    fig.subplots_adjust(right=0.946)
    if save:
        filename = 'PWV_anoms_{}_prior_tides.png'.format(days_prior)
        plt.savefig(savefig_path / filename, bbox_inches='tight', pad_inches=0.1)
    return ax_group


def produce_hydro_and_GNSS_stations_table(work_path=work_yuval,
                                          hydro_path=hydro_path, gis_path=gis_path):
    from PW_stations import produce_geo_gnss_solved_stations
    from hydro_procedures import hydro_pw_dict, hydro_st_name_dict
    from hydro_procedures import read_hydro_metadata
    from hydro_procedures import get_hydro_near_GNSS
    import xarray as xr
    import pandas as pd
    stns = [x for x in hydro_pw_dict.keys()]
    df_gnss = produce_geo_gnss_solved_stations(plot=False,
                                               add_distance_to_coast=False)
    df_gnss = df_gnss.loc[stns]
    df_gnss['ID'] = df_gnss.index.str.upper()
    pd.options.display.float_format = '{:.2f}'.format
    df = df_gnss[['name', 'ID', 'lat', 'lon', 'alt']]
    df['alt'] = df['alt'].map('{:,.0f}'.format)
    cols = ['GNSS Station name', 'Station ID', 'Latitude [N]',
            'Longitude [E]', 'Altitude [m a.s.l]']
    df.columns = cols
    # df.loc['spir', 'GNSS station name'] = 'Sapir'
    hydro_meta = read_hydro_metadata(hydro_path, gis_path, plot=False)
    hdf = hydro_meta.loc[:, ['id', 'alt', 'lat', 'lon']]
    hdf = hdf.set_index('id')
    hdf = hdf.loc[[x for x in hydro_pw_dict.values()], :]
    hdf['station_name'] = [x for x in hydro_st_name_dict.values()]
    hdf['nearest_gnss'] = [x.upper() for x in hydro_pw_dict.keys()]
    hdf1 = get_hydro_near_GNSS(radius=15, plot=False)
    li = []
    for st, hs_id in hydro_pw_dict.items():
        dis = hdf1[hdf1['id'] == hs_id].loc[:, st]
        li.append(dis.values[0])
    hdf['distance_to_gnss'] = [x/1000.0 for x in li]
    hdf['alt'] = hdf['alt'].map('{:,.0f}'.format)
    hdf['station_number'] = [int(x) for x in hydro_pw_dict.values()]
    hdf['distance_to_gnss'] = hdf['distance_to_gnss'].map('{:,.0f}'.format)
    # add tide events per station:
    file = hydro_path / 'hydro_tides_hourly_features_with_positives.nc'
    tides = xr.load_dataset(file)['Tides']
    tide_count = tides.to_dataset('GNSS').to_dataframe().count()
    hdf['tides'] = [x for x in tide_count]
    hdf = hdf[['station_name', 'station_number', 'lat', 'lon', 'alt', 'nearest_gnss', 'distance_to_gnss', 'tides']]
    hdf.columns = ['Hydro station name', 'Station ID', 'Latitude [N]',
            'Longitude [E]', 'Altitude [m a.s.l]', 'Nearest GNSS station', 'Distance to GNSS station [km]', 'Flood events near GNSS station']
    return df, hdf


def plot_hydro_events_climatology(hydro_path=hydro_path, fontsize=16, save=True):
    import xarray as xr
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    sns.set_style('whitegrid')
    sns.set_style('ticks')
    file = hydro_path / 'hydro_tides_hourly_features_with_positives.nc'
    X = xr.load_dataset(file)['X_pos']
    df = X['positive_sample'].groupby('positive_sample.month').count().to_dataframe()
    # add July and August:
    add = pd.DataFrame([0, 0], index=[7, 8])
    add.index.name = 'month'
    add.columns = ['positive_sample']
    df = df.append(add).sort_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df, x=df.index, y='positive_sample', ax=ax, color='tab:blue')
    ax.grid(True)
    ax.set_ylabel('Number of unique flood events [#]', fontsize=fontsize)
    ax.set_xlabel('month', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    fig.tight_layout()
    if save:
        filename = 'tides_count_climatology.png'
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fig


def plot_hydro_GNSS_periods_and_map(path=work_yuval, gis_path=gis_path,
                                    ims=False, dem_path=dem_path,
                                    hydro_path=hydro_path,
                                    fontsize=22, save=True):

    from aux_gps import gantt_chart
    import xarray as xr
    import pandas as pd
    import geopandas as gpd
    from PW_stations import produce_geo_gnss_solved_stations
    from aux_gps import geo_annotate
    from ims_procedures import produce_geo_ims
    from hydro_procedures import hydro_pw_dict
    import cartopy.crs as ccrs
    from hydro_procedures import read_hydro_metadata
    from hydro_procedures import prepare_tide_events_GNSS_dataset
    sns.set_style('whitegrid')
    sns.set_style('ticks')
    fig = plt.figure(figsize=(20, 15))
    grid = plt.GridSpec(1, 2, width_ratios=[
        5, 5], wspace=0.125)
    ax_gantt = fig.add_subplot(grid[0, 0])  # plt.subplot(221)
    ax_map = fig.add_subplot(grid[0, 1], projection=ccrs.PlateCarree())  # plt.subplot(122)
    extent = [34, 36.0, 29.2, 32.5]
    ax_map.set_extent(extent)
#    fig, ax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(20, 6))
    # RINEX gantt chart:
    file = hydro_path /  'hydro_tides_hourly_features.nc'
    ds = xr.open_dataset(file)
    just_pw = [x for x in hydro_pw_dict.keys()]
    ds = ds[just_pw]
    da = ds.to_array('station')
    da['station'] = [x.upper() for x in da.station.values]
    ds = da.to_dataset('station')
    # add tide events
    ds_events = prepare_tide_events_GNSS_dataset(hydro_path)
    # merge in couples for keeping the original order:
    li = []
    for pwv, tide in zip(ds, ds_events):
        first = ds[pwv]
        second = ds_events[tide]
        second.name = first.name + '*'
        li.append(first)
        li.append(second)
    ds = xr.merge(li)
    # colors:
    # title = 'Daily RINEX files availability for the Israeli GNSS stations'
    c = sns.color_palette('Dark2', n_colors=int(len(ds) / 2))
    colors = []
    for color in c:
        colors.append(color)
        colors.append(color)
    ax_gantt = gantt_chart(
        ds,
        ax=ax_gantt,
        fw='bold', grid=True,marker='x', marker_suffix='*',
        title='', colors=colors,
        pe_dict=None, fontsize=fontsize, linewidth=24, antialiased=False)
    years_fmt = mdates.DateFormatter('%Y')
    # ax_gantt.xaxis.set_major_locator(mdates.YearLocator())
    ax_gantt.xaxis.set_major_locator(mdates.YearLocator(4))
    ax_gantt.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax_gantt.xaxis.set_major_formatter(years_fmt)
    # ax_gantt.xaxis.set_minor_formatter(years_fmt)
    ax_gantt.tick_params(axis='x', labelrotation=0)
    # Israel gps ims map:
    ax_map = plot_israel_map(
        gis_path=gis_path, ax=ax_map, ticklabelsize=fontsize)
    # overlay with dem data:
    cmap = plt.get_cmap('terrain', 41)
    dem = xr.open_dataarray(dem_path / 'israel_dem_250_500.nc')
    # dem = xr.open_dataarray(dem_path / 'israel_dem_500_1000.nc')
    dem = dem.sel(lat=slice(29.2, 32.5), lon=slice(34, 36.3))
    fg = dem.plot.imshow(ax=ax_map, alpha=0.5, cmap=cmap,
                         vmin=dem.min(), vmax=dem.max(), add_colorbar=False)
#    scale_bar(ax_map, 50)
    cbar_kwargs = {'fraction': 0.1, 'aspect': 50, 'pad': 0.03}
    cb = plt.colorbar(fg, **cbar_kwargs)
    cb.set_label(label='meters above sea level',
                 size=fontsize, weight='normal')
    cb.ax.tick_params(labelsize=fontsize)
    ax_map.set_xlabel('')
    ax_map.set_ylabel('')
    # ax_map.xaxis.set_major_locator(ticker.MaxNLocator(2))
    # ax_map.yaxis.set_major_locator(ticker.MaxNLocator(5))
    # ax_map.yaxis.set_major_formatter(lat_formatter)
    # ax_map.xaxis.set_major_formatter(lon_formatter)
    # ax_map.gridlines(draw_labels=True, dms=False, x_inline=False,
    #                  y_inline=False, xformatter=lon_formatter, yformatter=lat_formatter,
    #                  xlocs=ticker.MaxNLocator(2), ylocs=ticker.MaxNLocator(5))
    # fig.canvas.draw()
    ax_map.set_xticks([34, 35, 36])
    ax_map.set_yticks([29.5, 30, 30.5, 31, 31.5, 32, 32.5])
    ax_map.tick_params(top=True, bottom=True, left=True, right=True,
                       direction='out', labelsize=fontsize)
    gps = produce_geo_gnss_solved_stations(path=gis_path, plot=False)
    gps = gps.loc[just_pw, :]
    # gps_list = [x for x in gps.index if x not in merged and x not in removed]
    gps.plot(ax=ax_map, edgecolor='black', marker='s',
             alpha=1.0, markersize=100, facecolor="None", linewidth=2, zorder=3)
    to_plot_offset = ['nizn', 'ramo', 'nrif']

    for x, y, label in zip(gps.lon, gps.lat, gps.index.str.upper()):
        if label.lower() in to_plot_offset:
            ax_map.annotate(label, xy=(x, y), xytext=(4, -15),
                            textcoords="offset points", color='k',
                            fontweight='bold', fontsize=fontsize - 2)
        else:
            ax_map.annotate(label, xy=(x, y), xytext=(3, 3),
                            textcoords="offset points", color='k',
                            fontweight='bold', fontsize=fontsize - 2)
#    geo_annotate(ax_map, gps_normal_anno.lon, gps_normal_anno.lat,
#                 gps_normal_anno.index.str.upper(), xytext=(3, 3), fmt=None,
#                 c='k', fw='normal', fs=10, colorupdown=False)
#    geo_annotate(ax_map, gps_offset_anno.lon, gps_offset_anno.lat,
#                 gps_offset_anno.index.str.upper(), xytext=(4, -6), fmt=None,
#                 c='k', fw='normal', fs=10, colorupdown=False)
    # plot bet-dagan:
    df = pd.Series([32.00, 34.81]).to_frame().T
    df.index = ['Bet-Dagan']
    df.columns = ['lat', 'lon']
    bet_dagan = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon,
                                                                 df.lat),
                                 crs=gps.crs)
    bet_dagan.plot(ax=ax_map, color='black', edgecolor='black',
                   marker='x', linewidth=2, zorder=2, markersize=100)
    geo_annotate(ax_map, bet_dagan.lon, bet_dagan.lat,
                 bet_dagan.index, xytext=(4, -6), fmt=None,
                 c='k', fw='bold', fs=fontsize - 2, colorupdown=False)
    # now add hydro stations:
    hydro_meta = read_hydro_metadata(hydro_path, gis_path, plot=False)
    hm = hydro_meta.loc[:, ['id', 'name', 'alt', 'lat', 'lon']]
    hm = hm.set_index('id')
    hm = hm.loc[[x for x in hydro_pw_dict.values()], :]
    hmgdf = gpd.GeoDataFrame(hm, geometry=gpd.points_from_xy(hm.lon, hm.lat), crs=gps.crs)
    hmgdf.plot(ax=ax_map, edgecolor='black', marker='o',
               alpha=1.0, markersize=100, facecolor='tab:pink', zorder=4)
#    plt.legend(['GNSS \nreceiver sites',
#                'removed \nGNSS sites',
#                'merged \nGNSS sites',
#                'radiosonde\nstation'],
#               loc='upper left', framealpha=0.7, fancybox=True,
#               handletextpad=0.2, handlelength=1.5)
    if ims:
        print('getting IMS temperature stations metadata...')
        ims = produce_geo_ims(path=gis_path, freq='10mins', plot=False)
        ims.plot(ax=ax_map, marker='o', edgecolor='tab:orange', alpha=1.0,
                 markersize=35, facecolor="tab:orange", zorder=1)
    # ims, gps = produce_geo_df(gis_path=gis_path, plot=False)
        print('getting solved GNSS israeli stations metadata...')
        plt.legend(['GNSS \nstations',
                    'radiosonde\nstation', 'IMS stations'],
                   loc='upper left', framealpha=0.7, fancybox=True,
                   handletextpad=0.2, handlelength=1.5, fontsize=fontsize - 2)
    else:
        plt.legend(['GNSS \nstations',
                    'radiosonde\nstation',
                    'hydrometric\nstations'],
                   loc='upper left', framealpha=0.7, fancybox=True,
                   handletextpad=0.2, handlelength=1.5, fontsize=fontsize - 2)
    fig.subplots_adjust(top=0.95,
                        bottom=0.11,
                        left=0.05,
                        right=0.95,
                        hspace=0.2,
                        wspace=0.2)
    # plt.legend(['IMS stations', 'GNSS stations'], loc='upper left')

    filename = 'hydro_israeli_gnss_map.png'
#    caption('Daily RINEX files availability for the Israeli GNSS station network at the SOPAC/GARNER website')
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fig


def produce_all_param_grid_tables():
    import pandas as pd
    mlp = produce_single_param_grid_table('MLP')
    rf = produce_single_param_grid_table('RF')
    svc = produce_single_param_grid_table('SVC')
    df = pd.concat([svc, rf, mlp], axis=1)
    df = df.fillna(' ')
    return df


def produce_single_param_grid_table(model='MLP'):
    import pandas as pd
    from hydro_procedures import ML_Classifier_Switcher
    numeric = ['C', 'alpha', 'gamma', 'max_depth', 'n_estimators']
    numeric_type = ['log', 'log', 'log', 'int', 'int']
    numeric_dict = dict(zip(numeric, numeric_type))
    ml = ML_Classifier_Switcher()
    ml.pick_model(model)
    params = ml.param_grid
    num_params = [x for x in params.keys() if x in numeric]
    num_dict = dict((k, params[k]) for k in num_params)
    other_params = [x for x in params.keys() if x not in numeric]
    other_dict = dict((k, params[k]) for k in other_params)
    di = {}
    for key, val in other_dict.items():
        val = [str(x) for x in val]
        di[key] = ', '.join(val)
    for key, val in num_dict.items():
        if numeric_dict[key] != 'log':
            val = '{} to {}'.format(val[0], val[-1])
        else:
            val = r'{} to {}'.format(sci_notation(val[0]), sci_notation(val[-1]))

        di[key] = val
    df = pd.Series(di).to_frame('Options')
    df['Parameters'] = df.index.str.replace('_', ' ')
    df = df[['Parameters', 'Options']]
    df = df.reset_index(drop=True)
    return df

