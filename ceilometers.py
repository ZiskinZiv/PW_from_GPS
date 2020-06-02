#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:41:11 2020

@author: shlomi
"""
from PW_paths import work_yuval
ceil_path = work_yuval / 'ceilometers'
# available stations: Jerousalem, Nevatim, Ramat_David, Tel_Aviv
stations_dict = {
    'Tel_Aviv': ['TLV', 34.8, 32.1, 5],
    'Nevatim': ['NV', 34.9, 31.2, 400],
    'Ramat_David': ['RD', 35.2, 32.7, 50],
    'Jerusalem': ['JR', 35.2, 31.8, 830]}


def plot_pw_mlh(path=work_yuval, ceil_path=ceil_path, kind='scatter'):
    import xarray as xr
    import matplotlib.pyplot as plt
    mlh = xr.load_dataset(ceil_path / 'MLH_from_ceilometers.nc')
    pw = xr.load_dataset(work_yuval / 'GNSS_PW_thresh_50_homogenized.nc')
    pw = pw[[x for x in pw if '_error' not in x]]
    pw = pw[['tela', 'klhv', 'jslm', 'nzrt', 'yrcm']]
    couples = [['tela', 'TLV'], ['klhv', 'NV'], ['jslm', 'JR']]
    if kind == 'scatter':
        fig, axes = plt.subplots(
            1, len(couples), sharey=True, sharex=True, figsize=(
                20, 5))
        for i, ax in enumerate(axes.flatten()):
            ax = scatter_plot_pw_mlh(
                pw[couples[i][0]], mlh[couples[i][1]], ax=ax)
    elif kind == 'diurnal':
        fig, axes = plt.subplots(
            len(couples), 2, sharey=False, sharex=False, figsize=(
                20, 15))
        for i, ax in enumerate(axes[:, 0].flatten()):
            ax = twin_hourly_mean_plot(
                pw[couples[i][0]], mlh[couples[i][1]], month=None, ax=ax, title=False)
        for i, ax in enumerate(axes[:, 1].flatten()):
            ax = scatter_plot_pw_mlh(pw[couples[i][0]],
                                     mlh[couples[i][1]],
                                     diurnal=True,
                                     ax=ax,
                                     title=False,
                                     leg_loc='lower right')
    fig.tight_layout()
    return fig


def scatter_plot_pw_mlh(pw, mlh, diurnal=False, ax=None, title=True,
                        leg_loc='best'):
    from aux_gps import dim_intersection
    import xarray as xr
    import numpy as np
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    from PW_stations import produce_geo_gnss_solved_stations
    df = produce_geo_gnss_solved_stations(plot=False)
    pw_alt = df.loc[pw.name, 'alt']
    newtime = dim_intersection([pw, mlh], 'time')
    pw = pw.sel(time=newtime)
    pw_attrs = pw.attrs
    mlh = mlh.sel(time=newtime)
    mlh_attrs = mlh.attrs
    if diurnal:
        pw = pw.groupby('time.hour').mean()
        pw.attrs = pw_attrs
        mlh = mlh.groupby('time.hour').mean()
        mlh.attrs = mlh_attrs
    ds = xr.merge([pw, mlh])
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ds.plot.scatter(pw.name, mlh.name, ax=ax)
    coefs = np.polyfit(pw.values, mlh.values, 1)
    x = np.linspace(pw.min().item(), pw.max().item(), 100)
    y = np.polyval(coefs, x)
    r2 = r2_score(mlh.values, np.polyval(coefs, pw.values))
#    coefs2 = np.polyfit(pw.values, mlh.values, 2)
#    y2 = np.polyval(coefs2, x)
#    r22 = r2_score(mlh.values,np.polyval(coefs2, pw.values))
    ax.plot(x, y, color='tab:red')
    # ax.plot(x, y2, color='tab:orange')
    ax.set_xlabel('PW [mm]')
    ax.set_ylabel('MLH [m]')
    ax.legend(['linear fit', 'data'], loc=leg_loc)
    textstr = '\n'.join(['n={}'.format(pw.size),
                         r'R$^2$={:.2f}'.format(r2),
                         'slope={:.1f} m/mm'.format(coefs[0])])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    mlh_name = mlh.attrs['station_full_name'].replace('_', '-')
    if title:
        ax.set_title(
            '{} ({:.0f} m) GNSS site PW vs. {} ({:.0f} m) Mixing Layer Height'.format(
                pw.name.upper(),
                pw_alt,
                mlh_name,
                mlh.attrs['alt']))
    return ax


def twin_hourly_mean_plot(pw, mlh, month=8, ax=None, title=True,
                          leg_loc='best'):
    from aux_gps import dim_intersection
    import matplotlib.pyplot as plt
    from calendar import month_abbr
    from PW_stations import produce_geo_gnss_solved_stations
    df = produce_geo_gnss_solved_stations(plot=False)
    pw_alt = df.loc[pw.name, 'alt']
    # first run multi-year month mean:
    if month is not None:
        pw_month = pw.sel(time=pw['time.month'] == month)
        pw_years = [
            pw_month.dropna('time').time.dt.year.min().item(),
            pw_month.dropna('time').time.dt.year.max().item()]
        pw_m_hour = pw_month.groupby('time.hour').mean()
    newtime = dim_intersection([pw, mlh], 'time')
    pw = pw.sel(time=newtime)
    mlh = mlh.sel(time=newtime)
    pw_hour = pw.groupby('time.hour').mean()
    pw_std = pw.groupby('time.hour').std()
    pw_hour_plus = (pw_hour + pw_std).values
    pw_hour_minus = (pw_hour - pw_std).values
    mlh_hour = mlh.groupby('time.hour').mean()
    mlh_std = mlh.groupby('time.hour').std()
    mlh_hour_minus = (mlh_hour - mlh_std).values
    mlh_hour_plus = (mlh_hour + mlh_std).values
    years = [mlh.time.dt.year.min().item(), mlh.time.dt.year.max().item()]
    mlh_month = mlh.time.dt.month.to_dataframe()['month'].value_counts().index[0]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    red = 'tab:red'
    blue = 'tab:blue'
    pwln = pw_hour.plot(color=blue, marker='s', ax=ax)
#        ax.errorbar(pw_hour.hour.values, pw_hour.values, pw_std.values,
#                    label='PW', color=blue, capsize=5, elinewidth=2,
#                    markeredgewidth=2)
    ax.fill_between(pw_hour.hour.values, pw_hour_minus, pw_hour_plus, color=blue, alpha=0.5)
    twin = ax.twinx()
#        twin.errorbar(mlh_hour.hour.values, mlh_hour.values, mlh_std.values,
#                      color=red, label='MLH', capsize=5, elinewidth=2,
#                      markeredgewidth=2)
    mlhln = mlh_hour.plot(color=red, marker='o', ax=twin)
    twin.fill_between(mlh_hour.hour.values, mlh_hour_minus, mlh_hour_plus, color=red, alpha=0.5)
#        handles, labels = ax.get_legend_handles_labels()
#        handles = [h[0] for h in handles]
#        handles1, labels1 = twin.get_legend_handles_labels()
#        handles1 = [h[0] for h in handles1]
#        hand = handles + handles1
#        labs = labels + labels1
    pw_label = 'PW: {}-{}, {} ({} pts)'.format(years[0], years[1], month_abbr[mlh_month], mlh.size)
    mlh_label = 'MLH: {}-{}, {} ({} pts)'.format(years[0], years[1], month_abbr[mlh_month], mlh.size)
    if month is not None:
        pwmln = pw_m_hour.plot(color='tab:orange', marker='^', ax=ax)
        pwm_label = 'PW: {}-{}, {} ({} pts)'.format(pw_years[0], pw_years[1], month_abbr[month], pw_month.dropna('time').size)
        ax.legend(pwln + mlhln + pwmln, [pw_label, mlh_label, pwm_label], loc=leg_loc)
    else:
        ax.legend(pwln + mlhln, [pw_label, mlh_label], loc=leg_loc)
    ax.tick_params(axis='y', colors=blue)
    twin.tick_params(axis='y', colors=red)
    ax.set_ylabel('PW [mm]', color=blue)
    twin.set_ylabel('MLH [m]', color=red)
    ax.set_xticks([x for x in range(24)])
    ax.set_xlabel('Hour of day [UTC]')
    mlh_name = mlh.attrs['station_full_name'].replace('_', '-')
    textstr = '{}, {}'.format(mlh_name, pw.name.upper())
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
    if title:
        ax.set_title('The diurnal cycle of {} Mixing Layer Height and {} GNSS site PW'.format(mlh_name, pw.name.upper()))
    return ax


def read_all_ceilometer_stations(path=ceil_path):
    import xarray as xr
    from aux_gps import save_ncfile
    stations = [x for x in stations_dict.keys()]
    da_list = []
    for station in stations:
        print('reading station {}'.format(station))
        da = read_ceilometer_station(path=path, name=station)
        da_list.append(da)
    ds = xr.merge(da_list)
    save_ncfile(ds, path, filename='MLH_from_ceilometers.nc')
    return ds


def read_ceilometer_station(path=ceil_path, name='Jerusalem'):
    from aux_gps import path_glob
    import pandas as pd
    files = path_glob(path, '{}_*.mat'.format(name))
    df_list = []
    for file in files:
        df_list.append(read_one_matfile_ceilometers(file))
    df = pd.concat(df_list, axis=0)
    df.index.name = 'time'
    df.drop_duplicates(inplace=True)
    da = df.to_xarray()
    da.name = stations_dict[name][0]
    da.attrs['full_name'] = 'Mixing Layer Height'
    da.attrs['name'] = 'MLH'
    da.attrs['units'] = 'm'
    da.attrs['station_full_name'] = name
    da.attrs['lon'] = stations_dict[name][1]
    da.attrs['lat'] = stations_dict[name][2]
    da.attrs['alt'] = stations_dict[name][3]
    return da


def read_one_matfile_ceilometers(file):
    from scipy.io import loadmat
    import pandas as pd
    mat = loadmat(file)
    name = [x for x in mat.keys()][-1]
    mdata = mat[name]
    li = []
    days = []
    for i in range(mdata.shape[0]):
        days.append([x.squeeze().item() for x in mdata[i, 0]])
        li.append([x.squeeze().item() for x in mdata[i, 1:]])
    days = [x[0] for x in days]
    df = pd.DataFrame(li[1:], index=days[1:])
    df.columns = [int(x) for x in li[0]]
    df.drop(df.tail(2).index, inplace=True)
    df = df.rename({'201508110': '20150811'}, axis=0)
    df = df.rename({'201608110': '20160811'}, axis=0)
    df.index = pd.to_datetime(df.index)
    # transform to time-series:
    df_list = []
    for date in df.index:
        dts = date + pd.Timedelta(1, unit='H')
        dates = pd.date_range(dts, periods=24, freq='H')
        df1 = pd.DataFrame(df.loc[date].values, index=dates)
        df_list.append(df1)
    s = pd.concat(df_list)[0]
    return s
