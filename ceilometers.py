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

pw_mlh_dict = {'tela': 'TLV', 'yrcm': 'NV', 'jslm': 'JR'}


def read_all_one_half_hours_csvs(path=ceil_path, plot=True):
    import pandas as pd
    from aux_gps import path_glob
    files = path_glob(path, '*_Check_Avg_high_peak.csv')
    df_list = []
    for file in files:
        df = read_one_half_hour_csv(file)
        df_list.append(df)
    df = pd.concat(df_list, axis=0)
    df = df.sort_index()
    if plot:
         ax = df['MLH'].plot(style='b-', marker='o', ms=5)
    return df


def read_one_half_hour_csv(file):
    import pandas as pd
    date = file.as_posix().split('/')[-1].split('_')[0]
    dt = pd.to_datetime(date, format='%d-%m-%Y')
    df = pd.read_csv(file, header=None)
    df = df.T
    df.columns = ['MLH']
    dts = pd.date_range(start=dt, periods=48, freq='30T')
    df.set_index(dts, inplace=True)
    return df


def align_pw_mlh(path=work_yuval, ceil_path=ceil_path, site='tela',
                 interpolate=None, plot=True, dt_range_str='2015'):
    import xarray as xr
    from aux_gps import dim_intersection
    from aux_gps import xr_reindex_with_date_range
    import pandas as pd
    import matplotlib.pyplot as plt

    def pw_mlh_to_df(pw_new, mlh_site):
        newtime = dim_intersection([pw_new, mlh_site])
        MLH = mlh_site.sel(time=newtime)
        PW = pw_new.sel(time=newtime)
        df = PW.to_dataframe()
        df[MLH.name] = MLH.to_dataframe()
        new_time = pd.date_range(df.index.min(), df.index.max(), freq='1H')
        df = df.reindex(new_time)
        df.index.name = 'time'
        return df

    mlh = xr.load_dataset(ceil_path / 'MLH_from_ceilometers.nc')
    mlh_site = xr_reindex_with_date_range(mlh[pw_mlh_dict.get(site)], freq='1H')
    if interpolate is not None:
        print('interpolating ceil-site {} with max-gap of {}.'.format(pw_mlh_dict.get(site), interpolate))
        attrs = mlh_site.attrs
        mlh_site_inter = mlh_site.interpolate_na('time', max_gap=interpolate,
                                                     method='cubic')
        mlh_site_inter.attrs = attrs
    pw = xr.open_dataset(work_yuval / 'GNSS_PW_hourly_thresh_50_homogenized.nc')
    pw = pw[['tela', 'klhv', 'jslm', 'nzrt', 'yrcm']]
    pw.load()
    pw_new = pw[site]
    pw_new = xr_reindex_with_date_range(pw_new, freq='1H')
    if interpolate is not None:
        print('interpolating pw-site {} with max-gap of {}.'.format(site, interpolate))
        attrs = pw_new.attrs
        pw_new_inter = pw_new.interpolate_na('time', max_gap=interpolate, method='cubic')
        pw_new_inter.attrs = attrs
    df = pw_mlh_to_df(pw_new, mlh_site)
    if interpolate is not None:
        df_inter = pw_mlh_to_df(pw_new_inter, mlh_site_inter)
    if dt_range_str is not None:
        df = df.loc[dt_range_str, :]
    if plot:
        fig, ax = plt.subplots(figsize=(18,5))
        if interpolate is not None:
            df_inter[pw_new.name].plot(style='b--', ax=ax)
            # same ax as above since it's automatically added on the right
            df_inter[mlh_site.name].plot(style='r--', secondary_y=True, ax=ax)
        ax = df[pw_new.name].plot(style='b-', marker='o', ax=ax, ms=5)
        # same ax as above since it's automatically added on the right
        ax_twin = df[mlh_site.name].plot(style='r-', marker='s', secondary_y=True, ax=ax, ms=5)
        if interpolate is not None:
            ax.legend(*[ax.get_lines() + ax.right_ax.get_lines()],
                       ['PW {} max interpolation'.format(interpolate), 'PW',
                        'MLH {} max interpolation'.format(interpolate), 'MLH'])
        else:
            ax.legend([ax.get_lines()[0], ax.right_ax.get_lines()[0]],
                       ['PW','MLH'])
        ax.set_title('MLH {} site and PW {} site'.format(pw_mlh_dict.get(site),site))
        ax.set_xlim(df.dropna().index.min(), df.dropna().index.max())
        ax.set_ylabel('PW [mm]')
        ax_twin.set_ylabel('MLH [m]')
        ax.grid(True, which='both', axis='x')
        fig.tight_layout()
    if interpolate is not None:
        ds = df_inter.to_xarray()
        ds[pw_new.name].attrs.update(pw_new.attrs)
        ds[mlh_site.name].attrs.update(mlh_site.attrs)
        return ds
    else:
        ds = df.to_xarray()
        ds[pw_new.name].attrs.update(pw_new.attrs)
        ds[mlh_site.name].attrs.update(mlh_site.attrs)
        return ds


def plot_pw_mlh(path=work_yuval, ceil_path=ceil_path, kind='scatter', month=None,
                ceil_interpolate=None):
    """use ceil_interpolate as  {'TLV': '6H'}, 6H being the map_gap overwhich
    to interpolate"""
    import xarray as xr
    import matplotlib.pyplot as plt
    mlh = xr.load_dataset(ceil_path / 'MLH_from_ceilometers.nc')
    if ceil_interpolate is not None:
        for site, max_gap in ceil_interpolate.items():
            print('interpolating ceil-site {} with max-gap of {}.'.format(site, max_gap))
            attrs = mlh[site].attrs
            mlh[site] = mlh[site].interpolate_na('time', max_gap=max_gap,
                                                 method='cubic')
            mlh[site].attrs = attrs
    pw = xr.load_dataset(work_yuval / 'GNSS_PW_thresh_50_homogenized.nc')
    pw = pw[[x for x in pw if '_error' not in x]]
    pw = pw[['tela', 'klhv', 'jslm', 'nzrt', 'yrcm']]
    couples = [['tela', 'TLV'], ['yrcm', 'NV'], ['jslm', 'JR']]
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
                pw[couples[i][0]], mlh[couples[i][1]], month=month, ax=ax, title=False)
        for i, ax in enumerate(axes[:, 1].flatten()):
            ax = scatter_plot_pw_mlh(pw[couples[i][0]],
                                     mlh[couples[i][1]],
                                     diurnal=True,
                                     month=month,
                                     ax=ax,
                                     title=False,
                                     leg_loc='lower right')
    fig.tight_layout()
    return fig


def scatter_plot_pw_mlh(pw, mlh, diurnal=False, ax=None, title=True,
                        leg_loc='best', month=None):
    from aux_gps import dim_intersection
    import xarray as xr
    import numpy as np
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    from PW_stations import produce_geo_gnss_solved_stations
    df = produce_geo_gnss_solved_stations(plot=False)
    pw_alt = df.loc[pw.name, 'alt']
    pw_attrs = pw.attrs
    mlh_attrs = mlh.attrs
    if diurnal:
        if month is not None:
            pw = pw.sel(time=pw['time.month'] == month)
        else:
            newtime = dim_intersection([pw, mlh], 'time')
            pw = pw.sel(time=newtime)
            mlh = mlh.sel(time=newtime)
        pw = pw.groupby('time.hour').mean()
        pw.attrs = pw_attrs
        mlh = mlh.groupby('time.hour').mean()
        mlh.attrs = mlh_attrs
    else:
        newtime = dim_intersection([pw, mlh], 'time')
        pw = pw.sel(time=newtime)
        mlh = mlh.sel(time=newtime)
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
#    from PW_stations import produce_geo_gnss_solved_stations
#    df = produce_geo_gnss_solved_stations(plot=False)
    # first run multi-year month mean:
    if month is not None:
        pw = pw.sel(time=pw['time.month'] == month).dropna('time')
    else:
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
    mlhyears = [mlh.time.dt.year.min().item(), mlh.time.dt.year.max().item()]
    pwyears = [pw.time.dt.year.min().item(), pw.time.dt.year.max().item()]
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
    pw_label = 'PW: {}-{}, {} ({} pts)'.format(pwyears[0], pwyears[1], month_abbr[mlh_month], pw.size)
    mlh_label = 'MLH: {}-{}, {} ({} pts)'.format(mlhyears[0], mlhyears[1], month_abbr[mlh_month], mlh.dropna('time').size)
#    if month is not None:
#        pwmln = pw_m_hour.plot(color='tab:orange', marker='^', ax=ax)
#        pwm_label = 'PW: {}-{}, {} ({} pts)'.format(pw_years[0], pw_years[1], month_abbr[month], pw_month.dropna('time').size)
#        ax.legend(pwln + mlhln + pwmln, [pw_label, mlh_label, pwm_label], loc=leg_loc)
#    else:
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
    return ax, twin


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


def shift_half_hour_lst(hours_back=3):
    import numpy as np
    hour1 = np.arange(24 - hours_back, 24, 0.5)
    hour2 = np.arange(0, 24-hours_back, 0.5)
    hour = np.append(hour1, hour2)
    return hour


def read_coastal_BL_levi_2011(path=ceil_path):
    import pandas as pd
    """Attached profiler data for the average diurnal boundary layers height 3
    km form the coast of Hadera for the 3 summers of 1997-1997.
    The data for July is in the tab  hour_july where MAX SNR is the height of
    the wind profiler signal-to-noise ratio peak. The wind profiler high
    signal-to-noise ratio is obtained near the BL top at the entrainment zone
    where inhomogeneities  due mixing of dry and humid air produce high values
    radar reflectivity.
    The Tv inversion is the inversion height of the virtual
    temperature profile measure by the wind profiler radio acoustic sounding
    system (RASS).
    The tab SNR JJAS has the diurnal boundary height at June, July, August and
    September as measured by the MAX SNR."""
    # read july 1997-1999 data:
    df_july = pd.read_excel(path/'coastal_BL_levi_2011.xls', sheet_name='hour_july')
    hour = shift_half_hour_lst(2)
    df_july.set_index(hour, inplace=True)
    df_july = df_july.sort_index()
    df_july.drop('hour', axis=1, inplace=True)
    df_july.columns = ['n_maxsnr', 'maxsnr', 'std_maxsnr', 'stderror_maxsnr', 'tv_inversion', 'std_tv200']
    # read 4 months data:
    df_JJAS = pd.read_excel(path/'coastal_BL_levi_2011.xls', sheet_name='SNR JJAS')
    df_JJAS.set_index(hour, inplace=True)
    df_JJAS = df_JJAS.sort_index()
    df_JJAS.drop('hour', axis=1, inplace=True)
    df = pd.concat([df_july, df_JJAS], axis=1)
    return df

    