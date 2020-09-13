#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:41:11 2020

@author: shlomi
"""
from PW_paths import work_yuval
from PW_paths import savefig_path
ceil_path = work_yuval / 'ceilometers'
# available stations: Jerousalem, Nevatim, Ramat_David, Tel_Aviv
stations_dict = {
    'Tel_Aviv': ['TLV', 34.8, 32.1, 5],
    'Nevatim': ['NV', 34.9, 31.2, 400],
    'Ramat_David': ['RD', 35.2, 32.7, 50],
    'Jerusalem': ['JR', 35.2, 31.8, 830]}

pw_mlh_dict = {'tela': 'TLV', 'yrcm': 'NV', 'jslm': 'JR', 'nzrt': 'RD'}


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
    pw = xr.open_dataset(work_yuval / 'GNSS_PW_thresh_50_homogenized.nc')
    pw = pw[['tela', 'klhv', 'jslm', 'nzrt', 'yrcm']]
    pw.load()
    pw_new = pw[site]
    if interpolate is not None:
        newtime = dim_intersection([pw_new, mlh_site_inter])
    else:
        newtime = dim_intersection([pw_new, mlh_site])
    pw_new = pw_new.sel(time=newtime)
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
                       ['PWV {} max interpolation'.format(interpolate), 'PWV',
                        'MLH {} max interpolation'.format(interpolate), 'MLH'],
                        loc='best')
        else:
            ax.legend([ax.get_lines()[0], ax.right_ax.get_lines()[0]],
                       ['PWV','MLH'], loc='upper center')
        ax.set_title('MLH {} site and PWV {} site'.format(pw_mlh_dict.get(site),site))
        ax.set_xlim(df.dropna().index.min(), df.dropna().index.max())
        ax.set_ylabel('PWV [mm]', color='b')
        ax_twin.set_ylabel('MLH [m]', color='r')
        ax.tick_params(axis='y', colors='b')
        ax_twin.tick_params(axis='y', colors='r')
        ax.grid(True, which='both', axis='x')
        fig.tight_layout()
        if interpolate is not None:
            filename = '{}-{}_{}_time_series_{}_max_gap_interpolation.png'.format(
                site, pw_mlh_dict.get(site), dt_range_str, interpolate)
        else:
            filename = '{}-{}_{}_time_series.png'.format(site, pw_mlh_dict.get(site), dt_range_str)
        plt.savefig(savefig_path / filename, orientation='portrait')
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
    couples = [['tela', 'TLV'], ['yrcm', 'NV'], ['jslm', 'JR'], ['nzrt', 'RD']]
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
                pw[couples[i][0]], mlh[couples[i][1]], month=month, ax=ax, title=False, unit='days')
        for i, ax in enumerate(axes[:, 1].flatten()):
            ax = scatter_plot_pw_mlh(pw[couples[i][0]],
                                     mlh[couples[i][1]],
                                     diurnal=True,
                                     month=month,
                                     ax=ax,
                                     title=False,
                                     leg_loc='lower right')
    fig.tight_layout()
    if ceil_interpolate is not None:
        filename = 'PW-MLH_{}_interpolate_max_gap_6H.png'.format(kind)
    else:
        filename = 'PW-MLH_{}.png'.format(kind)
    plt.savefig(savefig_path / filename, orientation='portrait')
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
    ax.set_xlabel('PWV [mm]')
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
                          leg_loc='best', unit='pts', sample_rate=24,
                          fontsize=14):
    from aux_gps import dim_intersection
    import matplotlib.pyplot as plt
    from calendar import month_abbr
#    from PW_stations import produce_geo_gnss_solved_stations
#    df = produce_geo_gnss_solved_stations(plot=False)
    # first run multi-year month mean:
    if month is not None:
        pw = pw.sel(time=pw['time.month'] == month).dropna('time')
        mlh = mlh.sel(time=mlh['time.month'] == month).dropna('time')
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
    if unit == 'pts':
        pw_pts = pw.dropna('time').size
        mlh_pts = mlh.dropna('time').size
    elif unit == 'days':
        pw_pts = int(pw.dropna('time').size / sample_rate)
        mlh_pts = int(mlh.dropna('time').size / sample_rate)
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
    if month is None:
        pw_label = 'PWV: {}-{} ({} {})'.format(pwyears[0], pwyears[1], pw_pts, unit)
        mlh_label = 'MLH: {}-{} ({} {})'.format(mlhyears[0], mlhyears[1], mlh_pts, unit)
    else:
        pw_pts = int(pw.dropna('time').size / 288)
        pw_label = 'PWV: {}-{}, {} ({} {})'.format(pwyears[0], pwyears[1], month_abbr[mlh_month], pw_pts, unit)
        mlh_label = 'MLH: {}-{}, {} ({} {})'.format(mlhyears[0], mlhyears[1], month_abbr[mlh_month], mlh_pts, unit)
#    if month is not None:
#        pwmln = pw_m_hour.plot(color='tab:orange', marker='^', ax=ax)
#        pwm_label = 'PW: {}-{}, {} ({} pts)'.format(pw_years[0], pw_years[1], month_abbr[month], pw_month.dropna('time').size)
#        ax.legend(pwln + mlhln + pwmln, [pw_label, mlh_label, pwm_label], loc=leg_loc)
#    else:
    ax.legend(pwln + mlhln, [pw_label, mlh_label], loc=leg_loc)
    ax.tick_params(axis='y', colors=blue, labelsize=fontsize)
    twin.tick_params(axis='y', colors=red, labelsize=fontsize)
    ax.set_ylabel('PWV [mm]', color=blue, fontsize=fontsize)
    twin.set_ylabel('MLH [m]', color=red, fontsize=fontsize)
    ax.set_xticks([x for x in range(24)])
    ax.set_xlabel('Hour of day [UTC]', fontsize=fontsize)
    mlh_name = mlh.attrs['station_full_name'].replace('_', '-')
    textstr = '{}, {}'.format(mlh_name, pw.name.upper())
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment='top', bbox=props)
    if title:
        ax.set_title('The diurnal cycle of {} Mixing Layer Height and {} GNSS site PWV'.format(mlh_name, pw.name.upper()))
    return ax, twin


def twin_hourly_mean_with_diurnal_mlh_plot(pw, mlh, month=None, ax=None,
                                           title=True, leg_loc='best',
                                           mlh_name='MLH', unit='days',
                                           mlh_station_name='Hadera'):
    import matplotlib.pyplot as plt
    from calendar import month_abbr
    mlh_std = mlh['{}_std'.format(mlh_name)]
    mlh_count = mlh['{}_count'.format(mlh_name)].mean().item()
    mlh_hour = mlh['{}_mean'.format(mlh_name)]
    pw_hour = pw.groupby('time.hour').mean()
    pw_std = pw.groupby('time.hour').std()
    pw_hour_plus = (pw_hour + pw_std).values
    pw_hour_minus = (pw_hour - pw_std).values
    if month is not None:
        pw = pw.sel(time=pw['time.month'] == month).dropna('time')
#    mlh_hour = mlh.groupby('time.hour').mean()
#    mlh_std = mlh.groupby('time.hour').std()
    mlh_hour_minus = (mlh_hour - mlh_std).values
    mlh_hour_plus = (mlh_hour + mlh_std).values
#    mlhyears = [mlh.time.dt.year.min().item(), mlh.time.dt.year.max().item()]
    pwyears = [pw.time.dt.year.min().item(), pw.time.dt.year.max().item()]
#    mlh_month = mlh.time.dt.month.to_dataframe()['month'].value_counts().index[0]
    if unit == 'pts':
        pw_pts = pw.dropna('time').size
        mlh_pts = mlh_count * 48
    elif unit == 'days':
        pw_pts = int(pw.dropna('time').size / 288)
        mlh_pts = int(mlh_count)
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
    twin.fill_between(mlh_hour['half_hour'].values, mlh_hour_minus, mlh_hour_plus, color=red, alpha=0.5)
#        handles, labels = ax.get_legend_handles_labels()
#        handles = [h[0] for h in handles]
#        handles1, labels1 = twin.get_legend_handles_labels()
#        handles1 = [h[0] for h in handles1]
#        hand = handles + handles1
#        labs = labels + labels1
    if month is None:
        pw_label = 'PWV: {}-{}, ({} {})'.format(pwyears[0], pwyears[1], pw_pts, unit)
        mlh_label = 'MLH: ({} {})'.format(mlh_pts, unit)
    else:
        pw_label = 'PWV: {}-{}, {} ({} {})'.format(pwyears[0], pwyears[1], month_abbr[month], pw_pts, unit)
        mlh_label = 'MLH: ({} {})'.format(mlh_pts, unit)
#    if month is not None:
#        pwmln = pw_m_hour.plot(color='tab:orange', marker='^', ax=ax)
#        pwm_label = 'PW: {}-{}, {} ({} pts)'.format(pw_years[0], pw_years[1], month_abbr[month], pw_month.dropna('time').size)
#        ax.legend(pwln + mlhln + pwmln, [pw_label, mlh_label, pwm_label], loc=leg_loc)
#    else:
    ax.legend(pwln + mlhln, [pw_label, mlh_label], loc=leg_loc)
    ax.tick_params(axis='y', colors=blue)
    twin.tick_params(axis='y', colors=red)
    ax.set_ylabel('PWV [mm]', color=blue)
    twin.set_ylabel('MLH [m]', color=red)
    ax.set_xticks([x for x in range(24)])
    ax.set_xlabel('Hour of day [UTC]')
    try:
        mlh_name = mlh.attrs['station_full_name'].replace('_', '-')
    except KeyError:
        mlh_name = mlh_station_name
    textstr = '{}, {}'.format(mlh_name, pw.name.upper())
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    if title:
        ax.set_title('The diurnal cycle of {} Mixing Layer Height and {} GNSS site PWV'.format(mlh_name, pw.name.upper()))
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


def read_BD_ceilometer_yoav_one_year_csv(file):
    import pandas as pd
    feet_to_m = 0.3048
    df = pd.read_csv(file, index_col='Date_Time[]', na_values='-999.0')
    df.index.name = 'time'
    octet_cols = [x for x in df.columns if 'octet' in x]
    df[octet_cols] = df[octet_cols].fillna(0)
    df[octet_cols] = df[octet_cols].astype(int)
    feet_cols = [x for x in df.columns if 'feet' in x]
    df[feet_cols] = df[feet_cols].mul(feet_to_m)
    cols = [x for x in df.columns]
    df.columns = [x.replace('feet', 'm') for x in cols]
    df.index = pd.to_datetime(df.index) - pd.Timedelta(2, unit='H')
    return df


def read_BD_ceilometer_yoav_all_years(path=ceil_path, savepath=None):
    from aux_gps import path_glob
    from aux_gps import save_ncfile
    import pandas as pd
    files = path_glob(path, 'ceilometer_BD*.csv')
    dfs = []
    for file in files:
        dfs.append(read_BD_ceilometer_yoav_one_year_csv(file))
    df = pd.concat(dfs)
    df = df.sort_index()
    names = [x.split('[')[0] for x in df.columns]
    units = [x.split('[')[1].split(']')[0] for x in df.columns]
    long_names = [
        'total cloud cover',
        'cloud cover of the most cloudy layer',
        'cloud cover of the 1st cloud layer',
        '1st cloud base height',
        'cloud cover of the 2nd cloud layer',
        '2nd cloud base height',
        'cloud cover of the 3rd cloud layer',
        '3rd cloud base height',
        'cloud cover of the 4th cloud layer',
        '4th cloud base height',
        'cloud cover of the 5th cloud layer',
        '5th cloud base height',
        'Mixing layer height']
    df.columns = names
    # fix cloud height to meters again for until 22-09-2013:
    hs = [x for x in df.columns if '_H' in x]
    df.loc[:'2013-09-22', hs] *= (1 / 0.3048)
    ds = df.to_xarray()
    for i, da in enumerate(ds):
        ds[da].attrs['units'] = units[i]
        ds[da].attrs['long_name'] = long_names[i]
    if savepath is not None:
        filename = 'BD_clouds_and_MLH_from_ceilometers.nc'
        save_ncfile(ds, savepath, filename)
    return ds


def plot_mlh_site_pw_station(ceil_path=ceil_path, path=work_yuval,
                             station='tela', mlh_site='BD', selection='syn',
                             max_gap_interpolate=None, srate=24):
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    month = None
    if mlh_site == 'BD':
        bd = read_BD_matfile(path=ceil_path, plot=False, add_syn=True)
        if selection == 'syn':
            # select PT's and High as synoptics:
            bd = bd['BD'].where((bd['syn'] == 'PT-W') | (bd['syn']
                                                         == 'PT-M') | (bd['syn'] == 'H_w'))
            print('selected synoptics.')
        elif isinstance(selection, int):
            # select all data for specific month:
            month = selection
            bd = bd['BD']
            print('selected month {}.'.format(selection))
        elif isinstance(selection, str) and selection.isupper():
            # select season:
            bd = bd['BD'].sel(time=bd['time.season'] == selection)
            print('selected season {}.'.format(selection))
        else:
            # select all data:
            bd = bd['BD']
        mlh = bd
    else:
        mlh = xr.load_dataset(ceil_path / 'MLH_from_ceilometers.nc')[mlh_site]
        if max_gap_interpolate is not None:
            print('interpolating ceil-site {} with max-gap of {}.'.format(mlh_site, max_gap_interpolate))
            attrs = mlh.attrs
            mlh = mlh.interpolate_na('time', max_gap=max_gap_interpolate,
                                     method='cubic')
            mlh.attrs = attrs
    pw = xr.open_dataset(path / 'GNSS_PW_thresh_50_for_diurnal_analysis.nc')[station]
    ax, twin = twin_hourly_mean_plot(pw, mlh, month=month, title=True, unit='days', sample_rate=srate)
    ax.grid()
    pw_data = ax.get_lines()[0].get_ydata()
    pwc = np.mean(pw_data)
    off = 9
    ax.vlines(2.75, ymin=pwc-off, ymax=pwc+off, color='k')
    ax.vlines(16.75, ymin=pwc-off, ymax=pwc+off, color='k')
    ax.text(x=1.5, y=pwc-off-0.5, s='mean sunrise')
    ax.text(x=15.5, y=pwc-off-0.5, s='mean sunset')
    ax.figure.tight_layout()
    if max_gap_interpolate is not None:
        title = ax.get_title()
        ax.set_title(title + '({} max gap cubic interpolation)'.format(max_gap_interpolate))
    if selection is None:
        filename = '{}-{}_max_gap_{}.png'.format(station, mlh_site, max_gap_interpolate)
    else:
        filename = '{}-{}_{}_max_gap_{}.png'.format(station, mlh_site, selection, max_gap_interpolate)
    plt.savefig(savefig_path / filename, orientation='portrait')
    return ax


def plot_profiler_hadera_pw(ceil_path=ceil_path, path=work_yuval,
                            station='csar', selection='JJA'):
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    ds = read_profiler_hadera(path=ceil_path, plot=False)
    pw = xr.open_dataset(path / 'GNSS_PW_thresh_50_for_diurnal_analysis.nc')[station]
    pw.load()
    month = None
    if isinstance(selection, int):
        month = selection
        add_title = ''
    elif isinstance(selection, str) and selection.isupper():
        pw = pw.sel(time=pw['time.season'] == selection).dropna('time')
        add_title = ' ({})'.format(selection)
    ax, twin = twin_hourly_mean_with_diurnal_mlh_plot(pw, ds, month=month, title=True)
    ax.grid()
    pw_data = ax.get_lines()[0].get_ydata()
    pwc = np.mean(pw_data)
    off = 5
    ax.vlines(2.75, ymin=pwc-off, ymax=pwc+off, color='k')
    ax.vlines(16.75, ymin=pwc-off, ymax=pwc+off, color='k')
    ax.text(x=1.5, y=pwc-off-0.5, s='mean sunrise')
    ax.text(x=15.5, y=pwc-off-0.5, s='mean sunset')
    title = ax.get_title()
    ax.set_title(title + add_title)
    ax.figure.tight_layout()
    filename = 'csar-HD_{}.png'.format(selection)
    plt.savefig(savefig_path / filename, orientation='portrait')
    return ax


def read_BD_matfile(path=ceil_path, plot=True, month=None, add_syn=True):
    from scipy.io import loadmat
    import pandas as pd
    from aux_gps import xr_reindex_with_date_range
    import matplotlib.pyplot as plt
    from aux_gps import dim_intersection
    from synoptic_procedures import read_synoptic_classification
    file = path / 'PBL_BD_LST.mat'
    mat = loadmat(file)
    mdata = mat['pblBD4shlomi']
    # mdata = mat['PBL_BD_LST']
    dates = mdata[:, :3]
    pbl = mdata[:, 3:]
    dates = dates.astype(str)
    dts = [pd.to_datetime(x[0] + '-' + x[1] + '-' + x[2]) for x in dates]
    dfs = []
    for i, dt in enumerate(dts):
        time = dt + pd.Timedelta(0.5, unit='H')
        times = pd.date_range(time, periods=48, freq='30T')
        df = pd.DataFrame(pbl[i], index=times)
        dfs.append(df)
    df = pd.concat(dfs)
    df.columns = ['MLH']
    df.index.name = 'time'
    # switch to UTC:
    df.index = df.index - pd.Timedelta(2, unit='H')
    da = df.to_xarray()['MLH']
    da.name = 'BD'
    da.attrs['full_name'] = 'Mixing Layer Height'
    da.attrs['name'] = 'MLH'
    da.attrs['units'] = 'm'
    da.attrs['station_full_name'] = 'Beit Dagan'
    da.attrs['lon'] = 34.81
    da.attrs['lat'] = 32.00
    da.attrs['alt'] = 34
    da = xr_reindex_with_date_range(da, freq='30T')
    # add synoptic data:
    syn = read_synoptic_classification().to_xarray()
    syn = syn.sel(time=slice('2015', '2016'))
    syn = syn.resample(time='30T').ffill()
    new_time = dim_intersection([da, syn])
    syn_da = syn.sel(time=new_time)
    syn_da = xr_reindex_with_date_range(syn_da, freq='30T')
    if plot:
        bd2015 = da.sel(time='2015').to_dataframe()
        bd2016 = da.sel(time='2016').to_dataframe()
        fig, axes = plt.subplots(2, 1, sharey=True, sharex=False,
                                 figsize=(15, 10))
        if add_syn:
            cmap = plt.get_cmap("tab10")
            syn_df = syn_da.to_dataframe()
            bd2015['synoptics'] = syn_df.loc['2015', 'class_abbr']
            groups = []
            for i, (index, group) in enumerate(bd2015.groupby('synoptics')):
                groups.append(index)
                d = xr_reindex_with_date_range(group['BD'].to_xarray(),
                                               freq='30T')
                d.to_dataframe().plot(x_compat=True, ms=10, color=cmap(i),
                                      ax=axes[0], xlim=['2015-06', '2015-10'])
            axes[0].legend(groups)
            bd2016['synoptics'] = syn_df.loc['2016', 'class_abbr']
            groups = []
            for i, (index, group) in enumerate(bd2016.groupby('synoptics')):
                groups.append(index)
                d = xr_reindex_with_date_range(group['BD'].to_xarray(),
                                               freq='30T')
                d.to_dataframe().plot(x_compat=True, ms=10, color=cmap(i),
                                      ax=axes[1], xlim=['2016-06', '2016-10'])
            axes[1].legend(groups)
        else:
            bd2015.plot(ax=axes[0], xlim=['2015-06', '2015-10'])
            bd2016.plot(ax=axes[1], xlim=['2016-06', '2016-10'])
        for ax in axes.flatten():
            ax.set_ylabel('MLH [m]')
            ax.set_xlabel('UTC')
            ax.grid()
        fig.tight_layout()
        fig.suptitle('MLH from Beit-Dagan ceilometer for 2015 and 2016')
        filename = 'MLH-BD_syn.png'
        plt.savefig(savefig_path / filename, orientation='portrait')
    if add_syn:
        ds = da.to_dataset(name='BD')
        ds['syn'] = syn_da['class_abbr']
        return ds
    else:
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


def read_profiler_hadera(path=ceil_path, plot=True):
    import pandas as pd
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt

    def read_hadera_synoptical(path, syn='Hw'):
        df = pd.read_excel(path / 'PBL_profiler_hadera.xlsx', sheet_name=syn).T
        df.columns = ['{}_mean'.format(syn), '{}_count'.format(syn),
                      '{}_median'.format(syn), '{}_std'.format(syn)]
        df.drop('Time (LST)', inplace=True)
        df.set_index(np.arange(0, 24, 0.5), inplace=True)
        hour = shift_half_hour_lst(2)
        df.set_index(hour, inplace=True)
        df = df.sort_index()
        df.index.name = 'half_hour'
        df = df.apply(pd.to_numeric)
        ds = df.to_xarray()
        return ds
    ds_hw = read_hadera_synoptical(path=path, syn='Hw')
    ds_ptw = read_hadera_synoptical(path=path, syn='PTw')
    ds_ptm = read_hadera_synoptical(path=path, syn='PTm')
    ds = xr.merge([ds_hw, ds_ptw, ds_ptm])
    mlh_mean = ds[['Hw_mean', 'PTw_mean', 'PTm_mean']].to_array('syn').mean('syn')
    mlh_std = ds[['Hw_std', 'PTw_std', 'PTm_std']].to_array('syn').mean('syn')
    mlh_count = ds[['Hw_count', 'PTw_count', 'PTm_count']].to_array('syn').sum('syn')
    ds['MLH_mean'] = mlh_mean
    ds['MLH_std'] = mlh_std
    ds['MLH_count'] = mlh_count
    if plot:
        fig, ax = plt.subplots(figsize=(12, 8))
        means = ds[[x for x in ds if '_mean' in x]]
        counts = ds[[x for x in ds if '_count' in x]]
        stds = ds[[x for x in ds if '_std' in x]]
        dfm = means.to_dataframe()
        cmeans = [x for x in counts.mean().to_array().values]
        cmeans.append(np.sum(cmeans))
        dfm.plot(ax=ax, style=['rd-','bo-','gX-','ks-'], markevery=2)
        ax.xaxis.set_ticks(np.arange(0, 24, 1))
        ax.set_xlabel('Hour of Day [UTC]')
        ax.grid()
        ax.set_ylabel('PBL height AGL [m]')
        labels = ['High-West: {:.0f} mean days'.format(cmeans[0])]
        labels.append('PT-Weak: {:.0f} mean days'.format(cmeans[1]))
        labels.append('PT-Medium: {:.0f} mean days'.format(cmeans[2]))
        labels.append('Simple Mean: {:.0f} mean days'.format(cmeans[3]))
        ax.legend(labels)
        ax.set_title('PBL average Height from 3.5 km east of the coast of Hadera (ORPP), between June - October, 1997-1999, 2002-2005')
        ax.fill_betweenx(y=[400, 800], x1=2.75, x2=16.75, color='y', alpha=0.5)
        fig.tight_layout()
        filename = 'MLH_HD_diurnal_syn.png'
        plt.savefig(savefig_path / filename, orientation='portrait')
    return ds


def shift_half_hour_lst(hours_back=3):
    import numpy as np
    hour1 = np.arange(24 - hours_back, 24, 0.5)
    hour2 = np.arange(0, 24-hours_back, 0.5)
    hour = np.append(hour1, hour2)
    return hour


def read_coastal_BL_levi_2011(path=ceil_path):
    import pandas as pd
    """Attached profiler data for the average diurnal boundary layers height 3
    km form the coast of Hadera for the 3 summers of 1997-1999.
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


def convert_to_numeric(large_string):
    import numpy as np
    s = large_string.strip()
    ss = [s[i:i + 5] for i in range(0, len(s), 5)]
    sint = [int(x, 16) for x in ss]
    sint = np.array(sint, dtype=np.int32)
    # correction:
    corr = sint > 2**19
    if corr.any():
        sint[corr] = -(2 ** 20 - sint[corr])
    return sint


def read_his_file(hfile):
    import pandas as pd
    import xarray as xr
    import numpy as np
    df = pd.read_csv(hfile, header=1)
    df.columns = [x.strip() for x in df.columns]
    df['profile'] = df['BS_PROFILE'].apply(convert_to_numeric)
    df.set_index(pd.to_datetime(df['CREATEDATE']), inplace=True)
    df.drop(['CREATEDATE', 'UNIXTIME', 'CEILOMETER', 'BS_PROFILE', 'PERIOD'],
            axis=1, inplace=True)
    df.index.name = 'time'
    vals = [df.values[x][0] for x in range(df.size)]
    da = xr.DataArray(vals, dims=['time', 'range'])
    da['time'] = df.index
    da['range'] = np.arange(10, 4510, 10)
    da = da.astype(np.float32)
    ds = da.to_dataset(name='rcs_0')
    ds['rcs_0'].attrs['long_name'] = 'normalized range corrected signal'
    ds['rcs_0'].attrs['units'] = '1e-8 sr^-1.m^-1'
    ds['range'].attrs['long_name'] = 'range'
    ds['range'].attrs['units'] = 'm'
    return ds


def compare_cloud_H1_to_BD_MLH_and_PW(path=ceil_path, pwv_path=work_yuval,
                                      plot=True):
    import xarray as xr
    import numpy as np
    from PW_from_gps_figures import plot_mean_std_count
    # load cloud height 1 from BD ceilometers:
    cld = read_BD_ceilometer_yoav_all_years(path=path)['cloud_H1']
    # load leenes MLH from BD:
#    bd_mlh = read_BD_matfile(plot=False, add_syn=False)
#    ds = bd_mlh.to_dataset(name='MLH')
#    ds['cloud_H1'] = cld
    # load PWV from TELA daily anomalies:
    ds = cld.to_dataset(name='cloud_H1')
    pwv = xr.open_dataset(
        pwv_path /
        'GNSS_PW_thresh_50_for_diurnal_analysis_removed_daily.nc')['tela']
    pwv = pwv.sel(time=slice('2015', '2016'))
    pwv.load()
    df = ds.to_dataframe()
    df['PWV_TELA'] = pwv.to_dataframe()
    df.index.name = 'time'
    # slice for only summer 2015:
    df = df.loc['2015-06':'2015-09']
    if plot:
        ax = df.plot(ylim=(0, 2000), secondary_y='PWV_TELA')
        ds = df.to_xarray()
        axes, ax2 = plot_mean_std_count(ds.dropna('time'), time_reduce='hour', reduce='mean')
        axes[1].set_xlabel('Hour of Day [UTC]')
        axes[1].set_ylabel('Points (10-mins sample rate) [#]')
        axes[0].set_ylabel('Cloud first layer height [m]')
        axes[0].set_title('June to September 2015')
        ax2.set_ylabel('TELA PWV anomalies [mm]')
        axes[0].figure.subplots_adjust(top=0.97,
                                       bottom=0.056,
                                       left=0.051,
                                       right=0.967,
                                       hspace=0.064,
                                       wspace=0.2)
    return df
