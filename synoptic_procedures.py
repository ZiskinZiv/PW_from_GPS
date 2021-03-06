#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:46:44 2020

@author: ziskin
"""

from pathlib import Path
from PW_paths import work_yuval
climate_path = work_yuval / 'climate'
#cwd = Path().cwd()


def choose_color_for_synoptic_classification():
    # import numpy as np
    # import seaborn as sns
    # from matplotlib.cm import tab20b
    # from matplotlib.cm import tab20c
    # from palettable.colorbrewer import sequential as seq
    from PW_from_gps_figures import create_enhanced_qualitative_color_map
    colors = create_enhanced_qualitative_color_map(plot=False)
    # syn_cls = np.arange(1, 20)
    # colors = sns.color_palette('tab20', 19)
    col_dict = {}
    edge_dict = {}
    east_color = colors[8]  # green
    west_color = colors[0]   # blue
    north_color = colors[28]  # gray
    south_color = colors[12]  # dark orange
    central_color = [0, 0, 0, 1]  # black
    # no_color = [0, 0, 1, 1]  # white
    RST_color = colors[4]  # light orange
    High_color = colors[16]  # purple
    cold_low_color = colors[39]
    sharav_color = colors[35]
    # RST colors:
    col_dict[1] = RST_color
    col_dict[2] = RST_color
    col_dict[3] = RST_color
    edge_dict[1] = east_color
    edge_dict[2] = west_color
    edge_dict[3] = central_color
    # PT colors:
    col_dict[4] = colors[10]
    col_dict[5] = colors[9]
    col_dict[6] = colors[8]
    edge_dict[4] = central_color
    edge_dict[5] = central_color
    edge_dict[6] = central_color
    # High colors:
    col_dict[7] = High_color
    col_dict[8] = High_color
    col_dict[9] = High_color
    col_dict[10] = High_color
    edge_dict[7] = east_color
    edge_dict[8] = west_color
    edge_dict[9] = north_color
    edge_dict[10] = central_color
    # low east deep:
    col_dict[11] = colors[28]
    edge_dict[11] = east_color
    # CL lows:
    col_dict[12] = colors[36]
    col_dict[13] = colors[38]
    col_dict[14] = colors[36]
    col_dict[15] = colors[38]
    edge_dict[12] = south_color
    edge_dict[13] = south_color
    edge_dict[14] = north_color
    edge_dict[15] = north_color
    # cold low west:
    col_dict[16] = cold_low_color
    edge_dict[16] = west_color
    # low east shallow:
    col_dict[17] = colors[29]
    edge_dict[17] = east_color
    # sharav Lows:
    col_dict[18] = sharav_color
    col_dict[19] = sharav_color
    edge_dict[18] = west_color
    edge_dict[19] = central_color
    return col_dict, edge_dict


def visualize_synoptic_class_on_time_series(da_ts, path=climate_path,
                                            ax=None, leg_ncol=1, add_mm=False,
                                            leg_loc=1, second_da_ts=None,
                                            twin=None):
    import xarray as xr
    import matplotlib.pyplot as plt
    from aux_gps import replace_time_series_with_its_group
    time_dim = list(set(da_ts.dims))[0]
    assert xr.infer_freq(da_ts[time_dim]) == 'D'
    if ax is None:
        fig, ax = plt.subplots()
    # also calc the monthly means:
    if add_mm:
        da_ts_mm = replace_time_series_with_its_group(da_ts, grp='month')
        da_ts_mm.plot.line('k-.', ax=ax)
    if isinstance(da_ts, xr.Dataset):
        styles = ['r-', 'g-', 'b-']
        lns = []
        for i, st in enumerate(da_ts):
            lbl = st.upper()
            ln = da_ts[st].plot.line(styles[i], lw=2, ax=ax, zorder=20, label=lbl)
            lns.append(ln)
        da_ts = da_ts[st]
    else:
        # plot daily values:
        da_ts.plot.line('k-', lw=2, ax=ax, zorder=20)
    if second_da_ts is not None:
        # record the corr between second_da_ts and da_ts:
        corr_all = xr.corr(da_ts, second_da_ts).item()
        corr_oct = xr.corr(da_ts.sel(time=da_ts['time.month'] == 10), second_da_ts.sel(
            time=second_da_ts['time.month'] == 10)).item()
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = '\n'.join([
            'r_all = {:.2f}'.format(corr_all),
            'r_just_Oct = {:.2f}'.format(corr_oct)])
        # textstr = 'r_all = {:.2f}'.format(corr)
        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
        try:
            if second_da_ts.attrs['units'] == da_ts.attrs['units']:
                second_da_ts.plot.line('k--', lw=2, ax=ax, marker='o')
            else:
                twinx = ax.twinx()
                second_da_ts.plot.line('k--', lw=2, ax=twinx, marker='o')
                if twin is not None:
                    twinx.set_ylim(*twin)
        except KeyError:
            twinx = ax.twinx()
            second_da_ts.plot.line('k--', lw=2, ax=twinx, marker='o')
            if twin is not None:
                twinx.set_ylim(*twin)
    # ymin, ymax = ax.get_ylim()
    df = read_synoptic_classification(path, report=False)
    ind = da_ts.to_dataframe().index
    da_ts = align_synoptic_class_with_daily_dataset(da_ts)
    df = df.loc[ind]
    color_dict, edge_dict = choose_color_for_synoptic_classification()
#    df['color'] = df['class'].map(color_dict)
    # monthly count of synoptics:
    month_counts = agg_month_count_syn_class(freq=False)
    min_year = da_ts[time_dim].min().dt.year.item()
    min_month = da_ts[time_dim].min().dt.month.item()
    max_year = da_ts[time_dim].max().dt.year.item()
    max_month = da_ts[time_dim].max().dt.month.item()
    min_dt = '{}-{}'.format(min_year, min_month)
    max_dt = '{}-{}'.format(max_year, max_month)
    month_counts = month_counts.sel(time=slice(min_dt, max_dt))
    # alternative count since we need not just monthly but by time slice:
    grp_dict = df.groupby('class').groups
    for key_class, key_ind in grp_dict.items():
        color = color_dict[key_class]
        edge_color = edge_dict[key_class]
        abbr = add_class_abbr(key_class)
        # abbr_count = month_counts.sel(syn_cls=key_class).sum().item()
        abbr_count = df[df['class'] == key_class].count().values[0]
        abbr_label = r'${{{}}}$: {}'.format(abbr, int(abbr_count))
#    for ind, row in df.iterrows():
        da_ts[da_ts['syn_class'] == key_class].plot.line(
            'k-', lw=0, ax=ax, marker='o', markersize=20,
            markerfacecolor=color, markeredgewidth=2,
            markeredgecolor=edge_color, label=abbr_label)
        # ax.vlines(key_ind, 0, 80, colors=color, alpha=0.4, lw=10,
        #           label=abbr_label)
    ax.legend(ncol=leg_ncol, labelspacing=1.5, fontsize=12, loc=leg_loc)
    ax.grid()
    return ax


def val_counts(ser):
    s = ser.value_counts()
    # s=s.drop(0)
    return s[:1].index.to_list()[0]


upper_class_dict = {'RST': [1, 2, 3], 'PT': [4, 5, 6], 'H': [7, 8, 9, 10],
                    'CL': [12, 13, 14, 15], 'DS': [17, 18, 19],
                    'Other': [11, 16]}

def class_to_upper(class_da, class_type='upper1', name='upper_class'):
    import numpy as np
    df = class_da.to_dataframe(name='class')
    if class_type == 'upper1':
        df[name] = np.ones(df['class'].shape) * np.nan
        df.loc[(df['class'] <= 3) & (df['class'] >= 1), name] = 'RST'
        df.loc[(df['class'] <= 6) & (df['class'] >= 4), name] = 'PT'
        df.loc[(df['class'] <= 10) & (df['class'] >= 7), name] = 'H'
        df.loc[(df['class'] <= 15) & (df['class'] >= 12), name] = 'CL'
        df[name] = df[name].fillna('Other')
    da = df.to_xarray()[name]
    return da


def align_synoptic_class_with_daily_dataset(ds, time_dim='time'):
    import xarray as xr
    assert xr.infer_freq(ds[time_dim]) == 'D'
#    ds = ds.resample({time_dim: '1D'}, keep_attrs=True).mean(keep_attrs=True)
    syn = read_synoptic_classification(report=False).to_xarray()
    ds['syn_class'] = syn['class']
    ds['upper_class'] = syn['upper_class']
    return ds


def align_synoptic_class_with_pw(path):
    import xarray as xr
    from aux_gps import dim_intersection
    from aux_gps import save_ncfile
    from aux_gps import xr_reindex_with_date_range
    pw = xr.load_dataset(path / 'GNSS_PW_thresh_50_homogenized.nc')
    pw = pw[[x for x in pw if '_error' not in x]]
    syn = read_synoptic_classification(report=False).to_xarray()
    # syn = syn.drop(['Name-EN', 'Name-HE'])
    syn = syn['class']
    syn = syn.sel(time=slice('1996', None))
    syn = syn.resample(time='5T').ffill()
    ds_list = []
    for sta in pw:
        print('aligning station {} with synoptics'.format(sta))
        new_time = dim_intersection([pw[sta], syn])
        syn_da = xr.DataArray(syn.sel(time=new_time))
        syn_da.name = '{}_class'.format(sta)
        syn_da = xr_reindex_with_date_range(syn_da)
        ds_list.append(syn_da)
    ds = xr.merge(ds_list)
    ds = ds.astype('int8')
    ds = ds.fillna(0)
    filename = 'GNSS_synoptic_class.nc'
    save_ncfile(ds, path, filename)
    return ds


def slice_xr_with_synoptic_class(pw, path=work_yuval, syn_class=1,
                                 plot=False):
    import xarray as xr
    import matplotlib.pyplot as plt
    from aux_gps import rename_data_vars
    name = pw.name
    syn = xr.load_dataset(path / 'GNSS_synoptic_class.nc')
    if isinstance(syn_class, int):
        if isinstance(pw, xr.DataArray):
            syn = syn['{}_class'.format(name)]
        elif isinstance(pw, xr.Dataset):
            syn = rename_data_vars(syn, suffix='_class', remove_suffix=True)
        syn = syn.to_dataframe()['{}_class'.format(name)]
        pw = pw.to_dataframe()
        pw = pw[syn==syn_class]
    elif isinstance(syn_class, str):
        if isinstance(pw, xr.DataArray):
            syn = syn['{}_class'.format(name)]
        elif isinstance(pw, xr.Dataset):
            syn = rename_data_vars(syn, suffix='_class', remove_suffix=True)
        syn = syn.to_dataframe()['{}_class'.format(name)]
        pw = pw.to_dataframe()
        pw = pw[syn.isin(upper_class_dict.get(syn_class))]
    pw = pw.to_xarray()[name]
    if plot:
        pw.plot()
        ax = plt.gca()
        ax.set_title('{} synoptic code selected'.format(syn_class))
    return pw


def add_class_abbr(class_num):
    import numpy as np
    classes = np.arange(1, 20)
    abbrs = ['RST_e', 'RST_w', 'RST_c', 'PT-W', 'PT-M',
             'PT-D', 'H_e', 'H_w', 'H_n', 'H_c', 'L_e-D', 'CL_s-D', 'CL_s-S',
             'CL_n-D', 'CL_n-S', 'L_w', 'L_e-S', 'SL_w', 'SL_c']
    class_abbr_dict = dict(zip(classes, abbrs))
    return class_abbr_dict.get(class_num)


def read_synoptic_classification(
        path=climate_path,
        filename='synoptic_classification_1948-8_May_2020.xls', report=True):
    import pandas as pd
    from aux_gps import path_glob
    import numpy as np
    from aux_gps import invert_dict
    synoptic_filename = path_glob(path, 'synoptic_classification_1948*.xls')
    # read excell:
    df = pd.read_excel(synoptic_filename[0])
    # read last date:
    last_day = df.Day[(df.iloc[:, -1].isnull() == True)
                      ].head(1).values.item() - 1
    last_month = df.Month[(df.iloc[:, -1].isnull() == True)
                          ].head(1).values.item()
    last_year = df.iloc[:, -1].name
    last_date = pd.to_datetime(
        '{}-{}-{}'.format(last_year, last_month, last_day))
    # produce datetime index:
    dt = pd.date_range('1948-01-01', last_date, freq='1D')
    # drop two first cols:
    df.drop(df.columns[0:2], axis=1, inplace=True)
    # melt the df:
    df = df.melt(var_name='year')
    df = df.dropna()
    df.drop('year', axis=1, inplace=True)
    # set new index:
    df.set_index(dt, inplace=True)
    df.columns = ['class']
    df['class'] = df['class'].astype(int)
    # load name table:
    class_df = pd.read_csv(path / 'Isabella_Names for the EM synoptic systems.csv')
    class_df.columns = ['class', 'Name-EN', 'Name-HE']
    class_df.set_index('class', inplace=True)
    # enter the names to df:
    df['Name-EN'] = class_df['Name-EN'].loc[df['class'].values].values
    df['Name-HE'] = class_df['Name-HE'].loc[df['class'].values].values
    # define upper level class:
    d = invert_dict(upper_class_dict)
    df['upper_class'] = df['class'].map(d)
#    df['upper_class'] = np.ones(df['class'].shape) * np.nan
#    df.loc[(df['class'] <= 3) & (df['class'] >= 1), 'upper_class'] = 'RST'
#    df.loc[(df['class'] <= 6) & (df['class'] >= 4), 'upper_class'] = 'PT'
#    df.loc[(df['class'] <= 10) & (df['class'] >= 7), 'upper_class'] = 'H'
#    df.loc[(df['class'] <= 15) & (df['class'] >= 12), 'upper_class'] = 'CL'
#    df['upper_class'] = df['upper_class'].fillna('Other')
    df.index.name = 'time'
    df['class_abbr'] = df['class'].apply(add_class_abbr)
    if report:
        for i, code in enumerate(sorted(df['class'].unique())):
            percent = 100 * (df['class'] == code).sum() / df['class'].size
            name = df['Name-EN'][df['class'] == code].unique().item()
            print('{}) {} : {:.1f} %'.format(i+1, name, percent))
    return df


def find_consecutive_classes(df):
    from aux_gps import find_consecutive_vals_df
    import numpy as np
    import pandas as pd
    classes = np.arange(1, 20, 1)
    sums = []
    for clas in classes:
        df_con = find_consecutive_vals_df(df, col='class', val=clas)
        if df_con.empty:
            sums.append(0)
        else:
            sums.append(df_con['2'][df_con['2'] > 1].sum())
    df_clas = pd.DataFrame(sums, index=classes)
    df_clas.columns = ['consecutive_class_days']
    return df_clas


def agg_month_consecutive_syn_class(path=climate_path, normalize=True):
    import numpy as np
    import pandas as pd
    from aux_gps import save_ncfile
    df = read_synoptic_classification(path=path, report=False)
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['months'] = df['year'].astype(str) + '-' + df['month'].astype(str)
    new_df = df.groupby(df['months']).apply(find_consecutive_classes)
    new_df.columns = ['class_sum']
    new_df = new_df.unstack()
    new_df.columns = np.arange(1, 20, 1)
    dt = pd.to_datetime(new_df.index)
    new_df.set_index(dt, inplace=True)
    new_df = new_df.sort_index()
    new_df.index.name = 'time'
    if normalize:
        new_df = new_df.divide(new_df.index.days_in_month, axis=0)
    da = new_df.to_xarray().to_array('class')
    ds = da.to_dataset(name='consecutive')
    filename = 'GNSS_synoptic_class_consecutive.nc'
    save_ncfile(ds, work_yuval, filename)
    return ds


def agg_month_count_syn_class(path=climate_path, syn_category='normal',
                              freq=True):
    # df.loc['2015-09']['Name-EN'].value_counts()
    import pandas as pd
    if syn_category == 'normal':
        syn_cat = 'class'
    elif syn_category == 'upper':
        syn_cat = 'upper_class'
    df = read_synoptic_classification(path=path, report=False)
    print('used {} synoptic category'.format(syn_category))
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['months'] = df['year'].astype(str) + '-' + df['month'].astype(str)
    new_df = df.groupby([df['months'], df[syn_cat]]).size().to_frame()
    new_df.columns = ['class_sum']
    dfmm = pd.pivot_table(new_df, index='months', columns=syn_cat)
    dfmm.set_index(pd.to_datetime(dfmm.index), inplace=True)
    dfmm.sort_index()
#    dfmm = dfmm.fillna(0)
#    dfmm = dfmm.astype(int)
    dfmm.columns = dfmm.columns.droplevel()
    dfmm.index.name = 'time'
    if freq:
        dfmm /= pd.DataFrame(dfmm.index.days_in_month.values, index=dfmm.index).values
    da = dfmm.to_xarray().to_array('syn_cls')
    da = da.sortby('time')
    da.attrs['units'] = 'counts in a month'
    if freq:
        da.attrs['units'] = 'relative frequency in a month'
    return da


def agg_month_syn_class_continous_variable_with_level(da, level_dim='level',
                                                      path=climate_path,
                                                      syn_cat='RST',
                                                      return_all_syn_cats=True):
    import xarray as xr
    ds_list = []
    for lev in da[level_dim]:
        ds = agg_month_syn_class_continous_variable(da.sel(
            {level_dim: lev}), syn_cat=syn_cat, return_all_syn_cats=return_all_syn_cats)
        ds_list.append(ds)
    dss = xr.concat(ds_list, level_dim)
    dss[level_dim] = da[level_dim]
    return dss

def agg_month_syn_class_continous_variable(
        da, path=climate_path, syn_cat='RST', return_all_syn_cats=False):
    import pandas as pd
    syn_da = align_synoptic_class_with_daily_dataset(da)
    df = syn_da.to_dataframe()
    if isinstance(syn_cat, int):
        df = df.drop('upper_class', axis=1)
    elif isinstance(syn_cat, str):
        df = df.drop('syn_class', axis=1)
        df = df.rename({'upper_class': 'syn_class'}, axis='columns')
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['months'] = df['year'].astype(str) + '-' + df['month'].astype(str)
    # do a mean on syn_class and months:
    new_df = df[da.name].groupby([df['syn_class'], df['months']]).mean()
    new_df = new_df.to_frame('class_mean')
    dfmm = pd.pivot_table(new_df, index='months', columns='syn_class')
    dfmm.set_index(pd.to_datetime(dfmm.index), inplace=True)
    dfmm = dfmm.sort_index()
    dfmm.columns = dfmm.columns.droplevel()
    dfmm.index.name = 'time'
    if return_all_syn_cats:
        da_agg = dfmm.to_xarray().to_array('syn_class')
        da_agg.name = 'syn_classes'
    else:
        da_agg = dfmm.to_xarray()[syn_cat]
    da_agg = da_agg.sortby('time')
    da_agg.name = str(da_agg.name)
    return da_agg
