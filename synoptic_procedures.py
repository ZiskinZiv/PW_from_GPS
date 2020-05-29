#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:46:44 2020

@author: ziskin
"""

from pathlib import Path
from PW_paths import work_yuval
cwd = Path().cwd()


def val_counts(ser):
    s = ser.value_counts()
    # s=s.drop(0)
    return s[:1].index.to_list()[0]


upper_class_dict = {'RST': [1, 2, 3], 'PT': [4, 5, 6], 'H': [7, 8, 9, 10],
                    'CL': [12, 13, 14, 15], 'Other': [11, 16, 17, 18, 19]}

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


def align_synoptic_class_with_pw(path):
    import xarray as xr
    from aux_gps import dim_intersection
    from aux_gps import save_ncfile
    from aux_gps import xr_reindex_with_date_range
    pw = xr.load_dataset(path / 'GNSS_PW_thresh_50_homogenized.nc')
    pw = pw[[x for x in pw if '_error' not in x]]
    syn = read_synoptic_classification().to_xarray()
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
    syn = xr.load_dataset(path / 'GNSS_synoptic_class.nc')
    if isinstance(syn_class, int):
        if isinstance(pw, xr.DataArray):
            syn = syn['{}_class'.format(pw.name)]
        elif isinstance(pw, xr.Dataset):
            syn = rename_data_vars(syn, suffix='_class', remove_suffix=True)
        syn = syn.to_dataframe()
        pw = pw.to_dataframe()
        pw = pw[syn==syn_class]
    elif isinstance(syn_class, str):
        if isinstance(pw, xr.DataArray):
            syn = syn['{}_class'.format(pw.name)]
        elif isinstance(pw, xr.Dataset):
            syn = rename_data_vars(syn, suffix='_class', remove_suffix=True)
        syn = syn.to_dataframe()
        pw = pw.to_dataframe()
        pw = pw[syn.isin(upper_class_dict.get(syn_class))]
    pw = pw.to_xarray()
    if plot:
        pw.plot()
        ax = plt.gca()
        ax.set_title('{} synoptic code selected'.format(syn_class))
    return pw


def read_synoptic_classification(
        path=cwd,
        filename='synoptic_classification_1948-8_May_2020.xls', report=True):
    import pandas as pd
    from aux_gps import path_glob
    import numpy as np
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
    df['upper_class'] = np.ones(df['class'].shape) * np.nan
    df.loc[(df['class'] <= 3) & (df['class'] >= 1), 'upper_class'] = 'RST'
    df.loc[(df['class'] <= 6) & (df['class'] >= 4), 'upper_class'] = 'PT'
    df.loc[(df['class'] <= 10) & (df['class'] >= 7), 'upper_class'] = 'H'
    df.loc[(df['class'] <= 15) & (df['class'] >= 12), 'upper_class'] = 'CL'
    df['upper_class'] = df['upper_class'].fillna('Other')
    df.index.name = 'time'
    if report:
        for code in sorted(df['class'].unique()):
            percent = 100 * (df['class'] == code).sum() / df['class'].size
            name = df['Name-EN'][df['class'] == code].unique().item()
            print('{} : {:.1f} %'.format(name, percent))
    return df


def agg_month_count_syn_class(path=cwd):
    # df.loc['2015-09']['Name-EN'].value_counts()
    import pandas as pd
    df = read_synoptic_classification(path=path)
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['months'] = df['year'].astype(str) + '-' + df['month'].astype(str)
    new_df = df.groupby([df['months'], df['class']]).size().to_frame()
    new_df.columns = ['class_sum']
    dfmm = pd.pivot_table(new_df, index='months', columns='class')
    dfmm.set_index(pd.to_datetime(dfmm.index), inplace=True)
    dfmm = dfmm.fillna(0)
    dfmm = dfmm.astype(int)
    dfmm.columns = [x + 1 for x in range(19)]
    return dfmm
    