#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:46:44 2020

@author: ziskin
"""

from pathlib import Path
cwd = Path().cwd()


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
    if report:
        for code in sorted(df['class'].unique()):
            percent = 100 * (df['class'] == code).sum() / df['class'].size
            name = df['Name-EN'][df['class'] == code].unique().item()
            print('{} : {:.1f} %'.format(name, percent))
    return df
