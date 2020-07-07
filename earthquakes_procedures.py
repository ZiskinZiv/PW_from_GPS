#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:11:38 2020

@author: shlomi
"""

from pathlib import Path
cwd = Path().cwd()
# TODO: Build mask = (e.index > st) & (e.index <= ed) for dates, choose dates:
# st=df.index[0]-np.timedelta64(1,'M') month each date or half month

def read_GNSS_station_position_velocity(path=cwd, station='ALON',
                                        return_pos=False):
    import pandas as pd
    df = pd.read_fwf(path / 'GNSS_pos_vel_all.txt')
    cols = [x for x in df.columns]
    cols[0] = 'station'
    cols[1] = 'pos_vel'
    df.columns = cols
    df = df[df['station'] == station]
    if return_pos:
        pos = df[df['pos_vel'] == 'POS'][['N', 'E']].values.squeeze()
        return pos
    else:
        return df


def read_GNSS_station_breakpoints(path=cwd, station='ALON'):
    from aux_gps import decimal_year_to_datetime
    import pandas as pd
    df = pd.read_fwf(path / 'GNSS_breakpoints_all.txt')
    cols = [x for x in df.columns]
    cols[0] = 'station'
    cols[1] = 'time'
    df.columns = cols
    df = df[df['station'] == station]
    df['time'] = df['time'].apply(decimal_year_to_datetime)
    df = df.set_index('time')
    return df


def read_earthquakes_IL(path=cwd):
    import pandas as pd
    df = pd.read_csv(cwd / 'earthquake_IL_1996-2020.csv')
    df = df.set_index('DateTime')
    df.index.name = 'time'
    df.index = pd.to_datetime(df.index)
    df = df.drop('epiid', axis=1)
    cols = [x for x in df.columns]
    cols = ['lat' if x == 'Lat' else x for x in cols]
    cols = ['lon' if x == 'Long' else x for x in cols]
    df.columns = cols
    df = df.sort_index()
    return df
