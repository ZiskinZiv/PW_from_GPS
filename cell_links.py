#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 09:20:37 2021

@author: shlomi
"""

from PW_paths import work_yuval
cell_path = work_yuval / 'cell_links'

def read_links_metadata(path=cell_path):
    import pandas as pd
    file = 'CML_TA.csv'
    df = pd.read_csv(path/file, header=0, index_col=False)
    return df


def read_link_humidity(path=cell_path):
    from aux_gps import path_glob
    import pandas as pd
    import xarray as xr
    files = sorted(path_glob(path, 'table*.csv'))
    da_list = []
    for i, file in enumerate(files):
        df = pd.read_csv(file, index_col='Time')
        df.index = pd.to_datetime(df.index)
        da = df.to_xarray().to_array('var')
        da.name = str(i)
        da_list.append(da)
    ds = xr.merge(da_list)
    return ds
