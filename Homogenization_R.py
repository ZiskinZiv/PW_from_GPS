#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:04:31 2020

@author: shlomi
"""
from PW_paths import work_yuval
import os
from aux_gps import path_glob
os.environ['R_HOME'] = '/home/shlomi/anaconda3/lib/R'

#def download_packages():
#    from rpy2.robjects.packages import importr
#    import rpy2.robjects.packages as rpackages
#    utils = rpackages.importr('utils')
#    utils.chooseCRANmirror(ind=1)
#    # utils.install_packages("remotes")
#    rm=importr("remotes")
#    rm.install_github('ECCM-CDAS/RHtests/V4_files')


def export_pw_station_to_csv(pw_da, savepath=work_yuval):
    import pandas as pd
    name = pw_da.name
    df = pw_da.to_dataframe()
    df['year'] = df.index.year
    df['month'] = df.index.month
    if pd.infer_freq(df.index) == 'D':
        df['day'] = df.index.day
        sample = 'daily_means'
    elif pd.infer_freq(df.index) == 'MS':
        df['day'] = 00
        df['day'] = df.day.map("{:02}".format)
        sample = 'monthly_means'
    else:
        raise ValueError('pls resample to MS or 1D for monthly or daily means')
    df = df[['year', 'month', 'day', name]]
    df = df.fillna(-999.0)
    # df[name] = df[name].map("{.:2}".format)
    df[name] = df[name].map("{0:.2f}".format)
    filename = 'PW_{}_{}_for_RHtests.csv'.format(name, sample)
    df.to_csv(savepath / filename, index=False, header=False, sep=',')
    return df


import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.rinterface import RRuntimeWarning
base = importr('base')
base.warnings()
r = robjects.r
r.source('RHtests.R')
monthly_files = path_glob(work_yuval, 'PW_*_monthly_means_for_RHtests.csv')
mm_file = "/mnt/DATA/Work_Files/PW_yuval/PW_tela_monthly_means_for_RHtests.csv"
# now r.FindU and other functions working
out="/mnt/DATA/Work_Files/PW_yuval/out/"
try:
    o = r.FindU(
            InSeries="/mnt/DATA/Work_Files/PW_yuval/InFile.csv",
            output="/mnt/DATA/Work_Files/PW_yuval/Out",
            MissingValueCode="-999.00",
            Mq=12)
except RRuntimeWarning as e:
    print(e)
