#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:04:31 2020

@author: shlomi
"""
#def download_packages():
#    from rpy2.robjects.packages import importr
#    import rpy2.robjects.packages as rpackages
#    utils = rpackages.importr('utils')
#    utils.chooseCRANmirror(ind=1)
#    # utils.install_packages("remotes")
#    rm=importr("remotes")
#    rm.install_github('ECCM-CDAS/RHtests/V4_files')


def read_dat_file(loadpath, station='tela', sample='monthly'):
    import pandas as pd
    from aux_gps import path_glob
    file = path_glob(
        loadpath, '{}_{}_means_out_*.dat'.format(station, sample))[0]
    df = pd.read_csv(
        file,
        header=None,
        delim_whitespace=True,
        na_values="-999.00")
    df.columns = [
        'ind',
        'date',
        station,
        'trend_shift',
        'mean_adjusted',
        '{}_anom'.format(station),
        'anom_trend_shift',
        'seasonal_trend_shift',
        'QM_adj',
        'anom_trend_no_shift']
    df['date'] = df['date'].astype(str)
    if sample == 'monthly':
        df['date']=df['date'].str[:6]
        df['date'] = pd.to_datetime(df['date'], format='%Y%m')
    else:
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.set_index(df['date'])
    df = df.drop(['ind', 'date'], axis=1)
    return df


def export_all(loadpath, savepath, sample='MS'):
    import xarray as xr
    pw = xr.load_dataset(loadpath / 'GNSS_PW_thresh_50.nc')
    if sample == 'MS':
        print('Monthly means selected:')
    elif sample == '1D':
        print('Daily means selected:')
    pw = pw[[x for x in pw.data_vars if '_error' not in x]]
    pw = pw.resample(time=sample).mean()
    for da in pw.data_vars:
        print('exporting {} station to csv'.format(da.upper()))
        _ = export_pw_station_to_csv(pw[da], savepath=savepath)
    print('Done!')
    return


def export_pw_station_to_csv(pw_da, savepath):
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


def run_RH_tests(station='tela', path=None, sample='monthly'):
    import rpy2.robjects as robjects
    from pathlib import Path
    #from rpy2.robjects.packages import importr
    #from rpy2.rinterface import RRuntimeWarning
    #base = importr('base')
    #base.warnings()
    r = robjects.r
    r.source('RHtests.R')
    in_file = "{}/PW_{}_{}_means_for_RHtests.csv".format(
        path.as_posix(), station, sample)
    if not Path(in_file).is_file():
        print('{} not found ...\n pls run export_pw_station_to_csv or export_all'.format(in_file))
        return
    # now r.FindU and other functions working
    out = "{}/{}_{}_means_out".format(path.as_posix(), station, sample)
    print('running FindU')
    r.FindU(InSeries=in_file,
            output=out,
            MissingValueCode="-999.00")
    print('running StepSize')
    r.StepSize(InSeries=in_file, output=out, MissingValueCode="-999.00",
               InCs=out + '_mCs.txt')
    return


if __name__ == '__main__':
    import argparse
    import sys
    from pathlib import Path
    from aux_gps import configure_logger
    from PW_paths import work_yuval
    import os
    homo_path = work_yuval / 'homogenization'
    os.environ['R_HOME'] = '/home/shlomi/anaconda3/lib/R'
    logger = configure_logger('RH_tests')
    savepath = Path(homo_path)
    parser = argparse.ArgumentParser(description='a command line tool for running the RHtests climatology homogenization procedures.')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # remove this line: optional = parser...
    required.add_argument('--station', help="GNSS 4 letter station", type=str)
    
    #optional.add_argument('--station', nargs='+',
    #                      help='GPS station name, 4 UPPERCASE letters',
    #                      type=check_station_name)
#                          metavar=str(cds.start_year) + ' to ' + str(cds.end_year))
#    optional.add_argument('--half', help='a spescific six months to download,\
#                          e.g, 1 or 2', type=int, choices=[1, 2],
#                          metavar='1 or 2')
    parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()
    # print(parser.format_help())
#    # print(vars(args))
    if args.station is None:
        print('station is a required argument, run with -h...')
        sys.exit()
    run_RH_tests(args.station, path=savepath, sample='monthly')
