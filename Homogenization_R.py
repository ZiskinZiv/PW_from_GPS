#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:04:31 2020
@author: shlomi
methodology: run FindU on ALT and find corresponding changepoints in PW 
(FindU, FindUD), also look for RINEX availability gaps for changepoints
# quationable stations:
# 'jslm': FindUD finds 3 points but not in sync with ALT
# 'katz': many point in ALT (UD), a few in PW , one corresponding
# Qmout error: FindU no 1 type and FindUD finds homogenoues, stepsize fails:
#: kabr, drag, csar, mrav, nizn
"""
#def download_packages():
#    from rpy2.robjects.packages import importr
#    import rpy2.robjects.packages as rpackages
#    utils = rpackages.importr('utils')
#    utils.chooseCRANmirror(ind=1)
#    # utils.install_packages("remotes")
#    rm=importr("remotes")
#    rm.install_github('ECCM-CDAS/RHtests/V4_files')
from PW_paths import work_yuval
adjusted_stations = ['bshm', 'dsea', 'elat', 'elro', 'katz', 'klhv', 'nrif',
                     'tela']
unchanged_stations = ['alon', 'csar', 'drag', 'jslm', 'kabr', 'mrav',
                      'nzrt', 'ramo', 'slom', 'yosh', 'yrcm']
gis_path = work_yuval / 'gis'
homo_path = work_yuval / 'homogenization'


def save_pw_monthly_means_and_anoms(loadpath=homo_path, savepath=work_yuval,
                                    thresh=50):
    import xarray as xr
    pw_o = xr.open_dataset(
                savepath /
                'GNSS_PW_thresh_{:.0f}.nc'.format(thresh))
    pw_o = pw_o[[x for x in pw_o.data_vars if '_error' not in x]]
    attrs_o = [x.attrs for x in pw_o.data_vars.values()]
    pw = load_adjusted_stations(loadpath, rename_adjusted=True,
                                return_ds='all+shifts')
    shifts = pw[[x for x in pw.data_vars if '_shift' in x]]
    pw = pw[[x for x in pw.data_vars if '_shift' not in x]]
    attrs = [x.attrs for x in pw.data_vars.values()]
    names = [x.name for x in pw.data_vars.values()]
    attrs = dict(zip(names, attrs))
    attrs_o = dict(zip(names, attrs_o))
    anoms = pw.groupby('time.month') - pw.groupby('time.month').mean('time')
    anoms = anoms.reset_coords(drop=True)
    for name in names:
        pw[name].attrs.update(attrs_o.get(name))
        anoms[name].attrs = attrs.get(name)
        anoms[name].attrs.update(attrs_o.get(name))
    filename = 'GNSS_PW_monthly_thresh_50_homogenized.nc'
    pw.to_netcdf(savepath/filename, 'w')
    print('{} was save to {}'.format(filename, savepath))
    filename = 'GNSS_PW_monthly_anoms_thresh_50_homogenized.nc'
    anoms.to_netcdf(savepath/filename, 'w')
    print('{} was save to {}'.format(filename, savepath))
    # now save shifts data:
    filename = 'GNSS_PW_monthly_shifts_thresh_50.nc'
    shifts.to_netcdf(savepath/filename, 'w')
    print('{} was save to {}'.format(filename, savepath))
    return


def compare_pw_trend_mann_kendall(loadpath=homo_path):
    from PW_stations import mann_kendall_trend_analysis
    pw = load_adjusted_stations(loadpath, rename_adjusted=True,
                                return_ds='all')
    attrs = [x.attrs for x in pw.data_vars.values()]
    names = [x.name for x in pw.data_vars.values()]
    attrs = dict(zip(names, attrs))
    anoms = pw.groupby('time.month') - pw.groupby('time.month').mean('time')
    anoms = anoms.reset_coords(drop=True)
    for name in names:
        anoms[name].attrs = attrs.get(name)
    anoms = anoms.map(mann_kendall_trend_analysis, verbose=False)
    return anoms


def compare_adj_pw(path=work_yuval, homo_path=homo_path, gis_path=gis_path,
                   stations=adjusted_stations, unchanged=unchanged_stations):
    import xarray as xr
    from PW_stations import produce_pw_statistics
    from PW_stations import mann_kendall_trend_analysis
    import numpy as np
    import pandas as pd
    adj_pw = load_adjusted_stations(
        homo_path, stations=stations, sample='monthly', field='PW',
        gis_path=gis_path, return_ds='adjusted')
    pw = xr.load_dataset(path / 'GNSS_PW_hourly_thresh_50.nc')
    pw = pw.resample(time='MS').mean('time')
    adjusted_names = [x.split('_')[0] for x in adj_pw.data_vars]
    originals = xr.merge([x for x in pw.data_vars.values() if x.name in adjusted_names])
    compare = xr.merge([adj_pw, originals])
    compare = compare[[x for x in sorted(compare)]]
    df_stats = produce_pw_statistics(resample_to_mm=False, pw_input=compare)
    attrs = [x.attrs for x in compare.data_vars.values() if '_mean_adj' in x.name]
    names = [x.name for x in compare.data_vars.values() if '_mean_adj' in x.name]
    attrs = dict(zip(names, attrs))
    anoms = compare.groupby('time.month') - compare.groupby('time.month').mean('time')
    for name in names:
        anoms[name].attrs = attrs.get(name)
    anoms = anoms.reset_coords(drop=True)
    anoms = anoms.map(mann_kendall_trend_analysis, alpha=0.05, verbose=False)
    mkt_trends = [anoms[x].attrs['mkt_trend'] for x in anoms.data_vars]
    mkt_bools = [anoms[x].attrs['mkt_h'] for x in anoms.data_vars]
    mkt_slopes = [anoms[x].attrs['mkt_slope'] for x in anoms.data_vars]
    trends = []
    for x in anoms.data_vars:
        try:
            trends.append(anoms[x].attrs['trend'])
        except KeyError:
            trends.append(np.nan)
    df_trends = pd.DataFrame(trends, index=[x for x in anoms.data_vars], columns=['trend'])
    df_trends['mkt_trend'] = mkt_trends
    df_trends['mkt_h'] = mkt_bools
    df_trends['mkt_slope'] = mkt_slopes
    df_trends['trend'] = df_trends['trend'].astype(float)
    return anoms, df_trends, df_stats


def load_adjusted_stations(
        loadpath, stations=adjusted_stations, sample='monthly', field='PW',
        gis_path=gis_path, return_ds='adjusted', rename_adjusted=False):
    # return_ds : adjusted, unchanged, all, all+shifts
    import xarray as xr
    from PW_stations import produce_geo_gnss_solved_stations
    # first assemble all adjusted stations:
    adj_list = []
    shift_list = []
    for station in stations:
        da = df_to_da_with_stats(loadpath, station, sample=sample,
                                 field=field, df_field='mean_adj',
                                 update_stats=True, rfunc='StepSize',
                                 plot=False)
        shift = df_to_da_with_stats(loadpath, station, sample=sample,
                                    field=field, df_field='shift',
                                    update_stats=False, rfunc='StepSize',
                                    plot=False)
        adj_list.append(da)
        shift_list.append(shift)
    # then assemble all other stations:
    df = produce_geo_gnss_solved_stations(path=gis_path, plot=False)
    all_stations = [x for x in df.index if x not in ['gilb', 'lhav', 'hrmn', 'nizn', 'spir']]
    other = sorted([x for x in all_stations if x not in adjusted_stations])
    other_list = []
    for station in other:
        da = df_to_da_with_stats(loadpath, station, sample=sample,
                                 field=field, rfunc='StepSize', df_field=station,
                                 update_stats=True, plot=False)
        other_list.append(da)
    if rename_adjusted:
        for da in adj_list:
            da.name = da.name.split('_')[0]
    if return_ds == 'adjusted':
#        adjusted_names = [x.name.split('_')[0] for x in adj_list]
#        print(adjusted_names)
#        originals = [x for x in other_list if x.name in adjusted_names]
#        print(originals)
        adj_pw = xr.merge(adj_list)
        adj_pw = adj_pw[[x for x in sorted(adj_pw)]]
    elif return_ds == 'unchanged':
        adj_pw = xr.merge(other_list)
        adj_pw = adj_pw[[x for x in sorted(adj_pw)]]
    elif return_ds == 'all':
        adj_pw = xr.merge(adj_list + other_list)
        adj_pw = adj_pw[[x for x in sorted(adj_pw)]]
    elif return_ds == 'all+shifts':
        adj_pw = xr.merge(adj_list + other_list + shift_list)
        adj_pw = adj_pw[[x for x in sorted(adj_pw)]]
    return adj_pw


def df_to_da_with_stats(loadpath, station='tela', sample='monthly', field='PW',
                        rfunc='StepSize', df_field='mean_adj',
                        update_stats=True, plot=True):
    kwargs = locals()
    [kwargs.pop(x) for x in ['plot', 'update_stats', 'df_field']]
    df, stats = read_dat_file(**kwargs)
    da = df[df_field].to_xarray()
    da = da.rename({'date': 'time'})
    if df_field == station:
        da.name = station
    else:
        da.name = '{}_{}'.format(station, df_field)
    da.attrs['units'] = 'mm'
    if update_stats:
        da.attrs.update(stats)
    if plot:
        da.plot()
    return da


def read_stat_txt_file(loadpath, station='tela', sample='monthly',
                       field='PW', rfunc='StepSize'):
    from aux_gps import path_glob
    import re
    rfunc_dict = {'FindU': 'Ustat', 'FindUD': 'UDstat', 'StepSize': 'FINAL_Fstat'}
    file = path_glob(
        loadpath, '{}_{}_{}_means_*_{}.txt'.format(station, field, sample, rfunc_dict.get(rfunc)))[0]
    f = open(file, 'r')
    lines = f.readlines()
    ##steps=  3 ; trend= 0.008495 ( 0.005017 , 0.011973 ) (p= 1 ); cor= 0.2084 ( 0.0786 , 0.3313 ) 0.9993 
    stats = {}
    for line in lines:
        if line.startswith('#steps'):
            intercept = line.split(';')[-1]
            inter_list = re.findall("[^a-zA-Z:]([-+]?\d+[\.]?\d*)", intercept)
            stats['interpect'] = inter_list[0]
            stats['interpect_95'] = inter_list[1:3]
            stats['interpect_pvalue'] = inter_list[-1]
            trend = line.split(';')[-2]
            trend_list = re.findall("[^a-zA-Z:]([-+]?\d+[\.]?\d*)", trend)
            stats['trend'] = trend_list[0]
            stats['trend_95'] = trend_list[1:3]
            stats['trend_pvalue'] = trend_list[-1]
    if sample == 'monthly':
        stats['trend_units'] = 'per_month'
    elif sample == 'daily':
        stats['trend_units'] = 'per_day'
    return stats


def read_dat_file(loadpath, station='tela', sample='monthly',
                  field='PW', rfunc='StepSize'):
    import pandas as pd
    from aux_gps import path_glob
    rfunc_dict = {'FindU': 'U', 'FindUD': 'UD', 'StepSize': 'FINAL_F'}
    file = path_glob(
        loadpath, '{}_{}_{}_means_*_{}.dat'.format(station, field, sample, rfunc_dict.get(rfunc)))[0]
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
        'mean_adj',
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
    df['shift'] = df['mean_adj'] - df[station]
    stats = read_stat_txt_file(loadpath, station=station, sample=sample,
                       field=field, rfunc=rfunc)
    return df, stats


def export_all(loadpath, savepath, sample='MS', field='PW'):
    import xarray as xr
    if sample == 'MS':
        print('Monthly means selected:')
    elif sample == '1D':
        print('Daily means selected:')
    if field == 'PW':
        ds = xr.load_dataset(loadpath / 'GNSS_PW_thresh_50.nc')
    elif field == 'ALT':
        ds = xr.load_dataset(loadpath / 'ALT_thresh_50.nc')
    print('{} field selected.'.format(field))
    ds = ds[[x for x in ds.data_vars if '_error' not in x]]
    ds = ds.resample(time=sample).mean()
    for da in ds.data_vars:
        print('exporting {} station to csv'.format(da.upper()))
        _ = export_time_series_station_to_csv(ds[da], savepath=savepath,
                                              field=field)
    print('Done!')
    return


def export_time_series_station_to_csv(da_ts, savepath, field='PW'):
    import pandas as pd
    name = da_ts.name
    df = da_ts.to_dataframe()
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
    filename = '{}_{}_{}_for_RHtests.csv'.format(field, name, sample)
    df.to_csv(savepath / filename, index=False, header=False, sep=',')
    return df


def check_station_name(name):
    # import os
    if isinstance(name, list):
        name = [str(x).lower() for x in name]
        for nm in name:
            if len(nm) != 4:
                raise argparse.ArgumentTypeError('{} should be 4 letters...'.format(nm))
        return name
    else:
        name = str(name).lower()
        if len(name) != 4:
            raise argparse.ArgumentTypeError(name + ' should be 4 letters...')
        return name


def run_RHtests_function(name='FindU', station='tela', sample='monthly',
                         field='PW', ref=None, path=None, params=None):
    import rpy2.robjects as robjects
    from pathlib import Path
    r = robjects.r
    r.source('RHtests.R')
    # options for name: FindU, FindUD, StepSize, FindU.wRef, FindUD.wRef, StepSize.wRef
    rfunc = r(name)
    if params is not None:
        if params.plev is not None:
            plev = params.plev
        else:
            plev = 0.95
        if params.Ny4a is not None:
            Ny4a = params.Ny4a
        else:
            Ny4a = 0
        if params.Mq is not None:
            Mq = params.Mq
        else:
            Mq = 12
    if ref is None:
        print('Running homogenization on {} {} {} means Without Reference Station:'.format(field, station, sample))
        print('with parameters: plev {}, Ny4a {}, Mq {}.'.format(plev, Ny4a, Mq))
        in_file = "{}/{}_{}_{}_means_for_RHtests.csv".format(
            path.as_posix(), field, station, sample)
        if not Path(in_file).is_file():
            print(
                '{} not found ...\n pls run export_time_series_station_to_csv or export_all'.format(in_file))
            return
        # now r.FindU and other functions working
        out = "{}/{}_{}_{}_means_plev{}_Mq{}_OUT".format(
            path.as_posix(), station, field, sample, round(100 * plev), Mq)
        print('running {}...'.format(name))
        if name == 'FindU':
            rfunc(InSeries=in_file,
                  output=out,
                  MissingValueCode="-999.00", p_lev=plev, Mq=Mq, Ny4a=Ny4a)
        elif name == 'FindUD':
            # InCs = out + '_mCs.txt'
            InCs = out + '_1Cs.txt'
            if not Path(InCs).is_file():
                print(
                        '{} not found ...\n pls run FindU first'.format(InCs))
                return
            rfunc(InSeries=in_file,
                  output=out, InCs=InCs,
                  MissingValueCode="-999.00", p_lev=plev, Mq=Mq, Ny4a=Ny4a)
        elif name == 'StepSize':
            InCs = out + '_mCs.txt'
            if not Path(InCs).is_file():
                print(
                        '{} not found ...\n pls run FindU first'.format(InCs))
                return
            rfunc(InSeries=in_file,
                  output=out.replace('OUT', 'FINAL'), InCs=InCs,
                  MissingValueCode="-999.00", p_lev=plev, Mq=Mq, Ny4a=Ny4a)
        print('')
    return


def prepare_pwv_for_climatol(path=work_yuval, freq='daily',
                             savepath=homo_path, first_year='1998',
                             last_year='2019', pwv_ds=None, group=None):
    """freq can be daily or monthly.
    climatol params used:
        std=2 since the PDF is positively skewed,
        na.strings="-999.9" this is for NA values,
        dz.max=7 this is 7 sigma std outliers max,
        homogen('PWV',1998,2019, na.strings="-999.9",dz.max=7,std=2)
        dahstat('PWV',1998,2019,stat='series',long=TRUE)"""
    import xarray as xr
    import csv
    from aux_gps import xr_reindex_with_date_range
    from PW_stations import produce_geo_gnss_solved_stations
    from PW_from_gps_figures import st_order_climate
    freq_dict = {'daily': 'D', 'monthly': 'MS'}
    if pwv_ds is not None:
        ds = pwv_ds
    else:
        ds = xr.load_dataset(path / 'GNSS_PW_{}_thresh_50.nc'.format(freq))
    ds = xr_reindex_with_date_range(
        ds,
        freq=freq_dict[freq],
        dt_min='{}-01-01'.format(first_year),
        dt_max='{}-12-31'.format(last_year),
        drop=False)
    df_gnss = produce_geo_gnss_solved_stations(plot=False)
#    sites = df.dropna()[['lat', 'alt', 'groups_annual']].sort_values(by=['groups_annual', 'lat'],ascending=[1,0]).index
    df = df_gnss.loc[st_order_climate, :]
    df['site'] = df.index
    df['name'] = df['site'].str.upper()
    df = df[['lat', 'lon', 'alt', 'site', 'name']]
    data = ds.to_dataframe().T
    if group is not None:  # can be 0 to 2
        inds = [x for x in df_gnss[df_gnss['groups_climate']==group].index]
        df = df.loc[inds, :]
        data = data.loc[inds, :]
    else:
        inds = [x for x in df_gnss.index if x in ds]
        df = df.loc[inds, :]
    if group is not None:
        if freq == 'daily':
            filename = 'PWV{}-d_{}-{}.est'.format(group, first_year, last_year)
        else:
            filename = 'PWV_{}_{}-{}.est'.format(group, first_year, last_year)
    else:
        filename = 'PWV_{}-{}.est'.format(first_year, last_year)
    df.to_csv(
        savepath /
        filename,
        sep=' ',
        index=False,
        header=False,
        quotechar='"',
        quoting=csv.QUOTE_NONNUMERIC)
    filename = filename.replace('.est', '.dat')
    df = data
    df = df.round(3)
    df.to_csv(
        savepath /
        filename,
        sep=' ',
        index=False,
        header=False, line_terminator='\n', na_rep=-999.9)
    return


def perform_pwv_filling_last_decade(path=work_yuval, fyear='2009', lyear='2019',
                                    drop=['slom', 'elro']):
    import xarray as xr
    from aux_gps import save_ncfile
    pw = xr.load_dataset(path / 'GNSS_PW_monthly_thresh_50.nc')
    pw = pw.sel(time=slice(fyear, lyear))
    pw = pw.drop_vars(drop)
    prepare_pwv_for_climatol(
        freq='monthly',
        first_year=fyear,
        last_year=lyear,
        pwv_ds=pw)
    # then run these two lines in R:
    # homogen('PWV',2009,2019, na.strings="-999.9",dz.max=7,std=2)
    # dahstat('PWV',2009,2019,stat='series',long=TRUE)
    ds, ds_flag = read_climatol_results(first_year=fyear, last_year=lyear)
    filename = 'GNSS_PW_monthly_homogenized_filled_{}-{}.nc'.format(fyear, lyear)
    save_ncfile(ds, path, filename)
    filename = 'GNSS_PW_monthly_homogenized_filled_flags_{}-{}.nc'.format(fyear, lyear)
    save_ncfile(ds_flag, path, filename)
    return


def read_climatol_results(var='PWV', first_year='1998', last_year='2019',
                          path=homo_path):
    import pandas as pd
    import xarray as xr
    ser_file = path / '{}_{}-{}_series.csv'.format(var, first_year, last_year)
    flag_file = path / '{}_{}-{}_flags.csv'.format(var, first_year, last_year)
    data = pd.read_csv(ser_file, header=0)
    flag = pd.read_csv(flag_file, header=0)
    data['Date'] = pd.to_datetime(data['Date'])
    flag['Date'] = pd.to_datetime(flag['Date'])
    data = data.set_index('Date')
    flag = flag.set_index('Date')
    data.index.name = 'time'
    flag.index.name = 'time'
#    flag.columns = [x + '_flag' for x in flag.columns]
    ds = data.to_xarray()
    ds_flag = flag.to_xarray()
    ds_flag.attrs['0'] = 'observed'
    ds_flag.attrs['1'] = 'filled in'
    ds_flag.attrs['2'] = 'corrected'
#    if remove_splits:
#        ds = ds[[x for x in ds if '-' not in x]]
#        ds_flag = ds_flag[[x for x in ds_flag if '-' not in x]]
    return ds, ds_flag
#def run_RH_tests(station='tela', path=None, sample='monthly', field='PW',
#                 args=None):
#    import rpy2.robjects as robjects
#    from pathlib import Path
#    r = robjects.r
#    r.source('RHtests.R')
#    FindU = r('FindU')
#    StepSize = r('StepSize')
#    FindU_wRef = r('FindU.wRef')
#    StepSize_wRef = r('StepSize.wRef')
#    #from rpy2.robjects.packages import importr
#    #from rpy2.rinterface import RRuntimeWarning
#    #base = importr('base')
#    # base.warnings()
#    if args is not None:
#        if args.plev is not None:
#            plev = args.plev
#        else:
#            plev = 0.95
#        if args.Ny4a is not None:
#            Ny4a = args.Ny4a
#        else:
#            Ny4a = 0
#        if args.Mq is not None:
#            Mq = args.Mq
#        else:
#            Mq = 12
#        if args.ref is not None:
#            ref = args.ref
#        else:
#            ref = None
#    if ref is None:
#        print('Running homogenization on {} {} {} means Without Reference Station:'.format(field, station, sample))
#        print('with parameters: plev {}, Ny4a {}, Mq {}.'.format(plev, Ny4a, Mq))
#        in_file = "{}/{}_{}_{}_means_for_RHtests.csv".format(
#            path.as_posix(), field, station, sample)
#        if not Path(in_file).is_file():
#            print(
#                '{} not found ...\n pls run export_time_series_station_to_csv or export_all'.format(in_file))
#            return
#        # now r.FindU and other functions working
#        out = "{}/{}_{}_{}_means_plev{}_Mq{}_out".format(
#            path.as_posix(), field, station, sample, round(100 * plev), Mq)
#        print('running FindU')
#        FindU(InSeries=in_file,
#              output=out,
#              MissingValueCode="-999.00", p_lev=plev, Mq=Mq, Ny4a=Ny4a)
#        print('')
#        print('running StepSize')
#        StepSize(InSeries=in_file, output=out, MissingValueCode="-999.00",
#                 InCs=out + '_mCs.txt', p_lev=plev, Mq=Mq, Ny4a=Ny4a)
#        print('')
#    else:
#        print('Running homogenization on {} {} {} means with reference station ():'.format(field, station, sample, ref))
#        print('with parameters: plev {}, Ny4a {}, Mq {}.'.format(plev, Ny4a, Mq))
#        B_file = "{}/{}_{}_{}_means_for_RHtests.csv".format(
#            path.as_posix(), field, station, sample)
#        R_file = "{}/{}_{}_{}_means_for_RHtests.csv".format(
#            path.as_posix(), field, ref, sample)
#        if not Path(B_file).is_file() or not Path(R_file).is_file():
#            print(
#                '{} not found ...\n pls run export_pw_station_to_csv or export_all'.format(in_file))
#            return
#        # now r.FindU and other functions working
#        out = "{}/{}_{}_{}_means_ref_{}_plev{}_Mq{}out".format(
#            path.as_posix(), field, station, sample, ref, round(100 * plev), Mq)
#        print('running FindU_wRef')
#        FindU_wRef(Bseries=B_file,
#                   output=out, Rseries=R_file,
#                   MissingValueCode="-999.00", p_lev=plev, Mq=Mq, Ny4a=Ny4a)
#        print('')
#        print('running StepSize_wRef')
#        StepSize_wRef(Bseries=B_file, output=out, MissingValueCode="-999.00",
#                      InCs=out + '_mCs.txt', p_lev=plev, Mq=Mq, Ny4a=Ny4a,
#                      Rseries=R_file)
#        print('')
#    return


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
    required.add_argument('--station', nargs='+', help="GNSS 4 letter station", type=check_station_name)
    required.add_argument('--field', help='Either PW or ALT for now', choices=['PW', 'ALT'])
    required.add_argument('--rfunc', help='R function from RHtests.R', choices=['FindU', 'FindUD', 'StepSize'])
    
    optional.add_argument('--sample', help='select monthly or daily',
                          type=str, choices=['monthly', 'daily'])
    optional.add_argument('--plev', help='pvalue significance',
                          type=float, choices=[0.75, 0.80, 0.90, 0.95, 0.99, 0.9999])
    optional.add_argument('--Mq', help='number of points(categories) for which the empirical PDF are to be estimated', type=int)
    optional.add_argument('--Ny4a', help='maximum number of years of data immidiately before or after a changepoint to be used to estimate the PDF', type=int)
    optional.add_argument('--ref', help='Reference station', type=check_station_name)
#                          metavar=str(cds.start_year) + ' to ' + str(cds.end_year))
#    optional.add_argument('--half', help='a spescific six months to download,\
#                          e.g, 1 or 2', type=int, choices=[1, 2],
#                          metavar='1 or 2')
    parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()
    if args.sample is None:
        args.sample = 'monthly'
    # print(parser.format_help())
#    # print(vars(args))
    if args.station is None:
        print('station is a required argument, run with -h...')
        sys.exit()
    if args.field is None:
        print('field is a required argument, run with -h...')
        sys.exit()
    if args.rfunc is None:
        print('rfunc is a required argument, run with -h...')
        sys.exit()
    for station in args.station:
        run_RHtests_function(name=args.rfunc, station=station, sample=args.sample,
                         field=args.field, ref=args.ref, path=savepath, params=args)

