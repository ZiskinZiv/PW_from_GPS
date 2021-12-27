#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 10:51:32 2021
Commands for processing T02 from ftp axis-gps:
1) copy the T02 files to my directory
# extract all multi-GNSS sats to tgd and dat files:
2) runpkr00 -d -g Bisa103V.T02
# convert to rinex 2: (input YY as year):
3) teqc -tr d Bisa103V.tgd > Bisa103V.21o
# hakatanaka compress:
4) RNX2CRX Bisa103V.21o
# finally gzip:
5) tar -czvf Bisa103V.21d.Z Bisa103V.21d
# for each step delete last file keep only Z file
# naming convension:
Bisa - four letter gnss station
103V - 13 of the month, V = 21-22 pm (hour) UTC
104H - 14 of the month, H = 07-08 am (hour) UTC
# Directory structure:
    Month.Apr
    Day.13
# recommended:
    need to convert to doy, keep station name add f
    keep all stations in same directory per day of year
    ssssdddf:
        ssss:  4-character station name designator
        ddd:  day of the year of first record
         f:  file sequence number/character within day
   |   |               daily file: f = 0
   |   |               hourly files:
   |   |               f = a:  1st hour 00h-01h; f = b: 2nd hour 01h-02h; ...
   |   |               f = x: 24th hour 23h-24h

Bisa103V.21d.Z
@author: shlomi
"""
from pathlib import Path
home_axis_path = Path('/home/axis-gps')
cwd = Path().cwd()


def analyse_T02_files(main_path=home_axis_path, start_date=None, savepath=cwd):
    """this will run every hour and write a csv file with dataframe containing the
    recent T02 file count"""
    import pandas as pd
    import calendar
    from aux_gps import path_glob
    from aux_gps import get_timedate_and_station_code_from_rinex
    today_date = pd.Timestamp.today()
    if start_date is not None:
        yesterday_date = pd.to_datetime(start_date)
    else:
        yesterday_date = today_date - pd.Timedelta(1, unit='d')
    dates = pd.date_range(yesterday_date, today_date, freq='1D')
    ddf = pd.DataFrame([x for x in dates.day], index=dates)
    ddf.columns = ['day']
    ddf['day'] = ddf['day'].astype(str).str.zfill(2)
    ddf['month'] = ddf.index.month
    ddf['month_name'] = ddf['month'].apply(lambda x: calendar.month_abbr[x])
    ddf['year'] = ddf.index.year
    dfs = []
    for day_dt, row in ddf.iterrows():
        month = row['month_name']
        day = row['day']
        file_path = main_path / 'Month.{}'.format(month) / 'Day.{}'.format(day)
        hourly_files = path_glob(file_path, '*.T02', return_empty_list=True)
        if not hourly_files:
            continue
        records = []
        for i, hour_file in enumerate(hourly_files):
            records.append(parse_T02_file(hour_file))
        df = pd.DataFrame(records, columns=['station_name', 'doy', 'hour_letter'])
        df['rfn'] = df['station_name'] + df['doy'].astype(str) + df['hour_letter'] + '.{}d'.format(str(row['year'])[2:])
        df['dt'] = df['rfn'].apply(get_timedate_and_station_code_from_rinex, just_dt=True)
        # df.to_csv('axis_try.csv')
        dfs.append(df)
    dff = pd.concat(dfs, axis=0)
    dff = dff.groupby('dt')['station_name'].count()
    dff.index=pd.to_datetime(dff.index)
    dff = dff.to_frame('total_files')
    full_dts = pd.date_range(dff.index[0],dff.index[-1], freq='1H')
    dff = dff.reindex(full_dts)
    dff = dff.fillna(0)
    dff['no_files'] = dff['total_files']==0
    dff.index.name = 'dt'
    dff.to_csv(savepath / 'T02_file_count.csv', index=True)
    return dff


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_path(path):
    import os
    from pathlib import Path
    path = str(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(path + ' does not exist...')
    return Path(path)


def check_year(year):
    import datetime
    now = datetime.datetime.now()
    year = int(year)
    if year < 1980 or year > now.year:
        raise argparse.ArgumentTypeError('{} is not a valid year'.format(year))
    return year


def check_for_missing_rinex_in_axis_path(args):
    import calendar
    import datetime
    from aux_gps import path_glob
    from pathlib import Path
    now = datetime.datetime.now()
    month_abr = calendar.month_abbr[now.month]
    month_path = 'Month.{}'.format(month_abr)
    T02_path = Path('/home/axis-gps/{}'.format(month_path))
    # in month transition:
    if not T02_path.is_dir():
        month_abr = calendar.month_abbr[now.month - 1]
        month_path = 'Month.{}'.format(month_abr)
        T02_path = Path('/home/axis-gps/{}'.format(month_path))
    days = sorted(path_glob(T02_path, '*/'))
    last_day = int(days[-1].as_posix().split('.')[-1])
    day_diff = now.day - last_day
    if day_diff <= 1:
        print('Last day is {}th in {}, no rinex gaps above 1 days.'.format(
            last_day, month_abr))
        return args
    else:
        print('Found rinex gaps of {} days, recovering last rinex day {}.'.format(
            day_diff, last_day))
        args.T02_path = days[-1]
        return args


def process_single_T02_file(T02_fileobj, year, savepath, verbose=False):
    """checks if file already exists in savepath, if not process it and save it there"""
    import logging
    import shutil
    import os
    import subprocess
    from pathlib import Path
    from subprocess import CalledProcessError
    logger = logging.getLogger('axis_rinex_processer')
    station, doy, hrl = parse_T02_file(T02_fileobj)
    yy = str(year)[2:4]
    filename = '{}{}{}.{}d.gz'.format(station, doy, hrl, yy)
    to_copy_path = savepath / str(year) / doy
    if not to_copy_path.is_dir():
        if verbose:
            logger.info('folder for doy {} not found, creating folder.'.format(doy))
        os.mkdir(savepath / str(year) / doy)
    # check for already copied file:
    if (to_copy_path / filename).is_file():
        if verbose:
            logger.warning('{} already exists in {}, skipping...'.format(filename, to_copy_path))
        return

    # if file above does not exist, first copy T02 to ziskin home folder:
    # for some strange reason, runpkr00 only runs on files from my home folder...
    T02_filename = T02_fileobj.as_posix().split('/')[-1]
    ziskin_home = Path('/home/ziskin')
    shutil.copy(T02_fileobj, ziskin_home / T02_filename)
    # convert to TGD:
    command = 'runpkr00 -d -g {}'.format(ziskin_home / T02_filename)
    try:
        subprocess.run(command, shell=True, check=True)
    except CalledProcessError:
        return CalledProcessError
    # delete T02:
    (ziskin_home / T02_filename).unlink()
    tgd_filename = T02_filename.replace('T02', 'tgd')
    shutil.copy(ziskin_home / tgd_filename, to_copy_path / tgd_filename)
    (ziskin_home / tgd_filename).unlink()
    if verbose:
        logger.info('{} was copied to {}'.format(tgd_filename, to_copy_path))
    # first, filenames:
    o_filename = T02_filename.replace('T02', '{}o'.format(yy))
    d_filename = T02_filename.replace('T02', '{}d'.format(yy))
    # dZ_filename = d_filename + '.Z'

    # run teqc to convert to o:
    command = 'teqc -tr d {} > {}'.format(to_copy_path / tgd_filename, to_copy_path / o_filename)
    try:
        subprocess.run(command, shell=True, check=True)
    except CalledProcessError:
        return CalledProcessError
    # delete tgd:
    try:
        (to_copy_path / tgd_filename).unlink()
    except FileNotFoundError:
        logger.warning('{} not found so no delete done.'.format(tgd_filename))
    # hatakanaka compress:
    command = 'RNX2CRX {}'.format(to_copy_path / o_filename)
    try:
        subprocess.run(command, shell=True, check=True)
    except CalledProcessError:
        return CalledProcessError
    # delete o:
    try:
        (to_copy_path / o_filename).unlink()
    except FileNotFoundError:
        logger.warning('{} not found so no delete done.'.format(o_filename))
    # gzip compress:
    # command = 'tar -czvf {} {}'.format(to_copy_path / dZ_filename, to_copy_path / d_filename)
    command = 'gzip {}'.format(to_copy_path / d_filename)
    try:
        subprocess.run(command, shell=True, check=True)
    except CalledProcessError:
        return CalledProcessError
    # delete d:
    try:
        (to_copy_path / d_filename).unlink()
    except FileNotFoundError:
        logger.warning('{} not found so no delete done.'.format(d_filename))
    return


def doy_to_datetime(doy, year, return_day_and_month=True):
    import datetime
    import calendar
    dt = datetime.datetime(year, 1, 1) + datetime.timedelta(int(doy) - 1)
    month_abr = calendar.month_abbr[dt.month]
    day_of_month = str(dt.day)
    if return_day_and_month:
        return month_abr, day_of_month
    else:
        return dt


def parse_T02_file(T02_fileobj):
    """input is T02_fileobj (path), output is station, doy and hour_letter"""
    if not T02_fileobj.is_file():
        return [None, None, None]
    T02 = T02_fileobj.as_posix().split('/')[-1]
    station = T02[:4]
    hour_letter = T02.split('.')[0][-1]
    day_of_year = T02[4:7]
    return station, day_of_year, hour_letter


def process_T02(args):
    import os
    import logging
    from pathlib import Path
    from aux_gps import path_glob
    from subprocess import CalledProcessError
    logger = logging.getLogger('axis_rinex_processer')
    # run analyse each hour:
    analyse_T02_files()
    if args.is_T02_path_year is None:
        path_year = True
    else:
        path_year = args.is_T02_path_year
    if args.mode is None:
        mode = 'whole'
    else:
        mode = args.mode
    if args.year is None:
        year = 2021
    else:
        year = args.year
    savepath = args.savepath
    if args.T02_path is None:
        T02_path = Path('/home/axis-gps')
    else:
        T02_path = args.T02_path
    # create year folder if not created yet:
    if not (savepath / str(year)).is_dir():
        logger.info('folder for year {} not found, creating folder.'.format(year))
        os.mkdir(savepath / str(year))
    # scan for files in T02_path:
    if path_year:
        months = path_glob(T02_path, 'Month.*')
        if mode == 'last_doy':
            # check for doy folders in savepath and flag last_doy:
            doys = sorted(path_glob(savepath / str(year), '*/'))
            doys = sorted([x for x in doys if x.is_dir()])
            last_doy = doys[-1].as_posix().split('/')[-1]
            month_abr, day_of_month = doy_to_datetime(last_doy, year)
            month = [x for x in months if month_abr in x.as_posix()][0]
            days = path_glob(month, 'Day.*')
            try:
                day = [x for x in days if day_of_month in x.as_posix()][0]
                T02s = path_glob(day, '*.T02')
                for T02 in T02s:
                    process_single_T02_file(T02, year, savepath, verbose=True)
            except IndexError:
                logger.warning('didnt find any files for {} doy.'.format(last_doy))
                return
            # also get doy+1 and check if any files exist:
            last_doyp1 = str(int(last_doy) + 1)
            month_abrp1, day_of_monthp1 = doy_to_datetime(last_doyp1, year)
            month = [x for x in months if month_abr in x.as_posix()][0]
            days = path_glob(month, 'Day.*')
            try:
                day = [x for x in days if day_of_monthp1 in x.as_posix()][0]
                T02s = path_glob(day, '*.T02')
                for T02 in T02s:
                    try:
                        process_single_T02_file(T02, year, savepath, verbose=True)
                    except CalledProcessError:
                        logger.warning('process failed on {}, skipping..'.format(T02))
            except IndexError:
                logger.warning('didnt find any files for {} doy.'.format(last_doyp1))
                return
        else:
            for month in months:
                days = path_glob(month, 'Day.*')
                for day in days:
                    T02s = path_glob(day, '*.T02')
                    for T02 in T02s:
                        if mode == 'single_file':
                            process_single_T02_file(T02, year, savepath, verbose=True)
                            return
                        elif mode == 'whole':
                            process_single_T02_file(T02, year, savepath, verbose=False)
    else:
        T02s = path_glob(T02_path, '*.T02')
        for T02 in T02s:
            try:
                process_single_T02_file(T02, year, savepath, verbose=True)
            except CalledProcessError:
                logger.warning('process failed on {}, skipping..'.format(T02))
    return


if __name__ == '__main__':
    import argparse
    import sys
    from aux_gps import configure_logger
    logger = configure_logger('axis_rinex_processer')
    parser = argparse.ArgumentParser(description='a command line tool for converting T02 to RINEX and saving it to directory structure')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # remove this line: optional = parser...
    required.add_argument('--savepath', help="a full path to main save directory, e.g., /home/ziskin/Work_Files/PW_yuval/axis-rinex", type=check_path)
    # required.add_argument('--channel', help="10 mins channel name , e.g., TD, BP or RH",
    #                       choices=channels)

#                          metavar=str(cds.start_year) + ' to ' + str(cds.end_year))
    optional.add_argument('--year', help='year of rinex files', type=check_year)
    optional.add_argument('--mode', help='which mode to run', type=str, choices=['last_doy', 'whole', 'single_file'])
    optional.add_argument('--T02_path', help='where the T02 files are', type=check_path)
    # optional.add_argument('--is_T02_path_year', help='are the T02 files in T02_path in year/month structure', type=bool)
    optional.add_argument("--is_T02_path_year", type=str2bool, nargs='?',
                          const=True, default=True,
                          help="are the T02 files in T02_path in year/month structure")
    parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()
    # print(parser.format_help())
#    # print(vars(args))
    if args.savepath is None:
        print('savepath is a required argument, run with -h...')
        sys.exit()
#    elif args.field is None:
#        print('field is a required argument, run with -h...')
#        sys.exit()
    args = check_for_missing_rinex_in_axis_path(args)
    print(args)
    process_T02(args)
