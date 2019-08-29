#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 13:58:05 2019

@author: ziskin
"""


def read_organize_rinex(path):
    """read and organize the rinex file names for 30 hour run"""
    from aux_gps import get_timedate_and_station_code_from_rinex
    import pandas as pd
    import numpy as np
    import logging
    logger = logging.getLogger('gipsyx')
    dts = []
    rfns = []
    logger.info('reading and organizing rinex files in {}'.format(path))
    for file_and_path in path.glob('*.Z'):
        filename = file_and_path.as_posix().split('/')[-1][0:12]
        dt, station = get_timedate_and_station_code_from_rinex(filename)
        dts.append(dt)
        rfns.append(filename)
    full_time = pd.date_range(dts[0], dts[-1], freq='1D')
    df = pd.DataFrame(data=rfns, index=dts)
    df = df.reindex(full_time)
    df = df.sort_index()
    df.columns = ['rinex']
    df.index.name = 'time'
    df['30hr'] = np.nan
    df.iat[0, 1] = 0
    df.iat[-1, 1] = 0
    for i in range(1, len(df)-1):
        nums = np.array([i-1, i, i+1])
        nan3days = df.iloc[nums, 0].isnull()
        if not nan3days[0] and not nan3days[1] and not nan3days[2]:
            # print('all')
            df.iat[i, 1] = 1
        elif not nan3days[0] and not nan3days[1] and nan3days[2]:
            # print('0')
            df.iat[i, 1] = 0
        elif nan3days[0] and not nan3days[1] and not nan3days[2]:
            # print('00')
            df.iat[i, 1] = 0
        elif not nan3days[1] and nan3days[0] and nan3days[2]:
            # print('000')
            df.iat[i, 1] = 0
        # print(i, nan3days, df.iat[i, 1])
        # input("Press Enter to continue...")
    return df


def get_var(varname):
    """get a linux shell var (without the $)"""
    import subprocess
    CMD = 'echo $%s' % varname
    p = subprocess.Popen(
        CMD,
        stdout=subprocess.PIPE,
        shell=True,
        executable='/bin/bash')
    return p.stdout.readlines()[0].strip().decode("utf-8")


def check_path(path):
    import os
    from pathlib import Path
    path = str(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(path + ' does not exist...')
    return Path(path)


def check_file_in_cwd(filename):
    from pathlib import Path
    cwd = Path().cwd()
    file_and_path = cwd / filename
    if not file_and_path.is_file():
        raise argparse.ArgumentTypeError(
            '{} does not exist at {}'.format(
                filename, cwd))
    return file_and_path


def prepare_gipsyx_for_run_one_station(rinexpath, staDb, hours):
    """rinex editing and merging command-line utility, two main modes:
        1)24hr run(single rinex file)
        2)30hr run(3 consecutive rinex files"""
    import subprocess
    from subprocess import CalledProcessError
    import logging
    import numpy as np
    from aux_gps import get_timedate_and_station_code_from_rinex
    logger = logging.getLogger('gipsyx')
    # first check for GCORE path:
    if len(get_var('GCORE')) == 0:
        raise ValueError('Run ~/GipsyX-1.1/rc_GipsyX.sh first !')
    logger.info(
        'starting rinex editing/merging utility for {} hours run.'.format(hours))
    logger.info('working with {}'.format(staDb))
    rinex_df = read_organize_rinex(rinexpath)
    if hours == '24':
        for date, row in rinex_df.iterrows():
            rfn = row['rinex']
            if rfn == np.nan:
                continue
            dt, station = get_timedate_and_station_code_from_rinex(rfn)
            hr24 = rinexpath / '24hr'
            try:
                hr24.mkdir()
            except FileExistsError:
                print('{} already exists, rewriting files...'.format(hr24))
            dr_path = hr24 / '{}.dr.gz'.format(rfn)
            filename = rfn + '.Z'
            file_and_path = rinexpath / filename
            logger.info('processing {} ({})'.format(filename,
                                                    dt.strftime('%Y-%m-%d')))
            command = 'rnxEditGde.py -data {} -out {} -staDb {} > {}.log 2>{}.err'.format(
                file_and_path.as_posix(), dr_path.as_posix(), staDb.as_posix(), rfn, rfn)
            try:
                subprocess.run(command, shell=True, check=True)
            except CalledProcessError:
                logger.error('rnxEditGde.py failed on {}...'.format(filename))
    logger.info('Done!')
    # elif hours == 30:
    return


def run_gipsyx_for_station(rinexpath, savepath, staDb, hours):
    """main running wrapper for gipsyX. two main modes:
        1)rnxEdit mode of 24hr run(single file)
        2)rnxEdit mode of 30hr run(three consecutive files)"""
    from pathlib import Path
    import subprocess
    from subprocess import CalledProcessError
    from subprocess import TimeoutExpired
    import logging
    import shutil
    from aux_gps import get_timedate_and_station_code_from_rinex
    logger = logging.getLogger('gipsyx')
    # first check for GCORE path:
    if len(get_var('GCORE')) == 0:
        raise ValueError('Run ~/GipsyX-1.1/rc_GipsyX.sh first !')
    logger.info('working with {}'.format(staDb))
    for file_and_path in rinexpath.glob('*.Z'):
        filename = file_and_path.as_posix().split('/')[-1][0:-2]
        dt, station = get_timedate_and_station_code_from_rinex(filename)
        dr_path = Path.cwd() / '{}_data.dr.gz'.format(station) 
        logger.info(
            'processing {} ({})'.format(
                filename,
                dt.strftime('%Y-%m-%d')))
#        orig_final = Path.cwd() / 'smoothFinal.tdp'
        final_tdp = filename + '_smoothFinal.tdp'
        if (savepath / final_tdp).is_file():
            logger.warning('{} already exists, skipping...'.format(final_tdp))
            continue
        if staDb is None:
            command = 'gd2e.py -rnxFile {} > {}.log 2>{}.err'.format(
                file_and_path.as_posix(), filename, filename)
        else:
            command0 = 'rnxEditGde.py -data {} -type rinex -out {} -staDb {} > {}_rnxEdit.log 2>{}_rnxEdit.err'.format(
                file_and_path.as_posix(), dr_path.as_posix(), staDb.as_posix(), filename, filename)
            orig_rnxedit_paths = ['{}_rnxEdit.log'.format(filename), '{}_rnxEdit.err'.format(filename)]
            orig_paths = [Path.cwd() / x for x in orig_rnxedit_paths]
            dest_paths = [savepath / x for x in orig_rnxedit_paths]
            try:
                subprocess.run(command0, shell=True, check=True)
                for orig, dest in zip(orig_paths, dest_paths):
                    shutil.move(orig.resolve(), dest.resolve())
            except:
                logger.error('rnxEditGde.py failed on {}, copying log files.'.format(filename))
                for orig, dest in zip(orig_paths, dest_paths):
                    shutil.move(orig.resolve(), dest.resolve())
                continue
            command = 'gd2e.py -drEditedFile {} -recList {} -staDb {} > {}.log 2>{}.err'.format(
                dr_path.as_posix(), station, staDb.as_posix(), filename, filename)
        orig_filenames = [
                filename + '.err',
                filename + '.log',
                'smoothFinal.tdp']
        try:
            subprocess.run(command, shell=True, check=True, timeout=300)
            orig_filenames_paths = [Path.cwd() / x for x in orig_filenames]
            dest_filenames = [filename + '.err', filename + '.log', final_tdp]
        except CalledProcessError:
            logger.error('gipsyx failed on {}, copying log files.'.format(filename))
            orig_filenames_paths = [Path.cwd() / x for x in orig_filenames[0:2]]
            dest_filenames = [filename + '.err', filename + '.log']
        except TimeoutExpired:
            logger.error('gipsyx timed out on {}, copying log files.'.format(filename))
            orig_filenames_paths = [Path.cwd() / x for x in orig_filenames[0:2]]
            with open(orig_filenames_paths[0], 'a') as f:
                f.write('GipsyX run has Timed out !')
            dest_filenames = [filename + '.err', filename + '.log']
        dest_filenames_paths = [savepath / x for x in dest_filenames]
        for orig, dest in zip(orig_filenames_paths, dest_filenames_paths):
            # orig.replace(dest)
            shutil.move(orig.resolve(), dest.resolve())
    logger.info('Done!')
    return


if __name__ == '__main__':
    import argparse
    import sys
    from aux_gps import configure_logger
    logger = configure_logger(name='gipsyx')
    parser = argparse.ArgumentParser(
        description='a command line tool for downloading all 10mins stations from the IMS with specific variable')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # remove this line: optional = parser...
    required.add_argument(
        '--savepath',
        help="a full path to save the raw output files, e.g., /home/ziskin/Work_Files/PW_yuval/gipsyx_resolved/TELA",
        type=check_path)
    required.add_argument(
        '--rinexpath',
        help="a full path to the rinex path of the station, /home/ziskin/Work_Files/PW_yuval/rinex/TELA",
        type=check_path)
    optional.add_argument(
        '--staDb',
        help='add a station DB file for antennas in rinexpath',
        type=check_file_in_cwd)
    optional.add_argument('--edit', help='call only rinex merge/edit utility',
                          choices=['True', 'False'], default=False)
    optional.add_argument('--hours', help='either 24 or 30 hours gipsyX run/rinex merge/edit',
                          choices=['24', '30'], default='24')
    parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()
    if args.rinexpath is None:
        print('rinexpath is a required argument, run with -h...')
        sys.exit()
    if args.staDb is None:
        args.staDb = '$GOA_VAR/sta_info/sta_db_qlflinn'
    if args.edit == 'True':
        prepare_gipsyx_for_run_one_station(args.rinexpath, args.staDb,
                                           args.hours)
    else:
        if args.savepath is None:
            print('savepath is a required argument, run with -h...')
            sys.exit()
        run_gipsyx_for_station(args.rinexpath, args.savepath, args.staDb,
                               args.hours)
