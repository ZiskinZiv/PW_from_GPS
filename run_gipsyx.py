#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 13:58:05 2019
Methodology:
    0) run code with --prep drdump on all rinex files
    1) run code with either --prep edit24hr or --prep edit30hr on all
    datarecords files
    2) run code with --gd2e 30hr or --gd2e 24hr for either solve ppp on 24hr
    folder or 30hr folder
@author: ziskin
"""


def move_files(path_orig, path_dest, files):
    """move files (a list containing the file names) and move them from
    path_orig to path_dest"""
    import shutil
    orig_filenames_paths = [path_orig / x for x in files]
    dest_filenames_paths = [path_dest / x for x in files]
    sizes = [x.stat().st_size for x in orig_filenames_paths]
    # delete files if size =0:
    for orig, dest, size in zip(
            orig_filenames_paths, dest_filenames_paths, sizes):
        if size == 0:
            orig.resolve().unlink()
        else:
            shutil.move(orig.resolve(), dest.resolve())
    return


def read_organize_rinex(path, glob_str='*.Z'):
    """read and organize the rinex file names for 30 hour run"""
    from aux_gps import get_timedate_and_station_code_from_rinex
    import pandas as pd
    import numpy as np
    import logging
    logger = logging.getLogger('gipsyx')
    dts = []
    rfns = []
    logger.info('reading and organizing rinex files in {}'.format(path))
    for file_and_path in path.glob(glob_str):
        filename = file_and_path.as_posix().split('/')[-1][0:12]
        dt, station = get_timedate_and_station_code_from_rinex(filename)
        dts.append(dt)
        rfns.append(filename)

    df = pd.DataFrame(data=rfns, index=dts)
    df = df.sort_index()
    full_time = pd.date_range(df.index[0], df.index[-1], freq='1D')
    df = df.reindex(full_time)
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


def prepare_gipsyx_for_run_one_station(rinexpath, staDb, prep, rewrite):
    """rinex editing and merging command-line utility, 3 values for prep:
        0) drdump: run dataRecordDump on all rinex files in rinexpath
        1) edit24hr: run rnxEditGde.py with staDb on all datarecords files in
            rinexpath / dr, savethem to rinexpath / 24hr
        2) edit30hr: run drMerge.py on 3 consecutive datarecords files in
            rinexpath / dr (rnxEditGde.py with staDb on lonely datarecords files)
            and then rnxEditGde.py with staDb on merged datarecords files,
            save them to rinexpath / 30hr
        rewrite: overwrite all files - supported with all modes of prep"""
    import subprocess
    from subprocess import CalledProcessError
    import logging
    import pandas as pd
    from itertools import count
    from pathlib import Path

    def run_dataRecorDump_on_all_files(rinexpath, dr_path, rewrite):
        try:
            dr_path.mkdir()
        except FileExistsError:
            logger.info(
                '{} already exists, using that folder.'.format(dr_path))
        for file_and_path in rinexpath.glob('*.Z'):
            filename = file_and_path.as_posix().split('/')[-1][0:12]
            dr_file = dr_path / '{}.dr.gz'.format(filename)
            if not rewrite:
                if (dr_file).is_file():
                    logger.warning(
                        '{} already exists in {}, skipping...'.format(
                            filename + '.dr.gz', dr_path))
                    continue
            logger.info('processing {}'.format(filename))
            files_to_move = [filename + x for x in ['.log', '.err']]
            command = 'dataRecordDump -rnx {} -drFileNmOut {} > {}.log 2>{}.err'.format(
                file_and_path.as_posix(), dr_file.as_posix(), filename, filename)
            try:
                subprocess.run(command, shell=True, check=True)
                next(succ)
            except CalledProcessError:
                logger.error('dataRecordDump failed on {}...'.format(filename))
                next(failed)
            move_files(Path().cwd(), dr_path, files_to_move)
        return

    def run_rnxEditGde(filename, dr_path, hr24, suffix=24, rewrite):
        rfn = filename[0:12]
        station = rfn[0:4].upper()
        dr_edited_file = hr24 / '{}_edited{}hr.dr.gz'.format(rfn, suffix)
        file_and_path = dr_path / filename
        if not rewrite:
            if (dr_edited_file).is_file():
                logger.warning(
                    '{} already exists in {}, skipping...'.format(
                        filename + '_edited24hr.dr.gz', hr24))
                return
        logger.info('processing {} ({})'.format(filename,
                                                date.strftime('%Y-%m-%d')))
        files_to_move = [rfn + x for x in ['.log', '.err']]
        command = 'rnxEditGde.py -type datarecord -recNm {} -data {} -out {} -staDb {} > {}.log 2>{}.err'.format(
            station, file_and_path.as_posix(), dr_edited_file.as_posix(), staDb.as_posix(), rfn, rfn)
        try:
            subprocess.run(command, shell=True, check=True)
            next(succ)
        except CalledProcessError:
            next(failed)
            logger.error('rnxEditGde.py failed on {}...'.format(filename))
        move_files(Path().cwd(), hr24, files_to_move)
        return

    def run_30hr_drMerge(rfns, dr_path, hr30, rewrite):
        from aux_gps import get_timedate_and_station_code_from_rinex
        dts = [get_timedate_and_station_code_from_rinex(x, True) for x in rfns]
        return
    logger = logging.getLogger('gipsyx')
    # first check for GCORE path:
    if len(get_var('GCORE')) == 0:
        raise ValueError('Run ~/GipsyX-1.1/rc_GipsyX.sh first !')
    logger.info(
        'starting preparation utility utility for gipsyX run.')
    logger.info('working with {}'.format(staDb))
    if rewrite:
        logger.warning('overwrite files mode initiated.')
    rinex_df = read_organize_rinex(rinexpath)
    succ = count(1)
    failed = count(1)
    # rinex_df = rinex_df.fillna(999)
    dr_path = rinexpath / 'dr'
    if prep == 'drdump':
        logger.info('running dataRecordDump for all files.')
        run_dataRecorDump_on_all_files(rinexpath, dr_path, rewrite)
    elif prep == 'edit24hr':
        logger.info('running rnxEditGde.py with 24hr setting for all files.')
        hr24 = rinexpath / '24hr'
        try:
            hr24.mkdir()
        except FileExistsError:
            logger.info('{} already exists, using that folder.'.format(hr24))
        for date, row in rinex_df.iterrows():
            rfn = row['rinex']
            if pd.isna(rfn):
                continue
            filename = rfn + '.dr.gz'
            run_rnxEditGde(filename, dr_path, hr24, rewrite)
    elif prep == 'edit30hr':
        logger.info(
            'running drMerge.py/rnxEditGde.py with 30hr setting for all files(when available).')
        hr30 = rinexpath / '30hr'
        try:
            hr30.mkdir()
        except FileExistsError:
            logger.info('{} already exists, using that folder.'.format(hr30))
        for i, date in enumerate(rinex_df.index):
            rfn = rinex_df.loc[date, 'rinex']
            if pd.isna(rfn):
                continue
            if rinex_df.loc[date, '30hr'] == 0:
                logger.warning(
                    '{} is lonely, doing 24hr prep only...'.format(rfn))
                run_rnxEditGde(rfn, dr_path, hr30, rewrite)
            elif rinex_df.loc[date, '30hr'] == 1:
                yesterday = rinex_df.index[i - 1]
                tommorow = rinex_df.index[i + 1]
                rfns = [rinex_df.loc[yesterday, 'rinex'],
                        rfn, rinex_df.loc[tommorow, 'rinex']]
                merged_file = run_30hr_drMerge(rfns, dr_path, hr30, rewrite)
                run_rnxEditGde(merged_file, Path().cwd(), hr30, 30, rewrite)
                # delete merged file
                print('1')
    logger.info('Done!')
    total = next(failed) + next(succ) - 2
    succses = next(succ) - 2
    failure = next(failed) - 2
    logger.info('Total files: {}, success: {}, failed: {}'.format(
            total, succses, failure))
    return


def run_gipsyx_for_station(rinexpath, savepath, staDb, rewrite):
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
    """--prep: prepare mode
        choices: 1) drdump 2) edit24hr 3) edit30hr"""
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
    optional.add_argument('--prep', help='call rinex rnxEditGde/drMerge or dataRecorDump',
                          choices=['drdump', 'edit24hr', 'edit30hr'], default='drdump')
    optional.add_argument(
            '--rewrite',
            dest='rewrite',
            action='store_true',
            help='overwrite files in prep/run mode')

    parser._action_groups.append(optional)  # added this line
    parser.set_defaults(rewrite=False)
    args = parser.parse_args()
    if args.rinexpath is None:
        print('rinexpath is a required argument, run with -h...')
        sys.exit()
    if args.staDb is None:
        args.staDb = '$GOA_VAR/sta_info/sta_db_qlflinn'
    if args.prep is not None:
        prepare_gipsyx_for_run_one_station(args.rinexpath, args.staDb,
                                           args.prep, args.rewrite)
    else:
        if args.savepath is None:
            print('savepath is a required argument, run with -h...')
            sys.exit()
        run_gipsyx_for_station(args.rinexpath, args.savepath, args.staDb,
                               args.rewrite)
