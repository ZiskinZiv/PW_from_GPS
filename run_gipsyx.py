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


def move_files(path_orig, path_dest, files, out_files=None, verbose=False):
    """move files (a list containing the file names) and move them from
    path_orig to path_dest"""
    import shutil
    import logging
    logger = logging.getLogger('gipsyx')
    if isinstance(files, str):
        files = [files]
    if out_files is not None:
        if isinstance(out_files, str):
            out_files = [out_files]
    orig_filenames_paths = [path_orig / x for x in files]
    if out_files is None:
        out_files = files
    dest_filenames_paths = [path_dest / x for x in out_files]
    # delete files if size =0:
    for file, orig, dest in zip(
            files, orig_filenames_paths, dest_filenames_paths):
        # check for file existance in orig:
        if not orig.is_file():
            if verbose:
                logger.warning('{} does not exist in {}'.format(file, orig))
            continue
        # check if its size is 0:
        if orig.stat().st_size == 0:
            orig.resolve().unlink()
        else:
            shutil.move(orig.resolve(), dest.resolve())
    return


def read_organize_rinex(path, glob_str='*.Z', date_range=None):
    """read and organize the rinex file names for 30 hour run"""
    from aux_gps import get_timedate_and_station_code_from_rinex
    from aux_gps import path_glob
    from aux_gps import slice_task_date_range
    import pandas as pd
    import numpy as np
    import logging
    logger = logging.getLogger('gipsyx')
    dts = []
    rfns = []
    stations = []
    logger.info('reading and organizing rinex files in {}'.format(path))
    files = path_glob(path, glob_str)
    if date_range is not None:
        files = slice_task_date_range(files, date_range, 'read_organize_rinex')
    for file_and_path in files:
        filename = file_and_path.as_posix().split('/')[-1][0:12]
        dt, station = get_timedate_and_station_code_from_rinex(filename)
        stations.append(station)
        dts.append(dt)
        rfns.append(filename)
    # check for more station than one:
    if len(set(stations)) > 1:
        raise Exception('mixed station names in folder {}'.format(path))
    df = pd.DataFrame(data=rfns, index=dts)
    df = df.sort_index()
    df = df[~df.index.duplicated()]
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


def prepare_gipsyx_for_run_one_station(rinexpath, staDb, prep, rewrite,
                                       date_range=None):
    """rinex editing and merging command-line utility, 3 values for prep:
        0) drdump: run dataRecordDump on all rinex files in rinexpath
        1) edit24hr: run rnxEditGde.py with staDb on all datarecords files in
            rinexpath / dr, savethem to rinexpath / 24hr
        2) edit30hr: run drMerge.py on 3 consecutive datarecords files in
            rinexpath / dr (rnxEditGde.py with staDb on lonely datarecords
            files)
            and then rnxEditGde.py with staDb on merged datarecords files,
            save them to rinexpath / 30hr
        rewrite: overwrite all files - supported with all modes of prep"""
    import subprocess
    from subprocess import CalledProcessError
    from subprocess import TimeoutExpired
    from aux_gps import slice_task_date_range
    from aux_gps import path_glob
    import logging
    import pandas as pd
    global cnt
    global tot
    # from itertools import count
    from pathlib import Path

    def run_dataRecorDump_on_all_files(rinexpath, out_path, rewrite,
                                       date_range=None):
        """runs dataRecordDump on all files in rinexpath(where all and only
        rinex files exist), saves the datarecord files to out_path. rewrite
        is a flag that overwrites the files in out_path even if they already
        exist there."""
        logger.info('running dataRecordDump...')
        est_time_per_single_run = 1.0  # seconds
        out_path.mkdir(parents=True, exist_ok=True)
        files = path_glob(rinexpath, '*.Z')
        files_already_done = path_glob(out_path, '*.dr.gz', True)
        if date_range is not None:
            files = slice_task_date_range(files, date_range, 'drdump')
            files_already_done = slice_task_date_range(files_already_done,
                                                       date_range,
                                                       'already done drdump')
        tot = len(files)
        logger.info('found {} rinex Z files in {} to run.'.format(tot,
                                                                  rinexpath))
        tot_final = len(files_already_done)
        logger.info('found {} data records dr.gz files in {}'.format(tot_final,
                    out_path))
        tot_to_run = tot - tot_final
        dtt = pd.to_timedelta(est_time_per_single_run, unit='s') * tot_to_run
        logger.info('estimated time to completion of run: {}'.format(dtt))
        logger.info('check again in {}'.format(pd.Timestamp.now() + dtt))
        for file_and_path in files:
            filename = file_and_path.as_posix().split('/')[-1][0:12]
            dr_file = out_path / '{}.dr.gz'.format(filename)
            if not rewrite:
                if (dr_file).is_file():
                    logger.warning(
                        '{} already exists in {}, skipping...'.format(
                            filename + '.dr.gz', out_path))
                    cnt['succ'] += 1
                    continue
            logger.info('processing {} ({}/{})'.format(
                filename, cnt['succ'] + cnt['failed'], tot))
            files_to_move = [filename + x for x in ['.log', '.err']]
            command = 'dataRecordDump -rnx {} -drFileNmOut {} > {}.log 2>{}.err'.format(
                file_and_path.as_posix(), dr_file.as_posix(), filename, filename)
            try:
                subprocess.run(command, shell=True, check=True)
#                next(succ)
                cnt['succ'] += 1
            except CalledProcessError:
                logger.error('dataRecordDump failed on {}...'.format(filename))
#                next(failed)
                cnt['failed'] += 1
            except TimeoutExpired:
                logger.error('dataRecordDump timed out on {}, copying log files.'.format(rfn))
                # next(failed)
                cnt['failed'] += 1
                with open(Path().cwd() / files_to_move[1], 'a') as f:
                    f.write('dataRecordDump has Timed out !')
            move_files(Path().cwd(), out_path, files_to_move)
        return

    def run_rnxEditGde(filename, in_path, out_path, rewrite, suffix=24):
        """runs rnxEditGde on filename that exists in in_path and writes the
        edited file (with suffix) to out_path. it first checks wether filename
        exists in out_path and if it is, it skipps this filename. rewrite flag
        overwrites the filename regardless."""
        rfn = filename[0:12]
        station = rfn[0:4].upper()
        dr_edited_file = out_path / '{}_edited{}hr.dr.gz'.format(rfn, suffix)
        file_and_path = in_path / filename
        if not rewrite:
            if (dr_edited_file).is_file():
                logger.warning(
                    '{} already exists in {}, skipping...'.format(
                        filename, out_path))
                cnt['succ'] += 1
                return
        logger.info(
            'processing {} ({}, {}/{})'.format(
                filename,
                date.strftime('%Y-%m-%d'), cnt['succ'] + cnt['failed'], tot))
        files_to_move = [rfn + x for x in ['.log', '.err']]
        command = 'rnxEditGde.py -type datarecord -recNm {} -data {} -out {} -staDb {} > {}.log 2>{}.err'.format(
            station, file_and_path.as_posix(), dr_edited_file.as_posix(),
            staDb.as_posix(), rfn, rfn)
        try:
            subprocess.run(command, shell=True, check=True)
#            next(succ)
            cnt['succ'] += 1
        except CalledProcessError:
#            next(failed)
            cnt['failed'] += 1
            logger.error('rnxEditGde.py failed on {}...'.format(filename))
        except TimeoutExpired:
            logger.error('rnxEditGde.py timed out on {}, copying log files.'.format(rfn))
            # next(failed)
            cnt['failed'] += 1
            with open(Path().cwd() / files_to_move[1], 'a') as f:
                f.write('rnxEditGde.py has Timed out !')
        move_files(Path().cwd(), out_path, files_to_move)
        return

    def run_drMerge(filenames, in_path, duration):
        from aux_gps import get_timedate_and_station_code_from_rinex
        rfns = [x[0:12] for x in filenames]
        dts = [get_timedate_and_station_code_from_rinex(x, True) for x in rfns]
        if duration == '30hr':
            start = dts[0].strftime('%Y-%m-%d') + ' 21:00:00'
            end = dts[2].strftime('%Y-%m-%d') + ' 03:00:00'
        dr_merged_file = Path().cwd() / '{}_merged.dr.gz'.format(rfns[1])
        logger.info('merging {}, {} and {} to {}'.format(*rfns,rfns[1] + '_merged.dr.gz'))
        f_and_paths = [in_path / x for x in filenames]
        files_to_move = [rfn + x for x in ['_drmerge.log', '_drmerge.err']]
        command = 'drMerge.py -inFiles {} {} {} -outFile {} -start {} -end {} > {}.log 2>{}.err'.format(
                f_and_paths[0].as_posix(), f_and_paths[1].as_posix(),
                f_and_paths[2].as_posix(), dr_merged_file.as_posix(),
                start, end, rfn + '_drmerge', rfn + '_drmerge')
        try:
            subprocess.run(command, shell=True, check=True, timeout=60)
        except CalledProcessError:
            logger.error('drMerge.py failed on {}...'.format(filenames))
        except TimeoutExpired:
            logger.error('drMerge.py timed out on {}, copying log files.'.format(filenames))
            # next(failed)
            cnt['failed'] += 1
            with open(Path().cwd() / files_to_move[1], 'a') as f:
                f.write('drMerge.py run has Timed out !')
            return None
        move_files(Path().cwd(), Path().cwd(), files_to_move)
        return rfns[1] + '_merged.dr.gz'
    logger = logging.getLogger('gipsyx')
    logger.info(
        'starting preparation utility utility for gipsyX run.')
    logger.info('working with {}'.format(staDb))
    if rewrite:
        logger.warning('overwrite files mode initiated.')
    rinex_df = read_organize_rinex(rinexpath, date_range=date_range)
    cnt = {'succ': 0, 'failed': 0}
#    succ = count(1)
#    failed = count(1)
    # rinex_df = rinex_df.fillna(999)
    dr_path = rinexpath / 'dr'
    if prep == 'drdump':
        run_dataRecorDump_on_all_files(rinexpath, dr_path, rewrite,
                                       date_range=None)
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
        est_time_per_single_run = 4.0  # seconds
        tot = rinex_df['30hr'].value_counts().sum()  # len(path_glob(dr_path, '*.dr.gz'))
        logger.info('found {} data records dr.gz files in {} to run.'.format(tot, dr_path))
        files_already_done = path_glob(hr30, '*.dr.gz', True)
        if date_range is not None:
            files_already_done = slice_task_date_range(files_already_done,
                                                       date_range,
                                                       'already done edit30hr')
        tot_final = len(files_already_done)
        logger.info('found {} edited and merged dr.gz files in {}'.format(tot_final,
                    hr30))
        tot_to_run = tot - tot_final
        dtt = pd.to_timedelta(est_time_per_single_run, unit='s') * tot_to_run
        logger.info('estimated time to completion of run: {}'.format(dtt))
        logger.info('check again in {}'.format(pd.Timestamp.now() + dtt))
        try:
            hr30.mkdir()
        except FileExistsError:
            logger.info('{} already exists, using that folder.'.format(hr30))
        for i, date in enumerate(rinex_df.index):
            rfn = rinex_df.loc[date, 'rinex']
            # missing datarecords files:
            if pd.isna(rfn):
                continue
            # check for non-consecutive datarecords files:
            if rinex_df.loc[date, '30hr'] == 0:
                logger.warning(
                    '{} is lonely, doing 24hr prep only...'.format(rfn))
                filename = rfn + '.dr.gz'
                run_rnxEditGde(filename, dr_path, hr30, rewrite)
            # check for 3 consecutive datarecords files:
            elif rinex_df.loc[date, '30hr'] == 1:
                merged_filename = '{}_edited30hr.dr.gz'.format(rfn)
                if not rewrite:
                    if (hr30 / merged_filename).is_file():
                        logger.warning(
                                '{} already merged and edited in {}, skipping...'.format(
                                        merged_filename, hr30))
                        cnt['succ'] += 1
                        continue
                yesterday = rinex_df.index[i - 1]
                tommorow = rinex_df.index[i + 1]
                rfns = [rinex_df.loc[yesterday, 'rinex'],
                        rfn, rinex_df.loc[tommorow, 'rinex']]
                filenames = [x + '.dr.gz' for x in rfns]
                # merge them from yesterday 21:00 to tommorow 03:00
                # i.e., 30 hr:
                merged_file = run_drMerge(filenames, dr_path, duration='30hr')
                if merged_file is None:
                    continue
                # rnxEditGde the merged datarecord with staDb and move
                # to 30hr folder:
                run_rnxEditGde(merged_file, Path().cwd(), hr30, rewrite, 30)
                merged_file_path = Path().cwd() / merged_file
                # delete the merged file:
                merged_file_path.resolve().unlink()
    logger.info('Done!')
#    total = next(failed) + next(succ) - 2
#    succses = next(succ) - 2
#    failure = next(failed) - 2
    total = cnt['failed'] + cnt['succ']
    logger.info('Total files: {}, success: {}, failed: {}'.format(
            total, cnt['succ'], cnt['failed']))
    return


def run_gd2e_for_one_station(dr_path, staDb, tree, rewrite, date_range=None):
    """runs gd2e.py for all datarecodrs in one folder(dr_path) with staDb.
    rewrite: overwrite the results tdp in dr_path / results."""
    from pathlib import Path
    import subprocess
    # from itertools import count
    from subprocess import CalledProcessError
    from subprocess import TimeoutExpired
    import logging
    from aux_gps import get_timedate_and_station_code_from_rinex
    from aux_gps import path_glob
    from aux_gps import slice_task_date_range
    import pandas as pd
    logger = logging.getLogger('gipsyx')
    logger.info(
        'starting gd2e.py main gipsyX run.')
    logger.info('working with {} station database'.format(staDb))
    if rewrite:
        logger.warning('overwrite files mode initiated.')
    results_path = dr_path / 'results'
    if tree.as_posix().strip():
        logger.info('working with {} tree'.format(tree))
    try:
        results_path.mkdir()
    except FileExistsError:
        logger.info(
            '{} already exists, using that folder.'.format(results_path))
#    succ = count(1)
#    failed = count(1)
    cnt = {'succ': 0, 'failed': 0}
    files = path_glob(dr_path, '*.dr.gz')
    if date_range is not None:
        files = slice_task_date_range(files, date_range, 'run')
    tot = len(files)
    logger.info('found {} dr.gz files in {} to run.'.format(tot, dr_path))
    tot_final = len(path_glob(results_path, '*_smoothFinal.tdp', True))
    logger.info('found {} _smoothFinal.tdp files in {}'.format(tot_final,
                results_path))
    tot_to_run = tot - tot_final
    est_time_per_single_run = 22.0  # seconds
    dtt = pd.to_timedelta(est_time_per_single_run, unit='s') * tot_to_run
    logger.info('estimated time to completion of run: {}'.format(dtt))
    logger.info('check again in {}'.format(pd.Timestamp.now() + dtt))
    for file_and_path in files:
        rfn = file_and_path.as_posix().split('/')[-1][0:12]
        dt, station = get_timedate_and_station_code_from_rinex(rfn)
        final_tdp = '{}_smoothFinal.tdp'.format(rfn)
        logger.info(
            'processing {} ({}, {}/{})'.format(
                rfn,
                dt.strftime('%Y-%m-%d'), cnt['succ'] + cnt['failed'], tot))
        if not rewrite:
            if (results_path / final_tdp).is_file():
                logger.warning(
                    '{} already exists in {}, skipping...'.format(
                        final_tdp, results_path))
                cnt['succ'] += 1
                continue
        command = 'gd2e.py -drEditedFile {} -recList {} -staDb {} -treeS {} \
        > {}.log 2>{}.err'.format(
            file_and_path.as_posix(), station, staDb.as_posix(), tree, rfn,
            rfn)
        files_to_move = ['{}{}'.format(rfn, x)
                         for x in ['.log', '.err']]
        more_files = ['finalResiduals.out', 'smoothFinal.tdp']
        more_files_rfn = ['{}_{}'.format(rfn, x) for x in more_files]
        try:
            subprocess.run(command, shell=True, check=True, timeout=300)
            move_files(Path().cwd(), results_path, more_files,
                       more_files_rfn)
            move_files(Path().cwd(), results_path, 'Summary',
                       '{}_Summary.txt'.format(rfn))
            # next(succ)
            cnt['succ'] += 1
        except CalledProcessError:
            logger.error('gipsyx failed on {}, copying log files.'.format(rfn))
            # next(failed)
            cnt['failed'] += 1
        except TimeoutExpired:
            logger.error('gipsyx timed out on {}, copying log files.'.format(rfn))
            # next(failed)
            cnt['failed'] += 1
            with open(Path().cwd() / files_to_move[1], 'a') as f:
                f.write('GipsyX run has Timed out !')
        move_files(Path().cwd(), results_path, files_to_move)
        move_files(Path().cwd(), results_path, 'debug.tree', '{}_debug.tree'.format(rfn))
    logger.info('Done!')
    # total = next(failed) + next(succ) - 2
    total = cnt['succ'] + cnt['failed']
#    succses = next(succ) - 2
#    failure = next(failed) - 2
    logger.info('Total files: {}, success: {}, failed: {}'.format(
            total, cnt['succ'], cnt['failed']))
    return


if __name__ == '__main__':
    """--prep: prepare mode
        choices: 1) drdump 2) edit24hr 3) edit30hr
        Note: running --rinexpath arg without --prep means gd2e.py run and
        rinexpath is actually where the datarecords are. e.g., /rinex/tela/24hr
        or /rinex/tela/30hr"""
    import argparse
    import sys
    from aux_gps import configure_logger
    from aux_gps import get_var
    from pathlib import Path
    # first check for GCORE path:
    if get_var('GCORE') is None:
        raise ValueError('Run source ~/GipsyX-1.1/rc_GipsyX.sh first !')
    logger = configure_logger(name='gipsyx')
    parser = argparse.ArgumentParser(
        description='a command line tool for preparing and running rinex files with gipsyX softwere.')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
#    required.add_argument(
#        '--savepath',
#        help="a full path to save the raw output files, e.g., /home/ziskin/Work_Files/PW_yuval/gipsyx_resolved/TELA",
#        type=check_path)
    required.add_argument(
        '--rinexpath',
        help="a full path to the rinex path of the station, /home/ziskin/Work_Files/PW_yuval/rinex/TELA",
        type=check_path)
    optional.add_argument(
        '--staDb',
        help='add a station DB file for antennas and receivers in rinexpath',
        type=check_file_in_cwd)
    optional.add_argument('--daterange', help='add specific date range, can be one day',
                          type=str, nargs=2)
    optional.add_argument('--prep', help='call rinex rnxEditGde/drMerge or dataRecorDump',
                          choices=['drdump', 'edit24hr', 'edit30hr'])
    optional.add_argument('--tree', help='gipsyX tree directory.',
                          type=check_path)
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
                                           args.prep, args.rewrite,
                                           args.daterange)
    elif args.prep is None:
        if args.tree is None:
            args.tree = Path(' ')
        run_gd2e_for_one_station(args.rinexpath, args.staDb, args.tree,
                                 args.rewrite, args.daterange)
