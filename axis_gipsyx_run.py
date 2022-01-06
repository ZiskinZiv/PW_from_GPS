#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:56:59 2021
dataRecordDump -rnx Ohad103D.21d.gz -drFileNmOut Ohad103D.21d.dr.gz
drMerge.py -i *.dr.gz -o merged.dr.gz
gd2e.py -drEditedFile merged.dr.gz -GNSSproducts ultra -staDb ~/Python_Projects/PW_from_GPS/AXIS.staDb -treeS ~/Python_Projects/PW_from_GPS/my_trees/AXISOcnld/ -recList RAMO OHAD
create folders in axis/2021/103 like final, rapid and ultra with solutions and extra files every hour for all stations
first do datarecorddump for all hourly files and then merge them together
use this func to do it seperately from main gipsyx run
strategy for final solutions of hourly axis network year/doy folders:
0) run count_organize_rinex to get all 30-hr with 6-h window per stations rindex hourly files:
1) for each station copy to axis_final folder under station name.
2) gunzip each rinex group files and crx2rnx.
2) for each station name teqc_concat to group file (30hr max)
4) delete all remaining files
5) rnxEditGde.py with gde tree
6) gd2e.py
    two main modes for final solutions, the strategey is per station:
        1) historic - loop over all rinex upuntil 3 weeks time from now and solve
    with 30hr window centered on midday of doy.
    2) recent, solve the last day after 3 weeks from now, a 30 hr window.
    solve it daily for all station
@author: ziskin
"""


def check_window(window):
    window = int(window)
    if window < 1 or window > 30:
        raise argparse.ArgumentTypeError('{} is not a valid window.'.format(window))
    return window


def check_end_datetime(end_dt):
    end_dt = str(end_dt)
    # if len(station) != 4:
    #     raise argparse.ArgumentTypeError('{} is not a valid station name.'.format(station))
    return end_dt


def check_station(station):
    station = str(station)
    if len(station) != 4:
        raise argparse.ArgumentTypeError('{} is not a valid station name.'.format(station))
    return station


def check_year(year):
    import datetime
    now = datetime.datetime.now()
    year = int(year)
    if year < 1980 or year > now.year:
        raise argparse.ArgumentTypeError('{} is not a valid year'.format(year))
    return year


def check_path(path):
    import os
    path = str(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(path + ' does not exist...')
    return Path(path)


def check_file_in_cwd(filename):
    cwd = Path().cwd()
    file_and_path = cwd / filename
    if not file_and_path.is_file():
        raise argparse.ArgumentTypeError(
            '{} does not exist at {}'.format(
                filename, cwd))
    return file_and_path

def check_file(file):
    if not file.is_file():
        raise argparse.ArgumentTypeError(
            '{} does not exist. {}'.format(
                file))
    return file


def create_rinex_df_per_station(rinex_df, station='Alon'):
    """produces input to create_windowed_chunk_rinex"""
    import numpy as np
    import pandas as pd
    st_df = rinex_df[station].to_frame()
    st_df['rfn'] = st_df[station].apply(lambda x: np.nan if pd.isnull(x) else x.name[0:12])
    inds = st_df[~st_df[station].isnull()].index
    st_df.loc[inds, 'year'] = st_df.loc[inds].index.year
    st_df.loc[inds, 'doy'] = st_df.loc[inds].index.dayofyear
    return st_df


def create_station_windowed_chunk_rinex(station_df, station='Alon', doy=117, year=2021):
    """take the year, doy and station and return a list of rinex files to concat
    the center is 12:00 UTC (M letter)"""
    import pandas as pd
    yr = str(year)[2:]
    center_ind = station_df[station_df['rfn']=='{}{}M.{}d'.format(station, doy, yr)].index
    if not center_ind.empty:
        ind = center_ind[0]
    else:
        # assume either no rinex at M (midday) or only part of rinex day exist
        # in this case, just get all daily rinex:
        daily = station_df.loc[(station_df['year']==year)&(station_df['doy']==doy)]
        logger.warning('No midday file found for {} in {} of {}, total hours ={}.'.format(station, doy, year, len(daily)))
        return daily[station].dropna().values
    start = ind - pd.Timedelta(15, unit='H')
    end = ind + pd.Timedelta(15, unit='H')
    sliced = station_df.loc[start:end]
    return sliced[station].dropna().values


# def record_dump_and_merge(args):
#     # import os
#     import logging
#     from aux_gps import path_glob
#     logger = logging.getLogger('axis-gipsyx')
#     if args.mode is None:
#         mode = 'whole'
#     else:
#         mode = args.mode
#     if args.year is None:
#         year = 2021
#     else:
#         year = args.year
#     if mode == 'last_doy':
#         # check for doy folders in args.rinexpath and flag last_doy:
#         doys = sorted(path_glob(args.rinexpath / str(year), '*/'))
#         doys = sorted([x for x in doys if x.is_dir()])
#         last_doy = doys[-1].as_posix().split('/')[-1]
#         logger.info('drRecordDump on year {}, doy {}, using last_doy'.format(year, last_doy))
#         record_dump_and_merge_at_single_folder(doys[0], args.drmerger, args.staDb)
#     elif mode == 'whole':
#         doys = sorted(path_glob(args.rinexpath / str(year), '*/'))
#         doys = sorted([x for x in doys if x.is_dir()])
#         for doy in doys:
#             current_doy = doy.as_posix().split('/')[-1]
#             logger.info('drRecordDump on year {}, doy {},using whole data'.format(year, current_doy))
#             record_dump_and_merge_at_single_folder(doy, args.drmerger, args.staDb)
#     return


# def record_dump_and_merge_at_single_folder(rinexpath, drmerger, staDb):
#     """search for hourly rinex in folder for all stations and convert them to dr.gz
#     then merge them together"""
#     import subprocess
#     import os
#     import logging
#     from aux_gps import path_glob
#     import string
#     from subprocess import CalledProcessError
#     # from subprocess import TimeoutExpired
#     logger = logging.getLogger('axis-gipsyx')
#     # cnt = {'succ': 0, 'failed': 0}
#     # first check for rinexpath / dr, if doesn't exist create it:
#     dr_path = rinexpath / 'dr'
#     if not (dr_path).is_dir():
#         os.mkdir(dr_path)
#     # next check if performing hourly all stations, or daily one/all stations:
#     logger.info('performing {} drmerger for all rinex files.'.format(drmerger))
#     if drmerger == 'hourly':
#         hours = [x.upper() for x in string.ascii_letters][:24]
#         for hour in hours:
#             try:
#                 files = path_glob(rinexpath, '*{}.*.gz'.format(hour))
#             except FileNotFoundError:
#                 logger.warning('hour {} not found in rinex files.'.format(hour))
#                 continue
#             dr_files = []
#             stations = []
#             doy = files[0].as_posix().split('/')[-1][0:12][4:7]
#             merged_file = dr_path / 'merged_{}_{}.dr.gz'.format(doy, hour)
#             if (merged_file).is_file():
#                 logger.warning('{} already exists in {}, skipping...'.format(merged_file, dr_path))
#                 continue
#             for file in files:
#                 # rinex 2.11 filename:
#                 filename = file.as_posix().split('/')[-1][0:12]
#                 stn = filename[0:4].upper()
#                 dr_file = dr_path / (filename + '.dr.gz')
#                 dr_filename = dr_path / filename
#                 command = 'dataRecordDump -rnx {} -drFileNmOut {} > {}.log 2>{}.err'.format(
#                     file.as_posix(), dr_file.as_posix(), dr_filename, dr_filename)
#                 try:
#                     subprocess.run(command, shell=True, check=True)
#                     dr_files.append(dr_file)
#                     # keep station names that were succesufully dred:
#                     stations.append(filename[0:4])
#                     # cnt['succ'] += 1
#                 except CalledProcessError:
#                     logger.error(
#                         'dataRecordDump failed on {}, deleting file.'.format(filename))
#                     dr_file.unlink()
#                 # now rnxedit:
#                 dr_edited_file = dr_path / (filename + '.dr_edited.gz')
#                 command = 'rnxEditGde.py -type datarecord -recNm {} -data {} -out {} -staDb {} > {}.log 2>{}.err'.format(
#                     stn, dr_file.as_posix(), dr_edited_file.as_posix(),
#                     staDb.as_posix(), filename, filename)
#                 try:
#                     subprocess.run(command, shell=True, check=True)
#     #                next(succ)
#                 # cnt['succ'] += 1
#                 except CalledProcessError:
#     #                next(failed)
#                     # cnt['failed'] += 1
#                     logger.error('rnxEditGde.py failed on {}...'.format(filename))
#                     dr_edited_file.unlink()
#                     # cnt['failed'] += 1
#             # now merge all dr files :
#             logger.info(
#                 'merging dr files in doy {}, hour {}.'.format(doy, hour))
#             merged_glob = dr_path / '*{}{}.*.dr_edited.gz'.format(doy, hour)
#             merged_filename = dr_path / 'merged_{}_{}'.format(doy, hour)
#             command = 'drMerge.py -i {} -o {} > {}.log 2>{}.err'.format(
#                 merged_glob.as_posix(), merged_file.as_posix(), merged_filename,
#                 merged_filename)
#             try:
#                 subprocess.run(command, shell=True, check=True)
#                 # also create txt file with station names inside merged file:
#                 with open('{}.txt'.format(merged_filename.as_posix()), "w") as outfile:
#                     outfile.write("\n".join(stations))
#             except CalledProcessError:
#                 logger.error('drMerge.py failed on {}...'.format(
#                     merged_glob.as_posix()))
#                 return
#             # now delete all single dr (whom i already merged):
#             [x.unlink() for x in dr_files]
#     elif drmerger == 'daily':
#         files = path_glob(rinexpath, '*.gz')
#         dr_files = []
#         stations = []
#         for file in files:
#             # rinex 2.11 filename:
#             filename = file.as_posix().split('/')[-1][0:12]
#             doy = filename[4:7]
#             merged_file = dr_path / 'merged_{}.dr.gz'.format(doy)
#             if (merged_file).is_file():
#                 logger.warning('{} already exists in {}, skipping...'.format(merged_file, dr_path))
#                 continue
#             dr_file = dr_path / (filename + '.dr.gz')
#             dr_filename = dr_path / filename
#             command = 'dataRecordDump -rnx {} -drFileNmOut {} > {}.log 2>{}.err'.format(
#                 file.as_posix(), dr_file.as_posix(), dr_filename, dr_filename)
#             try:
#                 subprocess.run(command, shell=True, check=True)
#                 dr_files.append(dr_file)
#                 stations.append(filename[0:4])
#                 # cnt['succ'] += 1
#             except CalledProcessError:
#                 logger.error('dataRecordDump failed on {}, deleting file.'.format(filename))
#                 dr_file.unlink()
#                 # cnt['failed'] += 1
#         # now merge all dr files :
#         logger.info('merging dr files in doy {}.'.format(doy))
#         merged_file = dr_path / 'merged_{}.dr.gz'.format(doy)
#         merged_filename = dr_path / 'merged_{}'.format(doy)
#         merged_glob = dr_path / '*{}*.*.dr.gz'.format(doy)
#         command = 'drMerge.py -i {} -o {} > {}.log 2>{}.err'.format(
#             merged_glob.as_posix(), merged_file.as_posix(), merged_filename, merged_filename)
#         try:
#             subprocess.run(command, shell=True, check=True)
#             with open('{}.txt'.format(merged_filename.as_posix()), "w") as outfile:
#                 outfile.write("\n".join(stations))
#         except CalledProcessError:
#             logger.error('drMerge.py failed on {}...'.format(
#                 merged_glob.as_posix()))
#         # now delete all single dr (whom i already merged):
#         [x.unlink() for x in dr_files]
#     # now delete all log and err files if empty:
#     log_files = path_glob(dr_path, '*.log')
#     for lfile in log_files:
#         if lfile.stat().st_size == 0:
#             lfile.unlink()
#     err_files = path_glob(dr_path, '*.err')
#     for efile in err_files:
#         if efile.stat().st_size == 0:
#             efile.unlink()
#     return


# def daily_prep_axis_final_solutions(doy_path, year=14):
#     from axis_process import copy_rinex_files_to_folder
#     from axis_process import run_rinex_compression_on_folder
#     from axis_process import get_unique_rfns_from_folder
#     from axis_process import teqc_concat_rinex
#     from aux_gps import path_glob
#     dr_path = doy_path / 'dr'
#     # copy gz RINEX to dr_path:
#     copy_rinex_files_to_folder(doy_path, dr_path)
#     # unzip and uncompress:
#     run_rinex_compression_on_folder(dr_path, command='gunzip', glob='*.gz')
#     run_rinex_compression_on_folder(dr_path, command='crx2rnx', glob='*.{}d'.format(year))
#     # delete d files:
#     files = path_glob(dr_path, '*.{}d'.format(year))
#     [x.unlink() for x in files]
#     # teqc concat for daily o per station:
#     fns = get_unique_rfns_from_folder(dr_path,'*.{}o'.format(year))
#     for fn in fns:
#         filename = '{}0.{}o'.format(fn, year)
#         teqc_concat_rinex(dr_path, rfn=filename,
#                           glob='{}*.{}o'.format(fn,year),
#                           delete_after_concat=True)
#         dataRecordDump_single_file(dr_path, filename)
#         file = dr_path / filename
#         file.unlink()
    # dataRecordDump to daily station file:




# def run_drMerge(filenames, in_path, duration):
#     import subprocess
#     from subprocess import CalledProcessError
#     from subprocess import TimeoutExpired
#     from aux_gps import get_timedate_and_station_code_from_rinex
#     rfns = [x[0:12] for x in filenames]
#     dts = [get_timedate_and_station_code_from_rinex(x, True) for x in rfns]
#     if duration == '30hr':
#         start = dts[0].strftime('%Y-%m-%d') + ' 21:00:00'
#         end = dts[2].strftime('%Y-%m-%d') + ' 03:00:00'
#     dr_merged_file = Path().cwd() / '{}_merged.dr.gz'.format(rfns[1])
#     logger.info('merging {}, {} and {} to {}'.format(*rfns,rfns[1] + '_merged.dr.gz'))
#     f_and_paths = [in_path / x for x in filenames]
#     files_to_move = [rfn + x for x in ['_drmerge.log', '_drmerge.err']]
#     command = 'drMerge.py -inFiles {} {} {} -outFile {} -start {} -end {} > {}.log 2>{}.err'.format(
#             f_and_paths[0].as_posix(), f_and_paths[1].as_posix(),
#             f_and_paths[2].as_posix(), dr_merged_file.as_posix(),
#             start, end, rfn + '_drmerge', rfn + '_drmerge')
#     try:
#         subprocess.run(command, shell=True, check=True, timeout=60)
#     except CalledProcessError:
#         logger.error('drMerge.py failed on {}...'.format(filenames))
#     except TimeoutExpired:
#         logger.error('drMerge.py timed out on {}, copying log files.'.format(filenames))
#         # next(failed)
#         cnt['failed'] += 1
#         with open(Path().cwd() / files_to_move[1], 'a') as f:
#             f.write('drMerge.py run has Timed out !')
#         return None
#     move_files(Path().cwd(), Path().cwd(), files_to_move)
#     return rfns[1] + '_merged.dr.gz'

def parse_hourly_range_rfn(hourly_rfn, return_mid_doy=True):
    station = hourly_rfn[:4]
    yr = int(hourly_rfn.split('.')[1][:2])
    if yr >= 0 and yr <= 79:
        year = int('20'+str(yr))
    elif yr <=99 and yr >=80:
        year = int('19'+str(yr))
    doy_start = int(hourly_rfn.split('-')[0][4:7])
    doy_end = int(hourly_rfn.split('-')[1][:3])
    mid_doy = int((doy_start + doy_end)/2)
    if return_mid_doy:
        return station, year, mid_doy

def final_historic_perp(rinexpath, rinexfinal_path, staDb, rinex_df, station='Alon', gde_tree=None):
    import os
    from aux_gps import path_glob
    st_df = create_rinex_df_per_station(rinex_df, station=station)
    mindex = st_df.groupby(['year','doy'])['rfn'].count()
    station_path = rinexfinal_path / station
    if not station_path.is_dir():
        logger.warning('{} is missing, creating it.'.format(station_path))
        os.mkdir(station_path)
    already_edited = sorted(path_glob(station_path, '{}*_edited.*.dr.gz'.format(station), return_empty_list=True))
    if already_edited:
        last_hourly_rfn = already_edited[-1].name
        _, last_year, last_doy  = parse_hourly_range_rfn(last_hourly_rfn)
        mindex = mindex.loc[slice(last_year, None), slice(last_doy, None), :]
        logger.info('found last RINEX: {}, year={}, doy={}.'.format(already_edited[-1].name, last_year, last_doy))
    for year, doy in mindex.index:
        files = create_station_windowed_chunk_rinex(st_df, station=station,
                                                    doy=int(doy), year=int(year))
        prep_30hr_all_steps(station_path, files, staDb, station=station, gde_tree=gde_tree)
    logger.info('Done prepring final {}.'.format(station))


def prep_30hr_all_steps(station_path, files, staDb, station='Alon',
                        gde_tree=None):
    import numpy as np
    from axis_process import copy_rinex_files_to_folder
    from axis_process import run_rinex_compression_on_folder
    from axis_process import teqc_concat_rinex
    from aux_gps import path_glob
    import os
    # first get station name from files and assert it is like station:
    station_from_files = np.unique([x.name.split('.')[0][:4] for x in files]).item()
    assert station_from_files == station
    # also get year:
    yrs = np.unique([x.name.split('.')[1][:2] for x in files]).astype(int)
    # now copy files to station path in rinexfinal_path:
    copy_rinex_files_to_folder(files, station_path)
    # unzip them and crx2rnx and delete d files:
    for yr in yrs:   # if there are more than one year (only in DEC-JAN):
        run_rinex_compression_on_folder(station_path, command='gunzip', glob='*.{}d.gz'.format(yr))
        run_rinex_compression_on_folder(station_path, command='crx2rnx', glob='*.{}d'.format(yr))
        dfiles = path_glob(station_path, '*.{}d'.format(yr))
        [x.unlink() for x in dfiles]
    # use new filename for concated rinex:
    yr = yrs[0]
    doy_hour_start = files[0].name[4:8]
    doy_hour_end = files[-1].name[4:8]
    filename = '{}{}-{}.{}o'.format(station, doy_hour_start, doy_hour_end, yr)
    if len(yrs) > 1 and (np.abs(np.diff(yrs))==1).item():
        teqc_concat_rinex(station_path, rfn=filename, glob='*.*o', cmd_path=None,
                          delete_after_concat=True)
    else:
        teqc_concat_rinex(station_path, rfn=filename, glob='*.{}o'.format(yr), cmd_path=None,
                          delete_after_concat=True)
    # now, dataRecordDump and delete o file:
    dataRecordDump_single_file(station_path, filename, rfn=filename[:17])
    file = station_path / filename
    file.unlink()
    # now rnxEditGde:
    dr_filename = filename[:17] + '.dr.gz'
    # new_filename = filename[:13] + '_edited' + filename[13:17] + '.dr.gz'
    rnxEditGde_single_file(station_path, dr_filename, staDb,
                           new_filename=filename[:13],
                           delete_dr_after=True, gde_tree=gde_tree)
    return

def daily_prep_all_steps(path, staDb, new_filename=False,
                         delete_last_rinex=False, gde_tree=None):
    from aux_gps import path_glob
    from aux_gps import replace_char_at_string_position
    try:
        files = sorted(path_glob(path, '*.gz'))
    except FileNotFoundError:
        files = sorted(path_glob(path, '*.Z'))
    rfn = files[0].as_posix().split('/')[-1]
    rfn_dfile = replace_char_at_string_position(rfn, pos=7, char='0')[0:12]
    last_rfn = files[-1].as_posix().split('/')[-1]
    rfn_dr_file = rfn[0:8] + '-' + last_rfn[4:8]
    if not new_filename:
        rfn_dr_file = rfn_dfile
    # 1) rinex concat and prep:
    daily_prep_and_concat_rinex(path)
    # 2) dataRecordDump:
    dataRecordDump_single_file(path/'dr', rfn_dfile + '.gz')
    # 3) rinex edit:
    rnxEditGde_single_file(path/'dr', rfn_dfile + '.dr.gz', staDb,
                           new_filename=rfn_dr_file, delete_dr_after=True,
                           gde_tree=gde_tree)
    if delete_last_rinex:
        #) finally, delete first rinex file (earliest):
        files[0].unlink()
        logger.info('{} has been deleted.'.format(files[0]))
    return


def daily_prep_drdump_and_rnxedit(path, staDb, gde_tree=None):
    from aux_gps import path_glob
    from aux_gps import replace_char_at_string_position
    import shutil
    import os
    try:
        files = sorted(path_glob(path, '*.gz'))
        suff = '.gz'
    except FileNotFoundError:
        files = sorted(path_glob(path, '*.Z'))
        suff = '.Z'
    rfn = files[0].as_posix().split('/')[-1]
    rfn_dfile = replace_char_at_string_position(rfn, pos=7, char='0')[0:12]
    dr_path = path / 'dr'
    if not dr_path.is_dir():
        os.mkdir(dr_path)
    # 0) move daily files to dr_path:
    file = dr_path / (rfn_dfile + suff)
    shutil.copy(files[0], file)
    # 1) dataRecordDump:
    dataRecordDump_single_file(dr_path, rfn_dfile + suff)
    # 3) rinex edit:
    rnxEditGde_single_file(dr_path, rfn_dfile + '.dr.gz', staDb,
                           gde_tree=gde_tree)
    return


def daily_prep_and_concat_rinex(path):
    import logging
    import os
    from axis_process import run_rinex_compression_on_file
    from axis_process import run_rinex_compression_on_folder
    from axis_process import teqc_concat_rinex
    from aux_gps import path_glob
    from aux_gps import replace_char_at_string_position
    from axis_process import move_files
    logger = logging.getLogger('axis-gipsyx')
    # get rfn,i.e., DSea2150.14o:
    try:
        files = sorted(path_glob(path, '*.gz'))
        unzip_glob = '*.gz'
    except FileNotFoundError:
        files = sorted(path_glob(path, '*.Z'))
        unzip_glob = '*.Z'
    rfn = files[0].as_posix().split('/')[-1]
    rfn_dfile = replace_char_at_string_position(rfn, pos=7, char='0')[0:12]
    rfn_ofile = replace_char_at_string_position(rfn_dfile, pos=-1, char='o')
    # create dr path if not exists:
    dr_path = path / 'dr'
    if not dr_path.is_dir():
        os.mkdir(dr_path)
    # 1) unzip all files:
    logger.info('unzipping {}'.format(path))
    run_rinex_compression_on_folder(path, command='gunzip', glob=unzip_glob)
    # 2) convert to obs files:
    logger.info('converting d to obs.')
    run_rinex_compression_on_folder(path, command='crx2rnx', glob='*.*d')
    # 3) concat using teqc:
    logger.info('teqc concating.')
    teqc_concat_rinex(path, rfn=rfn_ofile, glob='*.*o')
    # 4) convert to d file:
    logger.info('compressing concated file and moving to dr path.')
    run_rinex_compression_on_file(path, filename=rfn_ofile, command='rnx2crx')
    # 5) gzip d file:
    # rfn = replace_char_at_string_position(rfn, char='d', pos=-1)
    run_rinex_compression_on_file(path, rfn_dfile, command='gzip')
    # 6) move copressed file to dr_path and delete all other files except original rinex gz:
    move_files(path, dr_path, rfn_dfile + '.gz', rfn_dfile + '.gz')
    files = path_glob(path, '*.*o')
    [x.unlink() for x in files]
    # 7) gzip all d files:
    logger.info('gzipping {}'.format(path))
    run_rinex_compression_on_folder(path, command='gzip', glob='*.*d')
    # 8) dataRecordDump:
    logger.info('Done preping daily {} path.'.format(path))
    return


def dataRecordDump_single_file(path_dir, filename, rfn=None):
    import subprocess
    import logging
    logger = logging.getLogger('axis-gipsyx')
    from subprocess import CalledProcessError
    if not (path_dir/filename).is_file():
        raise FileNotFoundError
    logger.info('dataRecordDump on {}'.format(filename))
    if rfn is None:
        rfn = filename[0:12]
    cmd = 'dataRecordDump -rnx {} -drFileNmOut {}.dr.gz'.format(filename, rfn)
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=path_dir)
    except CalledProcessError:
        logger.error('{} failed !'.format(cmd))
        return


def rnxEditGde_single_file(path_dir, filename, staDb, new_filename=None,
                           delete_dr_after=True, gde_tree=None):
    from aux_gps import get_var
    import subprocess
    import logging
    from subprocess import CalledProcessError
    logger = logging.getLogger('axis-gipsyx')
    if not (path_dir/filename).is_file():
        raise FileNotFoundError
    logger.info('rnxEditGde on {} with {}.'.format(filename, staDb))
    if len(filename.split('.')[0]) == 13:
        rfn = filename[:17]
    elif len(filename.split('.')[0]) == 8:
        rfn = filename[0:12]
    if new_filename is None:
        rfn_edited = rfn.split('.')[0] + '_edited' + '.' + rfn.split('.')[-1] + '.dr.gz'
    else:
        rfn_edited = new_filename + '_edited' + '.' + rfn.split('.')[-1] + '.dr.gz'
    station = rfn[0:4].upper()
    if gde_tree is None:
        cmd = 'rnxEditGde.py -type datarecord -recNm {} -data {} -out {} -staDb {}'.format(station, filename, rfn_edited, staDb)
    else:
        cmd = 'rnxEditGde.py -type datarecord -recNm {} -data {} -out {} -staDb {} -gdeTree {}'.format(station, filename, rfn_edited, staDb, gde_tree)
        logger.info('using {} as gde tree.'.format(gde_tree))
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=path_dir)
    except CalledProcessError:
        print('{} failed !'.format(cmd))
        return
    if delete_dr_after:
        (path_dir/filename).unlink()
        logger.info('{} has been deleted.'.format(path_dir/filename))
    return


def main_program(args):
    import logging
    from aux_gps import path_glob
    from axis_process import copy_rinex_to_station_dir
    from axis_process import produce_rinex_filenames_at_time_window
    from axis_process import read_rinex_count_file
    import pandas as pd
    import numpy as np
    logger = logging.getLogger('axis-gipsyx')
    if args.year is None:
        year = 2022
    else:
        year = args.year
    if args.window is None:
        window = 30
    else:
        window = args.window
    if args.end_dt is None:
        end_dt = pd.Timestamp.utcnow().floor('H')
    else:
        end_dt = args.end_dt
    if args.mode == 'daily_prep':
        mode = 'daily_prep'
    else:
        mode = args.mode
    if mode == 'daily_prep_all':
        doys = sorted(path_glob(args.rinexpath / str(year), '*/'))
        for doypath in doys:
            logger.info('preping {}:'.format(doypath))
            daily_prep_all_steps(doypath, args.staDb, gde_tree=args.gde_tree)
        logger.info('Done preping all doys in {}.'.format(year))
    elif mode == 'daily_prep_drdump_rnxedit':
        doys = sorted(path_glob(args.rinexpath / str(year), '*/'))
        for doypath in doys:
            logger.info('preping {}:'.format(doypath))
            daily_prep_drdump_and_rnxedit(doypath, args.staDb,
                                          gde_tree=args.gde_tree)
        logger.info('Done preping all doys in {}.'.format(year))
    elif mode == 'daily_run':
        doys = sorted(path_glob(args.rinexpath / str(year), '*/'))
        for doypath in doys:
            logger.info('running GipsyX on {}:'.format(doypath))
            daily_gd2e(doypath, args.staDb, args.tree)
        logger.info('Done running GipsyX all doys in {}.'.format(year))
    elif mode == 'real-time':
        # 1) first find rinex of station the last 24 hours and copy to dir:
        fns = produce_rinex_filenames_at_time_window(
            args.station, end_dt=end_dt, window=window)
        copy_rinex_to_station_dir(args.rinexpath, fns)
        # # 2) then, concat them, and drdump and rnxedit to dr path inside station dir:
        daily_prep_all_steps(args.rinexpath / args.station, args.staDb,
                             new_filename=True, delete_last_rinex=True,
                             gde_tree=args.gde_tree)
        # # 3) run gd2e.py
        daily_gd2e(args.rinexpath / args.station, args.staDb, args.tree, args.accuracy,
                   extended_rfn=True)
    elif mode == 'final_prep_historic':
        # first read rinex_df file, it should be in rinexpath:
        rinex_df = read_rinex_count_file(args.rinexpath)
        final_historic_perp(args.rinexpath, args.rinexfinal_path,
                            args.staDb, rinex_df, args.station, args.gde_tree)
    elif mode == 'final_run_historic':
        files = path_glob(args.rinexfinal_path/args.station, '*.gz')
        for file in sorted(files):
            run_gd2e_on_single_file(args.rinexfinal_path/args.station,
                                    file, args.staDb, args.tree,
                                    extended_rfn=True)
        logger.info('Finished running gd2e on {}.'.format(args.station))
    # elif mode == 'daily_prep_final':
    #     year_path = args.rinexpath / str(year)
    #     year = year_path.name
    #     yr = year[2:]
    #     logger.info('preping for final solution year {}'.format(year))
    #     doy_paths = path_glob(year_path, '*/')
    #     doy_paths = np.array(sorted([x for x in doy_paths if x.name.isdigit()]))
    #     if args.end_dt is None:
    #         logger.warning('All year final perp selected.')
    #     else:
    #         end_doy = pd.to_datetime(end_dt).dayofyear
    #         logger.info('Final preping year {} until doy {}.'.format(year, end_doy))
    #         doys = np.array([int(x.name) for x in doy_paths])
    #         doy_paths = doy_paths[doys <= end_doy]
    #     for doy_path in doy_paths:
    #         doy = doy_path.name
    #         logger.info('preping for final solution doy {} in {}'.format(doy,year))
    #         daily_prep_axis_final_solutions(doy_path, year=yr)
    #     logger.info('Done prepring {} final solutions'.format(year))


def daily_gd2e(path, staDb, tree, acc='Final', extended_rfn=False):
    import logging
    from aux_gps import path_glob
    import subprocess
    import os
    from subprocess import CalledProcessError
    from subprocess import TimeoutExpired
    from axis_process import move_files
    from axis_process import read_multi_station_tdp_file
    logger = logging.getLogger('axis-gipsyx')
    dr_path = path / 'dr'
    res_path = dr_path / acc
    if not res_path.is_dir():
        os.mkdir(res_path)
    files = path_glob(dr_path, '*.gz')
    edited_list = sorted([x for x in files if 'edited' in x.as_posix()])
    # get the latest file:
    edited = max(edited_list, key=os.path.getctime)
    logger.info('started gd2e on {}'.format(edited))
    station = edited.as_posix().split('/')[-1][0:4].upper()
    rfn = edited.as_posix().split('/')[-1][0:8]
    if extended_rfn:
        rfn = edited.as_posix().split('/')[-1][0:13]
    if acc == 'Final':
        cmd = 'gd2e.py -drEditedFile {} -recList {} -staDb {} -treeS {}  > {}.log 2>{}.err'.format(
                edited.as_posix(), station, staDb.as_posix(), tree, rfn, rfn)
    elif acc == 'ql' or acc == 'ultra':
        cmd = 'gd2e.py -drEditedFile {} -recList {} -staDb {} -treeS {} -GNSSproducts {} > {}.log 2>{}.err'.format(
                edited.as_posix(), station, staDb.as_posix(), tree, acc, rfn, rfn)
    files_to_move = ['{}{}'.format(rfn, x)
                     for x in ['.log', '.err']]
    more_files = ['finalResiduals.out', 'smoothFinal.tdp']
    more_files_new_name = ['{}_{}'.format(rfn, x) for x in more_files]
    try:
        subprocess.run(cmd, shell=True, check=True, timeout=300, cwd=dr_path)
        ds = read_multi_station_tdp_file(dr_path/'smoothFinal.tdp',
                                         [station], savepath=res_path)
        move_files(res_path, res_path, 'smoothFinal.nc',
                   '{}_smoothFinal.nc'.format(rfn))
        move_files(Path().cwd(), res_path, more_files,
                   more_files_new_name)
        move_files(Path().cwd(), res_path, 'Summary',
                   '{}_Summary.txt'.format(rfn))
        # next(succ)
        # cnt['succ'] += 1
    except CalledProcessError:
        logger.error('gipsyx failed on {}, copying log files.'.format(rfn))
        # next(failed)
        # cnt['failed'] += 1
    except TimeoutExpired:
        logger.error('gipsyx timed out on {}, copying log files.'.format(rfn))
        # next(failed)
        # cnt['failed'] += 1
        # with open(Path().cwd() / files_to_move[1], 'a') as f:
        #     f.write('GipsyX run has Timed out !')
    move_files(Path().cwd(), res_path, files_to_move)
    move_files(Path().cwd(), res_path, 'debug.tree', '{}_debug.tree'.format(rfn))
    return


def run_gd2e_on_single_file(path, filename, staDb, tree, acc='Final', extended_rfn=False):
    import logging
    import subprocess
    import os
    from subprocess import CalledProcessError
    from subprocess import TimeoutExpired
    from axis_process import move_files
    from axis_process import read_multi_station_tdp_file
    logger = logging.getLogger('axis-gipsyx')
    res_path = path / acc
    if not res_path.is_dir():
        os.mkdir(res_path)
    file = path / filename
    if not file.is_file():
        raise FileNotFoundError
    logger.info('started gd2e on {}'.format(filename))
    station = filename.name[:4].upper()
    rfn = filename.name[0:8]
    if extended_rfn:
        rfn = filename.name[0:13]
    if acc == 'Final':
        cmd = 'gd2e.py -drEditedFile {} -recList {} -staDb {} -treeS {}  > {}.log 2>{}.err'.format(
                filename.as_posix(), station, staDb.as_posix(), tree, rfn, rfn)
    elif acc == 'ql' or acc == 'ultra':
        cmd = 'gd2e.py -drEditedFile {} -recList {} -staDb {} -treeS {} -GNSSproducts {} > {}.log 2>{}.err'.format(
                filename.as_posix(), station, staDb.as_posix(), tree, acc, rfn, rfn)
    files_to_move = ['{}{}'.format(rfn, x)
                     for x in ['.log', '.err']]
    more_files = ['finalResiduals.out', 'smoothFinal.tdp']
    more_files_new_name = ['{}_{}'.format(rfn, x) for x in more_files]
    try:
        subprocess.run(cmd, shell=True, check=True, timeout=300, cwd=path)
        ds = read_multi_station_tdp_file(path/'smoothFinal.tdp',
                                         [station], savepath=res_path)
        move_files(res_path, res_path, 'smoothFinal.nc',
                   '{}_smoothFinal.nc'.format(rfn))
        move_files(path, res_path, more_files,
                   more_files_new_name)
        move_files(path, res_path, 'Summary',
                   '{}_Summary.txt'.format(rfn))
        # next(succ)
        # cnt['succ'] += 1
    except CalledProcessError:
        logger.error('gipsyx failed on {}, copying log files.'.format(rfn))
        # next(failed)
        # cnt['failed'] += 1
    except TimeoutExpired:
        logger.error('gipsyx timed out on {}, copying log files.'.format(rfn))
        # next(failed)
        # cnt['failed'] += 1
        # with open(Path().cwd() / files_to_move[1], 'a') as f:
        #     f.write('GipsyX run has Timed out !')
    move_files(path, res_path, files_to_move)
    move_files(path, res_path, 'debug.tree', '{}_debug.tree'.format(rfn))
    return


def run_gd2e_single_dr_path(dr_path, staDb, tree, acc='ultra', n_proc=4, network_name='axis'):
    """runs gd2e.py for all datarecodrs in one folder(dr_path) with staDb.
    """
    from pathlib import Path
    import subprocess
    from axis_process import move_files
    # from itertools import count
    from subprocess import CalledProcessError
    from subprocess import TimeoutExpired
    import logging
    # from aux_gps import get_timedate_and_station_code_from_rinex
    from aux_gps import path_glob
    # from aux_gps import slice_task_date_range
    import pandas as pd
    from axis_process import read_multi_station_tdp_file
    logger = logging.getLogger('axis-gipsyx')
    logger.info(
        'starting gd2e.py main gipsyX run.')
    logger.info('working with {} station database'.format(staDb))
    results_path = dr_path / acc
    if tree.as_posix().strip():
        logger.info('working with {} tree'.format(tree))
    try:
        results_path.mkdir()
    except FileExistsError:
        logger.info(
            '{} already exists, using that folder.'.format(results_path))
#    succ = count(1)
#    failed = count(1)
    # cnt = {'succ': 0, 'failed': 0}
    files = path_glob(dr_path, '_edited*.dr.gz')
    tot = len(files)
    logger.info('found {} edited dr.gz files in {} to run.'.format(tot, dr_path))
    tot_final = len(path_glob(results_path, '*_smoothFinal.tdp', True))
    # logger.info('found {} _smoothFinal.tdp files in {}'.format(tot_final,
    # results_path))
    # tot_to_run = tot - tot_final
    # est_time_per_single_run = 22.0  # seconds
    # dtt = pd.to_timedelta(est_time_per_single_run, unit='s') * tot_to_run
    # logger.info('estimated time to completion of run: {}'.format(dtt))
    # logger.info('check again in {}'.format(pd.Timestamp.now() + dtt))
    for file_and_path in sorted(files):
        merged = file_and_path.as_posix().split('/')[-1][0:12].split('_')
        doy = merged[1]
        try:
            hour = merged[2]
            out_name = '{}_{}_{}'.format(network_name, doy, hour)
            # final_tdp = '{}_{}_{}_smoothFinal.tdp'.format(network_name, doy, hour)
            logger.info('running gd2e on merged dr file, doy {}, hour {}'.format(doy, hour))
        except IndexError:
            # final_tdp = '{}_{}_smoothFinal.tdp'.format(network_name, doy)
            out_name = '{}_{}'.format(network_name, doy)
            logger.info('running gd2e on merged dr file, doy {}'.format(doy))
        # read stations merged txt file:
        stns_path = Path(str(file_and_path).replace('.dr.gz', '.txt'))
        df = pd.read_csv(stns_path, names=['station'])
        stns = [x.upper() for x in df['station'].unique()]
        rec_list = '{}'.format(' '.join(stns))
        # dt, station = get_timedate_and_station_code_from_rinex(rfn)

        # logger.info(
        #     'processing {} ({}, {}/{})'.format(
        #         rfn,
        #         dt.strftime('%Y-%m-%d'), cnt['succ'] + cnt['failed'], tot))
        # if not rewrite:
        #     if (results_path / final_tdp).is_file():
        #         logger.warning(
        #             '{} already exists in {}, skipping...'.format(
        #                 final_tdp, results_path))
        #         cnt['succ'] += 1
        #         continue
        command = 'gd2e.py -drEditedFile {} -recList {} -staDb {} -treeS {} -GNSSproducts {} -nProcessors {} > {}.log 2>{}.err'.format(
            file_and_path.as_posix(), rec_list, staDb.as_posix(), tree, acc,
            n_proc, out_name, out_name)
        files_to_move = ['{}{}'.format(out_name, x)
                         for x in ['.log', '.err']]
        more_files = ['finalResiduals.out', 'smoothFinal.tdp']
        more_files_new_name = ['{}_{}'.format(out_name, x) for x in more_files]
        try:
            subprocess.run(command, shell=True, check=True, timeout=300)
            ds = read_multi_station_tdp_file(dr_path/'smoothFinal.tdp',
                                             stns, savepath=dr_path)
            move_files(Path().cwd(), results_path, 'smoothFinal.nc',
                       '{}_smoothFinal.nc'.format(out_name))

            move_files(Path().cwdf(), results_path, more_files,
                       more_files_new_name)
            move_files(Path().cwd(), results_path, 'Summary',
                       '{}_Summary.txt'.format(out_name))
            # next(succ)
            # cnt['succ'] += 1
        except CalledProcessError:
            logger.error('gipsyx failed on {}, copying log files.'.format(out_name))
            # next(failed)
            # cnt['failed'] += 1
        except TimeoutExpired:
            logger.error('gipsyx timed out on {}, copying log files.'.format(out_name))
            # next(failed)
            # cnt['failed'] += 1
            # with open(Path().cwd() / files_to_move[1], 'a') as f:
            #     f.write('GipsyX run has Timed out !')
        move_files(Path().cwd(), results_path, files_to_move)
        move_files(Path().cwd(), results_path, 'debug.tree', '{}_debug.tree'.format(out_name))
    logger.info('Done!')
    # total = next(failed) + next(succ) - 2
    # total = cnt['succ'] + cnt['failed']
#    succses = next(succ) - 2
#    failure = next(failed) - 2
    # logger.info('Total files: {}, success: {}, failed: {}'.format(
            # total, cnt['succ'], cnt['failed']))
    return

if __name__ == '__main__':
    """"""
    import argparse
    import sys
    from aux_gps import configure_logger
    from aux_gps import get_var
    from pathlib import Path
    # first check for GCORE path:
    if get_var('GCORE') is None:
        raise ValueError('Run source ~/GipsyX-1.1/rc_GipsyX.sh first !')
    logger = configure_logger(name='axis-gipsyx')
    parser = argparse.ArgumentParser(
        description='a command line tool for preparing and running rinex files with gipsyX software.')
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
    required.add_argument(
        '--mode',
        help="mode type",
        choices=['daily_run', 'daily_prep_all', 'daily_prep_drdump_rnxedit', 'real-time',
                 'final_prep_historic', 'final_prep', 'final_run_historic'])
    required.add_argument(
        '--station',
        help="GNSS 4 letter station code",
        type=check_station)
    required.add_argument(
        '--rinexfinal_path',
        help="a full path to the rinex path of the station for final solutions, /home/ziskin/Work_Files/PW_yuval/rinex/TELA",
        type=check_path)
    optional.add_argument(
        '--staDb',
        help='add a station DB file for antennas and receivers in rinexpath',
        type=check_file_in_cwd)
    optional.add_argument(
        '--accuracy',
        help='the orbit and clock accuracy products',
        type=str, choices=['Final', 'ql', 'ultra'])
    optional.add_argument(
        '--window',
        help='the window in hours to perform ppp solution. typically 24',
        type=check_window)
    optional.add_argument(
        '--end_dt',
        help='end datetime of window ppp solution. for real-time it is now.',
        type=check_end_datetime)
    optional.add_argument(
        '--imspath',
        help="a full path to the IMS real-time TD data, /home/ziskin/Work_Files/PW_yuval/IMS_T/10mins/real-time",
        type=check_path)
    optional.add_argument(
        '--awdpath',
        help="a full path to the Digital Elevetion Model (AWD) data, /home/ziskin/Work_Files/PW_yuval/gis/AW3D30",
        type=check_path)


    # optional.add_argument(
    #     '--drmerger',
    #     help='use this to just drRecordump to dr folder and merge all hourly files for all available stations or daily of one station',
    #     type=str, choices=['daily', 'hourly'])
    optional.add_argument('--tree', help='gipsyX tree directory.',
                          type=check_path)
    optional.add_argument('--gde_tree', help='gde tree file for rnxEditGde.',
                          type=check_path)

    # optional.add_argument('--mode', help='which mode to run', type=str,
    #                       choices=['last_doy', 'whole'])
    optional.add_argument('--year', help='year of rinex files', type=check_year)
    optional.add_argument('--n_proc', help='number of processors to solve rtgx', type=int)
    parser._action_groups.append(optional)  # added this line
    parser.set_defaults(rewrite=False)
    args = parser.parse_args()
    if args.accuracy is None:
        args.accuracy = 'ultra'
    if args.rinexpath is None:
        print('rinexpath is a required argument, run with -h...')
        sys.exit()
    if args.mode is None:
        print('mode is a required argument, run with -h...')
        sys.exit()
    if args.station is None:
        print('station is a required argument, run with -h...')
        sys.exit()
    if args.staDb is None:
        args.staDb = '$GOA_VAR/sta_info/sta_db_qlflinn'
    if args.gde_tree is None:
        args.gde_tree = Path().cwd() /'my_trees/gde_noclockprep.tree'
    main_program(args)
    # if args.drmerger is None:
    #     run_gd2e_single_dr_path(args.rinexpath, args.staDb, args.tree,
    #                             args.accuracy, n_proc=4, network_name='axis')
    #     # run_gd2e(args)
    # else:
    #     record_dump_and_merge(args)
