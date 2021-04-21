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
@author: ziskin
"""


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


def record_dump_and_merge(path):
    """search for hourly rinex in folder for all stations and convert them to dr.gz
    then merge them together"""
    import subprocess
    import logging
    from aux_gps import path_glob
    import string
    logger = logging.getLogger('axis-gipsyx')
    hours = [x.upper() for x in string.ascii_letters][:24]
    for hour in hours:
        files = path_glob(path, '*{}.*.gz'.format(hour))
    return


def run_gd2e_all_stations_one_hour(args):
    import subprocess
    import logging
    logger = logging.getLogger('axis-gipsyx')
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
    optional.add_argument(
        '--staDb',
        help='add a station DB file for antennas and receivers in rinexpath',
        type=check_file_in_cwd)
    optional.add_argument(
        '--accuracy',
        help='the orbit and clock accuracy products',
        type=str, choices=['final', 'ql', 'ultra'])
    optional.add_argument(
        '--drmerger',
        help='use this to just drRecordump to ultra/final/ql folder and merge all hourly files for all available stations',
        type=bool)
    optional.add_argument('--tree', help='gipsyX tree directory.',
                          type=check_path)
    parser._action_groups.append(optional)  # added this line
    parser.set_defaults(rewrite=False)
    args = parser.parse_args()
    if args.accuracy is None:
        args.accuracy = 'ultra'
    if args.rinexpath is None:
        print('rinexpath is a required argument, run with -h...')
        sys.exit()
    if args.staDb is None:
        args.staDb = '$GOA_VAR/sta_info/sta_db_qlflinn'
    if args.drmerger is None:
        run_gd2e_all_stations_one_hour(args)
    else:
        record_dump_and_merge(args)
