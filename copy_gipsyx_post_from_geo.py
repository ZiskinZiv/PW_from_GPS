#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 08:37:02 2019

@author: shlomi
"""


def copy_post_from_geo(remote_path, station):
    for curr_sta in station:
        if (workpath / curr_sta).is_dir():
            if (workpath / curr_sta / 'gipsyx_solutions').is_dir():
                copy(remote_path / )
    return

def check_python_version(min_major=3, min_minor=6):
    import sys
    major = sys.version_info[0]
    minor = sys.version_info[1]
    print('detecting python varsion: {}.{}'.format(major, minor))
    if major < min_major or minor < min_minor:
        raise ValueError('Python version needs to be at least {}.{} to run this script...'.format(min_major, min_minor))
    return


def check_path(path):
    import os
    path = str(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError('{} does not exist...'.format(path))
    return path


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


if __name__ == '__main__':
    import argparse
    import sys
    from pathlib import Path
    from aux_gps import get_var
    from aux_gps import path_glob
    import pandas as pd
    from PW_paths import geo_path
    global pwpath
    global workpath
    # main directive:
    # write a script called run_gipsyx_script.sh with:
    # cd to the workpath / station and run nohup with the usual args
    check_python_version(min_major=3, min_minor=6)
    parser = argparse.ArgumentParser(description='a command line tool for ' +
                                     'copying post proccessed gipsyx nc files' +
                                     'to home directory structure')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # remove this line: optional = parser...
    required.add_argument('--station', help="GPS station name four lowercase letters,",
                          nargs='+', type=check_station_name)
    parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()
    pwpath = Path(get_var('PWCORE'))
    workpath = Path(get_var('PWORK'))
    if pwpath is None:
        raise ValueError('Put source code folder at $PWCORE')
    # get all the names of israeli gnss stations:
    isr_stations = pd.read_csv(pwpath / 'stations_approx_loc.txt',
                               delim_whitespace=True)
    isr_stations = isr_stations.index.tolist()
    if workpath is None:
        raise ValueError('Put source code folder at $PWORK')
    # get the names of the stations in workpath:
    stations = path_glob(workpath, '*')
    stations = [x.as_posix().split('/')[-1] for x in stations if x.is_dir()]
    if args.station is None:
        print('station is a required argument, run with -h...')
        sys.exit()
    if args.station == ['isr1']:
        args.station = isr_stations
    # use ISR stations db for israeli stations and ocean loading also:
#    if all(a in isr_stations for a in args.station):
    remote_path = geo_path / 'Work_Files/PW_yuval/GNSS_stations'
    copy_post_from_geo(remote_path, args.station)
