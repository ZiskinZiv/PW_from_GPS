#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:21:21 2019

@author: shlomi
"""
# TODO: add backup option depending on task. use tarfile to compress

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


def generate_delete(station, task, dates):
    from aux_gps import query_yes_no
    from aux_gps import path_glob
    from aux_gps import slice_task_date_range
    for curr_sta in station:
        station_path = workpath / curr_sta
        if task == 'drdump':
            path = station_path / 'rinex/dr'
            glob_str = '*.dr.gz'
        elif task == 'edit30hr':
            path = station_path / 'rinex/30hr'
            glob_str = '*.dr.gz'
        elif task == 'run':
            path = station_path / 'rinex/30hr/results'
            glob_str = '*.tdp'
        elif task == 'post':
            path = station_path / 'gipsyx_solutions'
            glob_str = '*.nc'
        try:
            files_to_delete = path_glob(path, glob_str)
            if dates is not None:
                files_to_delete = slice_task_date_range(files_to_delete, dates,
                                                        'delete')
            fnum = len(files_to_delete)
        except FileNotFoundError:
            print('skipping {} , because its empty or not existant..'.format(path))
            continue
        suff = glob_str.split('.')[-1]
        print(
            'WARNING for {}, ALL {} files({}) in {} WILL BE DELETED!'.format(
                curr_sta, suff, fnum, path))
        to_delete = query_yes_no('ARE YOU SURE ?')
        if not to_delete:
            print('files NOT deleted...')
            continue
        else:
            [x.unlink() for x in files_to_delete]
            print('FILES DELETED!')
    return


def generate_backup(station, task, dates):
    from aux_gps import tar_dir
    from aux_gps import slice_task_date_range
    for curr_sta in station:
        station_path = workpath / curr_sta
        if task == 'drdump':
            path = station_path / 'rinex/dr'
            glob_str = '*.dr.gz'
        elif task == 'edit30hr':
            path = station_path / 'rinex/30hr'
            glob_str = '*.dr.gz'
        elif task == 'run':
            path = station_path / 'rinex/hr30/results'
            glob_str = '*.tdp'
        elif task == 'post':
            path = station_path / 'gipsyx_solutions'
            glob_str = '*.nc'
        filename = '{}_{}_backup.tar.gz'.format(curr_sta, task)
        savepath = station_path / 'backup'
        savepath.mkdir(parents=True, exist_ok=True)
        files_to_tar = path_glob(path, glob_str)
        if dates is not None:
            files_to_tar = slice_task_date_range(files_to_tar, dates, 'backup')
        try:
            tar_dir(files_to_tar, filename, savepath, compresslevel=None)
        except FileNotFoundError:
            print(
                'skipping {} , because no {} found in {}'.format(
                    curr_sta, glob_str, path))
            continue
        print('{} station {} files were backed up to {}'.format(
            curr_sta, glob_str, savepath / filename))
    return


def generate_rinex_reader(station, dates):
    from pathlib import Path
    lines = []
    cwd = Path().cwd()
    print('created rinex header reader script with the following parameters:')
    for curr_sta in station:
        station_path = workpath / curr_sta
        rinexpath = station_path / 'rinex'
        savepath = pwpath
        lines.append('cd {}'.format(station_path))
        line = 'nohup python -u {}/rinex_header_reader.py'.format(pwpath)\
            + ' --rinexpath {} --savepath {}'.format(rinexpath, savepath)
        if dates is not None:
            line += ' --daterange {} {}'.format(dates[0], dates[1])
        line += ' &>{}/nohup_{}_rnx_reader.txt&'.format(pwpath, curr_sta)
        lines.append(line)
        lines.append('cd {}'.format(pwpath))
        print('station: {}, savepath: {}'.format(curr_sta, savepath))
    with open(cwd / script_file, 'w') as file:
        for item in lines:
            file.write("%s\n" % item)
    print('run the script with source {} !'.format(script_file))
    return


def generate_rinex_download(station, mdt, db):
    from pathlib import Path
    lines = []
    cwd = Path().cwd()
    print('created rinex download script with the following parameters:')
    for curr_sta in station:
        station_path = workpath / curr_sta
        station_path.mkdir(parents=True, exist_ok=True)
        savepath = (station_path / 'rinex')
        savepath.mkdir(parents=True, exist_ok=True)
        lines.append('cd {}'.format(station_path))
        if db is None:
            db = 'garner'
        #     db_year = 1988
        # elif db == 'cddis':
        #     db_year = 1992
        line = 'nohup python -u {}/single_rinex_station_download_from_garner.py'.format(pwpath)\
                + ' --path {} --mode rinex --station {} --db {}'.format(savepath, curr_sta, db)\
                + '&>{}/nohup_{}_download.txt&'.format(pwpath, curr_sta)
                # + ' --mdt {}'.format(db_year)
        if mdt is not None:
            line = 'nohup python -u {}/single_rinex_station_download_from_garner.py'.format(pwpath)\
                + ' --path {} --mode rinex --station {} --db {}'.format(savepath, curr_sta, db)\
                + ' --mdt {} &>{}/nohup_{}_download.txt&'.format(mdt, pwpath, curr_sta)
        lines.append(line)
        lines.append('cd {}'.format(pwpath))
        print('station: {}, savepath: {}'.format(curr_sta, savepath))
        if mdt is not None:
            print('station: {}, datetime: {}'.format(curr_sta, mdt))
        if db is not None:
            print('station: {}, db: {}'.format(curr_sta, db))
    with open(cwd / script_file, 'w') as file:
        for item in lines:
            file.write("%s\n" % item)
    print('run the script with source {} !'.format(script_file))
    return


def generate_gipsyx_run(station, task, tree, staDb, dates):
    from pathlib import Path
    lines = []
    cwd = Path().cwd()
    print('created gipsyx run script with the following parameters:')
    for curr_sta in station:
        station_path = workpath / curr_sta
        rinexpath = station_path / 'rinex'
        runpath = station_path / 'rinex/30hr'
        lines.append('cd {}'.format(station_path))
        if task == 'drdump' or task == 'edit30hr':
            line = 'nohup python -u {}/run_gipsyx.py'.format(pwpath)\
                    + ' --rinexpath {} --prep {}'.format(rinexpath, task)\
                    + ' --staDb {}'.format(staDb)
            if tree is not None:
                line += ' --tree {}'.format(tree)
            if dates is not None:
                line += ' --daterange {} {}'.format(dates[0], dates[1])
            line += ' &>{}/nohup_{}_{}.txt&'.format(pwpath, curr_sta, task)
        elif task == 'run':
            line = 'nohup python -u {}/run_gipsyx.py'.format(pwpath)\
                    + ' --rinexpath {} --staDb {}'.format(runpath, staDb)
            if tree is not None:
                line += ' --tree {}'.format(tree)
            if dates is not None:
                line += ' --daterange {} {}'.format(dates[0], dates[1])
            line += ' &>{}/nohup_{}_{}.txt&'.format(pwpath, curr_sta, task)
        lines.append(line)
        lines.append('cd {}'.format(pwpath))
        print('station: {}, task: {}'.format(curr_sta, task))
        print('station: {}, rinexpath: {}'.format(curr_sta, rinexpath))
        print('station: {}, tree: {}'.format(curr_sta, tree))
        print('station: {}, staDb: {}'.format(curr_sta, staDb))
    with open(cwd / script_file, 'w') as file:
        for item in lines:
            file.write("%s\n" % item)
    print('run the script with source {} !'.format(script_file))
    return


def generate_gipsyx_post(station, iqr_k):
    from pathlib import Path
    lines = []
    cwd = Path().cwd()
    lines = []
    print('created gipsyx post procceser script with the following parameters:')
    for curr_sta in station:
        station_path = workpath / curr_sta
        savepath = station_path / 'gipsyx_solutions'
        tdppath = station_path / 'rinex/30hr/results'
        lines.append('cd {}'.format(station_path))
        if iqr_k is None:
            line = 'nohup python -u {}/gipsyx_post_proc.py'.format(pwpath)\
                + ' --tdppath {} --savepath {}'.format(tdppath, savepath)\
                + ' &>{}/nohup_{}_post.txt&'.format(pwpath, curr_sta)
        else:
            line = 'nohup python -u {}/gipsyx_post_proc.py'.format(pwpath)\
                + ' --tdppath {} --savepath {} --iqr_k {}'.format(tdppath, savepath, iqr_k)\
                + ' &>{}/nohup_{}_post.txt&'.format(pwpath, curr_sta)
        lines.append(line)
        lines.append('cd {}'.format(pwpath))
        print('station: {}, tdppath: {}'.format(curr_sta, tdppath))
        print('station:{}, savepath: {}'.format(curr_sta, savepath))
        if iqr_k is not None:
            print('station: {}, iqr_k: {}'.format(curr_sta, iqr_k))
    with open(cwd / script_file, 'w') as file:
        for item in lines:
            file.write("%s\n" % item)
    print('run the script with source {} !'.format(script_file))
    return


def task_switcher(args):
    if args.delete:
        generate_delete(args.station, args.task, args.daterange)
    elif args.backup:
        generate_backup(args.station, args.task, args.daterange)
    else:
        if args.task == 'rinex_download':
            generate_rinex_download(args.station, args.mdt, args.db)
        elif args.task == 'rinex_reader':
            generate_rinex_reader(args.station, args.daterange)
        elif args.task == 'drdump' or args.task == 'edit30hr' or args.task == 'run':
            generate_gipsyx_run(args.station, args.task, args.tree, args.staDb,
                                args.daterange)
        elif args.task == 'post':
            generate_gipsyx_post(args.station, args.iqr_k)
    return


def check_year(year):
    from datetime import datetime
    year = int(year)
    this_year = datetime.today().year
    if year < 1988:
        raise argparse.ArgumentTypeError('{} should be >= 1988'.format(year))
    if year > datetime.today().year:
        raise argparse.ArgumentTypeError(
            '{} should be <= {}'.format(
                year, this_year))
    return year


if __name__ == '__main__':
    import argparse
    import sys
    from pathlib import Path
    from aux_gps import configure_logger
    from aux_gps import get_var
    from aux_gps import path_glob
    import pandas as pd
    global script_file
    global pwpath
    global workpath
    # main directive:
    # write a script called run_gipsyx_script.sh with:
    # cd to the workpath / station and run nohup with the usual args
    script_file = 'gipsyx_pw_script.sh'
    check_python_version(min_major=3, min_minor=6)
    parser = argparse.ArgumentParser(description='a command line tool for ' +
                                     'generating pw shell script to run tasks' +
                                     'as download rinex files, running gipsyx etc...')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # remove this line: optional = parser...
    required.add_argument('--task', help="a task to preform, e.g., run_gipsyx, post_gipsyx, download_rinex."
                          , choices=['rinex_download', 'rinex_reader',
                                     'drdump', 'edit30hr', 'run', 'post'])
    required.add_argument('--station', help="GPS station name four lowercase letters,",
                          nargs='+', type=check_station_name)
    optional.add_argument('--mdt', help='minimum year-month to begin search in garner site.',
                          type=check_year)
    optional.add_argument('--daterange', help='add specific date range, can be one day',
                          type=str, nargs=2)
    optional.add_argument(
        '--staDb',
        help='add a station DB file for antennas and receivers in rinexpath',
        type=str)
    optional.add_argument('--tree', help='gipsyX tree directory.',
                          type=str)
    optional.add_argument('--iqr_k', help='iqr k data filter criterion',
                          type=float)
    optional.add_argument('--db', help='database to download rinex files from.',
                          choices=['garner', 'cddis'])
    required.add_argument('--delete', action='store_true')  # its False
    required.add_argument('--backup', action='store_true')  # its False
#                          metavar=str(cds.start_year) + ' to ' + str(cds.end_year))
#    optional.add_argument('--half', help='a spescific six months to download,\
#                          e.g, 1 or 2', type=int, choices=[1, 2],
#                          metavar='1 or 2')
    parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()
#    for arg in vars(args):
#        print(arg, getattr(args, arg))
#    sys.exit()
    pwpath = Path(get_var('PWCORE'))
    workpath = Path(get_var('PWORK'))
    if pwpath is None:
        raise ValueError('Put source code folder at $PWCORE')
    # get all the names of israeli gnss stations:
    isr_stations = pd.read_fwf(pwpath / 'stations_approx_loc.txt')
    isr_stations = isr_stations.iloc[:,0].tolist()
    if workpath is None:
        raise ValueError('Put source code folder at $PWORK')
    # get the names of the stations in workpath:
    stations = path_glob(workpath, '*')
    stations = [x.as_posix().split('/')[-1] for x in stations if x.is_dir()]
    if args.task is None:
        print('task is a required argument, run with -h...')
        sys.exit()
    if args.station is None:
        print('station is a required argument, run with -h...')
        sys.exit()
    if args.station == ['soin']:
        args.station = isr_stations
    # use soin (survey of israel network) stations db for israeli stations and ocean loading also:
    if all(a in isr_stations for a in args.station) and args.tree is None and args.staDb is None:
        args.tree = pwpath / 'my_trees/ISROcnld'
        args.staDb = pwpath / 'ALL.staDb'
    else:
        if args.staDb is not None:
            args.staDb = pwpath / args.staDb
        else:
            args.staDb = pwpath / 'ALL.staDb'
        if args.tree is not None:
            args.tree = pwpath / args.tree
    if (get_var('GCORE') is None and not args.delete or get_var('GCORE')
            is None and not args.backup):
        raise ValueError('Run source ~/GipsyX-1.1/rc_GipsyX.sh first !')


    task_switcher(args)
    # print(parser.format_help())
#    # print(vars(args))

#    elif args.field is None:
#        print('field is a required argument, run with -h...')
#        sys.exit()
