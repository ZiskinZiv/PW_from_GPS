#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:46:25 2019

@author: ziskin
"""
# TODO: improve command line tool, maybe use wget, for sure logger.


def generate_download_shell_script(station_list,
                                   script_file='rinex_download.sh'):
    from pathlib import Path
    lines = []
    cwd = Path().cwd()
    for station in station_list:
        line = 'nohup python -u single_download.py --path ~/Work_Files/PW_yuval/rinex_from_garner/ --mode rinex --station {} &>nohup_{}_download.txt&'.format(station, station)
        lines.append(line)
    with open(cwd / script_file, 'w') as file:
        for item in lines:
            file.write("%s\n" % item)
    print('generated download script file at {}'.format(cwd/script_file))
    return
    
    
def all_orbitals_download(save_dir, minimum_year=None, hr_only=None):
    import htmllistparse
    import requests
    import os
    print('Creating {}/{}'.format(save_dir, 'gipsy_orbitals'))
    savepath = save_dir / 'gipsy_orbitals'
    if not os.path.exists(savepath):
        try:
            os.makedirs(savepath)
        except OSError:
            print("Creation of the directory %s failed" % savepath)
        else:
            print("Successfully created the directory %s" % savepath)
    else:
        print('Folder {} already exists.'.format(savepath))
    command = 'https://sideshow.jpl.nasa.gov/pub/JPL_GPS_Products/Final/'
    cwd, listing = htmllistparse.fetch_listing(command, timeout=30)
    dirs = [f.name for f in listing if '/' in f.name]
    if minimum_year is not None:
        years = [int(x.split('/')[0]) for x in dirs]
        years = [x for x in years if x >= minimum_year]
        dirs = [str(x) + '/' for x in years]
        print('starting search from year {}'.format(minimum_year))
    for year in dirs:
        print(year)
        cwd, listing = htmllistparse.fetch_listing(command + year, timeout=30)
        files = [f.name for f in listing if f.size is not None]
#        2017-01-28.eo.gz
#        2017-01-28.shad.gz
#        2017-01-28_hr.tdp.gz
#        2017-01-28.ant.gz
#        2017-01-28.tdp.gz
#        2017-01-28.frame.gz
#        2017-01-28.pos.gz
#        2017-01-28.wlpb.gz
        if hr_only is None:
            suffixes = ['eo', 'shad', 'ant', 'tdp', 'frame', 'pos', 'wlpb']
            for suff in suffixes:
                found = [f for f in files if suff in f.split('.')[1] and '_' not in f]
                if found:
                    for filename in found:
                        print('Downloading {} to {}.'.format(filename, savepath))
                        r = requests.get(command + year + filename)
                        with open(savepath/filename, 'wb') as file:
                            file.write(r.content)
        else:
            pre_found = [f for f in files if '_' in f]
            if pre_found:
                found = [f for f in pre_found if f.split('.')[0].split('_')[1] == 'hr']
                if found:
                    for filename in found:
                        print('Downloading {} to {}.'.format(filename, savepath))
                        r = requests.get(command + year + filename)
                        with open(savepath/filename, 'w') as file:
                            file.write(r.content)
    return


def single_station_rinex_garner_download(save_dir, minimum_year=None,
                                         station='tela'):
    import htmllistparse
    import requests
    import os
    print('Creating {}/{}'.format(save_dir, station))
    savepath = save_dir / station
    if not os.path.exists(savepath):
        try:
            os.makedirs(savepath)
        except OSError:
            print("Creation of the directory %s failed" % savepath)
        else:
            print("Successfully created the directory %s" % savepath)
    else:
        print('Folder {} already exists.'.format(savepath))
    command = 'http://anonymous:shlomiziskin%40gmail.com@garner.ucsd.edu/pub/rinex/'
    cwd, listing = htmllistparse.fetch_listing(command, timeout=30)
    dirs = [f.name for f in listing if '/' in f.name]
    if minimum_year is not None:
        years = [int(x.split('/')[0]) for x in dirs]
        years = [x for x in years if x >= minimum_year]
        dirs = [str(x) + '/' for x in years]
        print('starting search from year {}'.format(minimum_year))
    for year in dirs:
        print(year)
        cwd, listing = htmllistparse.fetch_listing(command + year, timeout=30)
        days = [f.name for f in listing if '/' in f.name]
        for day in days:
            cwd, listing = htmllistparse.fetch_listing(
                command + year + day, timeout=30)
            files = [f.name for f in listing if f.size is not None]
            found = [f for f in files if station in f]
            if found:
                filename = found[0]
                saved_filename = savepath / filename
                if saved_filename.is_file():
                    print(
                        '{} already exists in {}, skipping...'.format(
                            filename, savepath))
                    continue
                print('Downloading {} to {}.'.format(filename, savepath))
                r = requests.get(command + year + day + filename)
                with open(saved_filename, 'wb') as file:
                    file.write(r.content)
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
        raise argparse.ArgumentTypeError(path + ' does not exist...')
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
    check_python_version(min_major=3, min_minor=6)
    parser = argparse.ArgumentParser(description='a command line tool for ' +
                                     'downloading a single station rinex files' +
                                     'from garner site and copy them to a single directory to' +
                                     ' be proccesed by gipsy')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # remove this line: optional = parser...
    required.add_argument('--path', help="a main path to save station rinex, the tool will create a folder in this path named as the station." +
                          " files, e.g., /home/ziskin/garner/", type=check_path)
    required.add_argument('--mode', help="choose either rinex or orbital",
                          choices=['rinex', 'orbital'])
    optional.add_argument('--station', help="GPS station name four lowercase letters,",
                          type=check_station_name)
    optional.add_argument('--myear', help='minimum year to begin search in garner site.',
                          type=check_year)
    optional.add_argument('--hr_only', help='download only _hr files...',
                          choices=['True'])
#                          metavar=str(cds.start_year) + ' to ' + str(cds.end_year))
#    optional.add_argument('--half', help='a spescific six months to download,\
#                          e.g, 1 or 2', type=int, choices=[1, 2],
#                          metavar='1 or 2')
    parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()
    # print(parser.format_help())
#    # print(vars(args))
    if args.path is None:
        print('path is a required argument, run with -h...')
        sys.exit()
#    elif args.field is None:
#        print('field is a required argument, run with -h...')
#        sys.exit()
    if args.mode == 'rinex':
        if args.station is not None:
            path = Path(args.path)
            if args.myear is not None:
                single_station_rinex_garner_download(path,
                                                     minimum_year=args.myear,
                                                     station=args.station)
            else:
                single_station_rinex_garner_download(path,
                                                     station=args.station)
        else:
            raise ValueError('need to specify station!')
    elif args.mode == 'orbital':
        path = Path(args.path)
        if args.myear is not None:
            if args.hr_only is not None:
                all_orbitals_download(path, minimum_year=args.myear,
                                      hr_only=True)
            else:
                all_orbitals_download(path, minimum_year=args.myear)
        else:
            if args.hr_only is not None:
                all_orbitals_download(path, hr_only=True)
            else:
                all_orbitals_download(path)
    else:
        raise ValueError('must choose mode!')
