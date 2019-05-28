#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:52:06 2019

@author: ziskin
"""


class Watcher:
    """ A simple class, set to watch its variable. """
    def __init__(self, value, length):
        self.variable = value
        self.length = length

    def set_value(self, new_value, length):
        if len(new_value) != length or not new_value.isdigit():
            raise ValueError('value length is not {} or it is not a digit'.format(self.length))
        if self.variable != new_value:
            # self.pre_change()
            self.variable = new_value
            return self.variable


#    def pre_change(self):
        # do stuff before variable is about to be changed
#        pass

#    def post_change(self):
        # do stuff right after variable has changed
#        print(self.variable)


def get_rinex_all_years(rinex_path, station_path, station='tela'):
    """copy all station rinex files to a single directory so it can be
    proccessed by gipsy"""
    import os
    import shutil
    print('Creating {}/{}'.format(station_path, station))
    savepath = station_path / station
    # year = Watcher('1991', 4)
    day = Watcher('001', 3)
    new_year = Watcher('1991', 4)
    cnt = 0
    if not os.path.exists(savepath):
        try:
            os.makedirs(savepath)
        except OSError:
            print("Creation of the directory %s failed" % savepath)
        else:
            print("Successfully created the directory %s" % savepath)
    else:
        print('Folder {} already exists.'.format(savepath))

    for path in rinex_path.rglob('*'):
        spath = path.as_posix()
        filename = spath.split('/')[-1].split('.')[0]
        my_file = spath.split('/')[-1]
        if filename.isdigit():
            continue

        year = spath.split('/')[-3]
        try:
            nyear = new_year.set_value(year, 4)
        except ValueError:
            continue
        if nyear is not None:
            print('scanning {} in year {}....'.format(station, nyear))
        if len(year) != 4 or not year.isdigit():
            continue
        try:
            new_day = day.set_value(spath.split('/')[-2], 3)
        except ValueError:
            continue
#        day = spath.split('/')[-2]
#        if len(day) != 3 or not day.isdigit():
#            continue
        if station in filename:
            to_file = savepath / my_file
            shutil.copy(path, to_file)
            print('Copied {} to {}'.format(my_file, savepath))
            cnt += 1
        else:
#            if new_day is not None:
#                print('{} not found in year {}, day {}'.format(station, year, new_day))
            continue
    print('Total {} were copied.'.format(cnt))
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

if __name__ == '__main__':
    import argparse
    import sys
    from pathlib import Path
    check_python_version(min_major=3, min_minor=6)
    parser = argparse.ArgumentParser(description='a command line tool for ' +
                                     'extracting a single station rinex files' +
                                     ' and copy them to a single directory to' +
                                     ' be proccesed by gipsy')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # remove this line: optional = parser...
    required.add_argument('--path', help="a full path to read station rinex" + 
                           " files, e.g., /home/ziskin/garner/", type=check_path)
    required.add_argument('--station', help="GPS station name four lowercase letters,",
                          type=check_station_name)
    required.add_argument('--savepath', help="a full savepath to save station rinex" + 
                           " files, e.g., /home/ziskin/garner/ WITHOUT station name", type=check_path)
    #optional.add_argument('--station', nargs='+',
    #                      help='GPS station name, 4 UPPERCASE letters',
    #                      type=check_station_name)
#                          metavar=str(cds.start_year) + ' to ' + str(cds.end_year))
#    optional.add_argument('--half', help='a spescific six months to download,\
#                          e.g, 1 or 2', type=int, choices=[1, 2],
#                          metavar='1 or 2')
    parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()
    # print(parser.format_help())
#    # print(vars(args))
    if args.path is None or args.savepath is None:
        print('path or savepath is a required argument, run with -h...')
        sys.exit()
#    elif args.field is None:
#        print('field is a required argument, run with -h...')
#        sys.exit()
    if args.station is not None:
        path = Path(args.path)
        savepath = Path(args.savepath)
        get_rinex_all_years(path, savepath, args.station)
    else:
        raise ValueError('need to specify station!')
