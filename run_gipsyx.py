#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 13:58:05 2019

@author: ziskin
"""


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


def run_gipsyx_for_station(rinexpath, savepath):
    from pathlib import Path
    import subprocess
    # first check for GCORE path:
    if not Path(get_var('GCORE')).is_dir():
        raise ValueError('Run ~/GipsyX-1.1/rc_GipsyX.sh first !')
    for file_and_path in sorted(rinexpath.glob('*.Z')):
        filename = file_and_path.as_posix().split('/')[-1][0:-2]
        print('processing {}'.format(filename))
        command = 'gd2e.py -rnxFile {} > {}.log 2>{}.err'.format(file_and_path.as_posix(), filename, filename)
        subprocess.run(command, shell=True)
        filenames = [filename + '.err', filename + '.log', 'smoothFinal.tdp']
        orig_filenames = [Path.cwd() / x for x in filenames]
        final_tdp = filename + '_smoothFinal.tdp'
        filenames = [filename + '.err', filename + '.log', final_tdp]
        print(filenames)
        dest_filenames = [savepath / x for x in filenames]
        for orig, dest in zip(orig_filenames, dest_filenames):
            orig.rename(dest)
    return


if __name__ == '__main__':
    import argparse
    import sys
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
    # optional.add_argument('--station', nargs='+',
    #                      help='GPS station name, 4 UPPERCASE letters',
    #                      type=check_station_name)
#                          metavar=str(cds.start_year) + ' to ' + str(cds.end_year))
#    optional.add_argument('--half', help='a spescific six months to download,\
#                          e.g, 1 or 2', type=int, choices=[1, 2],
#                          metavar='1 or 2')
    parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()

    if args.savepath is None:
        print('savepath is a required argument, run with -h...')
        sys.exit()
    if args.rinexpath is None:
        print('rinexpath is a required argument, run with -h...')
        sys.exit()
    run_gipsyx_for_station(args.rinexpath, args.savepath)