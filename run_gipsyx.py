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


def check_file_in_cwd(filename):
    from pathlib import Path
    cwd = Path().cwd()
    file_and_path = cwd / filename
    if not file_and_path.is_file():
        raise argparse.ArgumentTypeError(
            '{} does not exist at {}'.format(
                filename, cwd))
    return file_and_path


def run_gipsyx_for_station(rinexpath, savepath, staDb=None):
    from pathlib import Path
    import subprocess
    from subprocess import CalledProcessError
    import logging
    import shutil
    from aux_gps import get_timedate_and_station_code_from_rinex
    logger = logging.getLogger('gipsyx')
    # first check for GCORE path:
    if len(get_var('GCORE')) == 0:
        raise ValueError('Run ~/GipsyX-1.1/rc_GipsyX.sh first !')
    if staDb is not None:
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
            orig_rnxedit_paths = ['{}_rnxEdit.log'.format(station), '{}_rnxEdit.err'.format(station)]
            orig_paths = [Path.cwd() / x for x in orig_rnxedit_paths]
            dest_paths = [savepath / x for x in orig_rnxedit_paths]
            try:
                subprocess.run(command0, shell=True, check=True)
                for orig, dest in zip(orig_paths, dest_paths):
                    shutil.move(orig.resolve(), dest.resolve())
            except:
                logger.warning('rnxEditGde.py failed on {}, copying log files.'.format(filename))
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
            subprocess.run(command, shell=True, check=True)
            orig_filenames_paths = [Path.cwd() / x for x in orig_filenames]
            dest_filenames = [filename + '.err', filename + '.log', final_tdp]
        except CalledProcessError:
            logger.warning('gipsyx failed on {}, copying log files.'.format(filename))
            orig_filenames_paths = [Path.cwd() / x for x in orig_filenames[0:2]]
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
    parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()

    if args.savepath is None:
        print('savepath is a required argument, run with -h...')
        sys.exit()
    if args.rinexpath is None:
        print('rinexpath is a required argument, run with -h...')
        sys.exit()
    if args.staDb is None:
        run_gipsyx_for_station(args.rinexpath, args.savepath)
    else:
        run_gipsyx_for_station(args.rinexpath, args.savepath, args.staDb)
