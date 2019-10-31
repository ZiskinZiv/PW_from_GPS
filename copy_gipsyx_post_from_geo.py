#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 08:37:02 2019

@author: shlomi
"""


def progress_percentage(perc, width=None):
    # This will only work for python 3.3+ due to use of
    # os.get_terminal_size the print function etc.

    FULL_BLOCK = '█'
    # this is a gradient of incompleteness
    INCOMPLETE_BLOCK_GRAD = ['░', '▒', '▓']

    assert(isinstance(perc, float))
    assert(0. <= perc <= 100.)
    # if width unset use full terminal
    if width is None:
        width = os.get_terminal_size().columns
    # progress bar is block_widget separator perc_widget : ####### 30%
    max_perc_widget = '[100.00%]' # 100% is max
    separator = ' '
    blocks_widget_width = width - len(separator) - len(max_perc_widget)
    assert(blocks_widget_width >= 10) # not very meaningful if not
    perc_per_block = 100.0/blocks_widget_width
    # epsilon is the sensitivity of rendering a gradient block
    epsilon = 1e-6
    # number of blocks that should be represented as complete
    full_blocks = int((perc + epsilon)/perc_per_block)
    # the rest are "incomplete"
    empty_blocks = blocks_widget_width - full_blocks

    # build blocks widget
    blocks_widget = ([FULL_BLOCK] * full_blocks)
    blocks_widget.extend([INCOMPLETE_BLOCK_GRAD[0]] * empty_blocks)
    # marginal case - remainder due to how granular our blocks are
    remainder = perc - full_blocks*perc_per_block
    # epsilon needed for rounding errors (check would be != 0.)
    # based on reminder modify first empty block shading
    # depending on remainder
    if remainder > epsilon:
        grad_index = int((len(INCOMPLETE_BLOCK_GRAD) * remainder)/perc_per_block)
        blocks_widget[full_blocks] = INCOMPLETE_BLOCK_GRAD[grad_index]

    # build perc widget
    str_perc = '%.2f' % perc
    # -1 because the percentage sign is not included
    perc_widget = '[%s%%]' % str_perc.ljust(len(max_perc_widget) - 3)

    # form progressbar
    progress_bar = '%s%s%s' % (''.join(blocks_widget), separator, perc_widget)
    # return progressbar as string
    return ''.join(progress_bar)


def copy_progress(copied, total):
    print('\r' + progress_percentage(100*copied/total, width=30), end='')


def copyfile(src, dst, *, follow_symlinks=True):
    """Copy data from src to dst.

    If follow_symlinks is not set and src is a symbolic link, a new
    symlink will be created instead of copying the file it points to.

    """
    if shutil._samefile(src, dst):
        raise shutil.SameFileError("{!r} and {!r} are the same file".format(src, dst))

    for fn in [src, dst]:
        try:
            st = os.stat(fn)
        except OSError:
            # File most likely does not exist
            pass
        else:
            # XXX What about other special files? (sockets, devices...)
            if shutil.stat.S_ISFIFO(st.st_mode):
                raise shutil.SpecialFileError("`%s` is a named pipe" % fn)

    if not follow_symlinks and os.path.islink(src):
        os.symlink(os.readlink(src), dst)
    else:
        size = os.stat(src).st_size
        with open(src, 'rb') as fsrc:
            with open(dst, 'wb') as fdst:
                copyfileobj(fsrc, fdst, callback=copy_progress, total=size)
    return dst


def copyfileobj(fsrc, fdst, callback, total, length=16*1024):
    copied = 0
    while True:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)
        copied += len(buf)
        callback(copied, total=total)


def copy_with_progress(src, dst, *, follow_symlinks=True):
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    print('{} is being copied to {}'.format(src, dst))
    copyfile(src, dst, follow_symlinks=follow_symlinks)
    shutil.copymode(src, dst)
    print('\n')
    return


def copy_post_from_geo(remote_path, station):
    from aux_gps import path_glob
    for curr_sta in station:
        src_path = remote_path / curr_sta / 'gipsyx_solutions'
        try:
            filepaths = path_glob(src_path, '*PPP*.nc')
        except FileNotFoundError:
            print('{} final solution not found in {}'.format(curr_sta, src_path))
            continue
        for filepath in filepaths:
            filename = filepath.as_posix().split('/')[-1]
            src_path = src_path / filename
            dst_path = workpath / curr_sta / 'gipsyx_solutions'
            dst_path.mkdir(parents=True, exist_ok=True)
            copy_with_progress(src_path, dst_path)
    print('Done Copying GipsyX results!')
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
    import os
    import shutil
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
