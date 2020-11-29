#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:15:13 2019

@author: shlomi
"""

#
#def produce_log_report(df, station, savepath):
#    import logging
#    from pathlib import Path
#    logger = logging.getLogger('rinex_hdr_reader')
#    filename = '{}_log.txt'.format(station)
#    logger.info('saving raw log file ({}) to {}'.format(filename, savepath))
#    df.to_csv(Path(savepath) / filename, sep=' ')
#    return


def save_header_dataframe(df, savename, savepath):
    import logging
    from pathlib import Path
    logger = logging.getLogger('rinex_hdr_reader')
    if savename is not None:
        filename = '{}_rnxheaders.csv'.format(savename)
    else:
        filename = 'generic_rnxheaders.csv'
    logger.info('saving raw log file ({}) to {}'.format(filename, savepath))
    df.to_csv(Path(savepath) / filename, sep=',')
    return


def read_all_rinex_file_headers(rinexpath):
    from aux_gps import path_glob
    import pandas as pd
    import logging
    logger = logging.getLogger('rinex_hdr_reader')
    di_list = []
    files = path_glob(rinexpath, '*.Z')
    logger.info('staring to read rnx files headers.')
    logger.info('proccessing {}'.format(rinexpath))
    cnt = 0
    total = len(files)
    for rfn in sorted(files):
        filename = rfn.as_posix().split('/')[-1][0:12]
        logger.info('reading rnx header of {} ({}/{})'.format(filename, cnt,
                                                              total))
        try:
            dic = read_one_rinex_file(rfn)
            di_list.append(dic)
        except ValueError:
            logger.error('error parsing rnx header of {}'.format(filename))
            continue
        except OSError:
            logger.error('error parsing rnx header of {}'.format(filename))
            continue
        cnt += 1
    df = pd.DataFrame(di_list)
    logger.info('done reading {} rnx files.'.format(cnt))
    return df


#def read_all_rinex_headers(rinexpath, dates):
#    import pandas as pd
#    from aux_gps import path_glob
#    from aux_gps import slice_task_date_range
#    import logging
#    logger = logging.getLogger('rinex_hdr_reader')
#    ant_list = []
#    rec_list = []
#    name_list = []
#    aserial_list = []
#    rinex_list = []
#    dts = []
#    logger.info('staring to read rnx files headers.')
#    logger.info('proccessing {}'.format(rinexpath))
#    cnt = 0
#    files = path_glob(rinexpath)
#    if dates is not None:
#        files = slice_task_date_range(files, dates, 'rinex header reader')
#    total = len(files)
#    for rfn in sorted(files):
#        filename = rfn.as_posix().split('/')[-1][0:12]
#        try:
#            dic = read_one_rnx_file(rfn)
#        except ValueError:
#            logger.error('error parsing rnx header of {}'.format(filename))
#            continue
#        rinex_list.append(filename)
#        ant_list.append(dic['ant'])
#        aserial_list.append(dic['ant_serial'])
#        rec_list.append(dic['rec'])
#        name_list.append(dic['name'])
#        dts.append(dic['dt'])
#        logger.info('reading rnx header of {} ({}/{})'.format(filename, cnt,
#                    total))
#        cnt += 1
##    if len(set(name_list)) > 1:
##        raise Exception('path {} includes more than one station!'.format(rinexpath))
#    station = list(set(name_list))[0]
#    df = pd.DataFrame(rinex_list, index=dts, columns=['rfn'])
#    df['name'] = name_list
#    df['ant'] = ant_list
#    df['ant_seriel'] = aserial_list
#    df['rec'] = rec_list
#    df.sort_index(inplace=True)
#    logger.info('done reading {} rnx files for station {}.'.format(cnt, station))
#    return df, station


def read_one_rinex_file(rinex_file):
    from gcore.RinexHeader import RinexHeader
    header = RinexHeader(rinex_file)
    di = vars(header)['headerInfo']
    return di

#def read_one_rnx_file(rfn_with_path):
#    import georinex as gr
#    import pandas as pd
#    from pandas.errors import OutOfBoundsDatetime
#    from aux_gps import get_timedate_and_station_code_from_rinex
#
#    def parse_field(field):
#        field_new = [x.split(' ') for x in field]
#        flat = [item for sublist in field_new for item in sublist]
#        return [x for x in flat if len(x) > 1]
#    hdr = gr.rinexheader(rfn_with_path)
#    header = {}
#    ant = [val for key, val in hdr.items() if 'ANT' in key]
#    try:
#        header['ant'] = parse_field(ant)[1]
#    except IndexError:
#        header['ant'] = parse_field(ant)
#    try:
#        header['ant_serial'] = parse_field(ant)[0]
#    except IndexError:
#        header['ant_serial'] = parse_field(ant)
#    rec = [val for key, val in hdr.items() if 'REC' in key]
#    try:
#        rec = ' '.join(parse_field(rec)[1:3])
#    except IndexError:
#        rec = parse_field(rec)
#    header['rec'] = rec
#    name = [val for key, val in hdr.items() if 'NAME' in key]
#    try:
#        header['name'] = parse_field(name)[0]
#    except IndexError:
#        header['name'] = parse_field(name)
#    try:
#        dts = pd.to_datetime(hdr['t0'])
#    except OutOfBoundsDatetime:
#        rfn = rfn_with_path.as_posix().split('/')[-1][0:12]
#        dts = get_timedate_and_station_code_from_rinex(rfn, True)
#    except KeyError:
#        rfn = rfn_with_path.as_posix().split('/')[-1][0:12]
#        dts = get_timedate_and_station_code_from_rinex(rfn, True)
#    header['dt'] = dts
#    return header


def check_path(path):
    import os
    path = str(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(path + ' does not exist...')
    return path


if __name__ == '__main__':
    import argparse
    import sys, os
    from aux_gps import configure_logger
    logger = configure_logger(name='rinex_hdr_reader')
    sys.path.insert(0, "{}/lib/python{}.{}".format(os.environ['GCOREBUILD'], \
                    sys.version_info[0], sys.version_info[1]))
    parser = argparse.ArgumentParser(description='a command line tool for ' +
                                     'reading rnx file headers for one station' +
                                     'and produce a log file.')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # remove this line: optional = parser...
    required.add_argument('--rinexpath', help="a main path to the rinex files." +
                          " files, e.g., /home/ziskin/rinex/", type=check_path)
    required.add_argument('--savepath', help="a path to save the log report." +
                          " files, e.g., /home/ziskin/", type=check_path)
    optional.add_argument('--savename', help="a name for the save file",type=str)
#    optional.add_argument('--daterange', help='add specific date range, can be one day',
#                          type=str, nargs=2)
    # parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()
    # print(parser.format_help())
#    # print(vars(args))
    if args.rinexpath is None:
        print('rinexpath is a required argument, run with -h...')
        sys.exit()
    if args.savepath is None:
        print('savepath is a required argument, run with -h...')
        sys.exit()
#    df, station = read_all_rinex_headers(args.rinexpath, args.daterange)
    df = read_all_rinex_file_headers(args.rinexpath)
    save_header_dataframe(df, args.savename, args.savepath)
#    produce_log_report(df, station, args.savepath)
