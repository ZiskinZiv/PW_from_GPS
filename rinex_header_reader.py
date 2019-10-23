#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:15:13 2019

@author: shlomi
"""


def produce_log_report(df, station, savepath):
    import logging
    logger = logging.getLogger('rinex_hdr_reader')
    filename = '{}_log.txt'.format(station)
    logger.info('saving raw log file ({}) to {}'.format(filename, savepath))
    df.to_csv(savepath / filename, sep=(' '))
    return


def read_all_rinex_headers(rinexpath):
    import pandas as pd
    from aux_gps import path_glob
    import logging
    logger = logging.getLogger('rinex_hdr_reader')
    ant_list = []
    rec_list = []
    name_list = []
    aserial_list = []
    rinex_list = []
    dts = []
    logger.info('staring to read rnx files headers.')
    logger.info('proccessing {}'.format(rinexpath))
    cnt = 0
    total = len(path_glob(rinexpath))
    for rfn in sorted(path_glob(rinexpath)):
        filename = rfn.as_posix().split('/')[-1][0:12]
        rinex_list.append(filename)
        dic = read_one_rnx_file(rfn)
        ant_list.append(dic['ant'])
        aserial_list.append(dic['ant_serial'])
        rec_list.append(dic['rec'])
        name_list.append(dic['name'])
        dts.append(dic['dt'])
        logger.info('reading rnx header of {} ({}/{})'.format(filename, cnt,
                    total))
        cnt += 1
    if len(set(name_list)) > 1:
        raise Exception('path {} includes more than one station!'.format(rinexpath))
    station = list(set(name_list))[0]
    df = pd.DataFrame(rinex_list, index=dts, columns=['rfn'])
    df['ant'] = ant_list
    df['ant_seriel'] = aserial_list
    df['rec'] = rec_list
    df.sort_index(inplace=True)
    logger.info('done reading {} rnx files for station {}.'.format(cnt, station))
    return df, station


def read_one_rnx_file(rfn_with_path):
    import georinex as gr
    import pandas as pd

    def parse_field(field):
        field_new = [x.split(' ') for x in field]
        flat = [item for sublist in field_new for item in sublist]
        return [x for x in flat if len(x) > 1]
    hdr = gr.rinexheader(rfn_with_path)
    header = {}
    ant = [val for key, val in hdr.items() if 'ANT' in key]
    header['ant'] = parse_field(ant)[1]
    header['ant_serial'] = parse_field(ant)[0]
    rec = [val for key, val in hdr.items() if 'REC' in key]
    rec = ' '.join(parse_field(rec)[1:3])
    header['rec'] = rec
    name = [val for key, val in hdr.items() if 'NAME' in key]
    header['name'] = parse_field(name)[0]
    dts = pd.to_datetime(hdr['t0'])
    header['dt'] = dts
    return header


def check_path(path):
    import os
    path = str(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(path + ' does not exist...')
    return path


if __name__ == '__main__':
    import argparse
    import sys
    from aux_gps import configure_logger
    logger = configure_logger(name='rinex_hdr_reader')
    parser = argparse.ArgumentParser(description='a command line tool for ' +
                                     'reading rnx file headers for one station' +
                                     'and produce a log file.')
    # optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # remove this line: optional = parser...
    required.add_argument('--rinexpath', help="a main path to the rinex files." +
                          " files, e.g., /home/ziskin/rinex/", type=check_path)
    required.add_argument('--savepath', help="a path to save the log report." +
                          " files, e.g., /home/ziskin/", type=check_path)
    # parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()
    # print(parser.format_help())
#    # print(vars(args))
    if args.rinexpath is None:
        print('rinexpath is a required argument, run with -h...')
        sys.exit()
    df, station = read_all_rinex_headers(args.rinexpath)
    produce_log_report(df, station, args.savepath)
