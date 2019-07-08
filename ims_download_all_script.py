#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:29:10 2019

@author: ziskin
"""


def check_path(path):
    import os
    path = str(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(path + ' does not exist...')
    return path


def download_all_10mins_ims(savepath, channel_name='TD'):
    def get_already_dl_dict(savepath, channel_name):
        import collections
        dl_files = []
        search_str = '*' + channel_name + '_10mins.nc'
        for file in savepath.rglob(search_str):
            dl_files.append(file.as_posix().split('/')[-1])
        already_dl_id = [int(x.split('_')[1]) for x in dl_files]
        already_dl_names = [x.split('_')[0] for x in dl_files]
        already_dl = dict(zip(already_dl_id, already_dl_names))
        od = collections.OrderedDict(sorted(already_dl.items()))
        return od

    stations = ims_api_get_meta(active_only=True, channel_name=channel_name)
    for index, row in stations.iterrows():
        st_id = row['stationId']
        od = get_already_dl_dict(savepath, channel_name)
        if st_id not in od.keys():
            download_ims_single_station(savepath=savepath,
                                        channel_name=channel_name,
                                        stationid=st_id)
        else:
            print('station {} is already in {}, skipping...'.format(od[st_id],
                  savepath))
    return


if __name__ == '__main__':
    import argparse
    import sys
    from PW_startup import *
    from ims_procedures import ims_api_get_meta
    from ims_procedures import download_ims_single_station
    from pathlib import Path
    channels = ['BP', 'DiffR', 'Grad', 'NIP', 'Rain', 'RH', 'STDwd', 'TD',
                'TDmax', 'TDmin', 'TG', 'Time', 'WD', 'WDmax', 'WS', 'WS10mm',
                'WS1mm', 'WSmax']
    savepath = Path('/home/ziskin/Work_Files/PW_yuval/IMS_T/10mins')
    parser = argparse.ArgumentParser(description='a command line tool for downloading all 10mins stations from the IMS with specific variable')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # remove this line: optional = parser...
    required.add_argument('--savepath', help="a full path to download the files, e.g., /home/ziskin/Work_Files/PW_yuval/IMS_T/10mins", type=check_path)
    required.add_argument('--channel', help="10 mins channel name , e.g., TD, BP or RH",
                          choices=channels)
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
    if args.savepath is None:
        print('savepath is a required argument, run with -h...')
        sys.exit()
#    elif args.field is None:
#        print('field is a required argument, run with -h...')
#        sys.exit()
    if args.channel is not None:
        savepath = Path(args.savepath)
        download_all_10mins_ims(savepath, channel_name=args.channel)
    else:
        raise ValueError('need to specify channel name!')
