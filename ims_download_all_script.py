#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:29:10 2019

@author: ziskin
"""
# TODO: add logger 
# TODO: add option to just update recent data from 10 mins api

def check_path(path):
    import os
    from pathlib import Path
    path = str(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(path + ' does not exist...')
    return Path(path)


def generate_delete(savepath, channel_name):
    from aux_gps import query_yes_no
    from aux_gps import path_glob
    try:
        glob = '*_{}_10mins.nc'.format(channel_name)
        files_to_delete = path_glob(savepath, glob)
    except FileNotFoundError:
        print('skipping {} , because its empty or not existant..'.format(savepath))
        return
    print('WARNING for channel {}, ALL nc files in {} WILL BE DELETED!'.format(channel_name, savepath))
    to_delete = query_yes_no('ARE YOU SURE ?')
    if not to_delete:
        print('files NOT deleted...')
    else:
        [x.unlink() for x in files_to_delete]
        print('FILES DELETED!')
    return


def download_all_10mins_ims(savepath, channel_name='TD'):
    from aux_gps import path_glob
    import xarray as xr
    glob = '*_{}_10mins.nc'.format(channel_name)
    files = sorted(path_glob(savepath, glob, return_empty_list=True))
    files = [x for x in files if x.is_file()]
    time_dim = list(set(xr.open_dataarray(files[0]).dims))[0]
    last_dates = [xr.open_dataarray(x)[time_dim][-1].values.item() for x in files]
    st_id_downloaded = [int(x.as_posix().split('/')[-1].split('_')[1]) for x in files]
    d = dict(zip(st_id_downloaded, last_dates))
    stations = ims_api_get_meta(active_only=True, channel_name=channel_name)
    for index, row in stations.iterrows():
        st_id = row['stationId']
        if st_id not in d.keys():
            download_ims_single_station(savepath=savepath,
                                        channel_name=channel_name,
                                        stationid=st_id, update=None)
        elif st_id in d.keys():
            da = download_ims_single_station(savepath=savepath,
                                             channel_name=channel_name,
                                             stationid=st_id, update=d[st_id])
            file = path_glob(savepath, '*_{}_{}_10mins.nc'.format(st_id, channel_name))[0]
            da_old = xr.open_dataarray(file)
            da = xr.concat([da, da_old], time_dim)
            filename = '_'.join([stations['name'], str(st_id), channel_name,
                                 '10mins']) + '.nc'
            comp = dict(zlib=True, complevel=9)  # best compression
            encoding = {var: comp for var in da.to_dataset().data_vars}
            print('saving to {} to {}'.format(filename, savepath))
            da.to_netcdf(savepath / filename, 'w', encoding=encoding)
            print('done!')
        else:
            print('station {} is already in {}, skipping...'.format(st_id,
                  savepath))
    return


if __name__ == '__main__':
    import argparse
    import sys
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
    required.add_argument('--delete', action='store_true')  # its False
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
    if args.channel is not None and not args.delete:
        download_all_10mins_ims(args.savepath, channel_name=args.channel)
    elif args.delete:
        generate_delete(args.savepath, args.channel)
    else:
        raise ValueError('need to specify channel name!')
