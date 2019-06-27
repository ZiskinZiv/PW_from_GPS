#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:29:10 2019

@author: ziskin
"""
from PW_startup import *
from ims_procedures import ims_api_get_meta
from ims_procedures import download_ims_single_station
from pathlib import Path
savepath = Path('/home/ziskin/Work_Files/PW_yuval/IMS_T/10mins')


def download_all_10mins_ims(savepath):
    def get_already_dl_dict(savepath):
        import collections
        dl_files = []
        for file in savepath.rglob('*.nc'):
            dl_files.append(file.as_posix().split('/')[-1])
        already_dl_id = [int(x.split('_')[1]) for x in dl_files]
        already_dl_names = [x.split('_')[0] for x in dl_files]
        already_dl = dict(zip(already_dl_id, already_dl_names))
        od = collections.OrderedDict(sorted(already_dl.items()))
        return od

    stations = ims_api_get_meta(active_only=True, channel_name='TD')
    for index, row in stations.iterrows():
        st_id = row['stationId']
        od = get_already_dl_dict(savepath)
        if st_id not in od.keys():
            download_ims_single_station(savepath=savepath, channel_name='TD',
                                        stationid=st_id)
        else:
            print('station {} is already in {}, skipping...'.format(od[st_id],
                  savepath))
    return

download_all_10mins_ims(savepath)