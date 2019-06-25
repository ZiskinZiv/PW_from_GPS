#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:29:10 2019

@author: ziskin
"""
from ims_procedures import ims_api_get_meta
from ims_procedures import download_ims_single_station
from pathlib import Path
savepath = Path('/home/ziskin/Work_Files/PW_yuval/IMS_T')


def download_all_10mins_ims(savepath):
    stations = ims_api_get_meta(active_only=True, channel_name='TD')
    for index, row in stations.iterrows():
        download_ims_single_station(savepath=savepath, channel_name='TD',
                                    stationid=row['stationId'])
    return

download_all_10mins_ims(savepath)