#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 10:51:10 2018

@author: shlomi
"""
import platform
from pathlib import Path
path = Path().cwd()
if platform.system() == 'Linux':
    if platform.node() == 'ziskin-XPS-8700':
        work_path = Path('/home/ziskin/Work_Files/')
        work_yuval = work_path / 'PW_yuval'
        work_chaim = work_path / 'Chaim_Stratosphere_Data'
        cwd = Path().cwd()
        geo_path = Path('/home/ziskin/geo_ariel_home/')
        adams_path = Path('/home/ziskin/adams_home/')
        data11_path = Path('/home/ziskin/data11/')
    elif platform.node() == 'shlomipc':
        work_path = Path('/mnt/DATA/Work_Files/')
        work_yuval = work_path / 'PW_yuval'
        work_chaim = work_path / 'Chaim_Stratosphere_Data'
        cwd = Path().cwd()
        geo_path = Path('/home/shlomi/geo_ariel_home/')
        adams_path = Path('/home/shlomi/adams_home/')
        data11_path = Path('/home/shlomi/data11/')
    elif platform.node() == 'geophysics1.yosh.ac.il':
        work_path = Path('/home/ziskin/Work_Files/')
        work_yuval = work_path / 'PW_yuval'
        work_chaim = work_path / 'Chaim_Stratosphere_Data'
        cwd = Path().cwd()
        # geo_path = Path('/home/ziskin/geo_ariel_home/')
        # adams_path = Path('/home/ziskin/adams_home/')
        # data11_path = Path('/home/ziskin/data11/')
    else:
        work_path = Path('/home/ziskin/Work_Files/')
        work_yuval = work_path / 'PW_yuval'
        work_chaim = work_path / 'Chaim_Stratosphere_Data'
        cwd = Path().cwd()
elif platform.system() == 'Darwin':
    if platform.node() == 'Venus':
        work_path = Path('/Users/shlomi/Documents/')
        work_yuval = work_path / 'PW_yuval'
        work_chaim = work_path / 'Chaim_Stratosphere_Data'
        cwd = Path().cwd()
        geo_path = Path('/Users/shlomi/geo_ariel_home/')
        adams_path = Path('/Users/shlomi/adams_home/')
        data11_path = Path('/Users/shlomi/data11/')
