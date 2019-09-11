#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:40:34 2019

@author: shlomi
"""

from gipsyx_post_proc import post_procces_gipsyx_all_years
from pathlib import Path
tela = Path('/home/ziskin/Work_Files/PW_yuval/gipsyx_results/tela_all')
post_procces_gipsyx_all_years(tela)