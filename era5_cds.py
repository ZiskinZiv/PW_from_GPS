#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:21:11 2019

@author: ziskin
"""

import cdsapi
import numpy as np
c = cdsapi.Client()
decade = np.arange(1979, 2019, 10)
for dec in decade:
    years = [str(x) for x in np.arange(dec, dec + 10).tolist()]
    print('downloading years {} to {}'.format(years[0], years[-1]))
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type':'reanalysis',
            'format':'netcdf',
            'variable':'total_cloud_cover',
            'year':years,
            'month':[
                '01','02','03',
                '04','05','06',
                '07','08','09',
                '10','11','12'
            ],
            'day':[
                '01','02','03',
                '04','05','06',
                '07','08','09',
                '10','11','12',
                '13','14','15',
                '16','17','18',
                '19','20','21',
                '22','23','24',
                '25','26','27',
                '28','29','30',
                '31'
            ],
            'time':[
                '00:00','01:00','02:00',
                '03:00','04:00','05:00',
                '06:00','07:00','08:00',
                '09:00','10:00','11:00',
                '12:00','13:00','14:00',
                '15:00','16:00','17:00',
                '18:00','19:00','20:00',
                '21:00','22:00','23:00'
            ],
            'grid':[0.25, 0.25],
            'area':[34, 34, 29, 36]
        },
        'Total_cloud_cover_israel_' + years[0] + '-' + years[-1] + '.nc')
