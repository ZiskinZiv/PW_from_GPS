#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:24:40 2019

@author: ziskin
"""
from pathlib import Path


def proccess_sounding_json(savepath):
    import pandas as pd
    import json
    import xarray as xr
    with open(savepath/'bet_dagan_soundings.json') as file:
        dict_list = json.load(file)
    PWs = [float(x['precipitable_water_[mm]_for_entire_sounding']) for x
           in dict_list]
    # lookout: the %y (as opposed to %Y) is to read 2-digit year (%Y=4-digit)
    datetimes = [pd.to_datetime(x['observation_time'].split()[0],
                                format='%y%m%d/%H%M') for x in dict_list]
    station_number = int(dict_list[0]['station_number'].split()[0])
    station_lat = float(dict_list[0]['station_latitude'].split()[0])
    station_lon = float(dict_list[0]['station_longitude'].split()[0])
    station_alt = float(dict_list[0]['station_elevation'].split()[0])
    pw = xr.DataArray(PWs, dims=['time'])
    pw['time'] = datetimes
    pw.attrs['description'] = 'BET_DAGAN soundings of precipatable water'
    pw.attrs['units'] = 'mm'  # eqv. kg/m^2
    pw.attrs['station_number'] = station_number
    pw.attrs['station_lat'] = station_lat
    pw.attrs['station_lon'] = station_lon
    pw.attrs['station_alt'] = station_alt
    pw.to_netcdf(savepath / 'PW_bet_dagan_soundings.nc')
    return pw


def get_sounding_data(savepath, start_date='2005-01-01',
                      end_date='2019-04-15'):
    import requests
    from bs4 import BeautifulSoup as bs
    import pandas as pd
    import json
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    dates = pd.date_range(start_date, end_date, freq='12h')
    dict_list = []
    for date in dates:
        print('getting datetime: {}'.format(date.strftime('%Y-%m-%d:%H')))
        year = str(date.year)
        month = str(date.month)
        day = str(date.day)
        if date.hour == 0:
            hour = '0' + str(date.hour)
        elif date.hour == 12:
            hour = str(date.hour)
        url = ('http://weather.uwyo.edu/cgi-bin/sounding?region=mideast&'
               'TYPE=TEXT%3ALIST&YEAR=' + year + '&MONTH=' + month +
               '5&FROM=' + day + hour + '&TO=0100&STNM=40179')
        r = requests.get(url)
        soup = bs(r.text, "lxml")
        allLines = soup.text.split('\n')
        splice = allLines[49:74]
        keys = ['_'.join(x.split(':')[0].lower().split()) for x in splice]
        values = [x.split(':')[-1] for x in splice]
        dict_list.append(dict(zip(keys, values)))
    print('Saving list of dicts to: {}'.format(savepath))
    with open(savepath / 'bet_dagan_soundings.json', 'w') as fout:
        json.dump(dict_list, fout)
    print('Done!')
    return dict_list

di_list = get_sounding_data(Path('/home/ziskin/Work_Files/PW_yuval/sounding/'))