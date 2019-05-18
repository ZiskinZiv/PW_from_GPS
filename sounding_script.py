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
    # loop over lines lists in each year:
    pw_years = []
    for file in savepath.glob('bet_dagan*.json'):
        year = file.as_posix().split('.')[0].split('_')[-1]
        print('Opening json file year: {}'.format(year))
        with open(file) as read_file:
            lines_list = json.load(read_file)
        # loop over the lines list:
        pw_list = []
        dt_list = []
        for lines in lines_list:
            # print('.')
            try:
                pw = float([x for x in lines if '[mm]' in x][0].split(':')[-1])
                dt = [x for x in lines if 'Observation time' in
                      x][0].split(':')[-1].split()[0]
                # The %y (as opposed to %Y) is to read 2-digit year (%Y=4-digit)
                dt_list.append(pd.to_datetime(dt, format='%y%m%d/%H%M'))
                pw_list.append(pw)
                st_num = int([x for x in lines if 'Station number' in
                              x][0].split(':')[-1])
                st_lat = float([x for x in lines if 'Station latitude' in
                                x][0].split(':')[-1])
                st_lon = float([x for x in lines if 'Station longitude' in
                                x][0].split(':')[-1])
                st_alt = float([x for x in lines if 'Station elevation' in
                                x][0].split(':')[-1])
            except IndexError:
                print('no data found in lines entry...')
                continue
        pw_year = xr.DataArray(pw_list, dims=['time'])
        pw_year['time'] = dt_list
        pw_years.append(pw_year)
    pw = xr.concat(pw_years, 'time')
    pw.attrs['description'] = 'BET_DAGAN soundings of precipatable water'
    pw.attrs['units'] = 'mm'  # eqv. kg/m^2
    pw.attrs['station_number'] = st_num
    pw.attrs['station_lat'] = st_lat
    pw.attrs['station_lon'] = st_lon
    pw.attrs['station_alt'] = st_alt
    pw = pw.sortby('time')
    pw.to_netcdf(savepath / 'PW_bet_dagan_soundings.nc')
    return pw


def get_sounding_data(savepath, start_date='2005-01-01',
                      end_date='2019-04-30'):
    import requests
    from bs4 import BeautifulSoup as bs
    import pandas as pd
    import json
    import numpy as np
    import xarray as xr
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    dates = pd.date_range(start_date, end_date, freq='12h')
    years = np.arange(start_date.year, end_date.year + 1)
    time = xr.DataArray(dates, dims=['time'])
    for year in years:
        lines_list = []
        for date in time.sel(time=str(year)).values:
            date = pd.to_datetime(date)
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
                   '&FROM=' + day + hour + '&TO=0100&STNM=40179')
            r = requests.get(url)
            soup = bs(r.text, "lxml")
            allLines = soup.text.split('\n')
    #        splice = allLines[53:78]
    #        keys = ['_'.join(x.split(':')[0].lower().split()) for x in splice]
    #        values = [x.split(':')[-1] for x in splice]
    #        print(keys)
    #        dict_list.append(dict(zip(keys, values)))
            lines_list.append(allLines)
        print('Saving list of dicts to: {}'.format(savepath))
        filename = 'bet_dagan_soundings_' + str(year) + '.json'
        with open(savepath / filename, 'w') as fout:
            json.dump(lines_list, fout)
        print('Done!')
    return

# get_sounding_data(Path('/home/ziskin/Work_Files/PW_yuval/sounding/'))
# proccess_sounding_json(Path('/home/ziskin/Work_Files/PW_yuval/sounding/'))