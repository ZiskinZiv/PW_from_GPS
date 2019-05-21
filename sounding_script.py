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
    df_years = []
    bad_line = []
    for file in savepath.glob('bet_dagan*.json'):
        year = file.as_posix().split('.')[0].split('_')[-1]
        print('Opening json file year: {}'.format(year))
        with open(file) as read_file:
            lines_list = json.load(read_file)
        # loop over the lines list:
        pw_list = []
        dt_list = []
        df_list = []
        for lines in lines_list:
            # print('.')
            try:
                pw = float([x for x in lines if '[mm]' in x][0].split(':')[-1])
                dt = [x for x in lines if 'Observation time' in
                      x][0].split(':')[-1].split()[0]
                # The %y (as opposed to %Y) is to read 2-digit year
                # (%Y=4-digit)
                header_line = [
                    x for x in range(
                        len(lines)) if '40179  Bet Dagan Observations'
                    in lines[x]][0] + 3
                end_line = [x for x in range(len(lines)) if
                            'Station information and sounding indices'
                            in lines[x]][0]
                header = lines[header_line].split()
                units = lines[header_line + 1].split()
                df = pd.DataFrame(
                    [x.split() for x in lines[header_line + 3:end_line]],
                    columns=header)
                df = df.astype(float)
                dt_list.append(pd.to_datetime(dt, format='%y%m%d/%H%M'))
                pw_list.append(pw)
                df_list.append(df)
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
                bad_line.append(lines)
                continue
            except AssertionError:
                bad_line.append(lines)
                continue
        pw_year = xr.DataArray(pw_list, dims=['time'])
        df_year = [xr.DataArray(x,dims=['mpoint', 'var']) for x in df_list]
        df_year = xr.concat(df_year, 'time')
        df_year['time'] = dt_list
        df_year['var'] = header
        pw_year['time'] = dt_list
        pw_years.append(pw_year)
        df_years.append(df_year)
    pw = xr.concat(pw_years, 'time')
    da = xr.concat(df_years, 'time')
    da.attrs['description'] = 'BET_DAGAN soundings full profile'
    units_dict = dict(zip(header, units))
    for k, v in units_dict.items():
        da.attrs[k] = v
    pw.attrs['description'] = 'BET_DAGAN soundings of precipatable water'
    pw.attrs['units'] = 'mm'  # eqv. kg/m^2
    pw.attrs['station_number'] = st_num
    pw.attrs['station_lat'] = st_lat
    pw.attrs['station_lon'] = st_lon
    pw.attrs['station_alt'] = st_alt
    pw = pw.sortby('time')
    da = da.sortby('time')
    # drop 0 pw - not physical
    pw = pw.where(pw > 0, drop=True)
    pw.to_netcdf(savepath / 'PW_bet_dagan_soundings.nc')
    da.to_netcdf(savepath / 'ALL_bet_dagan_soundings.nc')
    return pw, da, bad_line


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