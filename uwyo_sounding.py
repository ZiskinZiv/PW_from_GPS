#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 16:30:02 2019

@author: shlomi
"""


def check_date(date):
    import pandas as pd
    return pd.to_datetime(date)


def check_path(path):
    import os
    from pathlib import Path
    path = str(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(path + ' does not exist...')
    return Path(path)


def DensHumid(tempc, pres, e):
    """Density of moist air.
    This is a bit more explicit and less confusing than the method below.
    INPUTS:
    tempc: Temperature (C)
    pres: static pressure (hPa)
    e: water vapor partial pressure (hPa)
    OUTPUTS:
    rho_air (kg/m^3)
    SOURCE: http://en.wikipedia.org/wiki/Density_of_air
    """
    tempk = tempc + 273.15
    prespa = pres * 100.0
    epa = e * 100.0
    Rs_v = 461.52  # Specific gas const for water vapour, J kg^{-1} K^{-1}
    Rs_da = 287.05  # Specific gas const for dry air, J kg^{-1} K^{-1}
    pres_da = prespa - epa
    rho_da = pres_da / (Rs_da * tempk)
    rho_wv = epa/(Rs_v * tempk)
    return rho_da + rho_wv


def precipitable_water(da):
    """Calculate Total Precipitable Water (TPW) for sounding.
    TPW is defined as the total column-integrated water vapour. I
    calculate it from the dew point temperature because this is the
    fundamental moisture variable in this module (even though it is RH
    that is usually measured directly)
    """
    import numpy as np
    tempk = da.sel(var='TEMP').dropna('mpoint').reset_coords(drop=True) + 273.15  # in K
    prespa = da.sel(var='PRES').dropna('mpoint').reset_coords(drop=True) * 100   # in Pa
    mixrkg = da.sel(var='MIXR').dropna('mpoint').reset_coords(drop=True) / 1000.0  # kg/kg
    dwptc = da.sel(var='DWPT').dropna('mpoint').reset_coords(drop=True)
    hghtm = da.sel(var='HGHT').dropna('mpoint').reset_coords(drop=True)
    min_size = min([tempk.size, prespa.size, mixrkg.size, dwptc.size, hghtm.size])
    tempk = tempk.sel(mpoint=slice(0, min_size - 1))
    prespa = prespa.sel(mpoint=slice(0, min_size - 1))
    mixrkg = mixrkg.sel(mpoint=slice(0, min_size - 1))
    dwptc = dwptc.sel(mpoint=slice(0, min_size - 1))
    hghtm = hghtm.sel(mpoint=slice(0, min_size - 1))
    # Get Water Vapour Mixing Ratio, by calculation
    # from dew point temperature
    vprespa = VaporPressure(dwptc)
    # mixrkg = MixRatio(vprespa, prespa)

    # Calculate density of air (accounting for moisture)
    rho = DensHumid(tempk, prespa, vprespa)
    # print('rho: {}, mix: {}, h: {}'.format(rho.shape,mixrkg.shape, hghtm.shape))
    # Trapezoidal rule to approximate TPW (units kg/m^2==mm)
    try:
        tpw = np.trapz(mixrkg * rho, hghtm)
    except ValueError:
        return np.nan
    return tpw


def VaporPressure(tempc, phase="liquid", units='hPa', method=None):
    import numpy as np
    """Water vapor pressure over liquid water or ice.
    INPUTS:
    tempc: (C) OR dwpt (C), if SATURATION vapour pressure is desired.
    phase: ['liquid'],'ice'. If 'liquid', do simple dew point. If 'ice',
    return saturation vapour pressure as follows:
    Tc>=0: es = es_liquid
    Tc <0: es = es_ice
    RETURNS: e_sat  (Pa) or (hPa) units parameter choice
    SOURCE: http://cires.colorado.edu/~voemel/vp.html (#2:
    CIMO guide (WMO 2008), modified to return values in Pa)
    This formulation is chosen because of its appealing simplicity,
    but it performs very well with respect to the reference forms
    at temperatures above -40 C. At some point I'll implement Goff-Gratch
    (from the same resource).
    """
    if units == 'hPa':
        unit = 1.0
    elif units == 'Pa':
        unit = 100.0
    if method is None:
        over_liquid = 6.112 * np.exp(17.67 * tempc / (tempc + 243.12)) * unit
        over_ice = 6.112 * np.exp(22.46 * tempc / (tempc + 272.62)) * unit
    elif method == 'Buck':
        over_liquid = 6.1121 * \
            np.exp((18.678 - tempc / 234.5) * (tempc / (257.4 + tempc))) * unit
        over_ice = 6.1125 * \
            np.exp((23.036 - tempc / 333.7) * (tempc / (279.82 + tempc))) * unit
    # return where(tempc<0,over_ice,over_liquid)

    if phase == "liquid":
        # return 6.112*exp(17.67*tempc/(tempc+243.12))*100.
        return over_liquid
    elif phase == "ice":
        # return 6.112*exp(22.46*tempc/(tempc+272.62))*100.
        return np.where(tempc < 0, over_ice, over_liquid)
    else:
        raise NotImplementedError


def Tm(da, from_raw_sounding=True):
    """ calculate the atmospheric mean temperature with pp as water
    vapor partial pressure and T deg in C. eq is Tm=int(pp/T)dz/int(pp/T^2)dz
    h is the heights vactor"""
    import numpy as np
    if from_raw_sounding:
        tempc = da.sel(var='TEMP').dropna('mpoint').reset_coords(drop=True)
        h = da.sel(var='HGHT').dropna('mpoint').reset_coords(drop=True)
        vp = VaporPressure(tempc, units='hPa')
        tempk = tempc + 273.15
        try:
            Tm = np.trapz(vp / tempk, h) / np.trapz(vp / tempk**2, h)
        except ValueError:
            return np.nan
    else:
        tempc = da
        h = da['height']
        vp = VaporPressure(tempc, units='hPa')
        tempk = tempc + 273.15
        Tm = np.trapz(vp / tempk, h) / np.trapz(vp / tempk**2, h)
    return Tm


def process_sounding_json(json_path, st_num):
    """process json files from sounding download and parse them to xarray"""
    import pandas as pd
    import json
    import xarray as xr
    import os
    import logging
    from aux_gps import path_glob
    logger = logging.getLogger('uwyo')
    # loop over lines lists in each year:
    pw_years = []
    df_years = []
    bad_line = []
    logger.info('proccessing station {} that was downloaded from UWYO website'.format(st_num))
    for file in path_glob(json_path, 'station_{}_soundings_*.json'.format(st_num)):
        year = file.as_posix().split('.')[0].split('_')[-1]
        logger.info('Opening json file year: {}'.format(year))
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
                        len(lines)) if 'Observations at'
                    in lines[x]][0] + 3
                end_line = [x for x in range(len(lines)) if
                            'Station information and sounding indices'
                            in lines[x]][0]
                header = lines[header_line].split()
                units = lines[header_line + 1].split()
                with open(json_path/'temp.txt', 'w') as f:
                    for item in lines[header_line + 3: end_line]:
                        f.write("%s\n" % item)
                df = pd.read_fwf(json_path / 'temp.txt', names=header)
                try:
                    os.remove(json_path / 'temp.txt')
                except OSError as e:  # if failed, report it back to the user
                    logger.error("Error: %s - %s." % (e.filename, e.strerror))
#                df = pd.DataFrame(
#                    [x.split() for x in lines[header_line + 3:end_line]],
#                    columns=header)
                df = df.astype(float)
                dt_list.append(pd.to_datetime(dt, format='%y%m%d/%H%M'))
                pw_list.append(pw)
                df_list.append(df)
#                st_num = int([x for x in lines if 'Station number' in
#                              x][0].split(':')[-1])
                st_lat = float([x for x in lines if 'Station latitude' in
                                x][0].split(':')[-1])
                st_lon = float([x for x in lines if 'Station longitude' in
                                x][0].split(':')[-1])
                st_alt = float([x for x in lines if 'Station elevation' in
                                x][0].split(':')[-1])
            except IndexError:
                logger.warning('no data found in lines entry...')
                bad_line.append(lines)
                continue
            except AssertionError:
                bad_line.append(lines)
                continue
        if not pw_list or not df_list or not dt_list:
            logger.warning('no entries in {}'.format(year))
            continue
        pw_year = xr.DataArray(pw_list, dims=['time'])
        df_year = [xr.DataArray(x, dims=['mpoint', 'var']) for x in df_list]
        df_year = xr.concat(df_year, 'time')
        df_year['time'] = dt_list
        df_year['var'] = header
        pw_year['time'] = dt_list
        pw_years.append(pw_year)
        df_years.append(df_year)
    pw = xr.concat(pw_years, 'time')
    da = xr.concat(df_years, 'time')
    da.attrs['description'] = 'station {} soundings full profile'.format(st_num)
    units_dict = dict(zip(header, units))
    for k, v in units_dict.items():
        da.attrs[k] = v
    pw.attrs['description'] = 'station {} soundings of precipatable water'.format(st_num)
    pw.attrs['units'] = 'mm'  # eqv. kg/m^2
    pw.attrs['station_number'] = st_num
    pw.attrs['station_lat'] = st_lat
    pw.attrs['station_lon'] = st_lon
    pw.attrs['station_alt'] = st_alt
    pw = pw.sortby('time')
    da = da.sortby('time')
    # drop 0 pw - not physical
    pw = pw.where(pw > 0, drop=True)
    pw_file = 'PW_{}_soundings.nc'.format(st_num)
    pw.to_netcdf(json_path / pw_file, 'w')
    logger.info('{} was saved to {}'.format(pw_file, json_path))
    all_file = 'ALL_{}_soundings.nc'.format(st_num)
    da.to_netcdf(json_path / all_file, 'w')
    logger.info('{} was saved to {}'.format(all_file, json_path))
    return pw, da, bad_line


def get_sounding_data_from_uwyo(savepath, st_num='40179',
                                start_date='2003-01-01',
                                end_date='2004-12-31'):
    """Download sounding data from bet_dagan station at two times:00 and 12"""
    import requests
    from bs4 import BeautifulSoup as bs
    import pandas as pd
    import json
    import numpy as np
    import xarray as xr
    import logging
    logger = logging.getLogger('uwyo')
    logger.info('Downloading station {} from UWYO'.format(st_num))
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    logger.info('start date : {}'.format(start_date.strftime('%Y-%m-%d')))
    logger.info('end date : {}'.format(end_date.strftime('%Y-%m-%d')))
    dates = pd.date_range(start_date, end_date, freq='12h')
    years = np.arange(start_date.year, end_date.year + 1)
    time = xr.DataArray(dates, dims=['time'])
    for year in years:
        lines_list = []
        for date in time.sel(time=str(year)).values:
            date = pd.to_datetime(date)
            logger.info('downloading datetime: {}'.format(date.strftime('%Y-%m-%d:%H')))
            year = str(date.year)
            month = str(date.month)
            day = str(date.day)
            if date.hour == 0:
                hour = '0' + str(date.hour)
            elif date.hour == 12:
                hour = str(date.hour)
            url = ('http://weather.uwyo.edu/cgi-bin/sounding?region=mideast&'
                   'TYPE=TEXT%3ALIST&YEAR=' + year + '&MONTH=' + month +
                   '&FROM=' + day + hour + '&TO=0100&STNM=' + st_num)
            r = requests.get(url)
            soup = bs(r.text, "lxml")
            allLines = soup.text.split('\n')
    #        splice = allLines[53:78]
    #        keys = ['_'.join(x.split(':')[0].lower().split()) for x in splice]
    #        values = [x.split(':')[-1] for x in splice]
    #        print(keys)
    #        dict_list.append(dict(zip(keys, values)))
            lines_list.append(allLines)
        logger.info('Saving list of dicts to: {}'.format(savepath))
        filename = 'station_{}_soundings_{}.json'.format(st_num, year)
        with open(savepath / filename, 'w') as fout:
            json.dump(lines_list, fout)
        print('Done!')
    return


def process_data_from_uwyo_sounding(path, st_num):
    """create tm, tpw from sounding station and also add surface temp and
    station caluculated ipw"""
    import xarray as xr
    from aux_gps import dim_intersection
    import numpy as np
    import logging
    logger = logging.getLogger('uwyo')
    # da = xr.open_dataarray(sound_path / 'ALL_bet_dagan_soundings.nc')
    pw_file = 'PW_{}_soundings.nc'.format(st_num)
    all_file = 'ALL_{}_soundings.nc'.format(st_num)
    da = xr.open_dataarray(path / all_file)
    pw = xr.open_dataarray(path / pw_file)
    new_time = dim_intersection([da, pw], 'time', dropna=False)
    logger.info('loaded {}'.format(pw_file))
    logger.info('loaded {}'.format(all_file))
    da = da.sel(time=new_time)
    pw = pw.sel(time=new_time)
    pw.load()
    da.load()
    logger.info('calculating pw and tm for station {}'.format(st_num))
    ts_list = []
    tpw_list = []
    tm_list = []
#    cld_list = []
    for date in da.time:
        ts_list.append(da.sel(var='TEMP', mpoint=0, time=date) + 273.15)
        tpw_list.append(precipitable_water(da.sel(time=date)))
        tm_list.append(Tm(da.sel(time=date)))
#        if np.isnan(ds.CLD.sel(time=date)).all():
#            cld_list.append(0)
#        else:
#            cld_list.append(1)
    tpw = xr.DataArray(tpw_list, dims='time')
    tm = xr.DataArray(tm_list, dims='time')
    tm.attrs['description'] = 'mean atmospheric temperature calculated by water vapor pressure weights'
    tm.attrs['units'] = 'K'
    ts = xr.concat(ts_list, 'time')
    ts.attrs['description'] = 'Surface temperature from {} station soundings'.format(st_num)
    ts.attrs['units'] = 'K'
    result = pw.to_dataset(name='pw')
    result['tpw'] = tpw
    result['tm'] = tm
    result['ts'] = ts
    result['tpw'].attrs['description'] = 'station {} percipatable water calculated from sounding by me'.format(st_num)
    result['tpw'].attrs['units'] = 'mm'
    result['season'] = result['time.season']
    result['hour'] = result['time.hour'].astype(str)
    result['hour'] = result.hour.where(result.hour != '12', 'noon')
    result['hour'] = result.hour.where(result.hour != '0', 'midnight')
#    result['any_cld'] = xr.DataArray(cld_list, dims='time')
    result = result.dropna('time')
    filename = 'station_{}_sounding_pw_Ts_Tk.nc'.format(st_num)
    logger.info('saving {} to {}'.format(filename, path))
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in result}
    result.to_netcdf(path / filename, 'w', encoding=encoding)
    logger.info('Done!')
    return


if __name__ == '__main__':
    """--mode: either download or post
    --path: where to save the downloaded files or where to save the post
    proccesed files.
   """
    import argparse
    import sys
    from aux_gps import configure_logger
    import pandas as pd
    logger = configure_logger(name='uwyo')
    parser = argparse.ArgumentParser(
        description='a command line tool for downloading and proccesing upper air soundings from university of Wyoming website.')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--path',
        help="a full path to the save location of the files",
        type=check_path)
    required.add_argument(
        '--mode',
        help='mode of the tool',
        choices=['download', 'post'])
    optional.add_argument('--station', help='WMO station number, use with mode=download',
                          type=str)
    optional.add_argument('--start', help='starting date, use with mode=download',
                          type=check_date)
    optional.add_argument(
            '--end',
            help='end date, use with mode=download', type=check_date)

    parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()
    if args.path is None:
        print('path is a required argument, run with -h...')
        sys.exit()
    if args.mode is None:
        print('mode is a required argument, run with -h...')
        sys.exit()
    if args.start is None:
        args.start = pd.to_datetime('2003-01-01')
    if args.end is None:
        args.end = pd.to_datetime('2019-01-01')
    if args.station is None:
        args.station = '40179'  # bet-dagan station
    if args.mode == 'download':
        get_sounding_data_from_uwyo(args.path, args.station, args.start,
                                    args.end)
    elif args.mode == 'post':
        process_sounding_json(args.path, args.station)
        process_data_from_uwyo_sounding(args.path, args.station)
