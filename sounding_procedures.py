#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:24:40 2019

@author: ziskin
"""
# from pathlib import Path
sound_path = work_yuval / 'sounding'


def plot_skew(sound_path=sound_path, date='2018-01-16T12:00'):
    from metpy.plots import SkewT
    from metpy.units import units
    import matplotlib.pyplot as plt
    import pandas as pd
    import xarray as xr
    da = xr.open_dataarray(sound_path / 'ALL_bet_dagan_soundings.nc')
    p = da.sel(time=date, var='PRES').values * units.hPa
    T = da.sel(time=date, var='TEMP').values * units.degC
    Td = da.sel(time=date, var='DWPT').values * units.degC
    Vp = VaporPressure(da.sel(time=date, var='TEMP').values) * units.Pa
    dt = pd.to_datetime(da.sel(time=date).time.values)
    fig = plt.figure(figsize=(9, 9))
    title = da.attrs['description'] + ' ' + dt.strftime('%Y-%m-%d %H:%M')
    skew = SkewT(fig)
    skew.plot(p, T, 'r', linewidth=2)
    skew.plot(p, Td, 'g', linewidth=2)
    # skew.ax.plot(p, Vp, 'k', linewidth=2)
    skew.ax.set_title(title)
    skew.ax.legend(['Temp', 'Dewpoint'])
    return


def process_sounding_json(savepath=sound_path):
    """process json files from sounding download and parse them to xarray"""
    import pandas as pd
    import json
    import xarray as xr
    import os
    # loop over lines lists in each year:
    pw_years = []
    df_years = []
    bad_line = []
    for file in sorted(savepath.glob('bet_dagan*.json')):
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
                with open(savepath/'temp.txt', 'w') as f:
                    for item in lines[header_line + 3: end_line]:
                        f.write("%s\n" % item)
                df = pd.read_fwf(savepath / 'temp.txt', names=header)
                try:
                    os.remove(savepath / 'temp.txt')
                except OSError as e:  # if failed, report it back to the user
                    print("Error: %s - %s." % (e.filename, e.strerror))
#                df = pd.DataFrame(
#                    [x.split() for x in lines[header_line + 3:end_line]],
#                    columns=header)
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
        df_year = [xr.DataArray(x, dims=['mpoint', 'var']) for x in df_list]
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
    pw.to_netcdf(savepath / 'PW_bet_dagan_soundings.nc', 'w')
    da.to_netcdf(savepath / 'ALL_bet_dagan_soundings.nc', 'w')
    return pw, da, bad_line


def get_sounding_data(savepath, start_date='2003-01-01',
                      end_date='2004-12-31'):
    """Download sounding data from bet_dagan station at two times:00 and 12"""
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


def process_data_from_sounding(sound_path=sound_path, savepath=None):
    """create tm, tpw from sounding station and also add surface temp and
    station caluculated ipw"""
    import xarray as xr
    from aux_gps import dim_intersection
    da = xr.open_dataarray(sound_path / 'ALL_bet_dagan_soundings.nc')
    pw = xr.open_dataarray(sound_path / 'PW_bet_dagan_soundings.nc')
    new_time = dim_intersection([da, pw], 'time', dropna=False)
    da = da.sel(time=new_time)
    pw = pw.sel(time=new_time)
    pw.load()
    da.load()
    print('calculating...')
    ts_list = [da.sel(mpoint=0, var='TEMP', time=x) + 273.15 for x in da.time]
    tpw_list = [precipitable_water(da.sel(time=x)) for x in da.time]
    tm_list = [Tm(da.sel(time=x)) for x in da.time]
    tpw = xr.DataArray(tpw_list, dims='time')
    tm = xr.DataArray(tm_list, dims='time')
    tm.attrs['description'] = 'mean atmospheric temperature calculated by water vapor pressure weights'
    tm.attrs['units'] = 'K'
    ts = xr.concat(ts_list, 'time')
    ts.attrs['description'] = 'Surface temperature from BET DAGAN soundings'
    ts.attrs['units'] = 'K'
    result = pw.to_dataset(name='pw')
    result['tpw'] = tpw
    result['tm'] = tm
    result['ts'] = ts
    result['tpw'].attrs['description'] = 'BET_DAGAN percipatable water calculated from sounding by me'
    result['tpw'].attrs['units'] = 'mm'
    result['season'] = result['time.season']
    result['hour'] = result['time.hour'].astype(str)
    result['hour'] = result.hour.where(result.hour != '12', 'noon')
    result['hour'] = result.hour.where(result.hour != '0', 'midnight')
    result = result.dropna('time')
    if savepath is not None:
        filename = 'bet_dagan_sounding_pw_Ts_Tk.nc'
        print('saving {} to {}'.format(filename, savepath))
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in result}
        result.to_netcdf(savepath / filename, 'w', encoding=encoding)
    print('Done!')
    return result


#def es(T):
#    """ARM function for water vapor saturation pressure"""
#    # T in celsius:
#    import numpy as np
#    es = 6.1094 * np.exp(17.625 * T / (T + 243.04))
#    # es in hPa
#    return es


def Tm(sounding):
    """ calculate the atmospheric mean temperature with pp as water
    vapor partial pressure and T deg in C. eq is Tm=int(pp/T)dz/int(pp/T^2)dz
    h is the heights vactor"""
    import numpy as np
    sounding = sounding.dropna('mpoint')
    tempc = sounding.sel(var='TEMP')
    h = sounding.sel(var='HGHT')
    vp = VaporPressure(tempc, units='hPa')
    tempk = tempc + 273.15
    Tm = np.trapz(vp / tempk, h) / np.trapz(vp / tempk**2, h)
    return Tm


def precipitable_water(sounding):
    """Calculate Total Precipitable Water (TPW) for sounding.
    TPW is defined as the total column-integrated water vapour. I
    calculate it from the dew point temperature because this is the
    fundamental moisture variable in this module (even though it is RH
    that is usually measured directly)
    """
    import numpy as np
    sounding = sounding.dropna('mpoint')
    tempk = sounding.sel(var='TEMP') + 273.15
    prespa = sounding.sel(var='PRES') * 100
    mixrkg = sounding.sel(var='MIXR') / 1000.0
    dwptc = sounding.sel(var='DWPT')
    hghtm = sounding.sel(var='HGHT')
    # Get Water Vapour Mixing Ratio, by calculation
    # from dew point temperature
    vprespa = VaporPressure(dwptc)
    # mixrkg = MixRatio(vprespa, prespa)

    # Calculate density of air (accounting for moisture)
    rho = DensHumid(tempk, prespa, vprespa)

    # Trapezoidal rule to approximate TPW (units kg/m^2==mm)

    tpw = np.trapz(mixrkg * rho, hghtm)
    return tpw


def MixRatio(e, p):
    """Mixing ratio of water vapour
    INPUTS
    e (Pa) Water vapor pressure
    p (Pa) Ambient pressure
    RETURNS
    qv (kg kg^-1) Water vapor mixing ratio`
    """
    Epsilon = 0.622  # Epsilon=Rs_da/Rs_v; The ratio of the gas constants
    return Epsilon * e / (p - e)


def DensHumid(tempk, pres, e):
    """Density of moist air.
    This is a bit more explicit and less confusing than the method below.
    INPUTS:
    tempk: Temperature (K)
    pres: static pressure (Pa)
    mixr: mixing ratio (kg/kg)
    OUTPUTS:
    rho_air (kg/m^3)
    SOURCE: http://en.wikipedia.org/wiki/Density_of_air
    """
    Rs_v = 461.51  # Specific gas const for water vapour, J kg^{-1} K^{-1}
    Rs_da = 287.05  # Specific gas const for dry air, J kg^{-1} K^{-1}
    pres_da = pres - e
    rho_da = pres_da / (Rs_da * tempk)
    rho_wv = e/(Rs_v * tempk)

    return rho_da + rho_wv


def VaporPressure(tempc, phase="liquid", units='hPa'):
    import numpy as np
    """Water vapor pressure over liquid water or ice.
    INPUTS:
    tempc: (C) OR dwpt (C), if SATURATION vapour pressure is desired.
    phase: ['liquid'],'ice'. If 'liquid', do simple dew point. If 'ice',
    return saturation vapour pressure as follows:
    Tc>=0: es = es_liquid
    Tc <0: es = es_ice
    RETURNS: e_sat  (Pa)
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
    over_liquid = 6.112 * np.exp(17.67 * tempc / (tempc + 243.12)) * unit
    over_ice = 6.112 * np.exp(22.46 * tempc / (tempc + 272.62)) * unit
    # return where(tempc<0,over_ice,over_liquid)

    if phase == "liquid":
        # return 6.112*exp(17.67*tempc/(tempc+243.12))*100.
        return over_liquid
    elif phase == "ice":
        # return 6.112*exp(22.46*tempc/(tempc+272.62))*100.
        return np.where(tempc < 0, over_ice, over_liquid)
    else:
        raise NotImplementedError


#def ea(T, rh):
#    """water vapor pressure function using ARM function for sat."""
#    # rh is relative humidity in %
#    return es(T) * rh / 100.0
