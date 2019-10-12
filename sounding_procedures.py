#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:24:40 2019

@author: ziskin
"""
# from pathlib import Path
from PW_paths import work_yuval
sound_path = work_yuval / 'sounding'

# write a command line tool to download data from Wyoming upper air soundings


def evaluate_sin_to_tmsearies(da, time_dim='time', plot=True, amp=6.0,
                              period=365.0, phase=30.0, offset=253.0,
                              just_eval=False):
    import pandas as pd
    import xarray as xr
    import numpy as np
    from scipy.optimize import curve_fit

    def func(time, amp, period, phase, offset):
        f = amp * np.sin(2 * np.pi * (time / period + 1.0 / phase)) + offset
        return f
    time = da[time_dim].dropna(time_dim).values
    time = pd.to_datetime(time)
    jul = time.to_julian_date()
    jul -= round(jul[0])
    if not just_eval:
        ydata = da.values
        popt, pcov = curve_fit(func, jul, ydata, p0=[amp, period, phase,
                                                     offset])
        amp = popt[0]
        period = popt[1]
        phase = popt[2]
        offset = popt[3]
        perr = np.sqrt(np.diag(pcov))
        print('amp: {:.2f} +- {:.2f}'.format(amp, perr[0]))
        print('period: {:.2f} +- {:.2f}'.format(period, perr[1]))
        print('phase: {:.2f} +- {:.2f}'.format(phase, perr[2]))
        print('offset: {:.2f} +- {:.2f}'.format(offset, perr[3]))
    new = func(jul, amp, period, phase, offset)
    new_da = xr.DataArray(new, dims=[time_dim])
    new_da[time_dim] = time
    if plot:
        da.plot.line(marker='.', linewidth=0.)
        new_da.plot.line(marker='.', linewidth=0.)
    return new_da


def move_bet_dagan_physical_to_main_path(bet_dagan_path):
    """rename bet_dagan physical radiosonde filenames and move them to main
    path i.e., bet_dagan_path! warning-DO ONCE """
    from aux_gps import path_glob
    import shutil
    year_dirs = sorted([x for x in path_glob(bet_dagan_path, '*/') if x.is_dir()])
    for year_dir in year_dirs:
        month_dirs = sorted([x for x in path_glob(year_dir, '*/') if x.is_dir()])
        for month_dir in month_dirs:
            day_dirs = sorted([x for x in path_glob(month_dir, '*/') if x.is_dir()])
            for day_dir in day_dirs:
                hour_dirs = sorted([x for x in path_glob(day_dir, '*/') if x.is_dir()])
                for hour_dir in hour_dirs:
                    file = [x for x in path_glob(hour_dir, '*/') if x.is_file()]
                    splitted = file[0].as_posix().split('/')
                    name = splitted[-1]
                    hour = splitted[-2]
                    day = splitted[-3]
                    month = splitted[-4]
                    year = splitted[-5]
                    filename = '{}_{}{}{}{}'.format(name, year, month, day, hour)
                    orig = file[0]
                    dest = bet_dagan_path / filename
                    shutil.move(orig.resolve(), dest.resolve())
                    print('moving {} to {}'.format(filename, bet_dagan_path))
    return year_dirs


def read_all_physical_radiosonde(path, savepath=None, lower_cutoff=None,
                                 upper_cutoff=None, verbose=True, plot=True):
    from aux_gps import path_glob
    import xarray as xr
    ds_list = []
    ds_extra_list = []
#    sound_list = []
#    tpw_list = []
#    cloud_list = []
#    dt_range_list = []
#    tm_list = []
    if lower_cutoff is not None:
        print('applying lower cutoff at {} meters for PW and Tm calculations.'.format(int(lower_cutoff)))
    if upper_cutoff is not None:
        print('applying upper cutoff at {} meters for PW and Tm calculations.'.format(int(upper_cutoff)))
    for path_file in sorted(path_glob(path, '*/')):
        if path_file.is_file():
            ds = read_one_physical_radiosonde_report(path_file,
                                                     lower_cutoff=lower_cutoff,
                                                     upper_cutoff=upper_cutoff)
            if ds is None:
                print('{} is corrupted...'.format(path_file.as_posix().split('/')[-1]))
                continue
            date = ds['sound_time'].dt.strftime('%Y-%m-%d %H:%M').values
            if verbose:
                print('reading {} physical radiosonde report'.format(date))
            ds_with_time_dim = [x for x in ds.data_vars if 'time' in ds[x].dims]
            ds_list.append(ds[ds_with_time_dim])
            ds_extra = [x for x in ds.data_vars if 'time' not in ds[x].dims]
            ds_extra_list.append(ds[ds_extra])
#            sound_list.append(ds['sound_time'])
#            tpw_list.append(ds['tpw'])
#            cloud_list.append(ds['cloud_code'])
#            dt_range_list.append(ds['dt_range'])
#            tm_list.append(ds['tm'])
#            ds_list.append(ds.drop(['tpw', 'cloud_code', 'sound_time',
#                                    'dt_range', 'tm']))
    dss = xr.concat(ds_list, 'time')
    dss_extra = xr.concat(ds_extra_list, 'sound_time')
#    cloud = xr.concat(cloud_list, 'sound_time')
#    tm = xr.concat(tm_list, 'sound_time')
#    dt_range = xr.DataArray(dt_range_list, dims=['bnd', 'sound_time'])
    dss = dss.merge(dss_extra)
#    dss['tpw'] = tpw
#    dss['tm'] = tm
#    dss['cloud_code'] = cloud
#    dss['sound_time'] = sound_list
#    dss['dt_range'] = dt_range
    if lower_cutoff is not None:
        dss['Tpw'].attrs['lower_cutoff'] = lower_cutoff
        dss['Tm'].attrs['lower_cutoff'] = lower_cutoff
    if upper_cutoff is not None:
        dss['Tpw'].attrs['upper_cutoff'] = upper_cutoff
        dss['Tm'].attrs['upper_cutoff'] = upper_cutoff
    if savepath is not None:
        yr_min = dss.time.min().dt.year.item()
        yr_max = dss.time.max().dt.year.item()
        filename = 'bet_dagan_phys_sounding_{}-{}.nc'.format(yr_min, yr_max)
        print('saving {} to {}'.format(filename, savepath))
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in dss}
        dss.to_netcdf(savepath / filename, 'w', encoding=encoding)
    print('Done!')
    return dss


def read_one_physical_radiosonde_report(path_file, lower_cutoff=None,
                                        upper_cutoff=None, verbose=False):
    """read one(12 or 00) physical bet dagan radiosonde reports and return a df
    containing time series, PW for the whole sounding and time span of the
    sounding"""
    def df_to_ds(df, df_extra, meta):
        import xarray as xr
        if not df.between_time('22:00', '02:00').empty:
            sound_time = pd.to_datetime(
                df.index[0].strftime('%Y-%m-%d')).replace(hour=0, minute=0)
        elif not df.between_time('10:00', '14:00').empty:
            sound_time = pd.to_datetime(
                df.index[0].strftime('%Y-%m-%d')).replace(hour=12, minute=0)
        elif (df.between_time('22:00', '02:00').empty and
              df.between_time('10:00', '14:00').empty):
            raise ValueError(
                '{} time is not midnight nor noon'.format(
                    df.index[0]))
        ds = df.to_xarray()
        for name in ds.data_vars:
            ds[name].attrs['units'] = meta['units'][name]
        ds.attrs['station_number'] = meta['station']
        for col in df_extra.columns:
            ds[col] = df_extra[col]
            ds[col].attrs['units'] = meta['units_extra'][col]
            ds[col] = ds[col].squeeze(drop=True)
        ds['dt_range'] = xr.DataArray(
            [ds.time.min().values, ds.time.max().values], dims='bnd')
        ds['bnd'] = ['Min', 'Max']
        ds['sound_time'] = sound_time
        return ds

    def calculate_tpw(df, upper=None, lower=None):
        if upper is not None:
            df = df[df['H-Msl'] <= upper]
        if lower is not None:
            df = df[df['H-Msl'] >= lower]
        specific_humidity = (df['Mixratio'] / 1000.0) / \
            (1 + 0.001 * df['Mixratio'] / 1000.0)
        try:
            tpw = np.trapz(specific_humidity * df['Rho'].values,
                           df['H-Msl'])
        except ValueError:
            return np.nan
        return tpw

    def calculate_tm(df, upper=None, lower=None):
        if upper is not None:
            df = df[df['H-Msl'] <= upper]
        if lower is not None:
            df = df[df['H-Msl'] >= lower]
        try:
            numerator = np.trapz(
                df['Press'] /
                (df['Temp'] +
                 273.15),
                df['H-Msl'])
            denominator = np.trapz(
                df['Press'] / (df['Temp'] + 273.15)**2.0, df['H-Msl'])
            tm = numerator / denominator
        except ValueError:
            return np.nan
        return tm

    import pandas as pd
    import numpy as np
    # TODO: recheck units, add to df and to units_dict
    df = pd.read_csv(
         path_file,
         header=None,
         encoding="windows-1252",
         delim_whitespace=True,
         skip_blank_lines=True,
         na_values=['/////'])
    if not df[df.iloc[:, 0].str.contains('PILOT')].empty:
        return None
    # drop last two cols:
    df.drop(df.iloc[:, -2:len(df)], inplace=True, axis=1)
    # extract datetime:
    date = df[df.loc[:, 0].str.contains("PHYSICAL")].loc[0, 3]
    time = df[df.loc[:, 0].str.contains("PHYSICAL")].loc[0, 4]
    dt = pd.to_datetime(date + 'T' + time, dayfirst=True)
    # extract station number:
    station_num = int(df[df.loc[:, 0].str.contains("STATION")].iloc[0, 3])
    # extract cloud code:
    cloud_code = df[df.loc[:, 0].str.contains("CLOUD")].iloc[0, 3]
    # change col names to:
    df.columns = ['Time', 'Temp', 'RH', 'Press', 'H-Sur', 'H-Msl', 'EL',
                  'AZ', 'W.D', 'W.S']
    units = dict(zip(df.columns.to_list(),
                     ['sec', 'deg_C', '%', 'mb', 'm', 'm', 'deg', 'deg', 'deg',
                      'knots']))
    # iterate over the cols and change all to numeric(or timedelta) values:
    for col in df.columns:
        if col == 'Time':
            df[col] = pd.to_timedelta(df[col], errors='coerce')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # filter all entries that the first col is not null:
    df = df[~df.loc[:, 'Time'].isnull()]
    # finally, take care of the datetime index:
    df.Time = dt + df.Time
    df = df.set_index('Time')
    df.index.name = 'time'
    # calculate total precipitaple water(tpw):
    # add cols to df that help calculate tpw:
    df['Dewpt'] = dewpoint_rh(df['Temp'], df['RH'])
    df['WVpress'] = VaporPressure(df['Dewpt'], units='hPa', method='Buck')
    df['Mixratio'] = MixRatio(df['WVpress'], df['Press'])  # both in hPa
    # Calculate density of air (accounting for moisture)
    df['Rho'] = DensHumid(df['Temp'], df['Press'], df['WVpress'])
    # print('rho: {}, mix: {}, h: {}'.format(rho.shape,mixrkg.shape, hghtm.shape))
    # Trapezoidal rule to approximate TPW (units kg/m^2==mm)
    tpw = calculate_tpw(df, lower=lower_cutoff, upper=upper_cutoff)
    # calculate the mean atmospheric temperature and get surface temp in K:
    tm = calculate_tm(df, lower=lower_cutoff, upper=upper_cutoff)
    ts = df['Temp'][0] + 273.15
    units.update(Dewpt='deg_C', WVpress='hPa', Mixratio='gr/kg', Rho='kg/m^3')
    extra = np.array([ts, tm, tpw, cloud_code]).reshape(1, -1)
    df_extra = pd.DataFrame(data=extra, columns=['Ts', 'Tm', 'Tpw', 'Cloud_code'])
    for col in ['Ts', 'Tm', 'Tpw']:
        df_extra[col] = pd.to_numeric(df_extra[col])
    units_extra = {'Ts': 'K', 'Tm': 'K', 'Tpw': 'kg/m^2', 'Cloud_code': ''}
    meta = {'units': units, 'units_extra': units_extra, 'station': station_num}
    if verbose:
        print('datetime: {}, TPW: {:.2f} '.format(df.index[0], tpw))
    ds = df_to_ds(df, df_extra, meta)
    return ds


def classify_clouds_from_sounding(sound_path=sound_path):
    import xarray as xr
    import numpy as np
    da = xr.open_dataarray(sound_path / 'ALL_bet_dagan_soundings.nc')
    ds = da.to_dataset(dim='var')
    cld_list = []
    for date in ds.time:
        h = ds['HGHT'].sel(time=date).dropna('mpoint').values
        T = ds['TEMP'].sel(time=date).dropna('mpoint').values
        dwT = ds['DWPT'].sel(time=date).dropna('mpoint').values
        h = h[0: len(dwT)]
        cld = np.empty(da.mpoint.size, dtype='float')
        cld[:] = np.nan
        T = T[0: len(dwT)]
        try:
            dT = np.abs(T - dwT)
        except ValueError:
            print('valueerror')
            cld_list.append(cld)
            continue
        found = h[dT < 0.5]
        found_pos = np.where(dT < 0.5)[0]
        if found.any():
            for i in range(len(found)):
                if found[i] < 2000:
                    cld[found_pos[i]] = 1.0  # 'LOW' clouds
                elif found[i] < 7000 and found[i] > 2000:
                    cld[found_pos[i]] = 2.0  # 'MIDDLE' clouds
                elif found[i] < 13000 and found[i] > 7000:
                    cld[found_pos[i]] = 3.0  # 'HIGH' clouds
        cld_list.append(cld)
    ds['CLD'] = ds['HGHT'].copy(deep=False, data=cld_list)
    ds.to_netcdf(sound_path / 'ALL_bet_dagan_soundings_with_clouds.nc', 'w')
    print('ALL_bet_dagan_soundings_with_clouds.nc saved to {}'.format(sound_path))
    return ds


def compare_interpolated_to_real(Tint, date='2014-03-02T12:00',
                                 sound_path=sound_path):
    from metpy.plots import SkewT
    from metpy.units import units
    import pandas as pd
    import xarray as xr
    import matplotlib.pyplot as plt
    da = xr.open_dataarray(sound_path / 'ALL_bet_dagan_soundings.nc')
    p = da.sel(time=date, var='PRES').values * units.hPa
    dt = pd.to_datetime(da.sel(time=date).time.values)
    T = da.sel(time=dt, var='TEMP').values * units.degC
    Tm = Tint.sel(time=dt).values * units.degC
    pm = Tint['pressure'].values * units.hPa
    fig = plt.figure(figsize=(9, 9))
    title = da.attrs['description'] + ' ' + dt.strftime('%Y-%m-%d')
    skew = SkewT(fig)
    skew.plot(p, T, 'r', linewidth=2)
    skew.plot(pm, Tm, 'g', linewidth=2)
    skew.ax.set_title(title)
    skew.ax.legend(['Original', 'Interpolated'])
    return


def average_meteogram(sound_path=sound_path, savepath=None):
    import xarray as xr
    import numpy as np
    from scipy.interpolate import interp1d
    da = xr.open_dataarray(sound_path / 'ALL_bet_dagan_soundings.nc')
    # pressure = np.linspace(1010, 20, 80)
    height = da.sel(var='HGHT').isel(time=0).dropna('mpoint').values
    T_list = []
    for i in range(da.time.size):
        x = da.isel(time=i).sel(var='HGHT').dropna('mpoint').values
        y = da.isel(time=i).sel(var='TEMP').dropna('mpoint').values
        x = x[0: len(y)]
        f = interp1d(x, y, kind='linear', fill_value='extrapolate')
        T_list.append(f(height))
    T = xr.DataArray(T_list, dims=['time', 'height'])
    ds = T.to_dataset(name='temperature')
    ds['height'] = height
    ds['time'] = da.time
    ds['temperature'].attrs['units'] = 'degC'
    ds['temperature'] = ds['temperature'].where(ds['temperature'].isel(height=0)>-13,drop=True)
    ds['height'].attrs['units'] = 'm'
    ts_list = [ds['temperature'].isel(height=0).sel(time=x) + 273.15 for x in
               ds.time]
    tm_list = [Tm(ds['temperature'].sel(time=x), from_raw_sounding=False)
               for x in ds.time]
    tm = xr.DataArray(tm_list, dims='time')
    tm.attrs['description'] = 'mean atmospheric temperature calculated by water vapor pressure weights'
    tm.attrs['units'] = 'K'
    ts = xr.concat(ts_list, 'time')
    ts.attrs['description'] = 'Surface temperature from BET DAGAN soundings'
    ts.attrs['units'] = 'K'
    hours = [12, 0]
    hr_dict = {12: 'noon', 0: 'midnight'}
    seasons = ['DJF', 'SON', 'MAM', 'JJA']
    for season in seasons:
        for hour in hours:
            da = ds['temperature'].sel(time=ds['time.season'] == season).where(
                ds['time.hour'] == hour).dropna('time').mean('time')
            name = ['T', season, hr_dict[hour]]
            ds['_'.join(name)] = da
    ds['season'] = ds['time.season']
    ds['hour'] = ds['time.hour'].astype(str)
    ds['hour'] = ds.hour.where(ds.hour != '12', 'noon')
    ds['hour'] = ds.hour.where(ds.hour != '0', 'midnight')
    ds['ts'] = ts
    ds['tm'] = tm
    ds = ds.dropna('time')
    if savepath is not None:
        filename = 'bet_dagan_sounding_pw_Ts_Tk1.nc'
        print('saving {} to {}'.format(filename, savepath))
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds}
        ds.to_netcdf(savepath / filename, 'w', encoding=encoding)
    print('Done!')
    return ds


def plot_skew(sound_path=sound_path, date='2018-01-16T12:00', two=False):
    from metpy.plots import SkewT
    from metpy.units import units
    import matplotlib.pyplot as plt
    import pandas as pd
    import xarray as xr
    da = xr.open_dataarray(sound_path / 'ALL_bet_dagan_soundings.nc')
    p = da.sel(time=date, var='PRES').values * units.hPa
    dt = pd.to_datetime(da.sel(time=date).time.values)
    if not two:
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
    elif two:
        dt1 = pd.to_datetime(dt.strftime('%Y-%m-%dT00:00'))
        dt2 = pd.to_datetime(dt.strftime('%Y-%m-%dT12:00'))
        T1 = da.sel(time=dt1, var='TEMP').values * units.degC
        T2 = da.sel(time=dt2, var='TEMP').values * units.degC
        fig = plt.figure(figsize=(9, 9))
        title = da.attrs['description'] + ' ' + dt.strftime('%Y-%m-%d')
        skew = SkewT(fig)
        skew.plot(p, T1, 'r', linewidth=2)
        skew.plot(p, T2, 'b', linewidth=2)
        # skew.ax.plot(p, Vp, 'k', linewidth=2)
        skew.ax.set_title(title)
        skew.ax.legend(['Temp at ' + dt1.strftime('%H:%M'),
                        'Temp at ' + dt2.strftime('%H:%M')])
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
        print('Saving list of dicts to: {}'.format(savepath))
        filename = 'station_{}_soundings_{}.json'.format(st_num, year)
        with open(savepath / filename, 'w') as fout:
            json.dump(lines_list, fout)
        print('Done!')
    return


def process_data_from_sounding(sound_path=sound_path, savepath=None):
    """create tm, tpw from sounding station and also add surface temp and
    station caluculated ipw"""
    import xarray as xr
    from aux_gps import dim_intersection
    import numpy as np
    # da = xr.open_dataarray(sound_path / 'ALL_bet_dagan_soundings.nc')
    ds = xr.open_dataset(sound_path / 'ALL_bet_dagan_soundings_with_clouds.nc')
    pw = xr.open_dataarray(sound_path / 'PW_bet_dagan_soundings.nc')
    new_time = dim_intersection([ds, pw], 'time', dropna=False)
    ds = ds.sel(time=new_time)
    pw = pw.sel(time=new_time)
    pw.load()
    ds.load()
    da = ds.to_array(dim='var')
    print('calculating...')
    ts_list = []
    tpw_list = []
    tm_list = []
    cld_list = []
    for date in ds.time:
        ts_list.append(ds['TEMP'].sel(mpoint=0, time=date) + 273.15)
        tpw_list.append(precipitable_water(da.sel(time=date)))
        tm_list.append(Tm(da.sel(time=date)))
        if np.isnan(ds.CLD.sel(time=date)).all():
            cld_list.append(0)
        else:
            cld_list.append(1)
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

    result['any_cld'] = xr.DataArray(cld_list, dims='time')
    result = result.dropna('time')
    if savepath is not None:
        filename = 'bet_dagan_sounding_pw_Ts_Tk_with_clouds.nc'
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


def MixRatio(e, p):
    """Mixing ratio of water vapour
    INPUTS
    e (Pa) Water vapor pressure
    p (Pa) Ambient pressure
    RETURNS
    qv (g kg^-1) Water vapor mixing ratio`
    """
    MW_dry_air = 28.9647  # gr/mol
    MW_water = 18.015  # gr/mol
    Epsilon = MW_water / MW_dry_air  # Epsilon=Rs_da/Rs_v;
    # The ratio of the gas constants
    return 1000 * Epsilon * e / (p - e)


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


def dewpoint(e):
    """Calculate the ambient dewpoint given the vapor pressure.

    Parameters
    ----------
    e : Water vapor partial pressure in mb (hPa)
    Returns
    -------
    Dew point temperature in deg_C
    See Also
    --------
    dewpoint_rh, saturation_vapor_pressure, vapor_pressure
    Notes
    -----
    This function inverts the [Bolton1980]_ formula for saturation vapor
    pressure to instead calculate the temperature. This yield the following
    formula for dewpoint in degrees Celsius:

    .. math:: T = \frac{243.5 log(e / 6.112)}{17.67 - log(e / 6.112)}

    """
    import numpy as np
    # units are degC
    sat_pressure_0c = 6.112  # in millibar
    val = np.log(e / sat_pressure_0c)
    return 243.5 * val / (17.67 - val)


def dewpoint_rh(temperature, rh):
    """Calculate the ambient dewpoint given air temperature and relative
    humidity.
    Parameters
    ----------
    temperature : in deg_C
        Air temperature
    rh : in %
        Relative Humidity
    Returns
    -------
       The dew point temperature
    See Also
    --------
    dewpoint, saturation_vapor_pressure
    """
    import numpy as np
    import warnings
    if np.any(rh > 120):
        warnings.warn('Relative humidity >120%, ensure proper units.')
    return dewpoint(rh / 100.0 * VaporPressure(temperature, units='hPa',
                    method='Buck'))

#def ea(T, rh):
#    """water vapor pressure function using ARM function for sat."""
#    # rh is relative humidity in %
#    return es(T) * rh / 100.0
