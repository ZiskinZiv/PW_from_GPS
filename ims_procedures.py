#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:22:51 2019
Work flow: to obtain the TD products for use with ZWD (after download):
    1)use fill_fix_all_10mins_IMS_stations() after copying the downloaded TD
    2)use IMS_interpolating_to_GNSS_stations_israel(dt=None, start_year=2019(latest))
    3)use resample_GNSS_TD(path=ims_path) to resample all TD
@author: ziskin
"""
from PW_paths import work_yuval
from pathlib import Path
ims_path = work_yuval / 'IMS_T'
gis_path = work_yuval / 'gis'
ims_10mins_path = ims_path / '10mins'
awd_path = work_yuval/'AW3D30'
cwd = Path().cwd()
# fill missing data:
#some_missing = ds.tmin.sel(time=ds['time.day'] > 15).reindex_like(ds)
#
#In [20]: filled = some_missing.groupby('time.month').fillna(climatology.tmin)
#
#In [21]: both = xr.Dataset({'some_missing': some_missing, 'filled': filled})

# kabr, nzrt, katz, elro, klhv, yrcm, slom have ims stations not close to them!
gnss_ims_dict = {
    'alon': 'ASHQELON-PORT', 'bshm': 'HAIFA-TECHNION', 'csar': 'HADERA-PORT',
    'tela': 'TEL-AVIV-COAST', 'slom': 'BESOR-FARM', 'kabr': 'SHAVE-ZIYYON',
    'nzrt': 'DEIR-HANNA', 'katz': 'GAMLA', 'elro': 'MEROM-GOLAN-PICMAN',
    'mrav': 'MAALE-GILBOA', 'yosh': 'ARIEL', 'jslm': 'JERUSALEM-GIVAT-RAM',
    'drag': 'METZOKE-DRAGOT', 'dsea': 'SEDOM', 'ramo': 'MIZPE-RAMON-20120927',
    'nrif': 'NEOT-SMADAR', 'elat': 'ELAT', 'klhv': 'SHANI',
    'yrcm': 'ZOMET-HANEGEV', 'spir': 'PARAN-20060124', 'nizn':'EZUZ'}

ims_units_dict = {
    'BP': 'hPa',
    'NIP': 'W/m^2',
    'Rain': 'mm',
    'TD': 'deg_C',
    'WD': 'deg',
    'WS': 'm/s',
    'U' : 'm/s',
    'V' : 'm/s',
    'G': ''}


def save_daily_IMS_params_at_GNSS_loc(ims_path=ims_path,
                                      param_name='WS', stations=[x for x in gnss_ims_dict.keys()]):
    import xarray as xr
    from aux_gps import save_ncfile
    param = xr.open_dataset(ims_path / 'IMS_{}_israeli_10mins.nc'.format(param_name))
    ims_stns = [gnss_ims_dict.get(x) for x in stations]
    param = param[ims_stns]
    param = param.resample(time='D', keep_attrs=True).mean(keep_attrs=True)
    inv_dict = {v: k for k, v in gnss_ims_dict.items()}
    for da in param:
        param = param.rename({da: inv_dict.get(da)})
    filename = 'GNSS_{}_daily.nc'.format(param_name)
    save_ncfile(param, ims_path, filename)
    return param


def produce_bet_dagan_long_term_pressure(path=ims_path, rate='1H',
                                         savepath=None, fill_from_jerusalem=True):
    import xarray as xr
    from aux_gps import xr_reindex_with_date_range
    from aux_gps import get_unique_index
    from aux_gps import save_ncfile
    from aux_gps import anomalize_xr
    # load manual old measurements and new 3 hr ones:
    bd_man = xr.open_dataset(
        path / 'IMS_hourly_03hr.nc')['BET-DAGAN-MAN_2520_ps']
    bd_auto = xr.open_dataset(path / 'IMS_hourly_03hr.nc')['BET-DAGAN_2523_ps']
    bd = xr.concat(
        [bd_man.dropna('time'), bd_auto.dropna('time')], 'time', join='inner')
    bd = get_unique_index(bd)
    bd = bd.sortby('time')
    bd = xr_reindex_with_date_range(bd, freq='1H')
    # remove dayofyear mean, interpolate and reconstruct signal to fill it with climatology:
    climatology = bd.groupby('time.dayofyear').mean(keep_attrs=True)
    bd_anoms = anomalize_xr(bd, freq='DOY')
    bd_inter = bd_anoms.interpolate_na('time', method='cubic', max_gap='24H', keep_attrs=True)
    # bd_inter = bd.interpolate_na('time', max_gap='3H', method='cubic')
    bd_inter = bd_inter.groupby('time.dayofyear') + climatology
    bd_inter = bd_inter.reset_coords(drop=True)
    # load 10-mins new measurements:
    bd_10 = xr.open_dataset(path / 'IMS_BP_israeli_hourly.nc')['BET-DAGAN']
    bd_10 = bd_10.dropna('time').sel(
        time=slice(
            '2019-06-30T00:00:00',
            None)).resample(
                time='1H').mean()
    bd_inter = xr.concat([bd_inter, bd_10], 'time', join='inner')
    bd_inter = get_unique_index(bd_inter)
    bd_inter = bd_inter.sortby('time')
    bd_inter.name = 'bet-dagan'
    bd_inter.attrs['action'] = 'interpolated from 3H'
    if fill_from_jerusalem:
        print('filling missing gaps from 2018 with jerusalem')
        jr_10 = xr.load_dataset(
            path / 'IMS_BP_israeli_hourly.nc')['JERUSALEM-CENTRE']
        climatology = bd_inter.groupby('time.dayofyear').mean(keep_attrs=True)
        jr_10_anoms = anomalize_xr(jr_10, 'DOY')
        bd_anoms = anomalize_xr(bd_inter, 'DOY')
        bd_anoms = xr.concat(
            [bd_anoms.dropna('time'), jr_10_anoms.dropna('time')], 'time', join='inner')
        bd_anoms = get_unique_index(bd_anoms)
        bd_anoms = bd_anoms.sortby('time')
        bd_anoms = xr_reindex_with_date_range(bd_anoms, freq='5T')
        bd_anoms = bd_anoms.interpolate_na('time', method='cubic', max_gap='2H')
        bd_anoms.name = 'bet-dagan'
        bd_anoms.attrs['action'] = 'interpolated from 3H'
        bd_anoms.attrs['filled'] = 'using Jerusalem-centre'
        bd_anoms.attrs['long_name'] = 'Pressure Anomalies'
        bd_anoms.attrs['units'] = 'hPa'
        bd_inter = bd_anoms.groupby('time.dayofyear') + climatology
        bd_inter = bd_inter.resample(time='1H', keep_attrs=True).mean(keep_attrs=True)
        # if savepath is not None:
        #     yr_min = bd_anoms.time.min().dt.year.item()
        #     yr_max = bd_anoms.time.max().dt.year.item()
        #     filename = 'IMS_BD_anoms_5min_ps_{}-{}.nc'.format(
        #         yr_min, yr_max)
        #     save_ncfile(bd_anoms, savepath, filename)
        # return bd_anoms
    if savepath is not None:
        # filename = 'IMS_BD_hourly_ps.nc'
        yr_min = bd_inter.time.min().dt.year.item()
        yr_max = bd_inter.time.max().dt.year.item()
        filename = 'IMS_BD_hourly_ps_{}-{}.nc'.format(yr_min, yr_max)
        save_ncfile(bd_inter, savepath, filename)
        bd_anoms = anomalize_xr(bd_inter, 'DOY', units='std')
        filename = 'IMS_BD_hourly_anoms_std_ps_{}-{}.nc'.format(yr_min, yr_max)
        save_ncfile(bd_anoms, savepath, filename)
        bd_anoms = anomalize_xr(bd_inter, 'DOY')
        filename = 'IMS_BD_hourly_anoms_ps_{}-{}.nc'.format(yr_min, yr_max)
        save_ncfile(bd_anoms, savepath, filename)
    return bd_inter


def transform_wind_speed_direction_to_u_v(path=ims_path, savepath=ims_path):
    import xarray as xr
    import numpy as np
    WS = xr.load_dataset(path / 'IMS_WS_israeli_10mins.nc')
    WD = xr.load_dataset(path / 'IMS_WD_israeli_10mins.nc')
    # change angles to math:
    WD = 270 - WD
    U = WS * np.cos(np.deg2rad(WD))
    V = WS * np.sin(np.deg2rad(WD))
    print('updating attrs...')
    for station in WS:
        attrs = WS[station].attrs
        attrs.update(channel_name='U')
        attrs.update(units='m/s')
        attrs.update(field_name='zonal velocity')
        U[station].attrs = attrs
        attrs.update(channel_name='V')
        attrs.update(field_name='meridional velocity')
        V[station].attrs = attrs
    if savepath is not None:
        filename = 'IMS_U_israeli_10mins.nc'
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in U.data_vars}
        U.to_netcdf(savepath / filename, 'w', encoding=encoding)
        filename = 'IMS_V_israeli_10mins.nc'
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in V.data_vars}
        V.to_netcdf(savepath / filename, 'w', encoding=encoding)
        print('Done!')
    return


def perform_harmonic_analysis_all_IMS(path=ims_path, var='BP', n=4,
                                      savepath=ims_path):
    import xarray as xr
    from aux_gps import harmonic_analysis_xr
    from aux_gps import keep_iqr
    ims = xr.load_dataset(path / 'IMS_{}_israeli_10mins.nc'.format(var))
    sites = [x for x in gnss_ims_dict.values()]
    ims_actual_sites = [x for x in ims if x in sites]
    ims = ims[ims_actual_sites]
    if var == 'NIP':
        ims = xr.merge([keep_iqr(ims[x]) for x in ims])
        max_nip = ims.to_array('site').max()
        ims /= max_nip
    dss_list = []
    for site in ims:
        da = ims[site]
        da = keep_iqr(da)
        print('performing harmonic analysis for IMS {} field at {} site:'.format(var, site))
        dss = harmonic_analysis_xr(da, n=n, anomalize=True, normalize=False)
        dss_list.append(dss)
    dss_all = xr.merge(dss_list)
    dss_all.attrs['field'] = var
    dss_all.attrs['units'] = ims_units_dict[var]
    if savepath is not None:
        filename = 'IMS_{}_harmonics_diurnal.nc'.format(var)
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in dss_all.data_vars}
        dss_all.to_netcdf(savepath / filename, 'w', encoding=encoding)
        print('Done!')
    return dss_all


def align_10mins_ims_to_gnss_and_save(ims_path=ims_path, field='G7',
                                      gnss_ims_dict=gnss_ims_dict,
                                      savepath=work_yuval):
    import xarray as xr
    d = dict(zip(gnss_ims_dict.values(), gnss_ims_dict.keys()))
    gnss_list = []
    for station, gnss_site in d.items():
        print('loading IMS station {}'.format(station))
        ims_field = xr.load_dataset(ims_path / 'IMS_{}_israeli_10mins.nc'.format(field))[station]
        gnss = ims_field.load()
        gnss.name = gnss_site
        gnss.attrs['IMS_station'] = station
        gnss_list.append(gnss)
    gnss_sites = xr.merge(gnss_list)
    if savepath is not None:
        filename = 'GNSS_IMS_{}_israeli_10mins.nc'.format(field)
        print('saving {} to {}'.format(filename, savepath))
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in gnss_sites.data_vars}
        gnss_sites.to_netcdf(savepath / filename, 'w', encoding=encoding)
        print('Done!')
    return gnss_sites


def produce_10mins_gustiness(path=ims_path, rolling=5):
    import xarray as xr
    from aux_gps import keep_iqr
    from aux_gps import xr_reindex_with_date_range
    ws = xr.load_dataset(path / 'IMS_WS_israeli_10mins.nc')
    stations = [x for x in ws.data_vars]
    g_list = []
    for station in stations:
        print('proccesing station {}'.format(station))
        attrs = ws[station].attrs
        g = ws[station].rolling(time=rolling, center=True).std() / ws[station].rolling(time=rolling, center=True).mean()
        g = keep_iqr(g)
        g = xr_reindex_with_date_range(g, freq='10min')
        g.name = station
        g.attrs = attrs
        g_list.append(g)
    G = xr.merge(g_list)
    filename = 'IMS_G{}_israeli_10mins.nc'.format(rolling)
    print('saving {} to {}'.format(filename, path))
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in G.data_vars}
    G.to_netcdf(path / filename, 'w', encoding=encoding)
    print('Done resampling!')
    return G


def produce_10mins_absolute_humidity(path=ims_path):
    from sounding_procedures import wrap_xr_metpy_mixing_ratio
    from aux_gps import dim_intersection
    import xarray as xr
    P = xr.load_dataset(path / 'IMS_BP_israeli_10mins.nc')
    stations = [x for x in P.data_vars]
    T = xr.open_dataset(path / 'IMS_TD_israeli_10mins.nc')
    T = T[stations].load()
    RH = xr.open_dataset(path / 'IMS_RH_israeli_10mins.nc')
    RH = RH[stations].load()
    mr_list = []
    for station in stations:
        print('proccesing station {}'.format(station))
        p = P[station]
        t = T[station]
        rh = RH[station]
        new_time = dim_intersection([p, t, rh])
        p = p.sel(time=new_time)
        rh = rh.sel(time=new_time)
        t = t.sel(time=new_time)
        mr = wrap_xr_metpy_mixing_ratio(p, t, rh, verbose=True)
        mr_list.append(mr)
    MR = xr.merge(mr_list)
    filename = 'IMS_MR_israeli_10mins.nc'
    print('saving {} to {}'.format(filename, path))
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in MR.data_vars}
    MR.to_netcdf(path / filename, 'w', encoding=encoding)
    print('Done resampling!')
    return MR


def produce_wind_frequency_gustiness(path=ims_path,
                                     station='TEL-AVIV-COAST',
                                     season='DJF', plot=True):
    import xarray as xr
    import matplotlib.pyplot as plt
    import numpy as np
    from aux_gps import keep_iqr
    ws = xr.open_dataset(path / 'IMS_WS_israeli_10mins.nc')[station]
    ws.load()
    ws = ws.sel(time=ws['time.season'] == season)
    gustiness = ws.rolling(time=5).std() / ws.rolling(time=5).mean()
    gustiness = keep_iqr(gustiness)
    gustiness_anoms = gustiness.groupby('time.month') - gustiness.groupby('time.month').mean('time')
    gustiness_anoms = gustiness_anoms.reset_coords(drop=True)
    G = gustiness_anoms.groupby('time.hour').mean('time')
    wd = xr.open_dataset(path / 'IMS_WD_israeli_10mins.nc')[station]
    wd.load()
    wd.name = 'WD'
    wd = wd.sel(time=wd['time.season'] == season)
    all_Q = wd.groupby('time.hour').count()
    Q1 = wd.where((wd >= 0) & (wd < 90)).dropna('time')
    Q2 = wd.where((wd >= 90) & (wd < 180)).dropna('time')
    Q3 = wd.where((wd >= 180.1) & (wd < 270)).dropna('time')
    Q4 = wd.where((wd >= 270) & (wd < 360)).dropna('time')
    Q = xr.concat([Q1, Q2, Q3, Q4], 'Q')
    Q['Q'] = [x + 1 for x in range(4)]
    Q_freq = 100.0 * (Q.groupby('time.hour').count() / all_Q)
    if plot:
        fig, ax = plt.subplots(figsize=(16, 8))
        for q in Q_freq['Q']:
            Q_freq.sel(Q=q).plot(ax=ax)
        ax.set_title(
            'Relative wind direction frequency in {} IMS station in {} season'.format(
                station, season))
        ax.set_ylabel('Relative frequency [%]')
        ax.set_xlabel('Time of day [UTC]')
        ax.set_xticks(np.arange(0, 24, step=1))
        ax.legend([r'0$\degree$-90$\degree$', r'90$\degree$-180$\degree$',
                   r'180$\degree$-270$\degree$', r'270$\degree$-360$\degree$'], loc='upper left')
        ax.grid()
        ax2 = ax.twinx()
        G.plot.line(ax=ax2, color='k', marker='o')
        ax2.axhline(0, color='k', linestyle='--')
        ax2.legend(['{} Gustiness anomalies'.format(station)], loc='upper right')
        ax2.set_ylabel('Gustiness anomalies')
    return


def produce_gustiness(path=ims_path,
                      station='TEL-AVIV-COAST',
                      season='DJF', pw_station='tela', temp=False,
                      ax=None):
    import xarray as xr
    import matplotlib.pyplot as plt
    import numpy as np
    from aux_gps import keep_iqr
    from aux_gps import groupby_date_xr
    from matplotlib.ticker import FixedLocator

    def align_yaxis(ax1, v1, ax2, v2):
        """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
        _, y1 = ax1.transData.transform((0, v1))
        _, y2 = ax2.transData.transform((0, v2))
        adjust_yaxis(ax2,(y1-y2)/2,v2)
        adjust_yaxis(ax1,(y2-y1)/2,v1)

    def adjust_yaxis(ax,ydif,v):
        """shift axis ax by ydiff, maintaining point v at the same location"""
        inv = ax.transData.inverted()
        _, dy = inv.transform((0, 0)) - inv.transform((0, ydif))
        miny, maxy = ax.get_ylim()
        miny, maxy = miny - v, maxy - v
        if -miny>maxy or (-miny==maxy and dy > 0):
            nminy = miny
            nmaxy = miny*(maxy+dy)/(miny+dy)
        else:
            nmaxy = maxy
            nminy = maxy*(miny+dy)/(maxy+dy)
        ax.set_ylim(nminy+v, nmaxy+v)

    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    print('loading {} IMS station...'.format(station))
    g = xr.open_dataset(path / 'IMS_G_israeli_10mins.nc')[station]
    g.load()
    g = g.sel(time=g['time.season'] == season)
    date = groupby_date_xr(g)
    # g_anoms = g.groupby('time.month') - g.groupby('time.month').mean('time')
    g_anoms = g.groupby(date) - g.groupby(date).mean('time')
    g_anoms = g_anoms.reset_coords(drop=True)
    G = g_anoms.groupby('time.hour').mean('time')
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 8))
    G.plot(ax=ax, color='b', marker='o')
    ax.set_title(
        'Gustiness {} IMS station in {} season'.format(
            station, season))
    ax.axhline(0, color='b', linestyle='--')
    ax.set_ylabel('Gustiness anomalies [dimensionless]', color='b')
    ax.set_xlabel('Time of day [UTC]')
    ax.set_xticks(np.arange(0, 24, step=1))
    ax.yaxis.label.set_color('b')
    ax.tick_params(axis='y', colors='b')
    ax.grid()
    if pw_station is not None:
        pw = xr.open_dataset(
            work_yuval /
            'GNSS_PW_thresh_50_homogenized.nc')[pw_station]
        pw.load().dropna('time')
        pw = pw.sel(time=pw['time.season'] == season)
        date = groupby_date_xr(pw)
        pw = pw.groupby(date) - pw.groupby(date).mean('time')
        pw = pw.reset_coords(drop=True)
        pw = pw.groupby('time.hour').mean()
        axpw = ax.twinx()
        pw.plot.line(ax=axpw, color='k', marker='o')
        axpw.axhline(0, color='k', linestyle='--')
        axpw.legend(['{} PW anomalies'.format(pw_station.upper())], loc='upper right')
        axpw.set_ylabel('PW anomalies [mm]')
        align_yaxis(ax, 0, axpw, 0)
        if temp:
            axt = ax.twinx()
            axt.spines["right"].set_position(("axes", 1.05))
            # Having been created by twinx, par2 has its frame off, so the line of its
            # detached spine is invisible.  First, activate the frame but make the patch
            # and spines invisible.
            make_patch_spines_invisible(axt)
            # Second, show the right spine.
            axt.spines["right"].set_visible(True)
            p3, = T.plot.line(ax=axt, marker='s',color='m', label="Temperature")
            axt.yaxis.label.set_color(p3.get_color())
            axt.tick_params(axis='y', colors=p3.get_color())
            axt.set_ylabel('Temperature anomalies [$C\degree$]')
    return G


def produce_relative_frequency_wind_direction(path=ims_path,
                                              station='TEL-AVIV-COAST',
                                              season='DJF', with_weights=False,
                                              pw_station='tela', temp=False,
                                              plot=True):
    import xarray as xr
    import matplotlib.pyplot as plt
    import numpy as np

    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    wd = xr.open_dataset(path / 'IMS_WD_israeli_10mins.nc')[station]
    wd.load()
    wd.name = 'WD'
    wd = wd.sel(time=wd['time.season'] == season)
    all_Q = wd.groupby('time.hour').count()
    Q1 = wd.where((wd >= 0) & (wd < 90)).dropna('time')
    Q2 = wd.where((wd >= 90) & (wd < 180)).dropna('time')
    Q3 = wd.where((wd >= 180.1) & (wd < 270)).dropna('time')
    Q4 = wd.where((wd >= 270) & (wd < 360)).dropna('time')
    Q = xr.concat([Q1, Q2, Q3, Q4], 'Q')
    Q['Q'] = [x + 1 for x in range(4)]
    Q_freq = 100.0 * (Q.groupby('time.hour').count() / all_Q)
    T = xr.open_dataset(path / 'IMS_TD_israeli_10mins.nc')[station]
    T.load()
    T = T.groupby('time.month') - T.groupby('time.month').mean('time')
    T = T.reset_coords(drop=True)
    T = T.sel(time=T['time.season'] == season)
    T = T.groupby('time.hour').mean('time')
    if with_weights:
        ws = xr.open_dataset(path / 'IMS_WS_israeli_10mins.nc')[station]
        ws.load()
        ws = ws.sel(time=ws['time.season'] == season)
        ws.name = 'WS'
        wind = xr.merge([ws, wd])
        wind = wind.dropna('time')
        all_Q = wind['WD'].groupby('time.hour').count()
        Q1 = wind['WS'].where(
            (wind['WD'] >= 0) & (wind['WD'] < 90)).dropna('time')
        Q2 = wind['WS'].where(
            (wind['WD'] >= 90) & (wind['WD'] < 180)).dropna('time')
        Q3 = wind['WS'].where(
            (wind['WD'] >= 180) & (wind['WD'] < 270)).dropna('time')
        Q4 = wind['WS'].where(
            (wind['WD'] >= 270) & (wind['WD'] < 360)).dropna('time')
        Q = xr.concat([Q1, Q2, Q3, Q4], 'Q')
        Q['Q'] = [x + 1 for x in range(4)]
        Q_ratio = (Q.groupby('time.hour').count() / all_Q)
        Q_mean = Q.groupby('time.hour').mean() / Q.groupby('time.hour').max()
        Q_freq = 100 * ((Q_mean * Q_ratio) / (Q_mean * Q_ratio).sum('Q'))
    if plot:
        fig, ax = plt.subplots(figsize=(16, 8))
        for q in Q_freq['Q']:
            Q_freq.sel(Q=q).plot(ax=ax)
        ax.set_title(
            'Relative wind direction frequency in {} IMS station in {} season'.format(
                station, season))
        ax.set_ylabel('Relative frequency [%]')
        ax.set_xlabel('Time of day [UTC]')
        ax.legend([r'0$\degree$-90$\degree$', r'90$\degree$-180$\degree$',
                   r'180$\degree$-270$\degree$', r'270$\degree$-360$\degree$'], loc='upper left')
        ax.set_xticks(np.arange(0, 24, step=1))
        ax.grid()
        if pw_station is not None:
            pw = xr.open_dataset(
                work_yuval /
                'GNSS_PW_thresh_50_homogenized.nc')[pw_station]
            pw.load().dropna('time')
            pw = pw.groupby('time.month') - pw.groupby('time.month').mean('time')
            pw = pw.reset_coords(drop=True)
            pw = pw.sel(time=pw['time.season'] == season)
            pw = pw.groupby('time.hour').mean()
            axpw = ax.twinx()
            pw.plot.line(ax=axpw, color='k', marker='o')
            axpw.axhline(0, color='k', linestyle='--')
            axpw.legend(['{} PW anomalies'.format(pw_station.upper())], loc='upper right')
            axpw.set_ylabel('PW anomalies [mm]')
            if temp:
                axt = ax.twinx()
                axt.spines["right"].set_position(("axes", 1.05))
                # Having been created by twinx, par2 has its frame off, so the line of its
                # detached spine is invisible.  First, activate the frame but make the patch
                # and spines invisible.
                make_patch_spines_invisible(axt)
                # Second, show the right spine.
                axt.spines["right"].set_visible(True)
                p3, = T.plot.line(ax=axt, marker='s',color='m', label="Temperature")
                axt.yaxis.label.set_color(p3.get_color())
                axt.tick_params(axis='y', colors=p3.get_color())
                axt.set_ylabel('Temperature anomalies [$C\degree$]')
    return Q_freq


def get_israeli_coast_line(path=gis_path, minx=34.0, miny=30.0, maxx=36.0,
                           maxy=34.0):
    from shapely.geometry import box
    import geopandas as gpd
    # create bounding box using shapely:
    bbox = box(minx, miny, maxx, maxy)
    # read world coast lines:
    coast = gpd.read_file(gis_path / 'ne_10m_coastline.shp')
    # clip:
    gdf = gpd.clip(coast, bbox)
    return gdf


def clip_raster(fp=awd_path/'Israel_Area.tif',
                out_tif=awd_path/'israel_dem.tif',
                minx=34.0, miny=29.0, maxx=36.5, maxy=34.0):
    def getFeatures(gdf):
        """Function to parse features from GeoDataFrame in such a manner that
        rasterio wants them"""
        import json
        return [json.loads(gdf.to_json())['features'][0]['geometry']]

    import rasterio
    from rasterio.plot import show
    from rasterio.plot import show_hist
    from rasterio.mask import mask
    from shapely.geometry import box
    import geopandas as gpd
    from fiona.crs import from_epsg
    import pycrs
    print('reading {}'.format(fp))
    data = rasterio.open(fp)
    # create bounding box using shapely:
    bbox = box(minx, miny, maxx, maxy)
    # insert the bbox into a geodataframe:
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(4326))
    # re-project with the same projection as the data:
    geo = geo.to_crs(crs=data.crs.data)
    # get the geometry coords:
    coords = getFeatures(geo)
    # clipping is done with mask:
    out_img, out_transform = mask(dataset=data, shapes=coords, crop=True)
    # copy meta data:
    out_meta = data.meta.copy()
    # parse the epsg code:
    epsg_code = int(data.crs.data['init'][5:])
    # update the meta data:
    out_meta.update({"driver": "GTiff",
                     "height": out_img.shape[1],
                     "width": out_img.shape[2],
                     "transform": out_transform,
                     "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4()})
    # save to disk:
    print('saving {} to disk.'.format(out_tif))
    with rasterio.open(out_tif, "w", **out_meta) as dest:
        dest.write(out_img)
    print('Done!')
    return


def create_israel_area_dem(path):
    """merge the raw DSM tif files from AW3D30 model of Israel area togather"""
    from aux_gps import path_glob
    import rasterio
    from rasterio.merge import merge
    src_files_to_mosaic = []
    files = path_glob(path, '*DSM*.tif')
    for fp in files:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     "crs": src.crs
                     }
                    )
    with rasterio.open(path/'Israel_Area.tif', "w", **out_meta) as dest:
        dest.write(mosaic)
    return


def parse_cv_results(grid_search_cv):
    from aux_gps import process_gridsearch_results
    """parse cv_results from GridsearchCV object"""
    # only supports neg-abs-mean-error with leaveoneout
    from sklearn.model_selection import LeaveOneOut
    if (isinstance(grid_search_cv.cv, LeaveOneOut)
            and grid_search_cv.scoring == 'neg_mean_absolute_error'):

        cds = process_gridsearch_results(grid_search_cv)
        cds = - cds
    return cds


def IMS_interpolating_to_GNSS_stations_israel(dt='2013-10-19T22:00:00',
                                              stations=None,
                                              lapse_rate='auto',
                                              method='okrig',
                                              variogram='spherical',
                                              n_neighbors=3,
                                              start_year='1996',
                                              cut_days_ago=3,
                                              plot=False,
                                              verbose=False,
                                              savepath=ims_path):
    """interpolate the IMS 10 mins field(e.g., TD) to the location
    of the GNSS sites in ISRAEL(use dt=None for this). other dt is treated
    as datetime str and will give the "snapshot" for the field for just this
    datetime"""
    from pykrige.rk import Krige
    import pandas as pd
    from aux_gps import path_glob
    import xarray as xr
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from sklearn.neighbors import KNeighborsRegressor
    # import time

    def pick_model(method, variogram, n_neighbors):
        if method == 'okrig':
            if variogram is not None:
                model = Krige(method='ordinary', variogram_model=variogram,
                              verbose=verbose)
            else:
                model = Krige(method='ordinary', variogram_model='linear',
                              verbose=verbose)
        elif method == 'knn':
            if n_neighbors is None:
                model = KNeighborsRegressor(n_neighbors=5, weights='distance')
            else:
                model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        else:
            raise Exception('{} is not supported yet...'.format(method))
        return model

    def prepare_Xy(ts_lr_neutral, T_lats, T_lons):
        import numpy as np
        df = ts_lr_neutral.to_frame()
        df['lat'] = T_lats
        df['lon'] = T_lons
        # df = df.dropna(axis=0)
        c = np.linspace(
            df['lat'].min(),
            df['lat'].max(),
            df['lat'].shape[0])
        r = np.linspace(
            df['lon'].min(),
            df['lon'].max(),
            df['lon'].shape[0])
        rr, cc = np.meshgrid(r, c)
        vals = ~np.isnan(ts_lr_neutral)
        X = np.column_stack([rr[vals, vals], cc[vals, vals]])
        # rr_cc_as_cols = np.column_stack([rr.flatten(), cc.flatten()])
        # y = da_scaled.values[vals]
        y = ts_lr_neutral[vals]
        return X, y

    def neutrilize_t(ts_vs_alt, lapse_rate):
        ts_lr_neutral = (ts_vs_alt +
                         lapse_rate *
                         ts_vs_alt.index /
                         1000.0)
        return ts_lr_neutral

    def choose_dt_and_lapse_rate(tdf, dt, T_alts, lapse_rate):
        ts = tdf.loc[dt, :]
        # dt_col = dt.strftime('%Y-%m-%d %H:%M')
        # ts.name = dt_col
        # Tloc_df = Tloc_df.join(ts, how='right')
        # Tloc_df = Tloc_df.dropna(axis=0)
        ts_vs_alt = pd.Series(ts.values, index=T_alts)
        ts_vs_alt_for_fit = ts_vs_alt.dropna()
#        try:
        [a, b] = np.polyfit(ts_vs_alt_for_fit.index.values,
                            ts_vs_alt_for_fit.values, 1)
#        except TypeError as e:
#            print('{}, dt: {}'.format(e, dt))
#            print(ts_vs_alt)
#            return
        if lapse_rate == 'auto':
            lapse_rate = np.abs(a) * 1000
            if lapse_rate < 5.0:
                lapse_rate = 5.0
            elif lapse_rate > 10.0:
                lapse_rate = 10.0
        return ts_vs_alt, lapse_rate
#    import time
    dt = pd.to_datetime(dt)
    # read Israeli GNSS sites coords:
    df = pd.read_csv(
            cwd /
            'israeli_gnss_coords.txt',
            delim_whitespace=True,
            header=0)
    # use station=None to pick all stations, otherwise pick one...
    if stations is not None:
        if isinstance(stations, str):
            stations = [stations]
        df = df.loc[stations, :]
        print('selected only {} stations'.format(stations))
    else:
        print('selected all israeli stations.')
    # prepare lats and lons of gnss sites:
    gps_lats = np.linspace(df.lat.min(), df.lat.max(), df.lat.values.shape[0])
    gps_lons = np.linspace(df.lon.min(), df.lon.max(), df.lon.values.shape[0])
    gps_lons_lats_as_cols = np.column_stack([gps_lons, gps_lats])
    # load IMS temp data:
    glob_str = 'IMS_TD_israeli_10mins*.nc'
    file = path_glob(ims_path, glob_str=glob_str)[0]
    ds = xr.open_dataset(file)
    time_dim = list(set(ds.dims))[0]
    # slice to a starting year(1996?):
    ds = ds.sel({time_dim: slice(start_year, None)})
    years = sorted(list(set(ds[time_dim].dt.year.values)))
    # get coords and alts of IMS stations:
    T_alts = np.array([ds[x].attrs['station_alt'] for x in ds])
    T_lats = np.array([ds[x].attrs['station_lat'] for x in ds])
    T_lons = np.array([ds[x].attrs['station_lon'] for x in ds])
    print('loading IMS_TD of israeli stations 10mins freq..')
    # transform to dataframe and add coords data to df:
    tdf = ds.to_dataframe()
    if cut_days_ago is not None:
        # use cut_days_ago to drop last x days of data:
        # this is vital bc towards the newest data, TD becomes scarce bc not
        # all of the stations data exists...
        n = cut_days_ago * 144
        tdf.drop(tdf.tail(n).index, inplace=True)
        print('last date to be handled is {}'.format(tdf.index[-1]))
    # use this to solve for a specific datetime:
    if dt is not None:
        dt_col = dt.strftime('%Y-%m-%d %H:%M')
#        t0 = time.time()
        # prepare the ims coords and temp df(Tloc_df) and the lapse rate:
        ts_vs_alt, lapse_rate = choose_dt_and_lapse_rate(tdf, dt, T_alts, lapse_rate)
        if plot:
            fig, ax_lapse = plt.subplots(figsize=(10, 6))
            sns.regplot(x=ts_vs_alt.index, y=ts_vs_alt.values, color='r',
                        scatter_kws={'color': 'b'}, ax=ax_lapse)
            suptitle = dt.strftime('%Y-%m-%d %H:%M')
            ax_lapse.set_xlabel('Altitude [m]')
            ax_lapse.set_ylabel('Temperature [degC]')
            ax_lapse.text(0.5, 0.95, 'Lapse_rate: {:.2f} degC/km'.format(lapse_rate),
                          horizontalalignment='center', verticalalignment='center',
                          transform=ax_lapse.transAxes, fontsize=12, color='k',
                          fontweight='bold')
            ax_lapse.grid()
            ax_lapse.set_title(suptitle, fontsize=14, fontweight='bold')
        # neutrilize the lapse rate effect:
        ts_lr_neutral = neutrilize_t(ts_vs_alt, lapse_rate)
        # prepare the regressors(IMS stations coords) and the
        # target(IMS temperature at the coords):
        X, y = prepare_Xy(ts_lr_neutral, T_lats, T_lons)
        # pick the model and params:
        model = pick_model(method, variogram, n_neighbors)
        # fit the model:
        model.fit(X, y)
        # predict at the GNSS stations coords:
        interpolated = model.predict(gps_lons_lats_as_cols).reshape((gps_lats.shape))
        # add prediction to df:
        df[dt_col] = interpolated
        # fix for lapse rate:
        df[dt_col] -= lapse_rate * df['alt'] / 1000.0
        # concat gnss stations and Tloc DataFrames:
        Tloc_df = pd.DataFrame(T_lats, index=tdf.columns)
        Tloc_df.columns = ['lat']
        Tloc_df['lon'] = T_lons
        Tloc_df['alt'] = T_alts
        all_df = pd.concat([df, Tloc_df],axis=0)
        # fname = gis_path / 'ne_10m_admin_0_sovereignty.shp'
        # fname = gis_path / 'gadm36_ISR_0.shp'
        # ax = plt.axes(projection=ccrs.PlateCarree())
        if plot:
            fig, ax = plt.subplots(figsize=(6, 10))
            # shdf = salem.read_shapefile(salem.get_demo_file('world_borders.shp'))
            # shdf = salem.read_shapefile(gis_path / 'Israel_and_Yosh.shp')
            isr = gpd.read_file(gis_path / 'Israel_and_Yosh.shp')
            # shdf = shdf.loc[shdf['CNTRY_NAME'] == 'Israel']  # remove other countries
            isr.crs = {'init': 'epsg:4326'}
            time_snap = gpd.GeoDataFrame(all_df, geometry=gpd.points_from_xy(all_df.lon,
                                                                             all_df.lat),
                                        crs=isr.crs)
            time_snap = gpd.sjoin(time_snap, isr, op='within')
            isr.plot(ax=ax)
            cmap = plt.get_cmap('rainbow', 10)
            time_snap.plot(ax=ax, column=dt_col, cmap=cmap,
                           edgecolor='black', legend=True)
            for x, y, label in zip(df.lon, df.lat,
                                   df.index):
                ax.annotate(label, xy=(x, y), xytext=(3, 3),
                            textcoords="offset points")
            suptitle = dt.strftime('%Y-%m-%d %H:%M')
            fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    else:
        # do the above (except plotting) for the entire data, saving each year:
        for year in years:
            dts = tdf.index[tdf.index.year == year]
            # read Israeli GNSS sites coords again:
            df = pd.read_csv(
                    cwd /
                    'israeli_gnss_coords.txt',
                    delim_whitespace=True,
                    header=0)
            cnt = 1
            dt_col_list = []
            inter_list = []
            # t0 = time.time()
#            t1 = time.time()
#            t2 = time.time()
#            t3 = time.time()
#            t4 = time.time()
#            t5 = time.time()
#            t6 = time.time()
#            t7 = time.time()
#            t8 = time.time()
            for dt in dts:
                dt_col = dt.strftime('%Y-%m-%d %H:%M')
                if np.mod(cnt, 144) == 0:
                    # t1 = time.time()
                    print('working on {}'.format(dt_col))
                    # print('time1:{:.2f} seconds'.format(t1-t0))
                    # t0 = time.time()
                # prepare the ims coords and temp df(Tloc_df) and
                # the lapse rate:
                ts_vs_alt, lapse_rate = choose_dt_and_lapse_rate(tdf, dt, T_alts, lapse_rate)
#                if np.mod(cnt, 144) == 0:
#                    t2 = time.time()
#                    print('time2: {:.4f}'.format((t2-t1)*144))
                # neutrilize the lapse rate effect:
                ts_lr_neutral = neutrilize_t(ts_vs_alt, lapse_rate)
                # prepare the regressors(IMS stations coords) and the
                # target(IMS temperature at the coords):
#                if np.mod(cnt, 144) == 0:
#                    t3 = time.time()
#                    print('time3: {:.4f}'.format((t3-t2)*144))
                X, y = prepare_Xy(ts_lr_neutral, T_lats, T_lons)
#                if np.mod(cnt, 144) == 0:
#                    t4 = time.time()
#                    print('time4: {:.4f}'.format((t4-t3)*144))
                # pick model and params:
                model = pick_model(method, variogram, n_neighbors)
#                if np.mod(cnt, 144) == 0:
#                    t5 = time.time()
#                    print('time5: {:.4f}'.format((t5-t4)*144))
                # fit the model:
                model.fit(X, y)
#                if np.mod(cnt, 144) == 0:
#                    t6 = time.time()
#                    print('time6: {:.4f}'.format((t6-t5)*144))
                # predict at the GNSS stations coords:
                interpolated = model.predict(gps_lons_lats_as_cols).reshape((gps_lats.shape))
#                if np.mod(cnt, 144) == 0:
#                    t7 = time.time()
#                    print('time7: {:.4f}'.format((t7-t6)*144))
                # fix for lapse rate:
                interpolated -= lapse_rate * df['alt'].values / 1000.0
#                if np.mod(cnt, 144) == 0:
#                    t8 = time.time()
#                    print('time8: {:.4f}'.format((t8-t7)*144))
                # add to list:
                dt_col_list.append(dt_col)
                inter_list.append(interpolated)
                cnt += 1
            # convert to dataset:
            # da = xr.DataArray(df.iloc[:, 3:].values, dims=['station', 'time'])
            da = xr.DataArray(inter_list, dims=['time', 'station'])
            da['station'] = df.index
            da['time'] = pd.to_datetime(dt_col_list)
            da = da.sortby('time')
            ds = da.to_dataset(dim='station')
            for da in ds:
                ds[da].attrs['units'] = 'degC'
            filename = 'GNSS_TD_{}.nc'.format(year)
            ds.to_netcdf(savepath / filename, 'w')
            print('saved {} to {}'.format(filename, savepath))
            # return
        print('concatenating all TD years...')
        concat_GNSS_TD(savepath)
#    t1 = time.time()
    # geo_snap = geo_pandas_time_snapshot(var='TD', datetime=dt, plot=False)
#    total = t1-t0
#    print(total)
    return


def resample_GNSS_TD(path=ims_path):
    from aux_gps import path_glob
    import xarray as xr
    from aux_gps import get_unique_index

    def resample_GNSS_TD(ds, path, sample, sample_rate='1H'):
        # station = da.name
        ds = get_unique_index(ds)
        print('resampaling all GNSS stations to {}'.format(sample[sample_rate]))
        years = [str(x)
                 for x in sorted(list(set(ds[time_dim].dt.year.values)))]
        ymin = ds[time_dim].min().dt.year.item()
        ymax = ds[time_dim].max().dt.year.item()
        years_str = '{}_{}'.format(ymin, ymax)
        if sample_rate == '1H' or sample_rate == '3H':
            dsr_list = []
            for year in years:
                print('resampling {} of year {}'.format(sample_rate, year))
                dsr = ds.sel({time_dim: year}).resample(
                         {time_dim: sample_rate}, keep_attrs=True, skipna=True).mean(keep_attrs=True)
                dsr_list.append(dsr)
            print('concatenating...')
            dsr = xr.concat(dsr_list, time_dim)
        else:
            if sample_rate == '5min':
                dsr = ds.resample({time_dim: sample_rate}, keep_attrs=True,
                                  skipna=True).ffill()
            else:
                dsr = ds.resample({time_dim: sample_rate},
                                  keep_attrs=True,
                                  skipna=True).mean(keep_attrs=True)
        new_filename = '_'.join(['GNSS', sample[sample_rate], 'TD_ALL',
                                 years_str])
        new_filename = new_filename + '.nc'
        print('saving all resmapled GNSS stations to {}'.format(path))
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in dsr.data_vars}
        dsr.to_netcdf(path / new_filename, 'w', encoding=encoding)
        print('Done resampling!')
        return
    # first, load GNSS_TD_ALL:
    str_glob = 'GNSS_TD_ALL*.nc'
    file = sorted(path_glob(path, str_glob))[-1]
    print(file)
    ds = xr.open_dataset(file)
    ds.load()
    time_dim = list(set(ds.dims))[0]
    sample = {'5min': '5mins', '1H': 'hourly', '3H': '3hourly',
              'D': 'Daily', 'W': 'weekly', 'MS': 'monthly'}
    for key in sample.keys():
        resample_GNSS_TD(ds, path, sample, sample_rate=key)

#    for sta in stations:
#        # take each station's TD and copy to GNSS folder 'temperature':
#        savepath = GNSS / sta / 'temperature'
#        savepath.mkdir(parents=True, exist_ok=True)
#        # first save a 5-min resampled version and save:
#        da = ds[sta].resample(time='5min').ffill()
#        ymin = da[time_dim].min().dt.year.item()
#        ymax = da[time_dim].max().dt.year.item()
#        years_str = '{}_{}'.format(ymin, ymax)
#        new_filename = '_'.join([sta.upper(), 'TD', years_str])
#        new_filename = new_filename + '.nc'
#        print('saving resmapled station {} to {}'.format(sta, savepath))
#        comp = dict(zlib=True, complevel=9)  # best compression
#        encoding = {var: comp for var in da.to_dataset(name=da.name).data_vars}
#        da.to_netcdf(savepath / new_filename, 'w', encoding=encoding)
#        print('Done resampling!')
#        # finally, resample to all samples and save:
#        for key in sample.keys():
#            resample_GNSS_TD(da, savepath, sample, sample_rate=key)
    return


def concat_GNSS_TD(path=ims_path):
    import xarray as xr
    from aux_gps import path_glob
    files = path_glob(path, 'GNSS_TD_*.nc')
    years = sorted([file.as_posix().split('/')[-1].split('_')[-1].split('.')[0]
                    for file in files])
    ds_list = [xr.open_dataset(x) for x in files]
    time_dim = list(set(ds_list[0].dims))[0]
    ds = xr.concat(ds_list, time_dim)
    ds = ds.sortby(time_dim)
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in ds.data_vars}
    filename = 'GNSS_TD_ALL_{}-{}.nc'.format(years[0], years[-1])
    print('saving...')
    ds.to_netcdf(path / filename, 'w', encoding=encoding)
    print('{} was saved to {}'.format(filename, path))
    return ds


def Interpolating_models_ims(time='2013-10-19T22:00:00', var='TD', plot=True,
                             gis_path=gis_path, method='okrig',
                             dem_path=work_yuval / 'AW3D30', lapse_rate=5.,
                             cv=None, rms=None, gridsearch=False):
    """main 2d_interpolation from stations to map"""
    # cv usage is {'kfold': 5} or {'rkfold': [2, 3]}
    # TODO: try 1d modeling first, like T=f(lat)
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from pykrige.rk import Krige
    import numpy as np
    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from scipy.spatial import Delaunay
    from scipy.interpolate import griddata
    from sklearn.metrics import mean_squared_error
    from aux_gps import coarse_dem
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pyproj
    from sklearn.utils.estimator_checks import check_estimator
    from pykrige.compat import GridSearchCV
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')

    def parse_cv(cv):
        from sklearn.model_selection import KFold
        from sklearn.model_selection import RepeatedKFold
        from sklearn.model_selection import LeaveOneOut
        """input:cv number or string"""
        # check for integer:
        if 'kfold' in cv.keys():
            n_splits = cv['kfold']
            print('CV is KFold with n_splits={}'.format(n_splits))
            return KFold(n_splits=n_splits)
        if 'rkfold' in cv.keys():
            n_splits = cv['rkfold'][0]
            n_repeats = cv['rkfold'][1]
            print('CV is ReapetedKFold with n_splits={},'.format(n_splits) +
                  ' n_repeates={}'.format(n_repeats))
            return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=42)
        if 'loo' in cv.keys():
            return LeaveOneOut()
    # from aux_gps import scale_xr
    da = create_lat_lon_mesh(points_per_degree=250)  # 500?
    awd = coarse_dem(da)
    awd = awd.values
    geo_snap = geo_pandas_time_snapshot(var=var, datetime=time, plot=False)
    if var == 'TD':
        [a, b] = np.polyfit(geo_snap['alt'].values, geo_snap['TD'].values, 1)
        if lapse_rate == 'auto':
            lapse_rate = np.abs(a) * 1000
        fig, ax_lapse = plt.subplots(figsize=(10, 6))
        sns.regplot(data=geo_snap, x='alt', y='TD', color='r',
                    scatter_kws={'color': 'b'}, ax=ax_lapse)
        suptitle = time.replace('T', ' ')
        ax_lapse.set_xlabel('Altitude [m]')
        ax_lapse.set_ylabel('Temperature [degC]')
        ax_lapse.text(0.5, 0.95, 'Lapse_rate: {:.2f} degC/km'.format(lapse_rate),
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax_lapse.transAxes, fontsize=12, color='k',
                      fontweight='bold')
        ax_lapse.grid()
        ax_lapse.set_title(suptitle, fontsize=14, fontweight='bold')
#     fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    alts = []
    for i, row in geo_snap.iterrows():
        lat = da.sel(lat=row['lat'], method='nearest').lat.values
        lon = da.sel(lon=row['lon'], method='nearest').lon.values
        alt = row['alt']
        if lapse_rate is not None and var == 'TD':
            da.loc[{'lat': lat, 'lon': lon}] = row[var] + \
                lapse_rate * alt / 1000.0
            alts.append(alt)
        elif lapse_rate is None or var != 'TD':
            da.loc[{'lat': lat, 'lon': lon}] = row[var]
            alts.append(alt)
    # da_scaled = scale_xr(da)
    c = np.linspace(min(da.lat.values), max(da.lat.values), da.shape[0])
    r = np.linspace(min(da.lon.values), max(da.lon.values), da.shape[1])
    rr, cc = np.meshgrid(r, c)
    vals = ~np.isnan(da.values)
    if lapse_rate is None:
        Xrr, Ycc, Z = pyproj.transform(
                lla, ecef, rr[vals], cc[vals], np.array(alts), radians=False)
        X = np.column_stack([Xrr, Ycc, Z])
        XX, YY, ZZ = pyproj.transform(lla, ecef, rr, cc, awd.values,
                                      radians=False)
        rr_cc_as_cols = np.column_stack([XX.flatten(), YY.flatten(), ZZ.flatten()])
    else:
        X = np.column_stack([rr[vals], cc[vals]])
        rr_cc_as_cols = np.column_stack([rr.flatten(), cc.flatten()])
    # y = da_scaled.values[vals]
    y = da.values[vals]
    if method == 'gp-rbf':
        from sklearn.gaussian_process.kernels import RBF
        from sklearn.gaussian_process.kernels import WhiteKernel
        kernel = 1.0 * RBF(length_scale=0.25, length_scale_bounds=(1e-2, 1e3)) \
            + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-10, 1e+1))
#        kernel = None
        model = GaussianProcessRegressor(alpha=0.0, kernel=kernel,
                                         n_restarts_optimizer=5,
                                         random_state=42, normalize_y=True)

    elif method == 'gp-qr':
        from sklearn.gaussian_process.kernels import RationalQuadratic
        from sklearn.gaussian_process.kernels import WhiteKernel
        kernel = RationalQuadratic(length_scale=100.0) \
            + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-10, 1e+1))
        model = GaussianProcessRegressor(alpha=0.0, kernel=kernel,
                                         n_restarts_optimizer=5,
                                         random_state=42, normalize_y=True)
    elif method == 'knn':
        model = KNeighborsRegressor(n_neighbors=5, weights='distance')
    elif method == 'svr':
        model = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
                    gamma='auto_deprecated', kernel='rbf', max_iter=-1,
                    shrinking=True, tol=0.001, verbose=False)
    elif method == 'okrig':
        model = Krige(method='ordinary', variogram_model='spherical',
                      verbose=True)
    elif method == 'ukrig':
        model = Krige(method='universal', variogram_model='linear',
                      verbose=True)
#    elif method == 'okrig3d':
#        # don't bother - MemoryError...
#        model = OrdinaryKriging3D(rr[vals], cc[vals], np.array(alts),
#                                  da.values[vals], variogram_model='linear',
#                                  verbose=True)
#        awd = coarse_dem(da)
#        interpolated, ss = model.execute('grid', r, c, awd['data'].values)
#    elif method == 'rkrig':
#        # est = LinearRegression()
#        est = RandomForestRegressor()
#        model = RegressionKriging(regression_model=est, n_closest_points=5,
#                                  verbose=True)
#        p = np.array(alts).reshape(-1, 1)
#        model.fit(p, X, y)
#        P = awd.flatten().reshape(-1, 1)
#        interpolated = model.predict(P, rr_cc_as_cols).reshape(da.values.shape)
#    try:
#        u = check_estimator(model)
#    except TypeError:
#        u = False
#        pass
    if cv is not None and not gridsearch:  # and u is None):
        # from sklearn.model_selection import cross_validate
        from sklearn import metrics
        cv = parse_cv(cv)
        ytests = []
        ypreds = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]  # requires arrays
            y_train, y_test = y[train_idx], y[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # there is only one y-test and y-pred per iteration over the loo.split,
            # so to get a proper graph, we append them to respective lists.
            ytests += list(y_test)
            ypreds += list(y_pred)
        true_vals = np.array(ytests)
        predicted = np.array(ypreds)
        r2 = metrics.r2_score(ytests, ypreds)
        ms_error = metrics.mean_squared_error(ytests, ypreds)
        print("R^2: {:.5f}%, MSE: {:.5f}".format(r2*100, ms_error))
    if gridsearch:
        cv = parse_cv(cv)
        param_dict = {"method": ["ordinary", "universal"],
                      "variogram_model": ["linear", "power", "gaussian",
                                          "spherical"],
                      # "nlags": [4, 6, 8],
                      # "weight": [True, False]
                      }
        estimator = GridSearchCV(Krige(), param_dict, verbose=True, cv=cv,
                                 scoring='neg_mean_absolute_error',
                                 return_train_score=True, n_jobs=1)
        estimator.fit(X, y)
        if hasattr(estimator, 'best_score_'):
            print('best_score = {:.3f}'.format(estimator.best_score_))
            print('best_params = ', estimator.best_params_)

        return estimator
#    if (cv is not None and not u):
#        from sklearn import metrics
#        cv = parse_cv(cv)
#        ytests = []
#        ypreds = []
#        for train_idx, test_idx in cv.split(X):
#            X_train, X_test = X[train_idx], X[test_idx]  # requires arrays
#            y_train, y_test = y[train_idx], y[test_idx]
##            model = UniversalKriging(X_train[:, 0], X_train[:, 1], y_train,
##                                     variogram_model='linear', verbose=False,
##                                     enable_plotting=False)
#            model.X_ORIG = X_train[:, 0]
#            model.X_ADJUSTED = model.X_ORIG
#            model.Y_ORIG = X_train[:, 1]
#            model.Y_ADJUSTED = model.Y_ORIG
#            model.Z = y_train
#            y_pred, ss = model.execute('points', X_test[0, 0],
#                                             X_test[0, 1])
#            # there is only one y-test and y-pred per iteration over the loo.split,
#            # so to get a proper graph, we append them to respective lists.
#            ytests += list(y_test)        cmap = plt.get_cmap('spring', 10)
        Q = ax.quiver(isr['X'], isr['Y'], isr['U'], isr['V'],
                      isr['cm_per_year'], cmap=cmap)
        fig.colorbar(Q, extend='max')

#            ypreds += list(y_pred)
#        true_vals = np.array(ytests)
#        predicted = np.array(ypreds)
#        r2 = metrics.r2_score(ytests, ypreds)
#        ms_error = metrics.mean_squared_error(ytests, ypreds)
#        print("R^2: {:.5f}%, MSE: {:.5f}".format(r2*100, ms_error))
#        cv_results = cross_validate(gp, X, y, cv=cv, scoring='mean_squared_error',
#                                    return_train_score=True, n_jobs=-1)
#        test = xr.DataArray(cv_results['test_score'], dims=['kfold'])
#        train = xr.DataArray(cv_results['train_score'], dims=['kfold'])
#        train.name = 'train'
#        cds = test.to_dataset(name='test')
#        cds['train'] = train
#        cds['kfold'] = np.arange(len(cv_results['test_score'])) + 1
#        cds['mean_train'] = cds.train.mean('kfold')
#        cds['mean_test'] = cds.test.mean('kfold')

    # interpolated=griddata(X, y, (rr, cc), method='nearest')
    model.fit(X, y)
    interpolated = model.predict(rr_cc_as_cols).reshape(da.values.shape)
    da_inter = da.copy(data=interpolated)
    if lapse_rate is not None and var == 'TD':
        da_inter -= lapse_rate * awd / 1000.0
    if (rms is not None and cv is None):  # or (rms is not None and not u):
        predicted = []
        true_vals = []
        for i, row in geo_snap.iterrows():
            lat = da.sel(lat=row['lat'], method='nearest').lat.values
            lon = da.sel(lon=row['lon'], method='nearest').lon.values
            pred = da_inter.loc[{'lat': lat, 'lon': lon}].values.item()
            true = row[var]
            predicted.append(pred)
            true_vals.append(true)
        predicted = np.array(predicted)
        true_vals = np.array(true_vals)
        ms_error = mean_squared_error(true_vals, predicted)
        print("MSE: {:.5f}".format(ms_error))
    if plot:
        import salem
        from salem import DataLevels, Map
        import cartopy.crs as ccrs
        # import cartopy.io.shapereader as shpreader
        import matplotlib.pyplot as plt
        # fname = gis_path / 'ne_10m_admin_0_sovereignty.shp'
        # fname = gis_path / 'gadm36_ISR_0.shp'
        # ax = plt.axes(projection=ccrs.PlateCarree())
        f, ax = plt.subplots(figsize=(6, 10))
        # shdf = salem.read_shapefile(salem.get_demo_file('world_borders.shp'))
        shdf = salem.read_shapefile(gis_path / 'Israel_and_Yosh.shp')
        # shdf = shdf.loc[shdf['CNTRY_NAME'] == 'Israel']  # remove other countries
        shdf.crs = {'init': 'epsg:4326'}
        dsr = da_inter.salem.roi(shape=shdf)
        grid = dsr.salem.grid
        grid = da_inter.salem.grid
        sm = Map(grid)
        # sm.set_shapefile(gis_path / 'Israel_and_Yosh.shp')
        # sm = dsr.salem.quick_map(ax=ax)
#        sm2 = salem.Map(grid, factor=1)
#        sm2.set_shapefile(gis_path/'gis_osm_water_a_free_1.shp',
#                          edgecolor='k')
        sm.set_data(dsr)
        # sm.set_nlevels(7)
        # sm.visualize(ax=ax, title='Israel {} interpolated temperature from IMS'.format(method),
        #             cbar_title='degC')
        sm.set_shapefile(gis_path/'gis_osm_water_a_free_1.shp',
                         edgecolor='k')  # , facecolor='aqua')
        # sm.set_topography(awd.values, crs=awd.crs)
        # sm.set_rgb(crs=shdf.crs, natural_earth='hr')  # ad
        # lakes = salem.read_shapefile(gis_path/'gis_osm_water_a_free_1.shp')
        sm.set_cmap(cm='rainbow')
        sm.visualize(ax=ax, title='Israel {} interpolated temperature from IMS'.format(method),
                     cbar_title='degC')
        dl = DataLevels(geo_snap[var], levels=sm.levels)
        dl.set_cmap(sm.cmap)
        x, y = sm.grid.transform(geo_snap.lon.values, geo_snap.lat.values)
        ax.scatter(x, y, color=dl.to_rgb(), s=20, edgecolors='k', linewidths=0.5)
        suptitle = time.replace('T', ' ')
        f.suptitle(suptitle, fontsize=14, fontweight='bold')
        if (rms is not None or cv is not None) and (not gridsearch):
            import seaborn as sns
            f, ax = plt.subplots(1, 2, figsize=(12, 6))
            sns.scatterplot(x=true_vals, y=predicted, ax=ax[0], marker='.',
                            s=100)
            resid = predicted - true_vals
            sns.distplot(resid, bins=5, color='c', label='residuals',
                         ax=ax[1])
            rmean = np.mean(resid)
            rstd = np.std(resid)
            rmedian = np.median(resid)
            rmse = np.sqrt(mean_squared_error(true_vals, predicted))
            plt.axvline(rmean, color='r', linestyle='dashed', linewidth=1)
            _, max_ = plt.ylim()
            plt.text(rmean + rmean / 10, max_ - max_ / 10,
                     'Mean: {:.2f}, RMSE: {:.2f}'.format(rmean, rmse))
            f.tight_layout()
        # lakes.plot(ax=ax, color='b', edgecolor='k')
        # lake_borders = gpd.overlay(countries, capitals, how='difference')
        # adm1_shapes = list(shpreader.Reader(fname).geometries())
        # ax = plt.axes(projection=ccrs.PlateCarree())
        # ax.coastlines(resolution='10m')
        # ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),
        #                  edgecolor='black', facecolor='gray', alpha=0.5)
        # da_inter.plot.pcolormesh('lon', 'lat', ax=ax)
        #geo_snap.plot(ax=ax, column=var, cmap='viridis', edgecolor='black',
        #              legend=False)
    return da_inter


def create_lat_lon_mesh(lats=[29.5, 33.5], lons=[34, 36],
                        points_per_degree=1000):
    import xarray as xr
    import numpy as np
    lat = np.arange(lats[0], lats[1], 1.0 / points_per_degree)
    lon = np.arange(lons[0], lons[1], 1.0 / points_per_degree)
    nans = np.nan * np.ones((len(lat), len(lon)))
    da = xr.DataArray(nans, dims=['lat', 'lon'])
    da['lat'] = lat
    da['lon'] = lon
    return da


#def read_save_ims_10mins(path=ims_10mins_path, var='TD'):
#    import xarray as xr
#    search_str = '*' + var + '_10mins.nc'
#    da_list = []
#    for file_and_path in path.glob(search_str):
#        da = xr.load_dataarray(file_and_path)
#        print('reading ims 10mins {} data for {} station'.format(var, da.name))
#        da_list.append(da)
#    print('merging...')
#    ds = xr.merge(da_list)
#    comp = dict(zlib=True, complevel=9)  # best compression
#    encoding = {var: comp for var in ds.data_vars}
#    filename = 'ims_' + var + '_10mins.nc'
#    print('saving...')
#    ds.to_netcdf(path / filename, 'w', encoding=encoding)
#    print('{} was saved to {}.'.format(filename, path))
#    return ds


def analyse_10mins_ims_field(path=ims_10mins_path, var='TD',
                             gis_path=gis_path, dem_path=work_yuval/'AW3D30'):
    import xarray as xr
    import collections
    import numpy as np
    # TODO: make 2d histogram of stations by altitude and time...
    awd = xr.open_rasterio(dem_path / 'israel_dem.tif')
    awd = awd.squeeze(drop=True)
    filename = 'ims_' + var + '_10mins.nc'
    ds = xr.open_dataset(path / filename)
    meta = read_ims_metadata_from_files(path=gis_path,
                                        filename='IMS_10mins_meta_data.xlsx')
    meta.index = meta.ID.astype('int')
    meta.drop('ID', axis=1, inplace=True)
    meta.sort_index(inplace=True)
    # there are some stations with the same altitude, i'm mapping them:
    duplicate_alts = [item for item, count in collections.Counter(
                        meta['alt']).items() if count > 1]
    print(duplicate_alts)
    # then replacing them with a 1-meter seperations:
    for dup in duplicate_alts:
        dup_size = len(meta.loc[meta['alt'] == dup, 'alt'])
        start_value = meta.loc[meta['alt'] == dup, 'alt'].values[0]
        replace_values = np.arange(start_value, start_value + dup_size)
        print(
                'duplicate {} has {} values, replacing with {}'.format(
                        dup,
                        dup_size,
                        replace_values))
        meta.loc[meta['alt'] == dup, 'alt'] = replace_values
    for da in ds.data_vars.keys():
        id_ = ds[da].attrs['station_id']
        try:
            lat = meta.loc[id_, 'lat']
            lon = meta.loc[id_, 'lon']
            alt = meta.loc[id_, 'alt']
        except KeyError:
            lat = ds[da].attrs['station_lat']
            lon = ds[da].attrs['station_lon']
            print('station {} keyerror.'.format(da))
            alt = 'None'
        try:
            alt = awd.sel(x=float(lon), y=float(lat), method='nearest').values.item()
        except ValueError:
            print('station {} has not known lat or lon...'.format(ds[da].attrs['station_name']))
        ds[da].attrs['station_lat'] = lat
        ds[da].attrs['station_lon'] = lon
        ds[da].attrs['station_alt'] = alt
    return ds


def geo_pandas_time_snapshot(path=ims_path, var='TD', freq='10mins',
                             datetime='2013-10-19T10:00:00',
                             gis_path=gis_path, plot=True):
    import xarray as xr
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from aux_gps import path_glob
    # TODO: add simple df support
    # first, read ims_10mins data for choice var:
    # file should be : 'IMS_TD_israeli_10mins_filled.nc'
    glob_str = 'IMS_{}_israeli_{}*.nc'.format(var, freq)
    file = path_glob(path, glob_str=glob_str)[0]
    ds = xr.open_dataset(file)
    ds = ds.sel(time=datetime)
#    meta = read_ims_metadata_from_files(path=gis_path, option='10mins')
#    meta.index = meta.ID.astype('int')
#    meta.drop('ID', axis=1, inplace=True)
#    meta.sort_index(inplace=True)
    cols_list = []
    for dvar in ds.data_vars.values():
        value = dvar.values.item()
        id_ = dvar.attrs['station_id']
        lat = dvar.attrs['station_lat']
        lon = dvar.attrs['station_lon']
        alt = dvar.attrs['station_alt']
        name = dvar.name
        var_ = dvar.attrs['channel_name']
        cols = [pd.to_datetime(datetime), name, id_, lat, lon, alt,
                var_, value]
        cols_list.append(cols)
    df = pd.DataFrame(cols_list)
    df.columns = ['time', 'name', 'id', 'lat', 'lon', 'alt', 'var_name', var_]
    df.dropna(inplace=True)
    df = df.astype({'lat': 'float64', 'lon': 'float64'})
    # geopandas part:
    isr = gpd.read_file(gis_path / 'Israel_demog_yosh.shp')
    isr.crs = {'init': 'epsg:4326'}
    geo_snap = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon,
                                                                df.lat),
                                crs=isr.crs)
    if plot:
        ax = isr.plot()
        geo_snap.plot(ax=ax, column=var_, cmap='viridis', edgecolor='black',
                      legend=True)
        plt.title(var_ + ' in ' + datetime)
    return geo_snap


def get_meta_data_hourly_ims_climate_database(ds):
    import pandas as pd
    name_list = []
    for name, da in ds.data_vars.items():
        data = [name.split('_')[0], da.attrs['station_id'], da.attrs['lat'],
                da.attrs['lon'], da.attrs['height']]
        name_list.append(data)
    df = pd.DataFrame(name_list)
    df.columns = ['name', 'id', 'lat', 'lon', 'height']
    return df


def proccess_hourly_ims_climate_database(path=ims_path, var='tas',
                                         times=('1996', '2019')):
    import xarray as xr
    import numpy as np
    ds = xr.open_dataset(path / 'hourly_ims.nc')
    if var is not None:
        ds = ds.sel({'var': var})
        print('selecting {} variables'.format(var))
        if times is not None:
            print('selecting times from {} to {}'.format(times[0], times[1]))
            ds = ds.sel(time=slice(times[0], times[1]))
            to_drop_list = []
            for name, da in ds.data_vars.items():
                if (np.isnan(da) == True).all().item():
                    to_drop_list.append(name)
            ds = ds.drop(to_drop_list)
    return ds


def read_all_hourly_ims_climate_database(path=ims_path / 'hourly', freq='03',
                                         savepath=None):
    """downloaded from tau...ds is a dataset of all stations,
    times is a time period"""
    import xarray as xr
    from aux_gps import save_ncfile
    ds_list = []
    for file in sorted(path.glob('*_{}hr_*.csv'.format(freq))):
        ds = read_one_ims_hourly_station_csv(file)
        ds_list.append(ds)
    dss = xr.merge(ds_list)
    print('Done!')
    if savepath is not None:
        save_ncfile(dss, savepath, filename='IMS_hourly_{}hr.nc'.format(freq))
    return dss


def read_one_ims_hourly_station_csv(file):
    import pandas as pd
    from aux_gps import xr_reindex_with_date_range
    from aux_gps import rename_data_vars
    name = file.as_posix().split('/')[-1].split('_')[0]
    sid = file.as_posix().split('/')[-1].split('_')[1]
    freq = file.as_posix().split('/')[-1].split('_')[2]
    freq = ''.join([x for x in freq if x.isdigit()]) + 'H'
    array_name = '_'.join([name, sid])
    print('reading {} station...'.format(array_name))
    df = pd.read_csv(file, index_col='time')
    df.index = pd.to_datetime(df.index)
    df.drop(labels=['Unnamed: 0', 'name'], axis=1, inplace=True)
    lat = df.loc[:, 'lat'][0]
    lon = df.loc[:, 'lon'][0]
    height = df.loc[:, 'height'][0]
    df.drop(labels=['lat', 'lon', 'height'], axis=1, inplace=True)
    ds = df.to_xarray()
    station_attrs = {
        'station_id': sid,
        'lat': lat,
        'lon': lon,
        'height': height}
    names_units_attrs = {
        'ps': {
            'long_name': 'surface_pressure', 'units': 'hPa'},
        'tas': {
            'long_name': 'surface_temperature', 'units': 'degC'},
        'rh': {
                'long_name': 'relative_humidity', 'units': '%'},
        'wind_dir': {
                    'long_name': 'wind_direction', 'units': 'deg'},
        'wind_spd': {
                        'long_name': 'wind_speed', 'units': 'm/s'}}
    to_drop = []
    for da in ds:
        # add var names and units:
        attr = names_units_attrs.get(da, {})
        ds[da].attrs = attr
        # add station attrs for each var:
        ds[da].attrs.update(station_attrs)
#        # rename var to include station name:
#        ds[da].name = array_name + '_' + da
        # last, drop all NaN vars:
        try:
            ds[da] = xr_reindex_with_date_range(ds[da], freq=freq)
        except ValueError:
            to_drop.append(da)
            continue
#        if ds[da].size == ds[da].isnull().sum().item():
#            to_drop.append(da)
    ds = ds[[x for x in ds if x not in to_drop]]
    ds = rename_data_vars(ds, suffix=None, prefix=array_name + '_', verbose=False)
    ds = ds.sortby('time')
#    ds = xr_reindex_with_date_range(ds, freq=freq)
    return ds


def interpolate_hourly_IMS(path=ims_path, freq='03', field='ps', max_gap='6H',
                           station='JERUSALEM-CENTRE-MAN_6770', k_iqr=2,
                           times=['1996', '2019'],
                           plot=True):
    from aux_gps import path_glob
    from aux_gps import xr_reindex_with_date_range
    from aux_gps import keep_iqr
    import xarray as xr
    import matplotlib.pyplot as plt
    file = path_glob(path, 'IMS_hourly_{}hr.nc'.format(freq))[0]
    ds = xr.open_dataset(file)
    name = '{}_{}'.format(station, field)
    da = ds[name]
    da = xr_reindex_with_date_range(da, freq='1H')
    da_inter = da.interpolate_na('time', max_gap=max_gap, method='cubic')
    if times is not None:
        da = da.sel(time=slice(*times))
        da_inter = da_inter.sel(time=slice(*times))
    if k_iqr is not None:
        da_inter = keep_iqr(da_inter, k=k_iqr)
    if plot:
        fig, ax = plt.subplots(figsize=(18, 5))
        df = da.to_dataframe()
        df_inter = da_inter.to_dataframe()
        df_inter.plot(style='b--', ax=ax)
        df.plot(style='b-', marker='o', ax=ax, ms=5)
        ax.legend(*[ax.get_lines()],
                  ['PWV {} max interpolation'.format(max_gap), 'PWV'],
                  loc='best')
    return da_inter


def read_ims_metadata_from_files(path=gis_path, freq='10mins'):
    # for longer climate archive data use filename = IMS_climate_archive_meta_data.xls
    import pandas as pd
    """parse ims stations meta-data"""
    if freq == '10mins':
        filename = 'IMS_10mins_meta_data.xlsx'
        ims = pd.read_excel(path / filename,
                            sheet_name='מטה-דטה', skiprows=1)
        # drop two last cols and two last rows:
        ims = ims.drop(ims.columns[[-1, -2]], axis=1)
        ims = ims.drop(ims.tail(2).index)
        cols = ['#', 'ID', 'name_hebrew', 'name_english', 'east', 'west',
                'lon', 'lat', 'alt', 'starting_date', 'variables', 'model',
                'eq_position', 'wind_meter_height', 'notes']
        ims.columns = cols
        ims.index = ims['#'].astype(int)
        ims = ims.drop('#', axis=1)
        # fix lat, lon cols:
        ims['lat'] = ims['lat'].str.replace(u'\xba', '').astype(float)
        ims['lon'] = ims['lon'].str.replace(u'\xba', '').astype(float)
        # fix alt col:
        ims['alt'] = ims['alt'].replace('~', '', regex=True).astype(float)
        # fix starting date col:
        ims['starting_date'] = pd.to_datetime(ims['starting_date'])
    elif freq == 'hourly':
        filename = 'IMS_climate_archive_meta_data.xls'
        ims = pd.read_excel(path / filename,
                            sheet_name='תחנות אקלים', skiprows=1)
        cols = ['ID', 'name_hebrew', 'name_english', 'station_type', 'east',
                'west', 'lon', 'lat', 'alt', 'starting_date', 'closing_date',
                'date_range']
        ims.columns = cols
        # ims.index = ims['ID'].astype(int)
        # ims = ims.drop('ID', axis=1)
        # fix lat, lon cols:
        ims['lat'] = ims['lat'].str.replace(u'\xba', '').astype(float)
        ims['lon'] = ims['lon'].str.replace(u'\xba', '').astype(float)
        # fix alt col:
        ims['alt'] = ims['alt'].replace('~', '', regex=True).astype(float)
        # fix starting date, closing_date col:
        ims['starting_date'] = pd.to_datetime(ims['starting_date'])
        ims['closing_date'] = pd.to_datetime(ims['closing_date'])
    return ims


def produce_geo_ims(path, freq='10mins',
                    closed_stations=False, plot=True):
    import geopandas as gpd
    import numpy as np
    isr = gpd.read_file(path / 'Israel_and_Yosh.shp')
    isr.crs = {'init': 'epsg:4326'}
    ims = read_ims_metadata_from_files(path=path, freq=freq)
    if closed_stations:
        ims = ims[np.isnat(ims.closing_date)]
    geo_ims = gpd.GeoDataFrame(ims, geometry=gpd.points_from_xy(ims.lon,
                                                                ims.lat),
                               crs=isr.crs)
    if plot:
        ax = isr.plot()
        geo_ims.plot(ax=ax, column='alt', cmap='Reds', edgecolor='black',
                     legend=True)
    return geo_ims


def ims_api_get_meta(active_only=True, channel_name='TD'):
    import requests
    import pandas as pd
    """get meta data on 10mins ims stations"""
    myToken = 'f058958a-d8bd-47cc-95d7-7ecf98610e47'
    headers = {'Authorization': 'ApiToken ' + myToken}
    r = requests.get('https://api.ims.gov.il/v1/envista/stations/',
                     headers=headers)
    stations_10mins = pd.DataFrame(r.json())
    # filter inactive stations:
    if active_only:
        stations_10mins = stations_10mins[stations_10mins.active]
    # arrange lat lon nicely and add channel num for dry temp:
    lat_ = []
    lon_ = []
    channelId_list = []
    for index, row in stations_10mins.iterrows():
        lat_.append(row['location']['latitude'])
        lon_.append(row['location']['longitude'])
        channel = [x['channelId'] for x in row.monitors if x['name'] ==
                   channel_name]
        if channel:
            channelId_list.append(channel[0])
        else:
            channelId_list.append(None)
    stations_10mins['lat'] = lat_
    stations_10mins['lon'] = lon_
    stations_10mins[channel_name + '_channel'] = channelId_list
    stations_10mins.drop(['location', 'StationTarget', 'stationsTag'],
                         axis=1, inplace=True)
    return stations_10mins


#def download_ims_data(geo_df, path, end_date='2019-04-15'):
#    import requests
#    import glob
#    import pandas as pd
#
#    def to_dataarray(df, index, row):
#        import pandas as pd
#        ds = df.to_xarray()
#        ds['time'] = pd.to_datetime(ds.time)
#        channel_name = ds.name.isel(time=0).values
#        channel_id = ds.id.isel(time=0).values
#        ds = ds.drop(['id', 'name'])
#        da = ds.to_array(dim='TD', name=str(index))
#        da.attrs['channel_id'] = channel_id.item()
#        da.attrs['channel_name'] = channel_name.item()
#        da.attrs['station_name'] = row.name_english
#        da.attrs['station_id'] = row.ID
#        da.attrs['station_lat'] = row.lat
#        da.attrs['station_lon'] = row.lon
#        da.attrs['station_alt'] = row.alt
#        return da
#
#    def get_dates_list(starting_date, end_date):
#        """divide the date span into full 1 years and a remainder, tolist"""
#        import numpy as np
#        import pandas as pd
#        end_date = pd.to_datetime(end_date)
#        s_year = starting_date.year
#        e_year = end_date.year
#        years = np.arange(s_year, e_year + 1)
#        dates = [starting_date.replace(year=x) for x in years]
#        if (end_date - dates[-1]).days > 0:
#            dates.append(end_date)
#        return dates
#
#    myToken = 'f058958a-d8bd-47cc-95d7-7ecf98610e47'
#    headers = {'Authorization': 'ApiToken ' + myToken}
#    already_dl = []
#    for paths in glob.glob(path+'*_TD.nc'):
#        already_dl.append(paths.split('/')[-1].split('.')[0].split('_')[0])
#        to_download = list(set(geo_df.index.values.tolist()
#                               ).difference(set(already_dl)))
#    if to_download:
#        geo_df = geo_df.loc[to_download]
#    for index, row in geo_df.iterrows():
#        # get a list of dates to download: (1 year old parts)
#        dates = get_dates_list(row.starting_date, end_date)
#        # get station id and channel id(only dry temperature):
#        name = row.name_english
#        station_id = row.ID
#        channel_id = row.channel_id
#        # if tempertue is not measuered in station , skip:
#        if channel_id == 0:
#            continue
#        print(
#            'Getting IMS data for {} station(ID={}) from channel {}'.format(
#                name,
#                station_id,
#                channel_id))
#        # loop over one year time span and download:
#        df_list = []
#        for i in range(len(dates) - 1):
#            first_date = dates[i].strftime('%Y/%m/%d')
#            last_date = dates[i + 1].strftime('%Y/%m/%d')
#            print('proccesing dates: {} to {}'.format(first_date, last_date))
#            dl_command = ('https://api.ims.gov.il/v1/envista/stations/' +
#                          str(station_id) + '/data/' + str(channel_id) +
#                          '?from=' + first_date + '&to=' + last_date)
#            r = requests.get(dl_command, headers=headers)
#            if r.status_code == 204:  # i.e., no content:
#                print('no content for this search, skipping...')
#                break
#            print('parsing to dataframe...')
#            df_list.append(parse_ims_to_df(r.json()['data']))
#        print('concatanating df and transforming to xarray...')
#        df_all = pd.concat(df_list)
#        # only valid results:
#        # df_valid = df_all[df_all['valid']]
#        df_all.index.name = 'time'
#        da = to_dataarray(df_all, index, row)
#        filename = index + '_TD.nc'
#        comp = dict(zlib=True, complevel=9)  # best compression
#        encoding = {var: comp for var in da.to_dataset().data_vars}
#        print('saving to {} to {}'.format(filename, path))
#        da.to_netcdf(path + filename, 'w', encoding=encoding)
#        print('done!')
#    #    return df_list
#    # pick station and time span
#    # download
#    # call parse_ims_to_df
#    # concatanate and save to nc
#    return


def fill_fix_all_10mins_IMS_stations(path=ims_10mins_path,
                                     savepath=ims_path,
                                     unique_index=True, field='TD',
                                     clim='dayofyear', fix_only=False):
    """loop over all TD 10mins stations and first fix their lat/lon/alt from
    metadata file and then fill them with clim, then save them
    use specific station names to slice irrelevant data"""
    import xarray as xr
    from aux_gps import path_glob
    from aux_gps import get_unique_index
    # TODO: redo this analysis with adding the hourly TD data
    meta = read_ims_metadata_from_files(freq='10mins')
    files = path_glob(path, '*{}_10mins.nc'.format(field))
    cnt = 1
    da_list = []
    for file_and_path in files:
        name = file_and_path.as_posix().split('/')[-1].split('.')[0].split('_')[0]
        print('post-proccessing {} data for {} station, ({}/{})'.format(field,
              name, cnt, len(files)))
        da = xr.open_dataarray(file_and_path)
        sid = da.attrs['station_id']
        row = meta[meta.ID == sid]
        if da.name == 'ARIEL':
            da = da.loc['2000-09-01':]
            print('{} station is sliced!'.format(da.name))
        elif da.name == 'TEL-YOSEF-20141223':
            da = da.loc['2007-10-01':]
            row = meta[meta.ID == 380]
            print('{} station is sliced and fixed!'.format(da.name))
        elif da.name == 'PARAN-20060124':
            da = da.loc['1995-04-01':]
            row = meta[meta.ID == 207]
            print('{} station is fixed!'.format(da.name))
        elif da.name == 'MIZPE-RAMON-20120927':
            row = meta[meta.ID == 379]
            print('{} station is fixed!'.format(da.name))
        elif da.name == 'SHANI':
            da = da.loc['1995-12-01':]
            print('{} station is sliced!'.format(da.name))
        elif da.name == 'BET-ZAYDA':
            da = da.loc['1995-12-01':]
            print('{} station is sliced!'.format(da.name))
        elif da.name == 'BEER-SHEVA-UNI':
            print('skipping {} station...'.format(da.name))
            continue
        no_row_in_meta = row.empty
        # assert not no_row_in_meta
        if field == 'Rain':
            if da.name == 'YOTVATA':
                da = da.loc['2009-09-01':]
                print('{} station is sliced!'.format(da.name))
            elif da.name == 'ELAT':
                da = da.loc['2002-11-25':]
                print('{} station is sliced!'.format(da.name))
            elif da.name == 'ELON':
                da = da.loc['1999-02-01':]
                print('{} station is sliced!'.format(da.name))
            elif da.name == 'QEVUZAT-YAVNE':
                da = da.loc['2000-02-05':]
                print('{} station is sliced!'.format(da.name))
            elif da.name == 'ZOMET-HANEGEV':
                da = da.loc['2005-11-21':]
                print('{} station is sliced!'.format(da.name))
            elif da.name == 'JERUSALEM-CENTRE':
                da = da.loc['1995-11-13':]
                print('{} station is sliced!'.format(da.name))
            elif da.name == 'NETIV-HALAMED-HE':
                da = da.loc['1995-10-15':]
                print('{} station is sliced!'.format(da.name))
            elif da.name == 'GAT':
                da = da.loc['2007-10-01':]
                print('{} station is sliced!'.format(da.name))
            elif da.name == 'AVNE-ETAN':
                da = da.loc['1993-07-01':]
                print('{} station is sliced!'.format(da.name))
            elif da.name == 'ROSH-HANIQRA':
                da = da.loc['2007-09-01':]
                print('{} station is sliced!'.format(da.name))
            elif da.name == 'TAVOR-KADOORIE':
                da = da.loc['1995-01-15':]
                print('{} station is sliced!'.format(da.name))
            elif da.name == 'EN-KARMEL':
                da = da.loc['1993-12-01':]
                print('{} station is sliced!'.format(da.name))
        if no_row_in_meta:
            print('{} not exist in meta'.format(da.name))
        else:
            da.attrs['station_lat'] = row.lat.values.item()
            da.attrs['station_lon'] = row.lon.values.item()
            da.attrs['station_alt'] = row.alt.values.item()
        if field == 'TD' and not fix_only:
            fill_missing_single_ims_station(da, unique_index=unique_index,
                                            clim_period=clim, savepath=path,
                                            verbose=False)
        elif field == 'TD' and fix_only:
            if unique_index:
                ind_diff = da.size - get_unique_index(da).size
                da = get_unique_index(da)
                if ind_diff > 0:
                    print('dropped {} non-unique datetime index.'.format(ind_diff))
            da_list.append(da)
        else:
            if unique_index:
                ind_diff = da.size - get_unique_index(da).size
                da = get_unique_index(da)
                if ind_diff > 0:
                    print('dropped {} non-unique datetime index.'.format(ind_diff))
            da_list.append(da)
        cnt += 1
    if field == 'TD' and not fix_only:
        print('Done filling all stations!')
        files = path_glob(path, '*TD_10mins_filled.nc')
        dsl = [xr.open_dataarray(file) for file in files]
    elif field == 'TD' and fix_only:
        dsl = da_list
    else:
        dsl = da_list
    print('merging all files...')
    dsl = [x.dropna('time') for x in dsl]
    ds = xr.merge(dsl)
    if savepath is not None:
        if field == 'TD' and not fix_only:
            filename = 'IMS_TD_israeli_10mins_filled.nc'
        elif field == 'TD' and fix_only:
            filename = 'IMS_{}_israeli_10mins.nc'.format(field)
        else:
            filename = 'IMS_{}_israeli_10mins.nc'.format(field)
        print('saving {} to {}'.format(filename, savepath))
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(savepath / filename, 'w', encoding=encoding)
        print('Done!')
    return ds


#def produce_T_dataset(path=ims_10mins_path, savepath=None, unique_index=True,
#                      clim_period='dayofyear'):
#    import xarray as xr
#    from aux_gps import path_glob
#    da_list = []
#    for file_and_path in path_glob(path, '*TD_10mins.nc'):
#        da = xr.open_dataarray(file_and_path)
#        print('post-proccessing temperature data for {} station'.format(da.name))
#        da_list.append(fill_missing_single_ims_station(da, unique_index,
#                                                       clim_period))
#    ds = xr.merge(da_list)
#    if savepath is not None:
#        filename = 'IMS_TD_israeli_10mins_filled.nc'
#        print('saving {} to {}'.format(filename, savepath))
#        comp = dict(zlib=True, complevel=9)  # best compression
#        encoding = {var: comp for var in ds.data_vars}
#        ds.to_netcdf(savepath / filename, 'w', encoding=encoding)
#        print('Done!')
#    return ds


def fill_missing_single_ims_station(da, unique_index=True,
                                    clim_period='dayofyear', savepath=None,
                                    verbose=True):
    """fill in the missing time data for the ims station of any variable with
    clim_period is the fine tuning of the data replaced, options are:
        month, weekofyear, dayofyear. return a dataset with original and filled
        dataarray"""
    # da should be dattaarray and not dataset!
    import pandas as pd
    import numpy as np
    import xarray as xr
    from aux_gps import get_unique_index
    print('filling in missing data for {}'.format(da.name))
    if unique_index:
        ind_diff = da.size - get_unique_index(da).size
        da = get_unique_index(da)
        if verbose:
            print('dropped {} non-unique datetime index.'.format(ind_diff))
    # make sure no coords are in xarray:
    da = da.reset_coords(drop=True)
    # make sure nans are dropped:
    nans_diff = da.size - da.dropna('time').size
    if verbose:
        print('dropped {} nans.'.format(nans_diff))
    da_no_nans = da.dropna('time')
    if clim_period == 'month':
        grpby = 'time.month'
        if verbose:
            print('long term monthly mean data replacment selected')
    elif clim_period == 'weekofyear':
        if verbose:
            print('long term weekly mean data replacment selected')
        grpby = 'time.weekofyear'
    elif clim_period == 'dayofyear':
        if verbose:
            print('long term daily mean data replacment selected')
        grpby = 'time.dayofyear'
    # first compute the climatology and the anomalies:
    if verbose:
        print('computing anomalies:')
    climatology = da_no_nans.groupby(grpby).mean('time')
    anom = da_no_nans.groupby(grpby) - climatology
    # then comupte the diurnal cycle:
    if verbose:
        print('computing diurnal change:')
    diurnal = anom.groupby('time.hour').mean('time')
    # assemble old and new time and comupte the difference:
    if verbose:
        print('assembeling missing data:')
    old_time = pd.to_datetime(da_no_nans.time.values)
    freq = pd.infer_freq(da.time.values)
    new_time = pd.date_range(da_no_nans.time.min().item(),
                             da_no_nans.time.max().item(), freq=freq)
    missing_time = pd.to_datetime(
        sorted(
            set(new_time).difference(
                set(old_time))))
    missing_data = np.empty((missing_time.shape))
    if verbose:
        print('proccessing missing data...')
    for i in range(len(missing_data)):
        # replace data as to monthly long term mean and diurnal hour:
        # missing_data[i] = (climatology.sel(month=missing_time[i].month) +
        missing_data[i] = (climatology.sel({clim_period: getattr(missing_time[i],
                                                                 clim_period)}) +
                           diurnal.sel(hour=missing_time[i].hour))
    series = pd.Series(data=missing_data, index=missing_time)
    series.index.name = 'time'
    mda = series.to_xarray()
    mda.name = da.name
    new_data = xr.concat([mda, da_no_nans], 'time')
    new_data = new_data.sortby('time')
    # copy attrs:
    new_data.attrs = da.attrs
    new_data.attrs['description'] = 'missing data was '\
                                    'replaced by using ' + clim_period \
                                    + ' mean and hourly signal.'
    # put new_data and missing data into a dataset:
    dataset = new_data.to_dataset(name=new_data.name)
    dataset[new_data.name + '_original'] = da_no_nans
    if verbose:
        print('done!')
    if savepath is not None:
        da = dataset[new_data.name]
        sid = da.attrs['station_id']
        cname = da.attrs['channel_name']
        name = da.name
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in da.to_dataset(name=name).data_vars}
        filename = '{}_{}_{}_10mins_filled.nc'.format(name, sid, cname)
        if verbose:
            print('saving...')
        da.to_netcdf(savepath / filename, 'w', encoding=encoding)
        print('{} was saved to {}.'.format(filename, savepath))
        return
    return dataset

#    # resample to 5min with resample_method: (interpolate is very slow)
#    print('resampling to 5 mins using {}'.format(resample_method))
#    # don't resample the missing data:
#    dataset = dataset.resample(time='5min').ffill()
