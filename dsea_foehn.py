#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:20:40 2020

@author: shlomi
"""
from PW_stations import work_yuval
from PW_paths import savefig_path
des_path = work_yuval / 'deserve'
ims_path = work_yuval / 'IMS_T'
dsea_gipsy_path = work_yuval / 'dsea_gipsyx'


def calculate_zenith_hydrostatic_delay_dsea(ims_path=ims_path):
    from PW_stations import calculate_ZHD
    from PW_stations import produce_geo_gnss_solved_stations
    from aux_gps import xr_reindex_with_date_range
    import xarray as xr
    pres = xr.open_dataset(ims_path / 'IMS_BP_israeli_10mins.nc')['SEDOM']
    p_sta_ht_km = pres.attrs['station_alt'] / 1000
    df = produce_geo_gnss_solved_stations(plot=False)
    lat = df.loc['dsea', 'lat']
    ht = df.loc['dsea', 'alt']
    zhd = calculate_ZHD(pres, lat=lat, ht_km=ht/1000,
                        pressure_station_height_km=p_sta_ht_km)
    zhd = xr_reindex_with_date_range(zhd, freq='5T')
    zhd = zhd.interpolate_na('time', max_gap='1H', method='linear')
    return zhd


def calibrate_zwd_with_ts_tm_from_deserve(path=work_yuval, des_path=des_path,
                                          ims_path=ims_path, zwd=None):
    import xarray as xr
    from PW_stations import ml_models_T_from_sounding
    from PW_stations import produce_kappa_ml_with_cats
    radio = xr.load_dataset(des_path/'massada_deserve_PW_Tm_Ts_2014-2014.nc')
    mda = ml_models_T_from_sounding(
        physical_file=radio, times=None, station='massada')
    if zwd is not None:
        dsea_zwd = zwd
    else:
        dsea_zwd = xr.open_dataset(path / 'ZWD_unselected_israel_1996-2020.nc')['dsea']
        dsea_zwd.load()
    ts = xr.open_dataset(ims_path / 'GNSS_5mins_TD_ALL_1996_2020.nc')['dsea']
    k, dk = produce_kappa_ml_with_cats(ts, mda=mda, model_name='TSEN')
    dsea = k * dsea_zwd
    return dsea


def compare_radio_and_wrf_massada(des_path=des_path, plot=True):
    import xarray as xr
    from aux_gps import dim_intersection
    import matplotlib.pyplot as plt
    radio = xr.load_dataset(des_path/'massada_deserve_PW_Tm_Ts_2014-2014.nc')
    radio = radio['PW'].rename({'sound_time': 'time'}).to_dataset(name='radiosonde')
    wrf = get_wrf_pw_at_dsea_gnss_coord(point=[31.3177, 35.3725])
    wrf = wrf['pw'].rename({'Time': 'time'}).to_dataset(name='wrf')
    # new_time = dim_intersection([ds, wrf])
    # wrf = wrf.sel(time=new_time)
    # ds = ds.sel(time=new_time)
    radio88 = radio.sel(time=slice('2014-08-07', '2014-08-08'))
    wrf88 = wrf.sel(time=slice('2014-08-07', '2014-08-08'))
    wrf1517 = wrf.sel(time=slice('2014-08-14', '2014-08-17'))
    radio1517 = radio.sel(time=slice('2014-08-14', '2014-08-17'))
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        radio88.to_dataframe()['radiosonde'].plot(ax=axes[1], marker='s', lw=2, markersize=5)
        wrf88.to_dataframe()['wrf'].plot(ax=axes[1], marker='o', lw=2, markersize=2)
        radio1517.to_dataframe()['radiosonde'].plot(ax=axes[0], marker='s', lw=2, markersize=5)
        wrf1517.to_dataframe()['wrf'].plot(ax=axes[0], marker='o', lw=2, markersize=2)
        axes[0].set_ylim(10, 45)
        axes[0].set_ylabel('PWV [mm]')
        axes[0].grid()
        axes[0].legend()
        axes[1].set_ylim(10, 45)
        axes[1].set_ylabel('PWV [mm]')
        axes[1].grid()
        axes[1].legend()
        fig.suptitle('WRF vs. Radiosonde PWV at massada station (31.3177N, 35.3725E)')
        fig.tight_layout()
    return


def compare_gnss_dsea_with_wrf(des_path=des_path, work_path=work_yuval, dsea_da=None, plot=True):
    import xarray as xr
    from aux_gps import dim_intersection
    import matplotlib.pyplot as plt
    if dsea_da is not None:
        gnss = dsea_da
    else:
        gnss = xr.open_dataset(work_path / 'GNSS_PW_thresh_50.nc')['dsea']
    gnss = gnss.to_dataset(name='gnss')
    wrf = get_wrf_pw_at_dsea_gnss_coord(des_path, work_path)
    wrf = wrf['pw'].rename({'Time': 'time'}).to_dataset(name='wrf')
    # new_time = dim_intersection([gnss, wrf])
    # wrf = wrf.sel(time=new_time)
    # ds = ds.sel(time=new_time)
    # ds['wrf'] = wrf
    gnss79 = gnss.sel(time=slice('2014-08-07', '2014-08-08'))
    gnss1517 = gnss.sel(time=slice('2014-08-15', '2014-08-16'))
    wrf79 = wrf.sel(time=slice('2014-08-07', '2014-08-08'))
    wrf1517 = wrf.sel(time=slice('2014-08-15', '2014-08-16'))

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        gnss79.to_dataframe()['gnss'].plot(ax=axes[1], lw=2)
        wrf79.to_dataframe()['wrf'].plot(ax=axes[1], lw=2)
        gnss1517.to_dataframe()['gnss'].plot(ax=axes[0], lw=2)
        wrf1517.to_dataframe()['wrf'].plot(ax=axes[0], lw=2)
        axes[0].set_ylim(10, 40)
        axes[0].set_ylabel('PWV [mm]')
        axes[0].grid()
        axes[0].legend()
        axes[1].set_ylim(10, 40)
        axes[1].set_ylabel('PWV [mm]')
        axes[1].grid()
        axes[1].legend()
        fig.suptitle('WRF vs. GNSS PWV at DSEA station (31.037N, 35.369E), ~31 km south to massada st.')
        fig.tight_layout()
    return


def get_wrf_pw_at_dsea_gnss_coord(path=des_path, work_path=work_yuval, point=None):
    from PW_stations import produce_geo_gnss_solved_stations
    import xarray as xr
    from aux_gps import get_nearest_lat_lon_for_xy
    from aux_gps import path_glob
    from aux_gps import get_unique_index
    df = produce_geo_gnss_solved_stations(path=work_path / 'gis', plot=False)
    dsea_point = df.loc['dsea'][['lat', 'lon']].astype(float).values
    files = path_glob(path, 'pw_wrfout*.nc')
    pw_list = []
    for file in files:
        pw_all = xr.load_dataset(file)
        freq = xr.infer_freq(pw_all['Time'])
        print(freq)
        if point is not None:
            print('looking for {} at wrf.'.format(point))
            dsea_point = point
        loc = get_nearest_lat_lon_for_xy(pw_all['XLAT'], pw_all['XLONG'], dsea_point)
        print(loc)
        pw = pw_all.isel(south_north=loc[0][0], west_east=loc[0][1])
        pw_list.append(pw)
    pw_ts = xr.concat(pw_list, 'Time')
    pw_ts = get_unique_index(pw_ts, dim='Time')
    return pw_ts


def load_wrf_output_and_save_field(path=des_path, varname="pw", savepath=None):
    """
    load WRF output field and save it to savepath

    Parameters
    ----------
    path : Path() or str, optional
        the WRF loadpath. The default is des_path.
    varname : str, optional
        can be 'temp', 'pres', etc.. The default is 'pw'.
    savepath : Path() or str, optional
        The field savepath. The default is None.

    Returns
    -------
    var_list : list
        field dataarrays list.

    """
    import wrf
    import xarray as xr
    from aux_gps import path_glob
    from aux_gps import save_ncfile
    files = path_glob(path, 'wrfout_*.nc')
    var_list = []
    for file in files:
        ds = xr.open_dataset(file)
        wrfin = ds._file_obj.ds
        wrfvar = wrf.getvar(wrfin=wrfin, varname=varname, timeidx=wrf.ALL_TIMES)
        if savepath is not None:
            if wrfvar.attrs['projection'] is not None:
                wrfvar.attrs['projection'] = wrfvar.attrs['projection'].proj4()
            filename_to_save = '{}_{}'.format(varname, file.as_posix().split('/')[-1])
            save_ncfile(wrfvar, savepath, filename_to_save)
        var_list.append(wrfvar)
    return var_list


def get_pwv_dsea_foehn_paper(pwv_dsea, pwv_dsea_error=None, plot=True,
                             xlims=(13, 19), ylims=(10,50), save=True):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    dsea8=pwv_dsea.sel(time='2014-08-8')
    dsea16=pwv_dsea.sel(time='2014-08-16')
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(6, 8))
        dsea8.plot(ax=axes[1], color='k', lw=2)
        dsea16.plot(ax=axes[0], color='k', lw=2)
        if xlims is not None:
            xlims8 = [pd.to_datetime('2014-08-08T{}:00:00'.format(xlims[0])),
                      pd.to_datetime('2014-08-08T{}:00:00'.format(xlims[1]))]
            xlims16 = [pd.to_datetime('2014-08-16T{}:00:00'.format(xlims[0])),
                      pd.to_datetime('2014-08-16T{}:00:00'.format(xlims[1]))]
            axes[1].set_xlim(*xlims8)
            axes[0].set_xlim(*xlims16)
        if pwv_dsea_error is not None:
            dsea8_error=pwv_dsea_error.sel(time='2014-08-8')
            dsea16_error=pwv_dsea_error.sel(time='2014-08-16')
            dsea8_h = (dsea8 + dsea8_error).values
            dsea8_l = (dsea8 - dsea8_error).values
            dsea16_h = (dsea16 + dsea16_error).values
            dsea16_l = (dsea16 - dsea16_error).values
            axes[1].fill_between(dsea8['time'].values, dsea8_l, dsea8_h,
                                 where=np.isfinite(dsea8.values),
                                 alpha=0.7)
            axes[0].fill_between(dsea16['time'].values, dsea16_l, dsea16_h,
                                 where=np.isfinite(dsea16.values),
                                 alpha=0.7)
        axes[0].grid()
        axes[1].grid()
        axes[0].set_xlabel('UTC')
        axes[1].set_xlabel('UTC')
        axes[0].set_ylim(*ylims)
        axes[1].set_ylim(*ylims)
        fig.tight_layout()
        fig.suptitle('GNSS DSEA PWV - 2014')
        fig.subplots_adjust(top=0.95)
        if save:
            filename = 'gnss_pwv_dsea_foehn_2014-08-08_16.png'
            plt.savefig(savefig_path / filename, orientation='portrait')
    return fig


def read_all_WRF_GNSS_files(path=des_path, var='pw', point=None):
    from aux_gps import path_glob
    from aux_gps import get_nearest_lat_lon_for_xy
    import xarray as xr
    import wrf
    files = path_glob(path, 'wrfout_d04_*_GNSS.nc')
    dsl = [xr.open_dataset(file) for file in files]
    if var is not None:
        var_list = []
        for ds in dsl:
            wrfin = ds._file_obj.ds
            wrfvar = wrf.getvar(wrfin=wrfin, varname=var, timeidx=wrf.ALL_TIMES)
            var_list.append(wrfvar)
        ds = xr.concat(var_list, 'Time')
        ds = ds.sortby('Time')
    if point is not None:
        print('looking for {} at wrf.'.format(point))
        loc = get_nearest_lat_lon_for_xy(ds['XLAT'], ds['XLONG'], point)
        print(loc)
        ds = ds.isel(south_north=loc[0][0], west_east=loc[0][1])
        ds = ds.sortby('Time')
    # ds = xr.concat(dsl, 'Time')
    return ds


def assemble_WRF_pwv(path=des_path, work_path=work_yuval, radius=1):
    from PW_stations import produce_geo_gnss_solved_stations
    import xarray as xr
    from aux_gps import save_ncfile
    from aux_gps import get_nearest_lat_lon_for_xy
    from aux_gps import get_unique_index
    df = produce_geo_gnss_solved_stations(path=work_path / 'gis', plot=False)
    dsea_point = df.loc['dsea'][['lat', 'lon']].astype(float).values
    if radius is not None:
        point = None
    else:
        point = dsea_point
    wrf_pw = read_all_WRF_GNSS_files(path, var='pw', point=point)
    wrf_pw8 = xr.load_dataarray(path / 'pw_wrfout_d04_2014-08-08_40lev.nc').sel(Time='2014-08-08')
    wrf_pw16 = xr.load_dataarray(path / 'pw_wrfout_d04_2014-08-16_40lev.nc').sel(Time='2014-08-16')
    wrf_pw_8_16 = xr.concat([wrf_pw8, wrf_pw16], 'Time')
    print('looking for {} at wrf.'.format(dsea_point))
    loc = get_nearest_lat_lon_for_xy(wrf_pw_8_16['XLAT'], wrf_pw_8_16['XLONG'], dsea_point)
    print(loc)
    if radius is not None:
        print('getting {} radius around {}.'.format(radius, dsea_point))
        lat_islice = [loc[0][0] - radius, loc[0][0] + radius + 1]
        lon_islice = [loc[0][1] - radius, loc[0][1] + radius + 1]
        wrf_pw_8_16 = wrf_pw_8_16.isel(south_north=slice(*lat_islice), west_east=slice(*lon_islice))
        loc = get_nearest_lat_lon_for_xy(wrf_pw['XLAT'], wrf_pw['XLONG'], dsea_point)
        lat_islice = [loc[0][0] - radius, loc[0][0] + radius + 1]
        lon_islice = [loc[0][1] - radius, loc[0][1] + radius + 1]
        wrf_pw = wrf_pw.isel(south_north=slice(*lat_islice), west_east=slice(*lon_islice))
    else:
        wrf_pw_8_16 = wrf_pw_8_16.isel(south_north=loc[0][0], west_east=loc[0][1])
    wrf_pw = xr.concat([wrf_pw, wrf_pw_8_16], 'Time')
    wrf_pw = wrf_pw.rename({'Time': 'time'})
    wrf_pw = wrf_pw.sortby('time')
    wrf_pw = get_unique_index(wrf_pw)
    if wrf_pw.attrs['projection'] is not None:
        wrf_pw.attrs['projection'] = wrf_pw.attrs['projection'].proj4()
    if radius is not None:
        filename = 'pwv_wrf_dsea_gnss_radius_{}_2014-08.nc'.format(radius)
    else:
        filename = 'pwv_wrf_dsea_gnss_point_2014-08.nc'
    save_ncfile(wrf_pw, des_path, filename)
    return wrf_pw


def compare_WRF_GNSS_pwv(path=des_path, work_path=work_yuval, plot=True):
    import xarray as xr
    import matplotlib.pyplot as plt
    import numpy as np
    from PW_from_gps_figures import plot_mean_with_fill_between_std
    # load GNSS dsea:
    gnss = xr.open_dataset(
        work_path / 'GNSS_PW_thresh_50_homogenized.nc')['dsea']
    # load WRF:
    wrf = xr.load_dataset(path / 'pwv_wrf_dsea_gnss_point_2014-08.nc')
    ds = xr.Dataset()
    ds['WRF'] = wrf['pw']
    ds['GNSS'] = gnss
    ds = ds.reset_coords(drop=True)
    if plot:
        fig, ax = plt.subplots(figsize=(18, 5.5))
        ds.to_dataframe().plot(ax=ax)
        ax.grid()
        ax.set_ylabel('PWV [mm]')
        fig.suptitle(
            'PWV Time series comparison (2014-08) between GNSS and WRF at DSEA point')
        fig.tight_layout()
        fig, ax = plt.subplots(figsize=(9, 7))
        ds_hour = ds.groupby('time.hour').mean()
        wrfln = plot_mean_with_fill_between_std(ds['WRF'], grp='hour', mean_dim='time', ax=ax,
                                                color='tab:blue', marker='s', alpha=0.2)
        gnssln = plot_mean_with_fill_between_std(ds['GNSS'], grp='hour', mean_dim='time', ax=ax,
                                                 color='tab:orange', marker='o', alpha=0.2)
        ax.xaxis.set_ticks(np.arange(0, 24, 1))
        ax.legend(wrfln+gnssln, ['WRF', 'GNSS'])
        ax.grid()
        ax.set_ylabel('PWV [mm]')
        ax.set_xlabel('hour of the day [UTC]')
        fig.suptitle(
            'Diurnal PWV comparison (2014-08) between GNSS and WRF at DSEA point')
        fig.tight_layout()
    return ds


def read_all_final_tdps_dsea(path=dsea_gipsy_path/'results', return_mean=True,
                             dryz=False):
    from aux_gps import path_glob
    from gipsyx_post_proc import process_one_day_gipsyx_output
    import xarray as xr
    files = path_glob(path, 'dsea*_smoothFinal.tdp')
    df_list = []
    for file in files:
        try:
            df, _ = process_one_day_gipsyx_output(file, dryz=dryz)
            if dryz:
                df_list.append(df[['DryZ', 'WetZ']])
            else:
                df_list.append(df['WetZ'])
        except AssertionError:
            continue
    # dts = []
    da_list = []
    for i, df in enumerate(df_list):
        # dt = df.index[0]
        # dts.append(dt)
        # df.index = df.index - dt
        da = df.to_xarray()
        if dryz:
            da = da.rename({'WetZ': 'dsea_WetZ_{}'.format(i), 'DryZ': 'dsea_DryZ_{}'.format(i)})
        else:
            da.name = 'dsea_WetZ_{}'.format(i)
        da_list.append(da)
    ds = xr.merge(da_list)
    # ds['datetime'] = dts
    # ds = ds.sortby('datetime')
    if return_mean:
        if dryz:
            dry = [x for x in ds if 'Dry' in x]
            dry_da = ds[dry].to_array('s').mean('s')
            dry_da.name = 'DryZ_mean'
        wet = [x for x in ds if 'Wet' in x]
        wet_da = ds[wet].to_array('s').mean('s')
        wet_da.name = 'WetZ_mean'
        if dryz:
            ds = xr.merge([dry_da, wet_da])
            ds['Total_mean'] = ds['DryZ_mean'] + ds['WetZ_mean']
        else:
            return wet_da
    return ds


def plot_gpt2_vmf1_means(path=dsea_gipsy_path):
    import xarray as xr
    import matplotlib.pyplot as plt
    zhd = calculate_zenith_hydrostatic_delay_dsea().sel(time='2014-08')
    gpt2_path = path / 'results-gpt2'
    vmf1_path = path / 'results-vmf1'
    gpt2 = read_all_final_tdps_dsea(gpt2_path, dryz=True)
    vmf1 = read_all_final_tdps_dsea(vmf1_path, dryz=True)
    dss = xr.concat([gpt2, vmf1], 'model')
    dss['model'] = ['GPT2', 'VMF1']
    fg = dss.to_array('delay_type').plot.line(
        row='delay_type', hue='model', sharey=False, figsize=(20, 17))
    for ax in fg.axes.flat:
        ax.set_ylabel('Path Delay [cm]')
        ax.grid()
    ln = zhd.plot(ax=fg.axes.flat[0])
    fg.axes.flat[0].legend(ln, ['empirical ZHD'])
    fg.fig.tight_layout()
    fg.fig.suptitle('DSEA')
    zwd = dss['Total_mean'] - zhd
    zwd.name = 'WetZ_after_eZHD_subtraction'
    ds = zwd.to_dataset('model')
    ds1, _ = compare_all_zwd(models=['GMF'])
    ds['GMF'] = ds1['WetZ_mean_GMF']
    fig, ax = plt.subplots(figsize=(18, 6))
    ds.to_array('model').plot.line(hue='model', ax=ax)
    ax.grid()
    return fg


def get_dryz_from_one_file(file):
    from aux_gps import line_and_num_for_phrase_in_file
    import re
    i, line = line_and_num_for_phrase_in_file('DryZ', file)
    zhd = re.findall("\d+\.\d+", line)[0]
    return float(zhd)


def get_dryz_from_one_station(path=dsea_gipsy_path/'results'):
    from aux_gps import path_glob
    from aux_gps import get_timedate_and_station_code_from_rinex
    import xarray as xr
    files = sorted(path_glob(path, '*_debug.tree'))
    dt_list = []
    zhd_list = []
    for file in files:
        rfn = file.as_posix().split('/')[-1][0:12]
        dt = get_timedate_and_station_code_from_rinex(rfn, just_dt=True)
        # print('datetime {}'.format(dt.strftime('%Y-%m-%d')))
        dt_list.append(dt)
        zhd = get_dryz_from_one_file(file)
        zhd_list.append(zhd)
    zhd_da = xr.DataArray(zhd_list, dims=['time'])
    zhd_da['time'] = dt_list
    zhd_da *= 100
    # zhd_da.name = station
    zhd_da.attrs['units'] = 'cm'
    zhd_da.attrs['long_name'] = 'Zenith Hydrostatic Delay'
    zhd_da = zhd_da.sortby('time')
    return zhd_da


def compare_all_zwd(path=dsea_gipsy_path,
                    models=['GPT2', 'VMF1', 'WAAS', 'NEILL', 'GMF']):
    import xarray as xr
    da_list = []
    dry_list = []
    for model in models:
        p = path / 'results-{}'.format(model)
        da = read_all_final_tdps_dsea(path=p, return_mean=True,
                                      dryz=False)
        da_dry = get_dryz_from_one_station(p)
        da_dry.name = 'DryZ_{}'.format(model)
        dry_list.append(da_dry)
        da.name = 'WetZ_mean_{}'.format(model)
        da_list.append(da)
    ds = xr.merge(da_list)
    ds_dry = xr.merge(dry_list)
    return ds, ds_dry

