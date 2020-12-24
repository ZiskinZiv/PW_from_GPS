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


def calibrate_zwd_with_ts_tm_from_deserve(path=work_yuval, des_path=des_path, ims_path=ims_path):
    import xarray as xr
    from PW_stations import ml_models_T_from_sounding
    from PW_stations import produce_kappa_ml_with_cats
    radio = xr.load_dataset(des_path/'massada_deserve_PW_Tm_Ts_2014-2014.nc')
    mda = ml_models_T_from_sounding(
        physical_file=radio, times=None, station='massada')
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


    