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
dem_path = work_yuval / 'AW3D30'
axis_path = work_yuval/'axis'

dsea_points = {'SEDOM': [31.0306, 35.3919], 'OPERA': [35.3725, 31.3177],
               'SOI-GNSS': [31.03691605029973, 35.36882566157563],
               'AXIS-GNSS': [31.153660742415386, 35.36488575585163]}


def plot_IMS_WRF_at_SEDOM(path=des_path, ims_path=ims_path):
    import matplotlib.pyplot as plt
    import xarray as xr
    # import pandas as pd
    wrf = process_wrf_data_at_point_on_both_dsea_foehns(path=path, point='SEDOM')
    ims = process_IMS_data_at_station_on_dsea_foehn_dates(path=path, ims_path=ims_path, station='SEDOM')
    fg = xr.plot.FacetGrid(wrf, col='date', row='var', sharex=True, sharey=False, figsize=(10, 9.5))
    var_order = ['WS', 'WD', 'T', 'RH', 'PWV']
    for i, row_ax in enumerate(fg.axes):
        ax8 = row_ax[0]
        ax16 = row_ax[1]
        wrf8 = wrf.sel(date='2014-08-08')
        wrf16 = wrf.sel(date='2014-08-16')
        ims8 = ims.sel(date='2014-08-08')
        ims16 = ims.sel(date='2014-08-16')
        wrf8.sel(var=var_order[i]).plot(ax=ax8, color='tab:red', label=['WRF'])
        wrf16.sel(var=var_order[i]).plot(ax=ax16, color='tab:red')
        ims8.sel(var=var_order[i]).plot(ax=ax8, color='tab:blue', label=['IMS'])
        ims16.sel(var=var_order[i]).plot(ax=ax16, color='tab:blue')
        ax8.grid(True)
        handles, labels = ax8.get_legend_handles_labels()
        ax16.grid(True)
        ax8.set_xlabel('')
        ax16.set_xlabel('')
        if i == 4:
            ax8.set_xlabel('Time [UTC]')
            ax16.set_xlabel('Time [UTC]')
    [fg.axes[0][x].set_ylim(0.0, 11) for x in [0, 1]]
    fg.axes[0][0].set_ylabel('Wind Speed [m/s]')
    [fg.axes[1][x].set_ylim(0, 360) for x in [0, 1]]
    fg.axes[1][0].set_ylabel(r'Wind Direction [$\degree$]')
    [fg.axes[2][x].set_ylim(28.5, 41.5) for x in [0, 1]]
    fg.axes[2][0].set_ylabel(r'Surface Temperature [$\degree$C]')
    [fg.axes[3][x].set_ylim(17.5, 60) for x in [0, 1]]
    fg.axes[3][0].set_ylabel('Relative Humidity [%]')
    [fg.axes[4][x].set_ylim(15, 27) for x in [0, 1]]
    fg.axes[4][0].set_ylabel('PWV [mm]')
    # fg.fig.tight_layout()
    # legend:
    fg.fig.legend(handles=handles, labels=['WRF', 'IMS'], prop={'size': 14}, edgecolor='k',
                  framealpha=0.5, fancybox=True, facecolor='white',
                  ncol=2, fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 1.005),
                  bbox_transform=plt.gcf().transFigure)
    fg.fig.suptitle('WRF vs. IMS vars at SEDOM point, PWV (IMS) from SOI-GNSS 2.3 kms west of SEDOM', y=0.95)
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(top=0.906)
    return


def process_wrf_data_at_point_on_both_dsea_foehns(path=des_path, point='SEDOM'):
    import numpy as np
    import xarray as xr
    import pandas as pd
    da8 = process_wrf_data_at_point_on_dsea_foehn_dates(
        path=path, dsea_point=point, date='2014-08-08')
    da16 = process_wrf_data_at_point_on_dsea_foehn_dates(
        path=path, dsea_point=point, date='2014-08-16')
    da8 = da8.sel(time=slice('2014-08-08T13:00:00',
                             '2014-08-08T19:00:00'))
    da8['time'] = np.linspace(13, 19, len(da8['time']))
    # da8['time'] = da8['time'].dt.time
    da16 = da16.sel(time=slice('2014-08-16T13:00:00',
                               '2014-08-16T19:00:00'))
    da16['time'] = np.linspace(13, 19, len(da16['time']))
    # da16['time'] = da16['time'].dt.time
    dss = xr.concat([da8, da16],'date')
    # dss['date'] = [pd.to_datetime(x).date for x in ['2014-08-08', '2014-08-16']]
    dss['date'] = ['2014-08-08', '2014-08-16']
    dss = dss.reset_coords(drop=True)
    return dss


def process_wrf_data_at_point_on_dsea_foehn_dates(path=des_path, dsea_point='SEDOM',
                                                  date='2014-08-08'):
    from aux_gps import convert_wind_direction
    point = dsea_points.get(dsea_point)
    ds = get_wrf_vars_with_a_specific_point(
        path/'wrfout_d04_{}_40lev_T2_U10_V10_pw_rh2.nc'.format(date),
        point=point, name=dsea_point)
    ds = ds.rename({'Time': 'time'})
    ws, wd = convert_wind_direction(u=ds['U10'], v=ds['V10'])
    ds = ds.rename({'U10': 'WS', 'V10': 'WD', 'T2': 'T',
                    'rh2': 'RH', 'pw': 'PWV'})
    ds['WD'] = wd
    ds['WS'] = ws
    ds['T'] -= 273.15
    ds['T'].attrs['units'] = 'deg_C'
    da = ds.to_array('var')
    da.attrs['description'] = 'WRF output vars'
    da.attrs['dsea_point_name'] = dsea_point
    da.attrs['dsea_point'] = point
    return da


def process_IMS_data_at_station_on_dsea_foehn_dates(ims_path=ims_path, station='SEDOM', path=des_path,
                                                    times=['2014-08-01', '2014-08-31']):
    import xarray as xr
    import pandas as pd
    import numpy as np
    # import matplotlib.pyplot as plt
    ws = xr.open_dataset(
        ims_path/'IMS_WS_israeli_10mins.nc')[station].sel(time=slice(*times))
    wd = xr.open_dataset(
        ims_path/'IMS_WD_israeli_10mins.nc')[station].sel(time=slice(*times))
    rh = xr.open_dataset(
        ims_path/'IMS_RH_israeli_10mins.nc')[station].sel(time=slice(*times))
    ts = xr.open_dataset(
        ims_path/'IMS_TD_israeli_10mins.nc')[station].sel(time=slice(*times))
    ds = xr.Dataset()
    ds['WS'] = ws
    ds['WD'] = wd
    ds['T'] = ts
    ds['RH'] = rh
    # convert to UTC, since IMS is always UTC+2
    new_time = ds['time'] - pd.Timedelta(2, units='H')
    new_time = new_time.dt.round('s')
    ds['time'] = new_time
    ds['PWV'] = xr.load_dataset(path / 'DSEA_PWV_GNSS_2014-08.nc')['pwv-soi']
    da8 = ds.sel(time=slice('2014-08-08T13:00:00',
                            '2014-08-08T19:00:00')).to_array('var')
    da8['time'] = np.linspace(13, 19, len(da8['time']))
    # da8['time'] = da8['time'].dt.time
    da16 = ds.sel(time=slice('2014-08-16T13:00:00',
                             '2014-08-16T19:00:00')).to_array('var')
    da16['time'] = np.linspace(13, 19, len(da16['time']))
    # da16['time'] = da16['time'].dt.time
    dss = xr.concat([da8, da16],'date')
    # dss['date'] = [pd.to_datetime(x).date() for x in ['2014-08-08', '2014-08-16']]
    dss['date'] = ['2014-08-08', '2014-08-16']
    return dss


def plot_gnss_and_radiometer_timeseries(path=work_yuval, des_path=des_path):
    import xarray as xr
    import matplotlib.pyplot as plt
    ds = xr.load_dataset(path / 'DSEA_PWV_GNSS_2014-08.nc')
    radio = read_radiometers(des_path)
    ds['pwv-radio'] = radio.resample(time='5T').mean()
    df = ds.to_dataframe().loc['2014-08-04': '2014-08-12']
    fig, ax = plt.subplots(figsize=(16, 5))
    df.plot(ax=ax)
    ax.set_ylabel('PWV [mm]')
    ax.grid()
    fig.tight_layout()
    return fig


def produce_and_save_soi_axis_pwv(axis_path=axis_path, soi_path=dsea_gipsy_path,
                                  ims_path=ims_path, savepath=work_yuval):
    import xarray as xr
    from aux_gps import save_ncfile
    soi_pwv = produce_pwv_from_dsea_axis_station(path=soi_path, ims_path=ims_path)
    axis_pwv = produce_pwv_from_dsea_axis_station(path=axis_path, ims_path=ims_path)
    soi_pwv.attrs['GNSS network'] = 'SOI-APN'
    soi_pwv.attrs['station'] = 'dsea'
    soi_pwv.attrs['units'] = 'mm'
    soi_pwv = soi_pwv.reset_coords(drop=True)
    axis_pwv.attrs['GNSS network'] = 'AXIS'
    axis_pwv = axis_pwv.reset_coords(drop=True)
    axis_pwv.attrs['station'] = 'dsea'
    axis_pwv.attrs['units'] = 'mm'
    ds = xr.Dataset()
    ds['pwv-soi'] = soi_pwv
    ds['pwv-axis'] = axis_pwv
    save_ncfile(ds, savepath, 'DSEA_PWV_GNSS_2014-08.nc')
    return ds



def produce_pwv_from_dsea_axis_station(path=axis_path, ims_path=ims_path):
    """use axis_path = work_yuval/dsea_gispyx for original soi-apn dsea station"""
    import xarray as xr
    from aux_gps import transform_ds_to_lat_lon_alt
    from aux_gps import get_unique_index
    ds = xr.load_dataset(path / 'smoothFinal_2014.nc').squeeze()
    ds = get_unique_index(ds)
    # for now cut:
    if 'axis' in path.as_posix():
        ds = ds.sel(time=slice(None, '2014-08-12'))
    ds = transform_ds_to_lat_lon_alt(ds)
    axis_zwd = ds['WetZ']
    ts = xr.open_dataset(ims_path/'IMS_TD_israeli_10mins.nc')['SEDOM']
    axis_pwv = produce_pwv_from_zwd_with_ts_tm_from_deserve(ts=ts, zwd=axis_zwd)
    if 'axis' in path.as_posix():
        axis_pwv.name = 'AXIS-DSEA'
    else:
        axis_pwv.name = 'SOI-DSEA'
    axis_pwv.attrs['lat'] = ds['lat'].values[0]
    axis_pwv.attrs['lon'] = ds['lon'].values[0]
    axis_pwv.attrs['alt'] = ds['alt'].values[0]
    return axis_pwv


def produce_final_dsea_pwv(ims_station=None, savepath=None, use_pressure=True):
    """use ims_station='SEDOM' to get close ts, pressure"""
    import xarray as xr
    from aux_gps import save_ncfile
    if ims_station is not None:
        ts = xr.open_dataset(ims_path/'IMS_TD_israeli_10mins.nc')[ims_station]
        pres = xr.open_dataset(ims_path/'IMS_BP_israeli_10mins.nc')[ims_station]
        ts.load()
        pres.load()
        ts = ts.resample(time='5T').ffill()
        pres = pres.resample(time='5T', keep_attrs=True).ffill()
    if use_pressure:
        wetz = produce_wetz_dsea_from_ztd(pres=pres)
    else:
        p = dsea_gipsy_path / 'results-{}'.format('GPT2')
        wetz = read_all_final_tdps_dsea(return_mean=True,
                                        dryz=False)
    pwv = produce_pwv_from_zwd_with_ts_tm_from_deserve(ts=ts, zwd=wetz)
    pwv.attrs['units'] = 'mm'
    pwv.attrs['long_name'] = 'precipitable water vapor'
    pwv.name = 'pwv'
    pwv.attrs['action'] = 'corrected wetz using surface pressure and ts-tm from radiosonde'
    if savepath is not None:
        if ims_station is not None:
            filename = 'DSEA_PWV_{}.nc'.format(ims_station)
            save_ncfile(pwv, savepath, filename)
    return pwv


def produce_wetz_dsea_from_ztd(path=dsea_gipsy_path,
                               ims_path=ims_path, pres=None,
                               solution='GPT2'):
    p = path / 'results-{}'.format(solution)
    wetz = read_all_final_tdps_dsea(path=p, return_mean=True,
                                    dryz=False)
    dryz = get_dryz_from_one_station(p)
    ztd = wetz + dryz.resample(time='5T').ffill()
    zhd = calculate_zenith_hydrostatic_delay_dsea(ims_path, pres)
    wetz = ztd - zhd
    return wetz


def calculate_zenith_hydrostatic_delay_dsea(ims_path=ims_path, pres=None):
    from PW_stations import calculate_ZHD
    from PW_stations import produce_geo_gnss_solved_stations
    from aux_gps import xr_reindex_with_date_range
    import xarray as xr
    if pres is None:
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


def produce_pwv_from_zwd_with_ts_tm_from_deserve(path=work_yuval,
                                                 des_path=des_path, ts=None,
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
    if ts is None:
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


def compare_gnss_dsea_with_wrf_radiometer(des_path=des_path, work_path=work_yuval,
                                          dsea_da=None, plot=True):
    import xarray as xr
    from aux_gps import dim_intersection
    import pandas as pd
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
    iwv_rad = read_radiometers(des_path)
    rad79 = iwv_rad.sel(time=slice('2014-08-07', '2014-08-08'))
    rad79.name = 'radiometer'
    rad1517 = iwv_rad.sel(time=slice('2014-08-15', '2014-08-16'))
    rad1517.name = 'radiometer'

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        gnss79.to_dataframe()['gnss'].plot(ax=axes[1], lw=2)
        wrf79.to_dataframe()['wrf'].plot(ax=axes[1], lw=2)
        rad79.to_dataframe()['radiometer'].plot(ax=axes[1], lw=2)
        gnss1517.to_dataframe()['gnss'].plot(ax=axes[0], lw=2)
        wrf1517.to_dataframe()['wrf'].plot(ax=axes[0], lw=2)
        rad1517.to_dataframe()['radiometer'].plot(ax=axes[0], lw=2)
        axes[0].set_ylim(0, 50)
        axes[0].set_ylabel('PWV [mm]')
        axes[0].grid()
        axes[0].legend()
        # axes[0].set_xlim(pd.to_datetime('2014-08-16T13:00:00'), pd.to_datetime('2014-08-16T19:00:00'))
        axes[1].set_ylim(0, 50)
        # axes[1].set_xlim(pd.to_datetime('2014-08-08T13:00:00'), pd.to_datetime('2014-08-08T19:00:00'))
        axes[1].set_ylabel('PWV [mm]')
        axes[1].grid()
        axes[1].legend()
        fig.suptitle('WRF vs.radiometer GNSS vs. PWV at DSEA station (31.037N, 35.369E), ~31 km south to massada st.')
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


def get_wrf_vars_with_a_specific_point(wrfile, point=[31.0306, 35.3919], name='SEDOM',
                                       savepath=None, ds=None):
    from aux_gps import get_nearest_lat_lon_for_xy
    import xarray as xr
    if ds is None:
        ds = xr.open_dataset(wrfile)
    print('looking for {} at wrf ().'.format(point, name))
    loc = get_nearest_lat_lon_for_xy(ds['XLAT'], ds['XLONG'], point)
    print(loc)
    ds = ds.isel(south_north=loc[0][0], west_east=loc[0][1])
    ds = ds.sortby('Time')
    ds.load()
    return ds


def concat_wrf_vars_same_date(path=des_path, date='2014-08-16'):
    import xarray as xr
    from aux_gps import path_glob
    from aux_gps import save_ncfile
    files = path_glob(path, 'wrfout_*_{}_*_*.nc'.format(date))
    dsl = [xr.open_dataset(x) for x in files]
    ds = xr.merge(dsl)
    varnames = '_'.join(sorted([x for x in ds]))
    name = files[0].as_posix().split('/')[-1].split('.')[0].split('_')[0:-1]
    filename = '_'.join(name) + '_{}'.format(varnames) + '.nc'
    save_ncfile(ds, path, filename)
    return


def load_wrf_var_from_wrf_file_and_save(file, varname="rh2", savepath=None):
    """load one wrfvar from wrf file and save it to savepath"""
    from netCDF4 import Dataset
    import wrf
    nc = Dataset(file)
    from aux_gps import save_ncfile
    name = file.as_posix().split('/')[-1].split('.')[0]
    filename = '{}_{}.nc'.format(name, varname)
    wrfvar = wrf.getvar(wrfin=nc, varname=varname, timeidx=wrf.ALL_TIMES)
    if savepath is not None:
        if wrfvar.attrs['projection'] is not None:
            wrfvar.attrs['projection'] = wrfvar.attrs['projection'].proj4()
        save_ncfile(wrfvar, savepath, filename)
    return wrfvar


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
    from netCDF4 import Dataset
    import wrf
    files = path_glob(path, 'wrfout_d04_*_GNSS.nc')
    dsl = [Dataset(file) for file in files]
    if var is not None:
        var_list = []
        for ds in dsl:
            # wrfin = ds._file_obj.ds
            wrfvar = wrf.getvar(wrfin=ds, varname=var, timeidx=wrf.ALL_TIMES)
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


def compare_WRF_GNSS_radiometer_pwv(path=des_path, work_path=work_yuval, plot=True):
    import xarray as xr
    import matplotlib.pyplot as plt
    import numpy as np
    from PW_from_gps_figures import plot_mean_with_fill_between_std
    # load GNSS dsea:
    # gnss = xr.open_dataset(
    #     work_path / 'GNSS_PW_thresh_50_homogenized.nc')['dsea']
    gnss = xr.load_dataset(work_path / 'DSEA_PWV_GNSS_2014-08.nc')['pwv-soi']
    # load WRF:
    wrf = xr.load_dataset(path / 'pwv_wrf_dsea_gnss_point_2014-08.nc')
    # load radiometer:
    radio = read_radiometers(path)
    ds = xr.Dataset()
    ds['WRF'] = wrf['pw']
    ds['GNSS'] = gnss
    ds['Radiometer'] = radio
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
        radioln = plot_mean_with_fill_between_std(ds['Radiometer'], grp='hour', mean_dim='time', ax=ax,
                                                  color='tab:green', marker='o', alpha=0.2)

        ax.xaxis.set_ticks(np.arange(0, 24, 1))
        ax.legend(wrfln+gnssln+radioln, ['WRF', 'GNSS', 'Radiometer'])
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


def read_radiometers(path=des_path):
    from aux_gps import path_glob
    import xarray as xr
    files = path_glob(des_path, 'KIT_HATPRO_*.nc')
    dsl = [xr.load_dataset(x) for x in files]
    ds = xr.concat(dsl, 'time')
    ds = ds.sortby('time')
    iwv = ds['iwv'].resample(time='5T', keep_attrs=True).mean(keep_attrs=True)
    return iwv


def read_kit_rs(path=des_path):
    """read kit_rs nc files, but they reach up to 12 kms while
    the original txt files for each profile goes up to 18 kms"""
    from aux_gps import path_glob
    import xarray as xr
    files = path_glob(des_path, 'KIT_RS_*.nc')
    dsl = [xr.load_dataset(x) for x in files]
    ds = xr.concat(dsl, 'time')
    ds['DewPoint'].attrs['units'] = 'degC'
    return ds


def calculate_pwv_from_kit_rs(path=des_path):
    """ for kit_rs that go up to 12kms"""
    from metpy.calc import precipitable_water
    import xarray as xr
    ds = read_kit_rs(path)
    dew = ds['DewPoint']
    P = ds['Pressure']
    times = []
    for dt in P.time:
        d = dew.sel(time=dt)
        p = P.sel(time=dt)
        pwv = precipitable_water(p, d).magnitude
        times.append(pwv)
    pwv_rs = xr.DataArray(times, dims=['time'])
    pwv_rs['time'] = P.time
    pwv_rs.name = 'pwv'
    pwv_rs.attrs['long_name'] = 'precipitable water'
    pwv_rs.attrs['units'] = 'mm'
    pwv_rs.attrs['action'] = 'proccesed by metpy.calc on KIT-RS data'
    return pwv_rs


def read_surface_pressure(path=des_path, dem_path=work_yuval/'AW3D30'):
    """taken from ein gedi spa 31.417313616189308, 35.378962961491474"""
    import pandas as pd
    from aux_gps import path_glob
    from aux_gps import get_unique_index
    import xarray as xr
    awd = xr.open_rasterio(dem_path / 'israel_dem.tif')
    awd = awd.squeeze(drop=True)
    alt = awd.sel(x=35.3789,y=31.4173,method='nearest').item()
    file = path_glob(path, 'EBS1_*_pressure.txt')[0]
    df = pd.read_csv(file)
    df['Time'] = pd.to_datetime(df['Time'], format='%d-%b-%Y %H:%M:%S')
    df = df.set_index('Time')
    df.index.name = 'time'
    da = df.to_xarray()['Press']
    da = get_unique_index(da)
    da.attrs['station_alt'] = alt
    da.attrs['lat'] = 31.4173
    da.attrs['lon'] = 35.3789
    return da


def wrap_xr_metpy_pw(dewpt, pressure, bottom=None, top=None, verbose=False,
                     cumulative=False):
    from metpy.calc import precipitable_water
    from metpy.units import units
    import numpy as np
    # try:
    #     T_unit = dewpt.attrs['units']
    #     assert T_unit == 'degC'
    # except KeyError:
    #     T_unit = 'degC'
    #     if verbose:
    #         print('assuming dewpoint units are degC...')
    dew_values = dewpt.values * units('K')
    try:
        P_unit = pressure.attrs['units']
        assert P_unit == 'hPa'
    except KeyError:
        P_unit = 'hPa'
        if verbose:
            print('assuming pressure units are hPa...')
    if top is not None:
        top_with_units = top * units(P_unit)
    else:
        top_with_units = None
    if bottom is not None:
        bottom_with_units = bottom * units(P_unit)
    else:
        bottom_with_units = None
    pressure_values = pressure.values * units(P_unit)
    if cumulative:
        pw_list = []
        # first value is nan:
        pw_list.append(np.nan)
        for pre_val in pressure_values[1:]:
            if np.isnan(pre_val):
                pw_list.append(np.nan)
                continue
            pw = precipitable_water(pressure_values, dew_values, bottom=None,
                                    top=pre_val)
            pw_units = pw.units.format_babel('~P')
            pw_list.append(pw.magnitude)
        pw = np.array(pw_list)
        return pw, pw_units
    else:
        pw = precipitable_water(pressure_values, dew_values,
                                bottom=bottom_with_units, top=top_with_units)
        pw_units = pw.units.format_babel('~P')
        return pw.magnitude, pw_units


def plot_line_from_dsea_opera_to_coast(dem_path=dem_path, work_path=work_yuval):
    from PW_from_gps_figures import plot_israel_map
    from PW_stations import produce_geo_gnss_solved_stations
    from ims_procedures import plot_closest_line_from_point_to_israeli_coast
    from shapely.geometry import Point
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from aux_gps import geo_annotate
    import xarray as xr
    # df = produce_geo_gnss_solved_stations(plot=False)
    gnss = xr.load_dataset(work_path / 'DSEA_PWV_GNSS_2014-08.nc')
    fig = plt.figure(figsize=(7, 15))
    ax = fig.add_subplot(projection=ccrs.PlateCarree())  # plt.subplot(122)
    # fig, ax = plt.subplots(projection=ccrs.PlateCarree())
    extent = [34.5, 35.75, 30.9, 31.89]
    ax.set_extent(extent)
    ax = plot_israel_map(ax=ax)
    cmap = plt.get_cmap('terrain', 41)
    dem = xr.open_dataarray(dem_path / 'israel_dem_250_500.nc')
    # dem = xr.open_dataarray(dem_path / 'israel_dem_500_1000.nc')
    dem = dem.sel(lat=slice(29.2, 32.5), lon=slice(34, 36.3))
    fg = dem.plot.imshow(ax=ax, alpha=0.5, cmap=cmap,
                         vmin=dem.min(), vmax=dem.max(), add_colorbar=False)
#    scale_bar(ax_map, 50)
    cbar_kwargs = {'fraction': 0.1, 'aspect': 50, 'pad': 0.03}
    cb = plt.colorbar(fg, **cbar_kwargs)
    cb.set_label(label='meters above sea level',
                 size=14, weight='normal')
    cb.ax.tick_params(labelsize=14)

    soi_point = Point(gnss['pwv-soi'].lon, gnss['pwv-soi'].lat)
    axis_point = Point(gnss['pwv-axis'].lon, gnss['pwv-axis'].lat)
    opera = Point(35.3725, 31.3177)
    ds1 = plot_closest_line_from_point_to_israeli_coast(soi_point, ax=ax, color='k')
    print('{} km of soi-point to coast.'.format(ds1))
    ax.plot(*opera.xy, marker='o', markersize=5, color='k')
    ax.plot(*soi_point.xy, marker='o', markersize=5, color='k')
    ax.plot(*axis_point.xy, marker='o', markersize=5, color='k')
    geo_annotate(ax, [soi_point.x], [soi_point.y],
                 ['GNSS-SOI ({:.0f} km)'.format(ds1)], xytext=(4, -6), fmt=None,
                 c='k', fw='bold', fs=14, colorupdown=False)
    ds3 = plot_closest_line_from_point_to_israeli_coast(axis_point, ax=ax, color='k')
    print('{} km of axis-point to coast.'.format(ds3))
    geo_annotate(ax, [axis_point.x], [axis_point.y],
                 ['GNSS-AXIS ({:.0f} km)'.format(ds3)], xytext=(4, -6), fmt=None,
                 c='k', fw='bold', fs=14, colorupdown=False)
    ds2 = plot_closest_line_from_point_to_israeli_coast(opera, ax=ax, color='k')
    print('{} km of OPERA to coast.'.format(ds2))
    geo_annotate(ax, [opera.x], [opera.y],
                 ['Opera-pt. ({:.0f} km)'.format(ds2)], xytext=(4, -6), fmt=None,
                 c='k', fw='bold', fs=14, colorupdown=False)
    ax.set_xticks([34, 35, 36])
    ax.set_yticks([29.5, 30, 30.5, 31, 31.5, 32, 32.5])
    ax.tick_params(top=True, bottom=True, left=True, right=True,
                   direction='out', labelsize=14)
    return ax
