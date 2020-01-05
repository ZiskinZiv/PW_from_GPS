#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 17:28:04 2020

@author: shlomi
"""
from PW_paths import work_yuval
tela_results_path = work_yuval / 'GNSS_stations/tela/rinex/30hr/results'
tela_solutions = work_yuval / 'GNSS_stations/tela/gipsyx_solutions'


def plot_figure_1(path=work_yuval):
    from aux_gps import gantt_chart
    import xarray as xr
    ds = xr.open_dataset(path / 'GNSS_PW.nc')
    just_pw = [x for x in ds if 'error' not in x]
    ds = ds[just_pw]
    title = 'RINEX files availability for the Israeli GNSS stations'
    ax = gantt_chart(ds, title)
    return ax


def plot_figure_2(path=tela_results_path, plot='WetZ'):
    from aux_gps import path_glob
    import matplotlib.pyplot as plt
    from gipsyx_post_proc import process_one_day_gipsyx_output
    filepath = path_glob(path, 'tela*_smoothFinal.tdp')[3]
    if plot is None:
        df, meta = process_one_day_gipsyx_output(filepath, True)
        return df, meta
    else:
        df, meta = process_one_day_gipsyx_output(filepath, False)
        if not isinstance(plot, str):
            raise ValueError('pls pick only one field to plot., e.g., WetZ')
    error_plot = '{}_error'.format(plot)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    desc = meta['desc'][plot]
    unit = meta['units'][plot]
    df[plot].plot(ax=ax, legend=True, color='k')
    ax.fill_between(df.index, df[plot] - df[error_plot],
                    df[plot] + df[error_plot], alpha=0.5)
    ax.grid()
    ax.set_title('{} from station TELA'.format(desc))
    ax.set_ylabel(unit)
    ax.set_xlabel('')
    ax.grid('on')
    fig.tight_layout()
    return ax


def plot_figure_3(path=tela_solutions, year=2004, field='WetZ',
                  middle_date='11-25', zooms=[10, 3, 0.5]):
    from gipsyx_post_proc import analyse_results_ds_one_station
    import xarray as xr
    import matplotlib.pyplot as plt
    import pandas as pd
    dss = xr.open_dataset(path / 'TELA_ppp_raw_{}.nc'.format(year))
    nums = sorted(list(set([int(x.split('-')[1])
                            for x in dss if x.split('-')[0] == field])))
    ds = dss[['{}-{}'.format(field, i) for i in nums]]
    da = analyse_results_ds_one_station(dss, field=field, plot=False)
    fig, axes = plt.subplots(ncols=1, nrows=3, sharex=False, figsize=(16, 10))
    for j, ax in enumerate(axes):
        start = pd.to_datetime('{}-{}'.format(year, middle_date)
                               ) - pd.Timedelta(zooms[j], unit='D')
        end = pd.to_datetime('{}-{}'.format(year, middle_date)
                             ) + pd.Timedelta(zooms[j], unit='D')
        daa = da.sel(time=slice(start, end))
        for i, ppp in enumerate(ds):
            ds['{}-{}'.format(field, i)].plot(ax=ax)
        daa.plot.line(marker='.', linewidth=0., ax=ax, color='k')
        axes[j].set_xlim(start, end)
        axes[j].set_ylim(daa.min() - 0.5, daa.max() + 0.5)
        try:
            axes[j - 1].axvline(x=start, color='r', alpha=0.7, linestyle='--')
            axes[j - 1].axvline(x=end, color='r', alpha=0.7, linestyle='--')
        except IndexError:
            pass
        units = ds.attrs['{}>units'.format(field)]
        sta = da.attrs['station']
        desc = da.attrs['{}>desc'.format(field)]
        ax.set_ylabel('{} [{}]'.format(field, units))
        ax.set_xlabel('')
        ax.grid()
    fig.suptitle(
        '30 hours stitched {} for GNSS station {}'.format(
            desc, sta), fontweight='bold')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    return axes

def plot_figure_4():
    from PW_stations import fit_ts_tm_produce_ipw_and_compare_TELA
    return fit_ts_tm_produce_ipw_and_compare_TELA()
