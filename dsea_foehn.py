#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:20:40 2020

@author: shlomi
"""
from PW_stations import work_yuval
from PW_paths import savefig_path
des_path = work_yuval / 'deserve'


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