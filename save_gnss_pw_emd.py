#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:52:06 2019

@author: ziskin
"""


if __name__ == '__main__':
    from PW_paths import work_yuval
    import xarray as xr
    import pandas as pd
    from aux_gps import configure_logger
    from PyEMD import EMD
    logger = configure_logger(name='emd_saver')
    samples = ['monthly', 'hourly', '3hourly', 'daily', 'weekly', None]
    logger.info('Starting to calculate IMFs for GNSS PW:')
    for sample in samples:
        if sample is None:
            GNSS_pw = xr.open_dataset(work_yuval / 'GNSS_PW.nc')
            logger.info('proccessing 5 min sample rate.')
        else:
            GNSS_pw = xr.open_dataset(work_yuval / 'GNSS_{}_PW.nc'.format(sample))
            logger.info('proccessing {} sample rate.'.format(sample))
        only_pw = [x for x in GNSS_pw if 'error' not in x]
        da_list = []
        for pw in GNSS_pw[only_pw].data_vars.values():
            logger.info('calculating EMD for {} station.'.format(pw.name))
            emd = EMD()
            imfs = emd(pw.dropna('time').values)
            da = xr.DataArray(imfs, dims=['mode', 'time'])
            da['mode'] = range(imfs.shape[0])
            da.name = pw.name
            da['time'] = pw.dropna('time').time
            da.attrs['freq'] = pd.infer_freq(pw.time.values)
            da_list.append(da)
        ds = xr.merge(da_list)
        if sample is None:
            filename = 'GNSS_PW_IMFs.nc'
        else:
            filename = 'GNSS_{}_PW_IMFs.nc'.format(sample)
        logger.info('saving {} to {}'.format(filename, work_yuval))
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds}
        ds.to_netcdf(work_yuval / filename, 'w', encoding=encoding)
    logger.info('Done!')