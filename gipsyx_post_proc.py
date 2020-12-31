#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:24:01 2019

@author: shlomi
"""


# MEmORY saving tips:
# 1   Avoid List Slicing:For example: for a = [0, 1, 2, 3, 4, 5],
#  a[1:4] allocates a new array [1, 2, 3]
# try to use function parameters or separate variables to track indices
#  instead of slicing or altering a list.

# 2Use List Indexing Sparingly
# try to use “for item in array” for loops over arrays, before using
#  “for index in range(len(array))”

# 3 String Concatenation
# Instead of “+” for string concatenation, USE ''.join(iterable_object) or
#  .format or % ! This makes a huge impact when the program deals with more
#  data and/or longer strings.
#4. Use Iterators and Generators
# 5. Make use of libraries when possible
def check_path(path):
    import os
    from pathlib import Path
    path = str(path)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            logger.error("Creation of the directory %s failed" % path)
        else:
            logger.info("Successfully created the directory %s" % path)
    return Path(path)


def check_abs_int(num):
    try:
        num = abs(int(num))
    except ValueError:
        print('{} needs to be a natural number (>0 and int)'.format(num))
    return num


def plot_gipsy_field(ds, fields='WetZ', with_error=False):
    import numpy as np
    import matplotlib.pyplot as plt
    if isinstance(fields, str):
        fields = [fields]
    try:
        station = ds['WetZ'].attrs['station']
    except KeyError:
        station = 'No Name'
    if fields is None:
        all_fields = sorted(list(set([x.split('_')[0] for x in ds.data_vars])))
    elif fields is not None and isinstance(fields, list):
        all_fields = sorted(fields)
    if len(all_fields) == 1:
        da = ds[all_fields[0]]
        error = da.name + '_error'
        ax = da.plot(figsize=(20, 4), color='b')[0].axes
        if with_error:
            ax.fill_between(da.time.values, da.values - ds[error].values,
                            da.values + ds[error].values,
                            where=np.isfinite(da.values),
                            alpha=0.5)
        ax.grid()
        ax.set_title('GPS station: {}'.format(station))
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        return ax
    else:
        da = ds[all_fields].to_array('var')
        fg = da.plot(row='var', sharex=True, sharey=False, figsize=(20, 15),
                     hue='var',color='k')
        for i, (ax, field) in enumerate(zip(fg.axes.flatten(), all_fields)):
            if with_error:
                ax.fill_between(da.time.values,
                                da.sel(var=field).values - ds[field+'_error'].values,
                                da.sel(var=field).values + ds[field+'_error'].values,
                                where=np.isfinite(da.sel(var=field).values),
                                alpha=0.5)
            try:
                ax.set_ylabel('[' + ds[field].attrs['units'] + ']')
            except IndexError:
                pass
            ax.lines[0].set_color('C{}'.format(i))
            ax.grid()
        fg.fig.suptitle('GPS station: {}'.format(station))
        fg.fig.subplots_adjust(left=0.1, top=0.93)
    return fg


def save_all_resampled_versions_gipsyx(load_path, sample):
    for key in sample.keys():
        save_resampled_versions_gispyx_results(load_path, sample, key)
    return


def save_resampled_versions_gispyx_results(load_path, sample,
                                           sample_rate='1H'):
    from aux_gps import path_glob
    import xarray as xr
    import logging
    """resample gipsyx results nc files and save them.options for
    sample_rate are in sample dict"""
    logger = logging.getLogger('gipsyx_post_proccesser')
    path = path_glob(load_path, '*.nc')[0]
    station = path.as_posix().split('/')[-1].split('_')[0]
    # path = GNSS / station / 'gipsyx_solutions'
    glob = '{}_PPP*.nc'.format(station.upper())
    try:
        file = path_glob(load_path, glob_str=glob)[0]
    except FileNotFoundError:
        logger.warning(
            'did not find {} in gipsyx_solutions dir, skipping...'.format(station))
        return
    filename = file.as_posix().split('/')[-1].split('.')[0]
    years_str = filename.split('_')[-1]
    ds = xr.open_dataset(file)
    time_dim = list(set(ds.dims))[0]
    logger.info('resampaling {} to {}'.format(station, sample[sample_rate]))
    years = [str(x) for x in sorted(list(set(ds[time_dim].dt.year.values)))]
    if sample_rate == '1H' or sample_rate == '3H':
        dsr_list = []
        for year in years:
            logger.info('resampling {} of year {}'.format(sample_rate, year))
            dsr = ds.sel({time_dim: year}).resample(
                {time_dim: sample_rate}, keep_attrs=True, skipna=True).mean(keep_attrs=True)
            dsr_list.append(dsr)
        dsr = xr.concat(dsr_list, time_dim)
    else:
        dsr = ds.resample({time_dim: sample_rate},
                          keep_attrs=True,
                          skipna=True).mean(keep_attrs=True)
    new_filename = '_'.join([station.upper(), sample[sample_rate], 'PPP',
                             years_str])
    new_filename = new_filename + '.nc'
    logger.info('saving resmapled station {} to {}'.format(station, load_path))
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in dsr.data_vars}
    dsr.to_netcdf(load_path / new_filename, 'w', encoding=encoding)
    logger.info('Done resampling!')
    return


def read_gipsyx_all_yearly_files(load_path, savepath=None, iqr_k=3.0,
                                 plot=False):
    """read, stitch and clean all yearly post proccessed ppp gipsyx solutions
    and concat them to a multiple fields time-series dataset"""
    from aux_gps import path_glob
    import xarray as xr
    from aux_gps import get_unique_index
    from aux_gps import dim_intersection
    import pandas as pd
    from aux_gps import filter_nan_errors
    from aux_gps import keep_iqr
    from aux_gps import xr_reindex_with_date_range
    from aux_gps import transform_ds_to_lat_lon_alt
    import logging

    def stitch_yearly_files(ds_list):
        """input is multiple field yearly dataset list and output is the same
        but with stitched discontinuieties"""
        fields = [x for x in ds_list[0].data_vars]
        for i, dss in enumerate(ds_list):
            if i == len(ds_list) - 1:
                break
            first_year = int(ds_list[i].time.dt.year.median().item())
            second_year = int(ds_list[i+1].time.dt.year.median().item())
            first_ds = ds_list[i].sel(time=slice('{}-12-31T18:00'.format(first_year),
                                      str(second_year)))
            second_ds = ds_list[i+1].sel(time=slice(str(first_year),
                                                    '{}-01-01T06:00'.format(second_year)))
            if dim_intersection([first_ds, second_ds], 'time') is None:
                logger.warning('skipping stitching years {} and {}...'.format(first_year, second_year))
                continue
            else:
                logger.info('stitching years {} and {}'.format(first_year, second_year))
            time = xr.concat([first_ds.time, second_ds.time], 'time')
            time = pd.to_datetime(get_unique_index(time).values)
            st_list = []
            for field in fields:
                df = first_ds[field].to_dataframe()
                df.columns = ['first']
                df = df.reindex(time)
                df['second'] = second_ds[field].to_dataframe()
                if field in ['X', 'Y', 'Z']:
                    method = 'simple_mean'
                elif field in ['GradNorth', 'GradEast', 'WetZ']:
                    method = 'smooth_mean'
                elif 'error' in field:
                    method = 'error_mean'
                dfs = stitch_two_cols(df, method=method)['stitched_signal']
                dfs.index.name = 'time'
                st = dfs.to_xarray()
                st.name = field
                st_list.append(st)
            # merge to all fields:
            st_ds = xr.merge(st_list)
            # replace stitched values to first ds and second ds:
            first_time = dim_intersection([ds_list[i], st_ds])
            vals_rpl = st_ds.sel(time=first_time)
            for field in ds_list[i].data_vars:
                ds_list[i][field].loc[{'time': first_time}] = vals_rpl[field]
            second_time = dim_intersection([ds_list[i+1], st_ds])
            vals_rpl = st_ds.sel(time=second_time)
            for field in ds_list[i+1].data_vars:
                ds_list[i+1][field].loc[{'time': second_time}] = vals_rpl[field]
        return ds_list
    logger = logging.getLogger('gipsyx_post_proccesser')
    files = sorted(path_glob(load_path, '*.nc'))
    ds_list = []
    for file in files:
        filename = file.as_posix().split('/')[-1]
        station = file.as_posix().split('/')[-1].split('_')[0]
        if 'ppp_post' not in filename:
            continue
        logger.info('reading {}'.format(filename))
        dss = xr.open_dataset(file)
        ds_list.append(dss)
    # now loop over ds_list and stitch yearly discontinuities:
    ds_list = stitch_yearly_files(ds_list)
    logger.info('merging all years...')
    ds = xr.merge(ds_list)
    logger.info('fixing meta-data...')
    for da in ds.data_vars:
        old_keys = [x for x in ds[da].attrs.keys()]
        vals = [x for x in ds[da].attrs.values()]
        new_keys = [x.split('>')[-1] for x in old_keys]
        ds[da].attrs = dict(zip(new_keys, vals))
        if 'desc' in ds[da].attrs.keys():
            ds[da].attrs['full_name'] = ds[da].attrs.pop('desc')
    logger.info('dropping duplicates time stamps...')
    ds = get_unique_index(ds)
    # clean with IQR all fields:
    logger.info('removing outliers with IQR of {}...'.format(iqr_k))
    ds = keep_iqr(ds, dim='time', qlow=0.25, qhigh=0.75, k=iqr_k)
    # filter the fields based on their errors not being NaNs:
    logger.info('filtering out fields if their errors are NaN...')
    ds = filter_nan_errors(ds, error_str='_error', dim='time')
    logger.info('transforming X, Y, Z coords to lat, lon and alt...')
    ds = transform_ds_to_lat_lon_alt(ds, ['X', 'Y', 'Z'], '_error', 'time')
    logger.info('reindexing fields with 5 mins frequency(i.e., inserting NaNs)')
    ds = xr_reindex_with_date_range(ds, 'time', '5min')
    ds.attrs['station'] = station
    if plot:
        plot_gipsy_field(ds, None)
    if savepath is not None:
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds.data_vars}
        ymin = ds.time.min().dt.year.item()
        ymax = ds.time.max().dt.year.item()
        new_filename = '{}_PPP_{}-{}.nc'.format(station, ymin, ymax)
        ds.to_netcdf(savepath / new_filename, 'w', encoding=encoding)
        logger.info('{} was saved to {}'.format(new_filename, savepath))
    logger.info('Done!')
    return ds


def post_procces_gipsyx_all_years(load_save_path, plot=False):
    from aux_gps import path_glob
    import logging
    logger = logging.getLogger('gipsyx_post_proccesser')
    files = sorted(path_glob(load_save_path, '*.nc'))
    for file in files:
        filename = file.as_posix().split('/')[-1]
        station = file.as_posix().split('/')[-1].split('_')[0]
        year = file.as_posix().split('/')[-1].split('_')[-1].split('.')[0]
        if 'raw' not in filename:
            continue
        new_filename = '{}_ppp_post_{}.nc'.format(station, year)
        if (load_save_path / new_filename).is_file():
            logger.warning('{} already exists in {}, skipping...'.format(new_filename,
                                                                         load_save_path))
            continue
        _ = post_procces_gipsyx_yearly_file(file, savepath=load_save_path,
                                            plot=False)
    return


def post_procces_gipsyx_yearly_file(path_file, savepath=None, plot=False,
                                    verbose=0):
    import xarray as xr
    # from aux_gps import get_unique_index
    import matplotlib.pyplot as plt
    import numpy as np
    import logging
    # import pandas as pd
#    from scipy import stats
#    import pandas as pd
#    import seaborn as sns
    logger = logging.getLogger('gipsyx_post_proccesser')
    station = path_file.as_posix().split('/')[-1].split('_')[0]
    year = path_file.as_posix().split('/')[-1].split('_')[-1].split('.')[0]
    logger.info('proccessing {} station in year: {}'.format(station, year))
    dss = xr.open_dataset(path_file)
    da_fs = []
    # attrs_list = []
    vars_list = list(set([x.split('-')[0] for x in dss.data_vars.keys()]))
    for field in vars_list:
        try:
            da_field = analyse_results_ds_one_station(dss, field, verbose=verbose)
        except ValueError as e:
            logger.warning('ValueError: {}, meaning only single 24hr day files in all {}'.format(e, year))
            return None
        da_year = replace_fields_in_ds(dss, da_field, field, verbose=verbose)
        da_fs.append(da_year)
        # attrs_list += [(x, y) for x, y in da_year.attrs.items()]
    # attrs = list(set(attrs_list))
    ds = xr.merge(da_fs)
    # convert attrs list after set to dict:
#    vars_attr = {}
#    for attr in attrs:
#        field = attr[0].split('>')[0]
#        val = attr[1]
#        if field == 'station':
#            ds.attrs['station'] = val
#            continue
#        attr_type = attr[0].split('>')[-1]
#        vars_attr[field] = {attr_type: val}
#    return vars_attr
#    # add attrs after conversion:
#    for field in ds.data_vars:
#        key = [x for x in vars_attr[field].keys()][0]
#        val = [x for x in vars_attr[field].values()][0]
#        ds[field].attrs[key] = val
#    df = get_unique_index(ds, 'time').to_dataframe()
#    st = df.index.min()
#    ed = df.index.max()
#    new_time = pd.date_range(st, ed, freq='5min')
#    df = df.reindex(new_time)
#    df.index.name = 'time'
#    ds = df.to_xarray()
    # filter outlies (zscore>3):
    # df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    # df = df[df > 0]
    # ds = df.to_xarray()
#    ds = get_unique_index(ds, 'time')
    if plot:
        fields = [x for x in ds.data_vars if 'error' not in x]
        desc = [ds[x].attrs[x+'>desc'] for x in fields]
        units = [ds[x].attrs[x+'>units'] for x in fields]
        fig, axes = plt.subplots(len(fields), 1, figsize=(20, 15), sharex=True)
        df = ds.to_dataframe()
        for ax, field, name, unit in zip(axes.flatten(), fields, desc, units):
            df[field].plot(ax=ax, style='.', linewidth=0., color='b')
            ax.fill_between(df.index,
                            df[field].values - df[field + '_error'].values,
                            df[field].values + df[field + '_error'].values,
                            where=np.isfinite(df['WetZ'].values),
                            alpha=0.5)
            ax.grid()
            ax.set_title(name)
            ax.set_ylabel(unit)
        fig.tight_layout()
    if savepath is not None:
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds.data_vars}
        new_filename = '{}_ppp_post_{}.nc'.format(station, year)
        ds.to_netcdf(savepath / new_filename, 'w', encoding=encoding)
        logger.info('{} was saved to {}'.format(new_filename, savepath))
    return ds


def replace_fields_in_ds(dss, da_repl, field='WetZ', verbose=None):
    """replaces dss overlapping field(and then some) with the stiched signal
    fron da_repl. be carful with the choices for field"""
    from aux_gps import get_unique_index
    import xarray as xr
    import logging
    logger = logging.getLogger('gipsyx_post_proccesser')
    if verbose == 0:
        logger.info('replacing {} field.'.format(field))
    # choose the field from the bigger dss:
    nums = sorted(list(set([int(x.split('-')[1])
                            for x in dss if x.split('-')[0] == field])))
    ds = dss[['{}-{}'.format(field, i) for i in nums]]
    da_list = []
    for i, _ in enumerate(ds):
        if i == len(ds) - 1:
            break
        first = ds['{}-{}'.format(field, i)]
        time0 = list(set(first.dims))[0]
        second = ds['{}-{}'.format(field, i+1)]
        time1 = list(set(second.dims))[0]
        try:
            min_time = first.dropna(time0)[time0].min()
            max_time = second.dropna(time1)[time1].max()
        except ValueError:
            if verbose == 1:
                logger.warning('item {}, {} - {} is lonely'.format(field, i, i+1))
            continue
        try:
            da = da_repl.sel(time=slice(min_time, max_time))
        except KeyError:
            if verbose == 1:
                logger.warning('item {}, {} - {} is lonely'.format(field, i, i+1))
            continue
        if verbose == 1:
            logger.info('proccesing {} and {}'.format(first.name, second.name))
        # utime = dim_union([first, second], 'time')
        first_time = set(first.dropna(time0)[time0].values).difference(set(da.time.values))
        second_time = set(second.dropna(time1)[time1].values).difference(set(da.time.values))
        first = first.sel({time0: list(first_time)})
        second = second.sel({time1: list(second_time)})
        first = first.rename({time0: 'time'})
        second = second.rename({time1: 'time'})
        da_list.append(xr.concat([first, da, second], 'time'))
    da_final = xr.concat(da_list, 'time')
    da_final = da_final.sortby('time')
    da_final.name = field
    da_final.attrs = da_repl.attrs
    da_final = get_unique_index(da_final, 'time')
    return da_final


def stitch_two_cols(df, window=25, order=3, method='smooth_mean',
                    cols=None):
    """Use smoothing with savgol filter on the mean of two overlapping
    signals. assume that df columns are : A, B, if cols=None
    means take A, B to be the first two cols of df"""
    from scipy.signal import savgol_filter
    import numpy as np
    if cols is None:
        cols = df.columns.values[0:2]
    if method == 'smooth_mean':
        df['Mean'] = df[cols].mean(axis=1)
        sav = savgol_filter(df.Mean.values, window, order)
        df['stitched_signal'] = sav
    elif method == 'error_mean':
        df['stitched_signal'] = np.sqrt(df[cols[0]].fillna(0)**2 +
                                        df[cols[1]].fillna(0)**2)
    elif method == 'simple_mean':
        df['stitched_signal'] = df[cols].mean(axis=1)
    return df


def analyse_results_ds_one_station(dss, field='WetZ', verbose=None,
                                   plot=False):
    """analyse and find an overlapping signal to fields 'WetZ' or 'WetZ_error'
    in dss"""
    # algorithm for zwd stitching of 30hrs gipsyx runs:
    # just take the mean of the two overlapping signals
    # and then smooth is with savgol_filter using 3 hours more data in each
    # direction...
    import matplotlib.pyplot as plt
    import pandas as pd
    import logging

    def select_two_ds_from_gipsyx_results(ds, names=['WetZ_0', 'WetZ_1'],
                                          hours_offset=None):
        """selects two dataarrays from the raw gipsyx results dataset"""
        import pandas as pd
        import xarray as xr
        time0 = list(set(ds[names[0]].dims))[0]
        time1 = list(set(ds[names[1]].dims))[0]
        time = list(set(ds[names[0]][time0].values).intersection(set(ds[names[1]][time1].values)))
        # time = dim_intersection([ds[names[0]], ds[names[1]]], dim='time')
        if not time:
            return None
        time = sorted(pd.to_datetime(time))
        if hours_offset is not None:
            # freq = pd.infer_freq(time)
            start = time[0] - pd.DateOffset(hours=hours_offset)
            end = time[-1] + pd.DateOffset(hours=hours_offset)
            # time = pd.date_range(start, end, freq=freq)
            first = ds[names[0]].sel({time0: slice(start, end)})
            second = ds[names[1]].sel({time1: slice(start, end)})
        else:
            first = ds[names[0]].sel({time0: time})
            second = ds[names[1]].sel({time1: time})
        first = first.rename({time0: 'time'})
        second = second.rename({time1: 'time'})
        two = xr.Dataset()
        two[first.name] = first
        two[second.name] = second
        df = two.to_dataframe()
        return df
    logger = logging.getLogger('gipsyx_post_proccesser')
    if verbose == 0:
        logger.info('analysing {} field.'.format(field))
    # first, group different vars for different stitching schemes:
    to_smooth = ['GradEast', 'GradNorth', 'WetZ']
    to_simple_mean = ['X', 'Y', 'Z']
    to_error_mean = [x + '_error' for x in to_smooth] + [x + '_error' for x in
                                                         to_simple_mean]
    # second, select the field to work on:
    nums = sorted(list(set([int(x.split('-')[1])
                            for x in dss if x.split('-')[0] == field])))
    ds = dss[['{}-{}'.format(field, i) for i in nums]]
    df_list = []
    for i, _ in enumerate(ds):
        if i == len(ds) - 1:
            break
        first = ds['{}-{}'.format(field, i)]
        second = ds['{}-{}'.format(field, i + 1)]
        if verbose == 1:
            print('proccesing {} and {}'.format(first.name, second.name))
        # 3 hours addition to each side:
        df = select_two_ds_from_gipsyx_results(ds, [first.name, second.name],
                                               3)
        if df is not None:
            if field in to_smooth:
                wn = 25
                order = 3
                stitched = stitch_two_cols(df, wn, order, method='smooth_mean')
                action = 'stitched and replaced daily discontinuities '\
                    'with smooth(savgol filter, window:{}, order:{}) mean'.format(wn, order)
            elif field in to_simple_mean:
                stitched = stitch_two_cols(df, method='simple_mean')
                action = 'stitched and replaced daily discontinuities '\
                    'with simple mean'
            elif field in to_error_mean:
                stitched = stitch_two_cols(df, method='error_mean')
                action = 'stitched and replaced daily discontinuities '\
                    'with error mean (sqrt(errorA^2 + errorB^2))'
            df_list.append(stitched)
            # df_list.append(find_cross_points(df, None))
        elif df is None:
            if verbose:
                logger.warning('skipping {} and {}...'.format(first.name, second.name))
    da = pd.concat([x['stitched_signal'] for x in df_list]).to_xarray()
    attrs_list = [(x, y)
                  for x, y in dss.attrs.items() if field == x.split('>')[0]]
    attrs_list.append(('{}>action'.format(field), action))
    for items in attrs_list:
        da.attrs[items[0]] = items[1]
    da.attrs['station'] = dss.attrs['station']
    if plot:
        fig, ax = plt.subplots(figsize=(16, 5))
        da.plot.line(marker='.', linewidth=0., ax=ax, color='k')
        for i, ppp in enumerate(ds):
            ds['{}-{}'.format(field, i)].plot(ax=ax)
        units = dss.attrs['{}>units'.format(field)]
        sta = da.attrs['station']
        desc = da.attrs['{}>desc'.format(field)]
        ax.set_ylabel('{} [{}]'.format(field, units))
        ax.set_xlabel('')
        fig.suptitle('30 hours stitched {} for GNSS station {}'.format(desc, sta), fontweight='bold')
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        ax.grid()
#    dfs = []
#    for df in df_list:
#        # check if there is an offset:
#        A = df.columns.values[0]
#        B = df.columns.values[1]
#        if all([x is None for x in df.Cross]):
#            offset = df.Diff.median()
#            df['{}_new'.format(B)] = df[B] + offset
#            dfs.append(df)
    return da


#def gipsyx_rnxedit_errors(df1, savepath=None):
#    """get the df output of gipsyx_runs_error_analysis and map out the reciever
#    error analysis using regex and print the output"""
#    df = df1.copy()
#    error_col = df.columns.values.item()
#    df['receiver'] = df[error_col].str.findall(r"'(.*?)'")
#    df['receiver'] = [x[0] if x is not None else None for x in df['receiver']]
#    text = [df.loc[i, error_col]
#            for i in df.index if df.loc[i, error_col] is not None][0]
#    station = error_col.split('_')[0]
#    if savepath is not None:
#        filename = station + '_rnxEdit_errors.txt'
#        with open(savepath / filename, 'a') as f:
#            f.write("%s\n" % text)
#            f.write("dataframe: \n")
#            df['receiver'].to_csv(f)
#            print('{} was saved to {}'.format(filename, savepath))
#    return df


#def gipsyx_runs_error_analysis(path):
#    from collections import Counter
#    from aux_gps import get_timedate_and_station_code_from_rinex
#
#    def further_filter(counter):
#        return c
#
#    def find_errors(content_list, name):
#        if len(content_list) <= 1:
#            return None
#        elif len(content_list) > 1:
#            keys = [x for x in content_list if 'KeyError' in x]
#            vals = [x for x in content_list if 'ValueError' in x]
#            excpt = [x for x in content_list if 'Exception' in x]
#            err = [x for x in content_list if 'Error' in x]
#            errors = keys + vals + excpt + err
#        if not errors:
#            dt, _ = get_timedate_and_station_code_from_rinex(name)
#            print('found new error on {} ({})'.format(name,  dt.strftime('%Y-%m-%d')))
#        return errors
#    edict = {}
#    good = 0
#    bad = 0
#    for file in path.glob('*.err'):
#        filename = file.as_posix().split('/')[-1][0:12]
#        if good == 0 and bad == 0:
#            print('running error analysis for station {}'.format(filename[0:4]))
#        with open(file) as f:
#            content = f.readlines()
#            # you may also want to remove whitespace characters like `\n` at
#            # the end of each line
#            content = [x.strip() for x in content]
#            errors = find_errors(content, filename)
#            if errors is not None:
#                edict[filename] = list(set(errors))
#                bad += 1
#            else:
#                good += 1
#    g = [get_timedate_and_station_code_from_rinex(x) for x in edict.keys()]
#    dts = [x[0] for x in g]
#    station = [x[1] for x in g][0]
#    df = pd.DataFrame(data=edict.values(), index=dts)
#    df = df.sort_index()
#    len_er = len(df.columns)
#    df.columns = [station + '_errors_' + str(i) for i in range(len_er)]
#    flat_list = [item for sublist in edict.values() for item in sublist]
#    counted_errors = Counter(flat_list)
#    print(
#        'total files: {}, good runs: {}, bad runs: {}'.format(
#            good +
#            bad,
#            good,
#            bad))
#    errors_sorted = sorted(counted_errors.items(), key=lambda x: x[1],
#                           reverse=True)
#    return errors_sorted, df


def save_yearly_gipsyx_results(path, savepath):
    """call read one station for each year and save the results, then
    concat and save to a bigger raw file, can add postproccess function"""
    from aux_gps import path_glob
    from aux_gps import get_timedate_and_station_code_from_rinex
    import logging
    import pandas as pd
    global cnt
    global tot
    logger = logging.getLogger('gipsyx_post_proccesser')
    files = path_glob(path, '*.tdp')
    tot = len(files)
    est_time_per_single_run = 0.3  # seconds
    logger.info('found {} _smoothFinal tdp files in {} to process.'.format(tot, path))
    dtt = pd.to_timedelta(est_time_per_single_run, unit='s') * tot
    extra_dtt = pd.to_timedelta(0.4, unit='s') * tot
    resample_dtt = pd.to_timedelta(0.75, unit='s') * tot
    dtt += extra_dtt
    dtt += resample_dtt
    logger.info('estimated time to completion of run: {}'.format(dtt))
    logger.info('check again in {}'.format(pd.Timestamp.now() + dtt))
    rfns = [x.as_posix().split('/')[-1][0:12] for x in files]
    dts = [get_timedate_and_station_code_from_rinex(rfn, just_dt=True) for
           rfn in rfns]
    _, station = get_timedate_and_station_code_from_rinex(rfns[0])
    years = list(set([dt.year for dt in dts]))
    cnt = {'succ': 0, 'failed': 0}
    for year in sorted(years):
        filename = '{}_ppp_raw_{}.nc'.format(station, year)
        if (savepath / filename).is_file():
            logger.warning('{} already in {}, skipping...'.format(filename, savepath))
            continue
        _, _ = read_one_station_gipsyx_results(path, savepath, year)
    total = cnt['failed'] + cnt['succ']
    logger.info('Total files: {}, success: {}, failed: {}'.format(
            total, cnt['succ'], cnt['failed']))
    return


def read_one_station_gipsyx_results(path, savepath=None,
                                    year=None):
    """read one station (all years) consisting of many tdp files"""
    import xarray as xr
    from aux_gps import get_timedate_and_station_code_from_rinex
    from aux_gps import path_glob
    import logging
    logger = logging.getLogger('gipsyx_post_proccesser')
    if year is not None:
        year = int(year)
        logger.info('getting tdp files from year {}'.format(year))
    df_list = []
    errors = []
    dts = []
    # logger.info('reading folder:{}'.format(path))
    files = path_glob(path, '*.tdp')
    for tdp_file in files:
        rfn = tdp_file.as_posix().split('/')[-1][0:12]
        dt, station = get_timedate_and_station_code_from_rinex(rfn)
        if year is not None:
            if dt.year != year:
                continue
            else:
                logger.info(
                        'processing {} ({}, {}/{})'.format(
                                rfn,
                                dt.strftime('%Y-%m-%d'), cnt['succ'] + cnt['failed'], tot))
                try:
                    df, meta = process_one_day_gipsyx_output(tdp_file)
                    dts.append(df.index[0])
                    cnt['succ'] += 1
                except TypeError:
                    logger.error('problem reading {}, appending to errors...'.format(rfn))
                    errors.append(rfn)
                    cnt['failed'] += 1
                    continue
                df_list.append(df)
        elif year is None:
            try:
                df, meta = process_one_day_gipsyx_output(tdp_file)
                dts.append(df.index[0])
                cnt['succ'] += 1
            except TypeError:
                logger.error('problem reading {}, appending to errors...'.format(rfn))
                errors.append(rfn)
                cnt['failed'] += 1
                continue
            df_list.append(df)
    # sort by first dates of each df:
    df_dict = dict(zip(dts, df_list))
    df_list = []
    for key in sorted(df_dict):
        df_list.append(df_dict[key])
    dss = [df.to_xarray() for df in df_list]
    dss_new = []
    for i, ds in enumerate(dss):
        keys_to_rename = [x for x in ds.data_vars.keys()]
        keys_to_rename.append('time')
        values_to_rename = [x + '-{}'.format(i) for x in keys_to_rename]
        dict_to_rename = dict(zip(keys_to_rename, values_to_rename))
        dss_new.append(ds.rename(dict_to_rename))
    ds = xr.merge(dss_new)
    ds.attrs['station'] = station
    for key, val in meta['units'].items():
        ds.attrs[key + '>units'] = val
    for key, val in meta['desc'].items():
        ds.attrs[key + '>desc'] = val
    if savepath is not None:
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds.data_vars}
        filename = '{}_ppp_raw_{}.nc'.format(station, year)
        ds.to_netcdf(savepath / filename, 'w', encoding=encoding)
        logger.info('{} was saved to {}'.format(filename, savepath))
    return ds, errors


def read_tropnominal_tdp_file(file, keys=['DryZ'], plot=False):
    import pandas as pd
    import matplotlib.pyplot as plt
    df_raw = pd.read_csv(file, header=None, delim_whitespace=True)
    # keys = ['WetZ', 'DryZ']
    # get all the vars from smoothFinal.tdp file and put it in a df_list:
    df_list = [df_raw[df_raw.iloc[:, -1].str.contains(x)] for x in keys]
    # make sure that all keys in df have the same length:
    assert len(set([len(x) for x in df_list])) == 1
    # translate the seconds col to datetime:
    seconds = df_list[0].iloc[:, 0]
    dt = pd.to_datetime('2000-01-01T12:00:00')
    time = dt + pd.to_timedelta(seconds, unit='sec')
    # build a new df that contains all the vars(from keys):
    ppp = pd.DataFrame(index=time)
    ppp.index.name = 'time'
    for i, df in enumerate(df_list):
        df.columns = ['seconds', 'to_drop', keys[i], keys[i] + '_error',
                      'meta']
        ppp[keys[i]] = df[keys[i]].values
        ppp[keys[i] + '_error'] = df[keys[i] + '_error'].values
    # desc = ['Zenith Wet Delay', 'Zenith Dry Delay']
    # units = ['cm', 'cm']
    # fields = ['WetZ', 'DryZ']
    # units_dict = dict(zip(fields, units))
    # desc_dict = dict(zip(fields, desc))
    # meta = {'units': units_dict, 'desc': desc_dict}
    # convert tropospheric products to cm, rest stay in meters:
    # trop_cols = ppp.columns.values[0:4]
    # ppp[trop_cols] = ppp[trop_cols].mul(100.0)
    ppp = ppp.mul(100.0)
    return ppp


def process_one_day_gipsyx_output(path_and_file, dryz=False, plot=False):
    # path_and_file = work_yuval / 'smoothFinal.tdp'
    import pandas as pd
    # import pyproj
    import matplotlib.pyplot as plt
    # from aux_gps import get_latlonalt_error_from_geocent_error
    df_raw = pd.read_csv(path_and_file, header=None, delim_whitespace=True)
    # get all the vars from smoothFinal.tdp file and put it in a df_list:
    if dryz:
        keys = ['DryZ', 'WetZ', 'GradNorth', 'GradEast', 'Pos.X', 'Pos.Y', 'Pos.Z']
    else:
        keys = ['WetZ', 'GradNorth', 'GradEast', 'Pos.X', 'Pos.Y', 'Pos.Z']
    df_list = [df_raw[df_raw.iloc[:, -1].str.contains(x)] for x in keys]
    # make sure that all keys in df have the same length:
    assert len(set([len(x) for x in df_list])) == 1
    # translate the seconds col to datetime:
    seconds = df_list[0].iloc[:, 0]
    dt = pd.to_datetime('2000-01-01T12:00:00')
    time = dt + pd.to_timedelta(seconds, unit='sec')
    # build a new df that contains all the vars(from keys):
    ppp = pd.DataFrame(index=time)
    ppp.index.name = 'time'
    for i, df in enumerate(df_list):
        df.columns = ['seconds', 'to_drop', keys[i], keys[i] + '_error',
                      'meta']
        ppp[keys[i]] = df[keys[i]].values
        ppp[keys[i] + '_error'] = df[keys[i] + '_error'].values
    # rename all the Pos. to nothing:
    ppp.columns = ppp.columns.str.replace('Pos.', '')
    if dryz:
        desc = ['Zenith Hydrostatic Delay', 'Zenith Wet Delay',
                'North Gradient of Zenith Wet Delay',
                'East Gradient of Zenith Wet Delay',
                'WGS84(geocentric) X coordinate',
                'WGS84(geocentric) Y coordinate', 'WGS84(geocentric) Z coordinate']
        units = ['cm', 'cm', 'cm/m', 'cm/m', 'm', 'm', 'm']
        fields = ['DryZ', 'WetZ', 'GradNorth', 'GradEast', 'X', 'Y', 'Z']
    else:
        desc = ['Zenith Wet Delay', 'North Gradient of Zenith Wet Delay',
                'East Gradient of Zenith Wet Delay',
                'WGS84(geocentric) X coordinate',
                'WGS84(geocentric) Y coordinate', 'WGS84(geocentric) Z coordinate']
        units = ['cm', 'cm/m', 'cm/m', 'm', 'm', 'm']
        fields = ['WetZ', 'GradNorth', 'GradEast', 'X', 'Y', 'Z']
    units_dict = dict(zip(fields, units))
    desc_dict = dict(zip(fields, desc))
    meta = {'units': units_dict, 'desc': desc_dict}
    # convert tropospheric products to cm, rest stay in meters:
    if dryz:
        trop_cols = ppp.columns.values[0:8]
    else:
        trop_cols = ppp.columns.values[0:6]
    ppp[trop_cols] = ppp[trop_cols].mul(100.0)
    if plot:
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

        for ax, field, name, unit in zip(axes.flatten(), fields, desc, units):
            ppp[field].plot(ax=ax, legend=True, color='k')
            ax.fill_between(ppp.index, ppp[field] - ppp[field + '_error'],
                            ppp[field] + ppp[field + '_error'], alpha=0.5)
            ax.grid()
            ax.set_title(name)
            ax.set_ylabel(unit)
    return ppp, meta


if __name__ == '__main__':
    """tdppath is where the gipsyx results are (tdp files).
    e.g., /rinex/tela/30hr/results. savepath is where the raw/final post
    proccessed results will be saved."""
    import argparse
    import sys
    from PW_paths import work_yuval
    from PW_paths import work_path
    from PW_paths import geo_path
    from PW_paths import cwd
    from aux_gps import configure_logger
    garner_path = work_yuval / 'garner'
    ims_path = work_yuval / 'IMS_T'
    gis_path = work_yuval / 'gis'
    sound_path = work_yuval / 'sounding'
    rinex_on_geo = geo_path / 'Work_Files/PW_yuval/rinex'
    logger = configure_logger('gipsyx_post_proccesser')
    parser = argparse.ArgumentParser(
        description='a command line tool for post proccessing PPP gipsyX results.')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--savepath',
        help="a full path to save the raw and final output files, e.g., /home/ziskin/Work_Files/PW_yuval/gipsyx_resolved/TELA",
        type=check_path)
    required.add_argument(
        '--tdppath',
        help="a full path to the tdp files path of the station, /home/ziskin/Work_Files/PW_yuval/rinex/tela/30hr/results",
        type=check_path)
    optional.add_argument('--iqr_k', help='InterQuartile Range multiplier parameter(e.g., 1.5), Defualt=3.0',
                          type=check_abs_int)

#    optional.add_argument(
#            '--rewrite',
#            dest='rewrite',
#            action='store_true',
#            help='overwrite files in prep/run mode')

    parser._action_groups.append(optional)  # added this line
#    parser.set_defaults(rewrite=False)
    args = parser.parse_args()
    if args.tdppath is None:
        print('tdppath is a required argument, run with -h...')
        sys.exit()
    station = args.tdppath.as_posix().split('/')[-4].upper()
    logger.info('Starting post proccessing {} station'.format(station))
    save_yearly_gipsyx_results(args.tdppath, args.savepath)
    post_procces_gipsyx_all_years(args.savepath, False)
    if args.iqr_k is None:
        iqr_k = 3.0
    else:
        iqr_k = args.iqr_k
    read_gipsyx_all_yearly_files(args.savepath, args.savepath, iqr_k, False)
    sample = {'1H': 'hourly', '3H': '3hourly', 'D': 'Daily', 'W': 'weekly',
              'MS': 'monthly'}
    save_all_resampled_versions_gipsyx(args.savepath, sample)
    logger.info('Done post proccessing station {}.'.format(station))
